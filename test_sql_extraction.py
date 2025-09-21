#!/usr/bin/env python3
"""
Quick test to verify SQL extraction and generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader
import torch

print("Testing SQL Generation with Trained Model")
print("="*50)

# Load model
model_wrapper = QwenModelWrapper(
    model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
    use_lora=True,
    load_in_8bit=False,
    device_map=None
)

print("Loading model...")
model_wrapper.load_model()

# Load checkpoint if available
checkpoint_path = Path("./checkpoints/sft/best.pt")
if checkpoint_path.exists():
    print("Loading trained checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "lora_state_dict" in checkpoint:
        model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])
    print("✅ Checkpoint loaded")

# Move to MPS if available
if torch.backends.mps.is_available():
    model_wrapper.model = model_wrapper.model.to("mps")
    print("✅ Using MPS acceleration")

# Test with simple prompt
query = "Show all active projects"
schema_loader = SchemaLoader()
schema_context = schema_loader.get_schema_context(query)

# Create a simpler, more direct prompt
simple_prompt = f"""<|im_start|>system
You are a SQL expert. Generate only the SQL query, no explanations.
<|im_end|>
<|im_start|>user
Database tables:
- PAC_MNT_PROJECTS: Project_ID, Project_Code, Project_Name, Status, Budget
- SRM_COMPANIES: Company_ID, Company_Code, Company_Name, Status

Query: {query}
<|im_end|>
<|im_start|>assistant
```sql"""

print(f"\nQuery: {query}")
print("\nGenerating SQL...")

# Generate with very focused parameters
gen_params = GenerationParams(
    temperature=0.01,
    max_new_tokens=100,
    do_sample=False
)

result = model_wrapper.generate(simple_prompt, gen_params)
print(f"\nRaw output:\n{result}")

# Extract SQL
sql = model_wrapper.extract_sql(result)
if sql:
    print(f"\nExtracted SQL:\n{sql}")
else:
    # Try manual extraction
    if "SELECT" in result.upper():
        for line in result.split('\n'):
            if "SELECT" in line.upper():
                sql = line
                break

    print(f"\nManually extracted: {sql if sql else 'No SQL found'}")

# Check if using correct tables
if "PAC_MNT_PROJECTS" in (sql if sql else result):
    print("\n✅ Using domain-specific table PAC_MNT_PROJECTS!")
else:
    print("\n❌ Not using domain tables")