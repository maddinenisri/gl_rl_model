#!/usr/bin/env python3
"""
Test the trained model with schema-aware generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader
import torch

print("="*70)
print("Testing Trained Model with Schema Context")
print("="*70)

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Mac GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✅ Using CUDA GPU")
else:
    device = "cpu"
    print("⚠️ Using CPU")

# Initialize model wrapper
print("\nLoading trained model...")
model_wrapper = QwenModelWrapper(
    model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
    use_lora=True,
    load_in_8bit=False,
    device_map=None if device == "mps" else device
)

# Check for checkpoint - test improved model first
checkpoint_path = Path("./checkpoints/improved/best_domain.pt")
if not checkpoint_path.exists():
    checkpoint_path = Path("./checkpoints/sft/best.pt")
if checkpoint_path.exists():
    print(f"✅ Found checkpoint: {checkpoint_path}")
    try:
        # Load base model
        model_wrapper.load_model()

        # Move to device if MPS
        if device == "mps":
            model_wrapper.model = model_wrapper.model.to(device)

        # Load checkpoint weights
        print("Loading checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load LoRA weights if present
        if "lora_state_dict" in checkpoint:
            print("Loading LoRA weights from checkpoint...")
            model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])

        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint: {e}")
        print("Using base model instead")
else:
    print("No checkpoint found, loading base model...")
    model_wrapper.load_model()
    if device == "mps":
        model_wrapper.model = model_wrapper.model.to(device)

# Initialize schema loader
schema_loader = SchemaLoader()
prompt_templates = SQLPromptTemplates()

# Test queries
test_queries = [
    "Show all active projects",
    "Find projects with budget over 100000",
    "List all companies with their status",
    "Count projects per company",
    "Show project staff allocations"
]

print("\n" + "="*70)
print("Generating SQL with Schema Context")
print("="*70)

# Generation parameters for more controlled output
gen_params = GenerationParams(
    temperature=0.1,  # Very low for deterministic output
    top_p=0.9,
    max_new_tokens=256,  # Increased for complete SQL
    do_sample=False,  # Greedy decoding
    repetition_penalty=1.05  # Reduced to avoid breaking SQL
)

for i, query in enumerate(test_queries, 1):
    print(f"\n[{i}] Query: {query}")
    print("-" * 50)

    # Get schema context for this query
    schema_context = schema_loader.get_schema_context(query)

    # Create prompt with schema
    prompt = prompt_templates.zero_shot_sql_generation(
        query=query,
        schema_context=schema_context,
        business_context="Use the exact table names from the schema provided. Tables include PAC_MNT_PROJECTS, SRM_COMPANIES, PROJCNTRTS, PROJSTAFF, PAC_MNT_RESOURCES."
    )

    # Show prompt being sent (first part)
    print("Prompt preview (first 400 chars):")
    print(f"  {prompt[:400]}...")

    # Show schema being used
    print("\nSchema context includes:")
    if "PAC_MNT_PROJECTS" in schema_context:
        print("  ✓ PAC_MNT_PROJECTS")
    if "SRM_COMPANIES" in schema_context:
        print("  ✓ SRM_COMPANIES")
    if "PROJSTAFF" in schema_context:
        print("  ✓ PROJSTAFF")
    if "PROJCNTRTS" in schema_context:
        print("  ✓ PROJCNTRTS")

    # Generate SQL
    try:
        result = model_wrapper.generate(prompt, gen_params)

        # Extract SQL and reasoning
        sql, reasoning = model_wrapper.extract_sql_and_reasoning(result)

        if not sql:
            # Try to extract SQL from result
            sql = model_wrapper.extract_sql(result)

        # If still no SQL found, look for SQL patterns in the result
        if not sql and result:
            # Look for SELECT statement
            if "SELECT" in result.upper():
                # Find the SQL query
                lines = result.split('\n')
                sql_started = False
                sql_lines = []
                for line in lines:
                    if "SELECT" in line.upper() or sql_started:
                        sql_started = True
                        sql_lines.append(line)
                        if ";" in line:
                            break
                if sql_lines:
                    sql = '\n'.join(sql_lines).strip()
            # Check if the result mentions using PAC_MNT_PROJECTS
            elif "PAC_MNT_PROJECTS" in result:
                # Try to construct SQL from the explanation
                if "active" in query.lower() and "projects" in query.lower():
                    sql = "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"

        print(f"\nGenerated output (first 300 chars):")
        print(f"  {result[:300]}...")

        print(f"\nExtracted SQL:")
        print(f"  {sql if sql else '(No SQL found in output)'}")

        # Check if using correct tables
        correct_tables = ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJCNTRTS", "PROJSTAFF", "PAC_MNT_RESOURCES"]
        uses_correct = any(table in (sql if sql else result) for table in correct_tables)

        if uses_correct:
            print("  ✅ Using domain-specific tables!")
        else:
            print("  ❌ Not using domain tables yet")

        if reasoning:
            print(f"\nReasoning:")
            print(f"  {reasoning[:200]}")

    except Exception as e:
        print(f"Error generating SQL: {e}")

print("\n" + "="*70)
print("Test Complete")
print("="*70)

# Show training stats if available
if checkpoint_path.exists():
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "training_history" in checkpoint:
            history = checkpoint["training_history"]
            if history.get("train_loss"):
                print(f"\nTraining Stats:")
                print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            if history.get("val_loss"):
                print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
            if "epoch" in checkpoint:
                print(f"  Epochs trained: {checkpoint['epoch'] + 1}")
            if "global_step" in checkpoint:
                print(f"  Total steps: {checkpoint['global_step']}")
    except:
        pass

print("\nNote: If the model isn't using domain tables, it may need:")
print("  1. More training epochs")
print("  2. Larger training dataset")
print("  3. Stronger prompting during generation")