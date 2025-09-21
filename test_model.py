#!/usr/bin/env python3
"""
Test the trained model or base model for SQL generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
import torch

print("="*60)
print("GL RL Model - SQL Generation Test")
print("="*60)

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device_map = "auto"
else:
    print("⚠️ No GPU found, using CPU (will be slower)")
    device_map = "cpu"

# Initialize model wrapper
print("\nInitializing model wrapper...")
# Don't use 8-bit quantization on CPU (requires CUDA)
use_8bit = torch.cuda.is_available()
model_wrapper = QwenModelWrapper(
    model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
    use_lora=True,
    load_in_8bit=use_8bit,  # Only use 8-bit if GPU is available
    device_map=device_map
)

# Check for checkpoint
checkpoint_path = Path("./checkpoints/sft/best.pt")
if checkpoint_path.exists():
    print(f"Found checkpoint at {checkpoint_path}")
    print("Loading fine-tuned model...")
    try:
        model_wrapper.load_model(checkpoint_path)
        print("✅ Loaded fine-tuned model")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Loading base model instead...")
        model_wrapper.load_model()
else:
    print("No checkpoint found, loading base model...")
    try:
        model_wrapper.load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nTo download the model, run:")
        print("python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')\"")
        sys.exit(1)

# Test queries
test_queries = [
    "Show all active projects",
    "Find projects with budget over 100000",
    "List companies and their contact information",
    "Count projects per company",
    "Show resource allocation for active projects"
]

print("\n" + "="*60)
print("Testing SQL Generation")
print("="*60)

# Set generation parameters
gen_params = GenerationParams(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256  # Changed from max_length to max_new_tokens
)

for i, query in enumerate(test_queries, 1):
    print(f"\n📝 Test {i}/{len(test_queries)}")
    print(f"Query: {query}")
    print("-" * 40)

    try:
        # Generate SQL
        result = model_wrapper.generate(query, gen_params)

        # Extract SQL and reasoning
        sql = model_wrapper.extract_sql(result)
        reasoning = model_wrapper.extract_reasoning(result)

        print(f"Generated SQL:")
        print(sql)

        if reasoning:
            print(f"\nReasoning:")
            print(reasoning)
    except Exception as e:
        print(f"❌ Error generating SQL: {e}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
