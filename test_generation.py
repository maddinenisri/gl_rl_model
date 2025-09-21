#!/usr/bin/env python3
"""
Simple test script for SQL generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
import torch

print("="*60)
print("Testing SQL Generation with GL RL Model")
print("="*60)

# Initialize model
print("\nLoading model...")
model = QwenModelWrapper(
    model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
    use_lora=True,
    load_in_8bit=False,  # Don't use 8-bit on CPU
    device_map="cpu" if not torch.cuda.is_available() else "auto"
)

try:
    model.load_model()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Test queries
test_queries = [
    "Show all active projects",
    "Find projects with budget over 100000",
    "List companies and their contact information",
]

print("\n" + "="*60)
print("Generating SQL for test queries...")
print("="*60)

# Create generation parameters
gen_params = GenerationParams(
    temperature=0.3,  # Lower temperature for more deterministic output
    top_p=0.95,
    max_new_tokens=256,
    do_sample=True
)

for i, query in enumerate(test_queries, 1):
    print(f"\n📝 Query {i}: {query}")
    print("-" * 40)

    try:
        # Generate SQL
        result = model.generate(query, gen_params)

        # Extract SQL from result
        sql = model.extract_sql(result)
        reasoning = model.extract_reasoning(result)

        if sql:
            print(f"Generated SQL:\n{sql}")
        else:
            print(f"Raw output:\n{result[:500]}...")

        if reasoning:
            print(f"\nReasoning:\n{reasoning}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Test Complete!")
print("="*60)