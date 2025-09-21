#!/usr/bin/env python3
"""
Test model loading and generation on CPU (no GPU required)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("GL RL Model - CPU Test (No GPU Required)")
print("="*60)

# First, let's check if the model is downloaded
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Step 1: Try to load tokenizer (small, quick test)
print("\n1. Testing tokenizer loading...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"❌ Tokenizer not found. Run: python download_model.py")
    print(f"Error: {e}")
    sys.exit(1)

# Step 2: Try to load model (this uses more memory)
print("\n2. Testing model loading (this may take a minute)...")
print("   Note: This requires ~15GB of RAM")
try:
    # Load with CPU-friendly settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Reduce memory usage during loading
    )
    print("✅ Model loaded successfully")

    # Get model info
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"   Model size: {param_count:.1f}B parameters")

except Exception as e:
    print(f"❌ Failed to load model")
    print(f"Error: {e}")
    print("\nPossible solutions:")
    print("1. Make sure you have enough RAM (need ~15GB free)")
    print("2. Close other applications to free memory")
    print("3. Run: python download_model.py")
    sys.exit(1)

# Step 3: Test simple generation
print("\n3. Testing SQL generation...")
test_query = "Show all active projects"
print(f"   Query: {test_query}")

try:
    # Create a simple prompt
    prompt = f"""You are a SQL expert. Convert the following natural language query to SQL:

Query: {test_query}

SQL:"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate (with minimal settings for CPU)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the SQL part (after "SQL:")
    if "SQL:" in generated_text:
        sql = generated_text.split("SQL:")[-1].strip()
        print(f"   Generated SQL: {sql[:200]}...")  # Show first 200 chars
    else:
        print(f"   Generated: {generated_text[:200]}...")

    print("\n✅ Generation successful!")

except Exception as e:
    print(f"❌ Generation failed: {e}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
print("\nNext steps:")
print("1. If model loaded successfully, you can run: python train_sft.py")
print("2. Training on CPU will be slow (4-6 hours per epoch)")
print("3. Consider using Google Colab or cloud GPU for faster training")