#!/usr/bin/env python3
"""
Download the Qwen2.5-Coder-7B-Instruct model from Hugging Face
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Set cache directory (optional)
# os.environ['HF_HOME'] = './model_cache'

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("="*60)
print("Downloading Qwen2.5-Coder-7B-Instruct Model")
print("="*60)
print("\nThis will download approximately 15GB of data.")
print("Make sure you have enough disk space and a stable internet connection.")
print("\n")

# Step 1: Download tokenizer
print("Step 1/2: Downloading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✅ Tokenizer downloaded successfully")
except Exception as e:
    print(f"❌ Failed to download tokenizer: {e}")
    exit(1)

# Step 2: Download model
print("\nStep 2/2: Downloading model (this may take 10-30 minutes)...")
try:
    # Download with CPU config first (uses less memory during download)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✅ Model downloaded successfully")

    # Save model info
    print("\nModel Information:")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")

except Exception as e:
    print(f"❌ Failed to download model: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Make sure you have enough disk space (>20GB)")
    print("3. Try setting HF_HOME environment variable to a different location")
    print("4. If you're behind a proxy, set HTTP_PROXY and HTTPS_PROXY")
    exit(1)

print("\n" + "="*60)
print("✅ Model download complete!")
print("="*60)
print("\nYou can now run the training scripts:")
print("  python train_sft.py    # For supervised fine-tuning")
print("  python test_model.py   # To test the model")
