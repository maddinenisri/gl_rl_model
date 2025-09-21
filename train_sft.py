#!/usr/bin/env python3
"""
SFT Training Script for GL RL Model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.training import SFTTrainer, SFTConfig, DatasetLoader
from gl_rl_model.models.qwen_wrapper import QwenModelWrapper
import logging

logging.basicConfig(level=logging.INFO)

# Configuration
config = SFTConfig(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    num_epochs=1,  # Start with 1 epoch for testing
    batch_size=1,  # Small batch for testing
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    use_lora=True,
    lora_rank=16,  # Smaller rank for faster training
    checkpoint_dir="./checkpoints/sft",
    log_dir="./logs/sft"
)

# Initialize dataset
print("Loading dataset...")
dataset = DatasetLoader(
    data_path="gl_rl_model/data/training/query_pairs.jsonl",
    val_split=0.1,
    test_split=0.1
)
print(f"Dataset loaded: {dataset.stats['total_examples']} examples")

# Initialize model wrapper separately
print("Initializing model wrapper...")
import torch
# Only use 8-bit quantization if CUDA is available
use_8bit = torch.cuda.is_available() and False  # Set to True if you have memory constraints
device_map = "auto" if torch.cuda.is_available() else "cpu"

model_wrapper = QwenModelWrapper(
    model_name_or_path=config.model_name,
    use_lora=config.use_lora,
    load_in_8bit=use_8bit,
    device_map=device_map
)

# Load the model
print("Loading model (this may take a few minutes)...")
try:
    model_wrapper.load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Continue anyway for testing...")
    model_wrapper = None

# Initialize trainer with the model wrapper
print("Initializing trainer...")
trainer = SFTTrainer(
    config=config,
    model=model_wrapper,
    dataset_loader=dataset
)

# Run training
print("\n" + "="*60)
print("Starting SFT training...")
print("="*60)
try:
    history = trainer.train()
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final loss: {history['train_loss'][-1] if history['train_loss'] else 'N/A'}")
    print("="*60)
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
