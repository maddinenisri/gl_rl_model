#!/usr/bin/env python3
"""
SFT Training Script optimized for Mac GPU (MPS)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.training import SFTTrainer, SFTConfig, DatasetLoader
from gl_rl_model.models.qwen_wrapper import QwenModelWrapper
import torch
import logging

logging.basicConfig(level=logging.INFO)

print("="*70)
print("SFT Training with Mac GPU (MPS) Support")
print("="*70)

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Mac GPU (MPS) is available!")
    print(f"   Using device: {device}")
    device_map = "mps"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA GPU is available: {torch.cuda.get_device_name(0)}")
    device_map = "cuda"
else:
    device = torch.device("cpu")
    print("⚠️ No GPU found, using CPU (will be slow)")
    device_map = "cpu"

# Configuration optimized for Mac GPU
config = SFTConfig(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    num_epochs=1,  # Start with 1 epoch
    batch_size=1,  # Small batch for memory efficiency
    gradient_accumulation_steps=4,  # Effective batch = 4
    learning_rate=2e-5,
    use_lora=True,  # LoRA for efficient training
    lora_rank=8,  # Small rank for faster training
    checkpoint_dir="./checkpoints/sft",
    log_dir="./logs/sft",
    val_check_interval=5,
    save_interval=10,
    max_sequence_length=512  # Shorter sequences for faster training
)

print(f"\nConfiguration:")
print(f"  Device: {device}")
print(f"  Model: {config.model_name}")
print(f"  LoRA rank: {config.lora_rank}")
print(f"  Batch size: {config.batch_size} (accumulation: {config.gradient_accumulation_steps})")
print(f"  Max sequence length: {config.max_sequence_length}")
print("="*70)

# Load dataset
print("\nLoading dataset...")
dataset = DatasetLoader(
    data_path="gl_rl_model/data/training/query_pairs.jsonl",
    val_split=0.15,
    test_split=0.15
)
print(f"✅ Dataset loaded: {dataset.stats['total_examples']} examples")

# Initialize model wrapper with MPS support
print("\nInitializing model with MPS support...")
print("(This may take a few minutes on first run)")

# For MPS, we cannot use 8-bit quantization
use_8bit = False

try:
    # First try to load model directly to MPS
    if device.type == "mps":
        # MPS requires special handling
        model_wrapper = QwenModelWrapper(
            model_name_or_path=config.model_name,
            use_lora=config.use_lora,
            load_in_8bit=False,  # MPS doesn't support 8-bit
            device_map=None  # We'll move to MPS manually
        )

        print("Loading model to CPU first, then moving to MPS...")
        model_wrapper.load_model()

        # Move model to MPS if possible
        if model_wrapper.model is not None:
            try:
                print("Moving model to MPS...")
                model_wrapper.model = model_wrapper.model.to(device)
                print("✅ Model successfully moved to MPS!")
            except Exception as e:
                print(f"⚠️ Could not move entire model to MPS: {e}")
                print("Model will run on CPU with MPS acceleration where possible")
    else:
        # For CUDA or CPU
        model_wrapper = QwenModelWrapper(
            model_name_or_path=config.model_name,
            use_lora=config.use_lora,
            load_in_8bit=use_8bit,
            device_map=device_map if device.type != "cpu" else "cpu"
        )
        model_wrapper.load_model()

    model_info = model_wrapper.get_model_info()
    print(f"✅ Model loaded successfully!")
    print(f"   Trainable params: {model_info.get('trainable_parameters', 0):,}")
    print(f"   Total params: {model_info.get('total_parameters', 0):,}")
    print(f"   Trainable %: {model_info.get('trainable_percentage', 0):.2f}%")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize trainer
print("\nInitializing trainer...")
trainer = SFTTrainer(
    config=config,
    model=model_wrapper,
    dataset_loader=dataset
)

# Start training
print("\n" + "="*70)
print("Starting SFT Training with Schema Context")
print("="*70)
print("The model will learn to generate SQL using your specific tables:")
print("  - PAC_MNT_PROJECTS")
print("  - SRM_COMPANIES")
print("  - PROJCNTRTS")
print("  - PROJSTAFF")
print("  - PAC_MNT_RESOURCES")

try:
    history = trainer.train()

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)

    if history['train_loss']:
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    # Quick test
    print("\nTesting trained model...")
    test_queries = [
        "Show all active projects",
        "Find projects with budget over 100000"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            sql, reasoning = trainer.generate_sql(query)
            print(f"Generated SQL: {sql}")

            # Check if using domain tables
            if any(table in sql for table in ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJCNTRTS"]):
                print("✅ Using domain-specific tables!")
        except Exception as e:
            print(f"Generation error: {e}")

    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()