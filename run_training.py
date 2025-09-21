#!/usr/bin/env python3
"""
Complete training pipeline for GL RL Model with schema-aware SQL generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging
from gl_rl_model.training import SFTTrainer, SFTConfig, DatasetLoader
from gl_rl_model.models.qwen_wrapper import QwenModelWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*70)
print("GL RL Model - Schema-Aware SQL Generation Training")
print("="*70)

# Check device availability
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    device_map = "auto"
    use_8bit = False  # Set to True if memory constrained
else:
    print("⚠️ No GPU found, using CPU (training will be slower)")
    device_map = "cpu"
    use_8bit = False  # Cannot use 8-bit on CPU

# Training configuration
config = SFTConfig(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    num_epochs=1,  # Start with 1 epoch for testing
    batch_size=1,  # Small batch size for CPU/limited GPU memory
    gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 = 8
    learning_rate=2e-5,
    use_lora=True,  # Use LoRA for efficient fine-tuning
    lora_rank=8,  # Small rank for faster training
    checkpoint_dir="./checkpoints/sft",
    log_dir="./logs/sft",
    val_check_interval=5,  # Check validation every 5 steps
    save_interval=10,  # Save checkpoint every 10 steps
)

print("\n" + "="*70)
print("Configuration:")
print(f"  Model: {config.model_name}")
print(f"  LoRA: {'Enabled' if config.use_lora else 'Disabled'} (rank={config.lora_rank})")
print(f"  Epochs: {config.num_epochs}")
print(f"  Batch size: {config.batch_size} (accumulation: {config.gradient_accumulation_steps})")
print(f"  Learning rate: {config.learning_rate}")
print("="*70)

# Load dataset
print("\n[1/3] Loading training data...")
dataset = DatasetLoader(
    data_path="gl_rl_model/data/training/query_pairs.jsonl",
    val_split=0.15,
    test_split=0.15
)
print(f"✅ Dataset loaded: {dataset.stats['total_examples']} examples")
print(f"   Train: {dataset.stats['train_size']}, Val: {dataset.stats['val_size']}, Test: {dataset.stats['test_size']}")

# Initialize model
print("\n[2/3] Loading Qwen model...")
print("(This may take a few minutes on first run to download the model)")

model_wrapper = QwenModelWrapper(
    model_name_or_path=config.model_name,
    use_lora=config.use_lora,
    load_in_8bit=use_8bit,
    device_map=device_map
)

try:
    model_wrapper.load_model()
    model_info = model_wrapper.get_model_info()
    print(f"✅ Model loaded successfully!")
    if model_info.get("trainable_parameters"):
        print(f"   Trainable params: {model_info['trainable_parameters']:,} "
              f"({model_info['trainable_percentage']:.2f}% of total)")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\nTo download the model manually, run:")
    print(f"python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
          f"AutoTokenizer.from_pretrained('{config.model_name}'); "
          f"AutoModelForCausalLM.from_pretrained('{config.model_name}')\"")
    sys.exit(1)

# Initialize trainer
print("\n[3/3] Setting up trainer...")
trainer = SFTTrainer(
    config=config,
    model=model_wrapper,
    dataset_loader=dataset
)

# Start training
print("\n" + "="*70)
print("Starting Schema-Aware SFT Training")
print("="*70)
print("\n⚠️ Important: The model is now learning with domain-specific schema context!")
print("It will learn to generate SQL using actual table names like PAC_MNT_PROJECTS")

try:
    history = trainer.train()

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)

    if history['train_loss']:
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: {config.log_dir}")

    # Quick test on the trained model
    print("\n" + "="*70)
    print("Testing Trained Model")
    print("="*70)

    test_queries = [
        "Show all active projects",
        "Find projects with budget over 100000",
        "List companies and their contacts"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        sql, reasoning = trainer.generate_sql(query)
        print(f"Generated SQL: {sql}")
        if reasoning:
            print(f"Reasoning: {reasoning[:200]}...")

    print("\n" + "="*70)
    print("✅ Training pipeline complete!")
    print("\nNext steps:")
    print("1. Test the model more thoroughly: python test_model.py")
    print("2. Run GRPO training for further optimization")
    print("3. Deploy the model for production use")
    print("="*70)

except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user")
    print("Checkpoints have been saved. You can resume training later.")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)