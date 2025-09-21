#!/usr/bin/env python3
"""
Improved SFT Training Script with Better Hyperparameters and Monitoring
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging
from gl_rl_model.training import SFTTrainer, SFTConfig, DatasetLoader
from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

print("="*70)
print("Improved SFT Training with Domain-Specific SQL Focus")
print("="*70)

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Mac GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"✅ Using CUDA GPU")
else:
    device = "cpu"
    print("⚠️ Using CPU (will be slow)")

# IMPROVED Configuration
config = SFTConfig(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",

    # Training hyperparameters (IMPROVED)
    num_epochs=5,  # Increased from 1 to 5
    batch_size=2,  # Increased from 1 to 2
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    learning_rate=2e-5,  # Increased from 1e-6 to 2e-5
    warmup_steps=100,
    max_grad_norm=1.0,
    weight_decay=0.01,

    # LoRA configuration (IMPROVED)
    use_lora=True,
    lora_rank=32,  # Increased from 8 to 32 for better capacity
    lora_alpha=64,  # Adjusted alpha
    lora_dropout=0.05,  # Reduced dropout

    # Data configuration
    max_sequence_length=768,  # Increased from 512
    val_check_interval=20,  # Check validation more frequently
    save_interval=50,

    # Curriculum learning
    use_curriculum=True,
    curriculum_schedule={
        "easy": 1,    # Epoch 1: Simple SELECT queries
        "medium": 3,  # Epochs 2-3: JOINs and aggregations
        "hard": 5     # Epochs 4-5: Complex queries
    },

    # Paths
    checkpoint_dir="./checkpoints/improved",
    log_dir="./logs/improved"
)

print("\n" + "="*70)
print("Configuration:")
print(f"  Device: {device}")
print(f"  Model: {config.model_name}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Learning Rate: {config.learning_rate}")
print(f"  LoRA Rank: {config.lora_rank}")
print(f"  Batch Size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
print("="*70)

# Load EXPANDED dataset
print("\n[1/3] Loading expanded dataset...")
dataset = DatasetLoader(
    data_path="gl_rl_model/data/training/query_pairs_expanded.jsonl",  # Use expanded dataset
    val_split=0.1,
    test_split=0.1
)
print(f"✅ Dataset loaded: {dataset.stats['total_examples']} examples")
print(f"   Train: {dataset.stats['train_size']}")
print(f"   Val: {dataset.stats['val_size']}")
print(f"   Test: {dataset.stats['test_size']}")

# Initialize model
print("\n[2/3] Initializing model with improved LoRA configuration...")
model_wrapper = QwenModelWrapper(
    model_name_or_path=config.model_name,
    use_lora=config.use_lora,
    load_in_8bit=False,  # Don't use 8-bit on MPS
    device_map=None if device == "mps" else device
)

try:
    model_wrapper.load_model()

    # Move to MPS if available
    if device == "mps":
        model_wrapper.model = model_wrapper.model.to(device)
        print("✅ Model moved to MPS")

    model_info = model_wrapper.get_model_info()
    print(f"✅ Model loaded successfully")
    print(f"   Trainable params: {model_info.get('trainable_parameters', 0):,}")
    print(f"   Total params: {model_info.get('total_parameters', 0):,}")
    print(f"   Trainable %: {model_info.get('trainable_percentage', 0):.2f}%")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Custom trainer with monitoring
class MonitoredSFTTrainer(SFTTrainer):
    """Extended trainer with domain-specific monitoring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_tables = [
            "PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJSTAFF",
            "PROJCNTRTS", "PAC_MNT_RESOURCES", "SRM_CONTACTS"
        ]
        self.domain_usage_history = []

    def check_domain_usage(self, generated_sql: str) -> bool:
        """Check if generated SQL uses domain tables."""
        return any(table in generated_sql for table in self.domain_tables)

    def _validate(self) -> float:
        """Enhanced validation with domain checking."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        domain_correct = 0
        total_checked = 0

        prompt_templates = SQLPromptTemplates()

        with torch.no_grad():
            for i in range(min(5, self.dataset_loader.stats['val_size'])):  # Check 5 validation examples
                batch = self.dataset_loader.get_sft_batch(
                    batch_size=1,
                    split="val"
                )

                if batch is None:
                    break

                # Calculate loss
                loss = self._compute_batch_loss(batch)
                total_loss += loss.item()
                num_batches += 1

                # Generate SQL to check domain usage
                query = batch.queries[0]

                # Generate SQL with the current model
                try:
                    # Create prompt with explicit domain instruction
                    test_prompt = f"""<|im_start|>system
You are a SQL expert. Generate SQL using ONLY these tables:
- PAC_MNT_PROJECTS
- SRM_COMPANIES
- PROJSTAFF
- PROJCNTRTS
- PAC_MNT_RESOURCES
- SRM_CONTACTS
NEVER use generic table names like 'projects' or 'companies'.
<|im_end|>
<|im_start|>user
Query: {query}
<|im_end|>
<|im_start|>assistant
```sql"""

                    gen_params = GenerationParams(
                        temperature=0.1,
                        max_new_tokens=150,
                        do_sample=False
                    )

                    generated = self.model_wrapper.generate(test_prompt, gen_params)

                    # Check domain usage
                    if self.check_domain_usage(generated):
                        domain_correct += 1
                    else:
                        self.logger.warning(f"❌ Not using domain tables for: {query}")
                        self.logger.warning(f"   Generated: {generated[:100]}...")

                    total_checked += 1

                except Exception as e:
                    self.logger.warning(f"Generation error during validation: {e}")

        # Log domain usage rate
        if total_checked > 0:
            domain_rate = domain_correct / total_checked
            self.domain_usage_history.append(domain_rate)
            self.logger.info(f"📊 Domain table usage rate: {domain_rate:.1%} ({domain_correct}/{total_checked})")

            # Save best model based on domain usage
            if domain_rate > 0.8:  # If using domain tables >80% of the time
                self._save_checkpoint("best_domain")
                self.logger.info("🎯 Saved checkpoint with good domain usage!")

        return total_loss / max(num_batches, 1)

# Initialize trainer with monitoring
print("\n[3/3] Setting up trainer with monitoring...")
trainer = MonitoredSFTTrainer(
    config=config,
    model=model_wrapper,
    dataset_loader=dataset
)

# Training with progress monitoring
print("\n" + "="*70)
print("Starting Improved Training")
print("="*70)
print("\n🎯 Training Goals:")
print("  • Learn to use domain-specific tables (PAC_MNT_PROJECTS, etc.)")
print("  • Generate syntactically correct SQL")
print("  • Handle various query patterns")
print("  • Maintain reasoning capability")

try:
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)

    if history:
        if history.get("train_loss"):
            print(f"Initial loss: {history['train_loss'][0]:.4f}")
            print(f"Final loss: {history['train_loss'][-1]:.4f}")
            print(f"Loss reduction: {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.1f}%")

        if history.get("val_loss"):
            print(f"Best validation loss: {min(history['val_loss']):.4f}")

    print(f"\nTraining time: {training_time/60:.1f} minutes")

    # Test the improved model
    print("\n" + "="*70)
    print("Testing Improved Model")
    print("="*70)

    test_queries = [
        "Show all active projects",
        "Find projects with budget over 100000",
        "List all companies",
        "Count projects per department",
        "Show project staff allocations",
        "Find resources with high cost rate",
        "Get company contacts"
    ]

    prompt_templates = SQLPromptTemplates()
    gen_params = GenerationParams(
        temperature=0.1,
        max_new_tokens=150,
        do_sample=False
    )

    correct_domain = 0
    for query in test_queries:
        print(f"\nQuery: {query}")

        # Create prompt with strong domain emphasis
        test_prompt = f"""<|im_start|>system
You are a SQL expert for a GL/ERP database.
ALWAYS use these exact table names:
- PAC_MNT_PROJECTS (for projects)
- SRM_COMPANIES (for companies)
- PROJSTAFF (for staff)
- PROJCNTRTS (for contracts)
- PAC_MNT_RESOURCES (for resources)
- SRM_CONTACTS (for contacts)
<|im_end|>
<|im_start|>user
Query: {query}
<|im_end|>
<|im_start|>assistant
```sql"""

        try:
            result = model_wrapper.generate(test_prompt, gen_params)
            sql = model_wrapper.extract_sql(result)

            if not sql and result:
                # Try to extract SQL from result
                if "SELECT" in result:
                    sql = result.split("```")[0].strip()

            print(f"Generated SQL: {sql if sql else result[:150]}")

            # Check domain usage
            if trainer.check_domain_usage(sql if sql else result):
                print("✅ Using domain-specific tables!")
                correct_domain += 1
            else:
                print("❌ Not using domain tables")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\n📊 Domain Usage Score: {correct_domain}/{len(test_queries)} ({correct_domain/len(test_queries)*100:.0f}%)")

    if correct_domain < len(test_queries) * 0.5:
        print("\n⚠️ Model still not using domain tables consistently.")
        print("Consider:")
        print("  1. Running more epochs")
        print("  2. Increasing learning rate")
        print("  3. Adding more explicit domain examples")
    else:
        print("\n✅ Model successfully learned domain-specific SQL!")

    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
    print(f"Best checkpoint: {config.checkpoint_dir}/best.pt")

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Next Steps:")
print("1. If domain usage < 80%, run more training epochs")
print("2. Test with: python test_model.py")
print("3. Evaluate with: python evaluate_model.py --checkpoint ./checkpoints/improved/best.pt")
print("="*70)