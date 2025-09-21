#!/usr/bin/env python3
"""
GRPO Training Script - Optimized for Mac GPU (MPS)
Fixed version that properly handles LoRA models
"""
import sys
import asyncio
from pathlib import Path
import torch
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.training.grpo_trainer import GRPOConfig
from gl_rl_model.training.grpo_trainer_mps import GRPOTrainerMPS
from gl_rl_model.training.dataset_loader import DatasetLoader
from gl_rl_model.models.qwen_wrapper import QwenModelWrapper
from gl_rl_model.agents.reward_evaluator import RewardEvaluatorAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/grpo_training_{datetime.now():%Y%m%d_%H%M%S}.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main training function."""
    print("="*70)
    print("GRPO Training - Mac GPU Optimized")
    print("="*70)
    print("\nReinforcement Learning optimization for SQL generation")
    print("Fixed for LoRA model compatibility")
    print("="*70)

    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Mac GPU (MPS)")
    else:
        device = "cpu"
        print("⚠️ MPS not available, using CPU")

    # Configuration
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        num_iterations=10,  # Reduced for faster testing
        batch_size=1,  # Smaller batch for MPS memory
        num_candidates_per_prompt=2,  # Fewer candidates to reduce memory
        kl_coefficient=0.05,
        learning_rate=2e-6,
        gradient_accumulation_steps=4,
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.1,
        save_interval=2,
        val_check_interval=2,
        checkpoint_dir="./checkpoints/grpo_mps"
    )

    print("\n" + "="*70)
    print("Configuration:")
    print(f"  Starting from SFT checkpoint: ./checkpoints/sft/best.pt")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Candidates per prompt: {config.num_candidates_per_prompt}")
    print(f"  KL coefficient: {config.kl_coefficient}")
    print(f"  Learning rate: {config.learning_rate}")
    print("="*70)

    # Load dataset (use expanded for better training)
    print("\n[1/4] Loading dataset...")
    dataset = DatasetLoader(
        data_path="gl_rl_model/data/training/query_pairs_expanded.jsonl"
    )
    print(f"✅ Dataset loaded: {dataset.stats['total_examples']} examples")

    # Initialize policy model
    print("\n[2/4] Loading SFT-trained model...")
    policy_model = QwenModelWrapper(
        model_name_or_path=config.model_name,
        use_lora=True,
        load_in_8bit=False,
        device_map=None if device == "mps" else device
    )

    # Load the model
    policy_model.load_model()

    # Move to MPS if available
    if device == "mps":
        policy_model.model = policy_model.model.to(device)
        print("✅ Model moved to MPS")

    # Load SFT checkpoint if it exists
    sft_checkpoint = Path("./checkpoints/sft/best.pt")
    if not sft_checkpoint.exists():
        # Try improved checkpoint
        sft_checkpoint = Path("./checkpoints/improved/best_domain.pt")

    if sft_checkpoint.exists():
        print(f"Loading SFT checkpoint from {sft_checkpoint}")
        checkpoint = torch.load(sft_checkpoint, map_location=device)

        # Load LoRA weights
        if "lora_state_dict" in checkpoint:
            policy_model.load_lora_state_dict(checkpoint["lora_state_dict"])
            print("✅ Loaded SFT-trained LoRA weights")
        else:
            logger.warning("No LoRA weights in checkpoint, using base model")

    print("✅ Model loaded successfully")
    trainable_params = sum(p.numel() for p in policy_model.model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable_params:,}")

    # Initialize reward evaluator
    print("\n[3/4] Initializing reward evaluator...")
    reward_evaluator = RewardEvaluatorAgent()
    await reward_evaluator.initialize()
    print("✅ Reward evaluator initialized")

    # Initialize GRPO trainer
    print("\n[4/4] Setting up GRPO trainer...")
    trainer = GRPOTrainerMPS(
        config=config,
        model=policy_model,
        reward_evaluator=reward_evaluator,
        dataset_loader=dataset,
        device=device
    )
    print("✅ Trainer initialized")

    # Start training
    print("\n" + "="*70)
    print("Starting GRPO Training")
    print("="*70)

    try:
        history = await trainer.train()

        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)

        # Print final statistics
        if history["avg_reward"]:
            print(f"\n📊 Training Statistics:")
            print(f"  Final avg reward: {history['avg_reward'][-1]:.3f}")
            print(f"  Best avg reward: {max(history['avg_reward']):.3f}")
            print(f"  Final policy loss: {history['policy_loss'][-1]:.4f}")

            # Check for improvement
            initial_reward = history['avg_reward'][0] if history['avg_reward'] else 0
            final_reward = history['avg_reward'][-1] if history['avg_reward'] else 0
            improvement = final_reward - initial_reward

            if improvement > 0:
                print(f"  ✅ Reward improved by: {improvement:.3f}")
            else:
                print(f"  ⚠️ Reward decreased by: {abs(improvement):.3f}")

        print(f"\n✅ Checkpoints saved to: {config.checkpoint_dir}")
        print("\nTo test the GRPO model:")
        print(f"  python test_model.py --checkpoint {config.checkpoint_dir}/best.pt")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)

    # Run training
    asyncio.run(main())