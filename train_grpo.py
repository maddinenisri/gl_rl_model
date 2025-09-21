#!/usr/bin/env python3
"""
GRPO Training Script for GL RL Model
Reinforcement Learning optimization after SFT training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import torch
import logging
from gl_rl_model.training import GRPOTrainer, GRPOConfig, DatasetLoader
from gl_rl_model.models.qwen_wrapper import QwenModelWrapper
from gl_rl_model.agents.reward_evaluator import RewardEvaluatorAgent

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

print("="*70)
print("GRPO Training - Reinforcement Learning Optimization")
print("="*70)
print("\nThis will optimize the SFT-trained model using reward signals")
print("to improve SQL generation quality.")
print("="*70)

async def main():
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

    # GRPO Configuration
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        sft_checkpoint="./checkpoints/sft/best.pt",  # Start from SFT model
        num_iterations=20,  # Fewer iterations for testing
        batch_size=2,  # Small batch for memory
        num_candidates_per_prompt=3,  # Generate 3 candidates per query
        kl_coefficient=0.05,  # KL penalty to prevent drift from SFT
        entropy_coefficient=0.01,
        learning_rate=5e-6,  # Lower LR for fine-tuning
        use_lora=True,
        lora_rank=8,
        checkpoint_dir="./checkpoints/grpo",
        log_dir="./logs/grpo",
        generation_temperature=0.7,
        generation_top_p=0.9,
        val_check_interval=5,
        save_interval=10
    )

    print("\n" + "="*70)
    print("Configuration:")
    print(f"  Starting from SFT checkpoint: {config.sft_checkpoint}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Candidates per prompt: {config.num_candidates_per_prompt}")
    print(f"  KL coefficient: {config.kl_coefficient}")
    print(f"  Learning rate: {config.learning_rate}")
    print("="*70)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = DatasetLoader(
        data_path="gl_rl_model/data/training/query_pairs.jsonl",
        val_split=0.15,
        test_split=0.15
    )
    print(f"✅ Dataset loaded: {dataset.stats['total_examples']} examples")

    # Initialize model from SFT checkpoint
    print("\n[2/4] Loading SFT-trained model...")
    model_wrapper = QwenModelWrapper(
        model_name_or_path=config.model_name,
        use_lora=config.use_lora,
        load_in_8bit=False,
        device_map=None if device == "mps" else device
    )

    try:
        # Load base model
        model_wrapper.load_model()

        # Move to MPS if available
        if device == "mps":
            model_wrapper.model = model_wrapper.model.to(device)
            print("✅ Model moved to MPS")

        # Load SFT checkpoint if available
        sft_checkpoint_path = Path(config.sft_checkpoint)
        if sft_checkpoint_path.exists():
            print(f"Loading SFT checkpoint from {sft_checkpoint_path}")
            checkpoint = torch.load(sft_checkpoint_path, map_location=device)

            # Load LoRA weights from SFT
            if "lora_state_dict" in checkpoint:
                model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])
                print("✅ Loaded SFT-trained LoRA weights")

            # Log SFT training stats
            if "training_history" in checkpoint:
                history = checkpoint["training_history"]
                if history.get("train_loss"):
                    print(f"   SFT final loss: {history['train_loss'][-1]:.4f}")
        else:
            print("⚠️ No SFT checkpoint found, starting from base model")

        model_info = model_wrapper.get_model_info()
        print(f"✅ Model loaded successfully")
        print(f"   Trainable params: {model_info.get('trainable_parameters', 0):,}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize reward evaluator
    print("\n[3/4] Initializing reward evaluator...")
    reward_evaluator = RewardEvaluatorAgent()
    await reward_evaluator.initialize()
    print("✅ Reward evaluator initialized")

    # Initialize GRPO trainer
    print("\n[4/4] Setting up GRPO trainer...")
    trainer = GRPOTrainer(
        config=config,
        model=model_wrapper,
        reward_evaluator=reward_evaluator,
        dataset_loader=dataset
    )

    # Run GRPO training
    print("\n" + "="*70)
    print("Starting GRPO Training")
    print("="*70)
    print("\nThe model will be optimized based on reward signals:")
    print("  • SQL correctness and syntax")
    print("  • Use of correct domain tables (PAC_MNT_PROJECTS, etc.)")
    print("  • Query complexity handling")
    print("  • Reasoning quality")

    try:
        # Run async training
        history = await trainer.train()

        print("\n" + "="*70)
        print("✅ GRPO Training Complete!")
        print("="*70)

        # Show training results
        if history:
            if "avg_reward" in history and history["avg_reward"]:
                print(f"Final average reward: {history['avg_reward'][-1]:.3f}")
                print(f"Best average reward: {max(history['avg_reward']):.3f}")

            if "policy_loss" in history and history["policy_loss"]:
                print(f"Final policy loss: {history['policy_loss'][-1]:.4f}")

            # Show best SQL examples if available
            if "best_sql_examples" in history and history["best_sql_examples"]:
                print("\nBest generated SQL examples:")
                for i, example in enumerate(history["best_sql_examples"][-3:], 1):
                    print(f"\n{i}. Query: {example['prompt']}")
                    print(f"   SQL: {example['sql']}")
                    print(f"   Reward: {example['reward']:.2f}")

        print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
        print(f"Logs saved to: {config.log_dir}")

        # Test the GRPO-optimized model
        print("\n" + "="*70)
        print("Testing GRPO-Optimized Model")
        print("="*70)

        test_queries = [
            "Show all active projects with budget over 500000",
            "List companies with their project counts",
            "Find staff allocations for high-budget projects"
        ]

        for query in test_queries[:2]:  # Test first 2 queries
            print(f"\nQuery: {query}")
            try:
                # Generate SQL with optimized model
                result = await trainer.generate_sql(query)
                if isinstance(result, tuple):
                    sql, reasoning = result
                else:
                    sql = result
                    reasoning = ""

                print(f"Generated SQL: {sql}")

                # Check if using domain tables
                if any(table in sql for table in ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJSTAFF"]):
                    print("✅ Using domain-specific tables!")

                # Evaluate the generated SQL
                eval_result = await reward_evaluator.process({
                    "query": query,
                    "sql": sql,
                    "reasoning": reasoning,
                    "mode": "single"
                })

                if "rewards" in eval_result:
                    print(f"Reward score: {eval_result['rewards'].get('total_reward', 0):.2f}")

            except Exception as e:
                print(f"Error testing query: {e}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if reward_evaluator:
            await reward_evaluator.shutdown()

    print("\n" + "="*70)
    print("Next Steps:")
    print("1. Test the optimized model: python test_model.py")
    print("2. Compare SFT vs GRPO performance")
    print("3. Deploy via Production API")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())