"""
GRPO Trainer - Mac GPU Optimized Version
Fixed to properly handle LoRA models
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy

from .grpo_trainer import GRPOConfig
from ..models.qwen_wrapper import QwenModelWrapper, GenerationParams
from ..agents.reward_evaluator import RewardEvaluatorAgent
from ..agents.query_generator import QueryGeneratorAgent
from .dataset_loader import DatasetLoader
from ..utils.prompt_templates import SQLPromptTemplates

logger = logging.getLogger(__name__)


class GRPOTrainerMPS:
    """
    GRPO Trainer optimized for Mac GPU (MPS).
    Fixes LoRA model compatibility issues.
    """

    def __init__(
        self,
        config: GRPOConfig,
        model: QwenModelWrapper = None,
        reward_evaluator: RewardEvaluatorAgent = None,
        dataset_loader: DatasetLoader = None,
        device: str = "mps"
    ):
        """
        Initialize GRPO trainer with MPS optimizations.
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize or use provided model
        if model is None:
            self.logger.info("Initializing policy model...")
            self.policy_model = QwenModelWrapper(
                model_name_or_path=self.config.model_name,
                use_lora=self.config.use_lora,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout
            )
            self.policy_model.load_model()
        else:
            self.policy_model = model

        # Create reference model (frozen copy for KL divergence)
        # For LoRA models, we'll use the base model as reference
        self.logger.info("Creating reference model...")
        self.reference_model = self._create_reference_model()

        # Initialize reward evaluator
        if reward_evaluator is None:
            self.logger.info("Initializing reward evaluator...")
            self.reward_evaluator = RewardEvaluatorAgent()
            asyncio.run(self.reward_evaluator.initialize())
        else:
            self.reward_evaluator = reward_evaluator

        # Initialize dataset
        if dataset_loader is None:
            self.logger.info("Initializing dataset loader...")
            self.dataset_loader = DatasetLoader()
        else:
            self.dataset_loader = dataset_loader

        # Initialize query generator
        self.query_generator = QueryGeneratorAgent()

        # Training components
        self.optimizer = None
        self.best_avg_reward = -float('inf')
        self.global_step = 0
        self.current_iteration = 0

        # Metrics tracking
        self.training_history = {
            "iteration": [],
            "avg_reward": [],
            "avg_advantage": [],
            "policy_loss": [],
            "kl_divergence": [],
            "learning_rate": [],
            "best_sql_examples": []
        }

    def _create_reference_model(self) -> QwenModelWrapper:
        """
        Create reference model for KL divergence calculation.
        For LoRA models, we use a separate instance with frozen weights.
        """
        # Create a new model instance without LoRA
        reference_model = QwenModelWrapper(
            model_name_or_path=self.config.model_name,
            use_lora=False,  # No LoRA for reference
            load_in_8bit=False,
            device_map=None if self.device == "mps" else self.device
        )
        reference_model.load_model()

        # Move to device
        if self.device == "mps":
            reference_model.model = reference_model.model.to(self.device)

        # If policy model has LoRA, get base model state for reference
        if self.config.use_lora and hasattr(self.policy_model.model, 'get_base_model'):
            # Get base model without LoRA adapters
            base_model = self.policy_model.model.get_base_model()
            # Copy base weights to reference
            reference_model.model.load_state_dict(base_model.state_dict())
        elif not self.config.use_lora:
            # If no LoRA, just copy the policy model weights
            reference_model.model.load_state_dict(
                self.policy_model.model.state_dict()
            )

        # Freeze reference model
        for param in reference_model.model.parameters():
            param.requires_grad = False

        reference_model.model.eval()  # Always in eval mode

        return reference_model

    def setup_training(self):
        """Setup training components."""
        # Setup optimizer
        if self.config.use_lora:
            # Only optimize LoRA parameters
            params = [p for p in self.policy_model.model.parameters() if p.requires_grad]
        else:
            params = self.policy_model.model.parameters()

        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.logger.info(f"Optimizer setup with {len(params)} parameter groups")

    async def train(self):
        """Main GRPO training loop."""
        self.logger.info("Starting GRPO training...")
        self.setup_training()

        for iteration in range(self.config.num_iterations):
            self.current_iteration = iteration
            self.logger.info(f"\nIteration {iteration + 1}/{self.config.num_iterations}")

            # Collect trajectories
            trajectories = await self._collect_trajectories()

            if not trajectories:
                self.logger.warning("No trajectories collected, skipping iteration")
                continue

            # Compute advantages
            advantages = self._compute_advantages(trajectories)

            # Policy optimization
            policy_loss, kl_div = self._optimize_policy(trajectories, advantages)

            # Log metrics
            avg_reward = np.mean([t["reward"] for t in trajectories])
            avg_advantage = np.mean(advantages)

            self.logger.info(f"Avg reward: {avg_reward:.3f}")
            self.logger.info(f"Avg advantage: {avg_advantage:.3f}")
            self.logger.info(f"Policy loss: {policy_loss:.4f}")
            self.logger.info(f"KL divergence: {kl_div:.4f}")

            # Track history
            self.training_history["iteration"].append(iteration + 1)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["avg_advantage"].append(avg_advantage)
            self.training_history["policy_loss"].append(policy_loss)
            self.training_history["kl_divergence"].append(kl_div)

            # Save best model
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self._save_checkpoint("best")
                self.logger.info(f"New best model! Avg reward: {avg_reward:.3f}")

            # Regular checkpoint
            if (iteration + 1) % self.config.save_interval == 0:
                self._save_checkpoint(f"iter_{iteration + 1}")

            # Validation
            if (iteration + 1) % self.config.val_check_interval == 0:
                await self._validate()

        # Save final model
        self._save_checkpoint("final")
        self._save_training_history()

        self.logger.info("GRPO training complete!")
        return self.training_history

    async def _collect_trajectories(self) -> List[Dict[str, Any]]:
        """
        Collect trajectories by generating SQL for prompts and evaluating rewards.
        """
        trajectories = []
        prompt_templates = SQLPromptTemplates()

        # Get batch of prompts
        batch = self.dataset_loader.get_sft_batch(
            batch_size=self.config.batch_size,
            split="train"
        )

        if not batch:
            return trajectories

        for i in range(len(batch.queries)):
            query = batch.queries[i]
            expected_sql = batch.sqls[i]

            # Create prompt with schema context
            prompt = prompt_templates.zero_shot_sql_generation(
                query=query,
                schema_context=batch.schemas[i] if i < len(batch.schemas) else "",
                business_context="Use domain-specific table names"
            )

            # Generate multiple candidates
            candidates = []
            for _ in range(self.config.num_candidates_per_prompt):
                gen_params = GenerationParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=256,
                    do_sample=True
                )

                generated = self.policy_model.generate(prompt, gen_params)
                sql, _ = self.policy_model.extract_sql_and_reasoning(generated)

                if not sql:
                    sql = self.policy_model.extract_sql(generated)

                candidates.append({
                    "sql": sql if sql else "",
                    "full_response": generated
                })

            # Evaluate rewards for candidates
            for candidate in candidates:
                # Get reward from evaluator
                reward_result = await self.reward_evaluator.evaluate_sql(
                    query=query,
                    generated_sql=candidate["sql"],
                    expected_sql=expected_sql,
                    schema_context=batch.schemas[i] if i < len(batch.schemas) else ""
                )

                reward = reward_result.get("total_reward", 0.0)

                # Compute log probabilities for KL divergence
                policy_logprobs = self._get_log_probs(
                    self.policy_model,
                    prompt,
                    candidate["full_response"]
                )

                ref_logprobs = self._get_log_probs(
                    self.reference_model,
                    prompt,
                    candidate["full_response"]
                )

                trajectories.append({
                    "prompt": prompt,
                    "query": query,
                    "response": candidate["full_response"],
                    "sql": candidate["sql"],
                    "reward": reward,
                    "policy_logprobs": policy_logprobs,
                    "ref_logprobs": ref_logprobs
                })

                # Log best examples
                if reward > 0.8:
                    self.logger.info(f"High reward SQL: {candidate['sql'][:100]}")

        return trajectories

    def _get_log_probs(
        self,
        model: QwenModelWrapper,
        prompt: str,
        response: str
    ) -> torch.Tensor:
        """
        Calculate log probabilities for a response.
        Simplified version for MPS compatibility.
        """
        try:
            # Tokenize input
            full_text = prompt + response
            inputs = model.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Move to device
            if self.device == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get logits
            with torch.no_grad():
                outputs = model.model(**inputs)
                logits = outputs.logits

            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get the log prob of the generated tokens
            # This is simplified - in practice you'd want token-by-token probs
            mean_log_prob = log_probs.mean()

            return mean_log_prob

        except Exception as e:
            self.logger.warning(f"Error calculating log probs: {e}")
            return torch.tensor(0.0, device=self.device)

    def _compute_advantages(self, trajectories: List[Dict]) -> np.ndarray:
        """
        Compute advantages using reward signals.
        """
        rewards = np.array([t["reward"] for t in trajectories])

        # Normalize advantages
        advantages = rewards - rewards.mean()
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)

        return advantages

    def _optimize_policy(
        self,
        trajectories: List[Dict],
        advantages: np.ndarray
    ) -> tuple:
        """
        Optimize policy using GRPO objective.
        """
        self.policy_model.model.train()

        total_loss = 0
        total_kl = 0
        num_batches = 0

        for i, trajectory in enumerate(trajectories):
            # Calculate KL divergence
            kl_div = trajectory["policy_logprobs"] - trajectory["ref_logprobs"]

            # GRPO loss = -advantage * log_prob + kl_penalty
            loss = -advantages[i] * trajectory["policy_logprobs"]
            loss += self.config.kl_coefficient * kl_div

            # Backward pass
            loss.backward()

            # Accumulate gradients
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            total_kl += kl_div.item()
            num_batches += 1

        # Final optimizer step if needed
        if num_batches % self.config.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        avg_kl = total_kl / max(num_batches, 1)

        self.global_step += 1

        return avg_loss, avg_kl

    async def _validate(self):
        """
        Validate model on validation set.
        """
        self.logger.info("Running validation...")
        self.policy_model.model.eval()

        val_rewards = []
        val_batch = self.dataset_loader.get_sft_batch(
            batch_size=min(5, self.dataset_loader.stats.get('val_size', 3)),
            split="val"
        )

        if not val_batch:
            return

        prompt_templates = SQLPromptTemplates()

        for i in range(len(val_batch.queries)):
            query = val_batch.queries[i]
            expected_sql = val_batch.sqls[i]

            prompt = prompt_templates.zero_shot_sql_generation(
                query=query,
                schema_context=val_batch.schemas[i] if i < len(val_batch.schemas) else ""
            )

            gen_params = GenerationParams(
                temperature=0.1,
                top_p=0.9,
                max_new_tokens=256,
                do_sample=False
            )

            generated = self.policy_model.generate(prompt, gen_params)
            sql, _ = self.policy_model.extract_sql_and_reasoning(generated)

            if not sql:
                sql = self.policy_model.extract_sql(generated)

            # Evaluate
            reward_result = await self.reward_evaluator.evaluate_sql(
                query=query,
                generated_sql=sql if sql else "",
                expected_sql=expected_sql
            )

            reward = reward_result.get("total_reward", 0.0)
            val_rewards.append(reward)

            self.logger.info(f"Val Query: {query[:50]}... Reward: {reward:.3f}")

        avg_val_reward = np.mean(val_rewards) if val_rewards else 0
        self.logger.info(f"Average validation reward: {avg_val_reward:.3f}")

        return avg_val_reward

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "iteration": self.current_iteration,
            "global_step": self.global_step,
            "model_state_dict": self.policy_model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "training_history": self.training_history,
            "config": self.config.__dict__,
            "best_avg_reward": self.best_avg_reward
        }

        # Save LoRA weights if using LoRA
        if self.config.use_lora:
            checkpoint["lora_state_dict"] = self.policy_model.get_lora_state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _save_training_history(self):
        """Save training history."""
        import json

        history_path = Path(self.config.checkpoint_dir) / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Saved training history: {history_path}")