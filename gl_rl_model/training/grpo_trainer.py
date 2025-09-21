"""
Group Relative Policy Optimization (GRPO) trainer for GL RL Model.

This module implements GRPO training for SQL generation reinforcement learning.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
import asyncio

from ..models.qwen_wrapper import QwenModelWrapper, GenerationParams
from ..agents.query_generator import QueryGeneratorAgent
from ..agents.reward_evaluator import RewardEvaluatorAgent
from .dataset_loader import DatasetLoader, GRPOBatch

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    # GRPO specific
    num_iterations: int = 100
    batch_size: int = 4
    num_candidates_per_prompt: int = 4
    kl_coefficient: float = 0.1
    entropy_coefficient: float = 0.01
    advantage_clip: float = 5.0
    gradient_accumulation_steps: int = 4

    # Optimization
    learning_rate: float = 1e-5
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Generation
    generation_temperature: float = 0.8
    generation_top_p: float = 0.9
    generation_max_length: int = 512

    # Validation
    val_check_interval: int = 10
    save_interval: int = 20

    # Paths
    checkpoint_dir: str = None
    log_dir: str = None
    sft_checkpoint: str = None  # Pre-trained SFT model

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(__file__).parent / "checkpoints" / "grpo"

        if self.log_dir is None:
            self.log_dir = Path(__file__).parent / "logs" / "grpo"

class GRPOTrainer:
    """
    GRPO trainer for SQL generation.

    This trainer implements Group Relative Policy Optimization for
    reinforcement learning from rewards.
    """

    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        model: Optional[QwenModelWrapper] = None,
        reward_evaluator: Optional[RewardEvaluatorAgent] = None,
        dataset_loader: Optional[DatasetLoader] = None
    ):
        """
        Initialize the GRPO trainer.

        Args:
            config: Training configuration
            model: Qwen model wrapper
            reward_evaluator: Reward evaluator agent
            dataset_loader: Dataset loader
        """
        self.config = config or GRPOConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        if model is None:
            self.logger.info("Initializing policy model...")
            self.policy_model = QwenModelWrapper(
                model_name=self.config.model_name,
                use_lora=self.config.use_lora,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout
            )
            self.policy_model.load_model()

            # Load SFT checkpoint if available
            if self.config.sft_checkpoint and Path(self.config.sft_checkpoint).exists():
                self.logger.info(f"Loading SFT checkpoint: {self.config.sft_checkpoint}")
                self._load_sft_checkpoint()
        else:
            self.policy_model = model

        # Initialize reference model (frozen copy for KL divergence)
        self.logger.info("Creating reference model...")
        self.reference_model = QwenModelWrapper(
            model_name_or_path=self.config.model_name,
            use_lora=False  # Reference model doesn't need LoRA
        )
        self.reference_model.load_model()

        # Copy weights from policy to reference
        self._sync_reference_model()

        # Freeze reference model
        for param in self.reference_model.model.parameters():
            param.requires_grad = False

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

        # Initialize query generator for sampling
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
            "value_loss": [],
            "kl_divergence": [],
            "learning_rate": [],
            "best_sql_examples": []
        }

    def setup_training(self):
        """Setup training components."""
        # Setup optimizer
        if self.config.use_lora:
            params = [p for p in self.policy_model.model.parameters() if p.requires_grad]
        else:
            params = self.policy_model.model.parameters()

        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.logger.info("Training setup complete")

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
            policy_loss = self._optimize_policy(trajectories, advantages)

            # Log metrics
            avg_reward = np.mean([t["reward"] for t in trajectories])
            avg_advantage = np.mean(advantages)

            self.logger.info(f"Avg reward: {avg_reward:.3f}")
            self.logger.info(f"Avg advantage: {avg_advantage:.3f}")
            self.logger.info(f"Policy loss: {policy_loss:.4f}")

            # Track history
            self.training_history["iteration"].append(iteration + 1)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["avg_advantage"].append(avg_advantage)
            self.training_history["policy_loss"].append(policy_loss)

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

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []

        # Get batch of prompts
        batch = self.dataset_loader.get_sft_batch(
            batch_size=self.config.batch_size,
            split="train"
        )

        if not batch:
            return trajectories

        for i in range(len(batch.queries)):
            prompt = batch.queries[i]
            expected_sql = batch.sqls[i]

            # Generate multiple candidates
            candidates = await self._generate_candidates(
                prompt=prompt,
                num_candidates=self.config.num_candidates_per_prompt
            )

            # Evaluate candidates
            evaluation_result = await self.reward_evaluator.evaluate_batch(
                prompt=prompt,
                candidates=[{"sql": c["sql"], "reasoning": c.get("reasoning", "")}
                          for c in candidates],
                expected_sql=expected_sql
            )

            # Select best and worst for contrastive learning
            rewards = [eval.rewards.total_reward for eval in evaluation_result.candidates]
            best_idx = np.argmax(rewards)
            worst_idx = np.argmin(rewards)

            # Create trajectory for best candidate
            best_trajectory = {
                "prompt": prompt,
                "sql": candidates[best_idx]["sql"],
                "reasoning": candidates[best_idx].get("reasoning", ""),
                "reward": rewards[best_idx],
                "advantage": evaluation_result.advantages[best_idx],
                "log_probs": candidates[best_idx].get("log_probs"),
                "expected_sql": expected_sql
            }
            trajectories.append(best_trajectory)

            # Create trajectory for worst candidate (negative example)
            worst_trajectory = {
                "prompt": prompt,
                "sql": candidates[worst_idx]["sql"],
                "reasoning": candidates[worst_idx].get("reasoning", ""),
                "reward": rewards[worst_idx],
                "advantage": evaluation_result.advantages[worst_idx],
                "log_probs": candidates[worst_idx].get("log_probs"),
                "expected_sql": expected_sql
            }
            trajectories.append(worst_trajectory)

            # Log best example
            if rewards[best_idx] > 5.0:  # High-quality example
                self.training_history["best_sql_examples"].append({
                    "iteration": self.current_iteration,
                    "prompt": prompt,
                    "sql": candidates[best_idx]["sql"],
                    "reward": rewards[best_idx]
                })

        return trajectories

    async def _generate_candidates(
        self,
        prompt: str,
        num_candidates: int
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple SQL candidates for a prompt.

        Args:
            prompt: Natural language query
            num_candidates: Number of candidates to generate

        Returns:
            List of candidate dictionaries
        """
        candidates = []

        # Generation parameters for diversity
        gen_params = GenerationParams(
            temperature=self.config.generation_temperature,
            top_p=self.config.generation_top_p,
            max_length=self.config.generation_max_length
        )

        for _ in range(num_candidates):
            # Generate with policy model
            generated, log_probs = self.policy_model.generate(
                prompt=prompt,
                params=gen_params,
                return_tokens=True
            )

            # Extract SQL and reasoning
            sql = self.policy_model.extract_sql(generated)
            reasoning = self.policy_model.extract_reasoning(generated)

            candidates.append({
                "sql": sql,
                "reasoning": reasoning,
                "generated": generated,
                "log_probs": log_probs if isinstance(log_probs, torch.Tensor) else None
            })

        return candidates

    def _compute_advantages(self, trajectories: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute advantages for trajectories using group normalization.

        Args:
            trajectories: List of trajectory dictionaries

        Returns:
            Array of advantages
        """
        rewards = np.array([t["reward"] for t in trajectories])

        # Group normalization
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8

        advantages = (rewards - mean_reward) / std_reward

        # Clip advantages
        advantages = np.clip(advantages, -self.config.advantage_clip, self.config.advantage_clip)

        return advantages

    def _optimize_policy(
        self,
        trajectories: List[Dict[str, Any]],
        advantages: np.ndarray
    ) -> float:
        """
        Optimize policy using GRPO objective.

        Args:
            trajectories: List of trajectories
            advantages: Computed advantages

        Returns:
            Average policy loss
        """
        self.policy_model.model.train()
        total_loss = 0
        num_updates = 0

        for i, trajectory in enumerate(trajectories):
            if trajectory.get("log_probs") is None:
                continue

            # Compute policy loss
            advantage = torch.tensor(advantages[i], dtype=torch.float32).to(self.policy_model.model.device)

            # Get current policy log probs
            current_log_probs = self._get_log_probs(
                trajectory["prompt"],
                trajectory["sql"],
                trajectory["reasoning"]
            )

            if current_log_probs is None:
                continue

            # Compute KL divergence with reference model
            with torch.no_grad():
                ref_log_probs = self._get_reference_log_probs(
                    trajectory["prompt"],
                    trajectory["sql"],
                    trajectory["reasoning"]
                )

            if ref_log_probs is not None:
                kl_div = current_log_probs - ref_log_probs
                kl_penalty = self.config.kl_coefficient * kl_div.mean()
            else:
                kl_penalty = 0

            # Policy gradient loss
            policy_loss = -(advantage * current_log_probs.mean())

            # Add KL penalty
            loss = policy_loss + kl_penalty

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                num_updates += 1
                self.global_step += 1

            total_loss += loss.item()

        return total_loss / max(len(trajectories), 1)

    def _get_log_probs(
        self,
        prompt: str,
        sql: str,
        reasoning: str
    ) -> Optional[torch.Tensor]:
        """Get log probabilities from current policy."""
        try:
            # Format input
            from ..utils.prompt_templates import SQLPromptTemplates
            prompt_gen = SQLPromptTemplates()
            full_prompt = prompt_gen.generate_training_prompt(
                query=prompt,
                sql=sql,
                reasoning=reasoning
            )

            # Tokenize
            inputs = self.policy_model.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.generation_max_length
            ).to(self.policy_model.model.device)

            # Get logits
            with torch.cuda.amp.autocast():
                outputs = self.policy_model.model(**inputs)
                logits = outputs.logits

            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get log probs for generated tokens
            labels = inputs["input_ids"][:, 1:]  # Shift for next token prediction
            log_probs = log_probs[:, :-1]  # Remove last position

            # Gather log probs for actual tokens
            gathered_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)

            return gathered_log_probs

        except Exception as e:
            self.logger.warning(f"Error computing log probs: {e}")
            return None

    def _get_reference_log_probs(
        self,
        prompt: str,
        sql: str,
        reasoning: str
    ) -> Optional[torch.Tensor]:
        """Get log probabilities from reference model."""
        try:
            # Similar to _get_log_probs but using reference model
            from ..utils.prompt_templates import SQLPromptTemplates
            prompt_gen = SQLPromptTemplates()
            full_prompt = prompt_gen.generate_training_prompt(
                query=prompt,
                sql=sql,
                reasoning=reasoning
            )

            inputs = self.reference_model.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.generation_max_length
            ).to(self.reference_model.model.device)

            with torch.no_grad():
                outputs = self.reference_model.model(**inputs)
                logits = outputs.logits

            log_probs = F.log_softmax(logits, dim=-1)
            labels = inputs["input_ids"][:, 1:]
            log_probs = log_probs[:, :-1]

            gathered_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)

            return gathered_log_probs

        except Exception as e:
            self.logger.warning(f"Error computing reference log probs: {e}")
            return None

    async def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.logger.info("Running validation...")

        val_trajectories = []
        val_rewards = []

        # Collect validation trajectories
        for _ in range(5):  # 5 validation batches
            batch = self.dataset_loader.get_sft_batch(
                batch_size=self.config.batch_size,
                split="val"
            )

            if not batch:
                break

            for i in range(len(batch.queries)):
                # Generate SQL
                candidates = await self._generate_candidates(
                    prompt=batch.queries[i],
                    num_candidates=1
                )

                if candidates:
                    # Evaluate
                    result = await self.reward_evaluator.evaluate_generation(
                        query=batch.queries[i],
                        sql=candidates[0]["sql"],
                        reasoning=candidates[0].get("reasoning", ""),
                        expected_sql=batch.sqls[i]
                    )

                    val_rewards.append(result.rewards.total_reward)

        if val_rewards:
            avg_val_reward = np.mean(val_rewards)
            self.logger.info(f"Validation avg reward: {avg_val_reward:.3f}")
            return {"avg_reward": avg_val_reward}

        return {"avg_reward": 0}

    def _sync_reference_model(self):
        """Sync reference model weights with policy model."""
        self.reference_model.model.load_state_dict(
            self.policy_model.model.state_dict()
        )

    def _load_sft_checkpoint(self):
        """Load pre-trained SFT checkpoint."""
        checkpoint = torch.load(self.config.sft_checkpoint, map_location="cpu")

        if self.config.use_lora and "lora_state_dict" in checkpoint:
            self.policy_model.load_lora_state_dict(checkpoint["lora_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.policy_model.model.load_state_dict(checkpoint["model_state_dict"])

        self.logger.info(f"Loaded SFT checkpoint from {self.config.sft_checkpoint}")

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}.pt"

        checkpoint = {
            "iteration": self.current_iteration,
            "global_step": self.global_step,
            "model_state_dict": self.policy_model.model.state_dict() if not self.config.use_lora else None,
            "lora_state_dict": self.policy_model.get_lora_state_dict() if self.config.use_lora else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "best_avg_reward": self.best_avg_reward,
            "config": asdict(self.config),
            "training_history": self.training_history
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if self.config.use_lora and "lora_state_dict" in checkpoint:
            self.policy_model.load_lora_state_dict(checkpoint["lora_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.policy_model.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_iteration = checkpoint.get("iteration", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_avg_reward = checkpoint.get("best_avg_reward", -float('inf'))
        self.training_history = checkpoint.get("training_history", self.training_history)

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = Path(self.config.log_dir) / "grpo_training_history.json"

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Saved training history: {history_path}")