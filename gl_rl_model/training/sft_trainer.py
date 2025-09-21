"""
Supervised Fine-Tuning (SFT) trainer for GL RL Model.

This module implements supervised fine-tuning for the Qwen model on SQL generation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict

from ..models.qwen_wrapper import QwenModelWrapper
from ..utils.prompt_templates import SQLPromptTemplates
from .dataset_loader import DatasetLoader, SFTBatch

@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Data
    max_sequence_length: int = 2048
    val_check_interval: int = 100
    save_interval: int = 500

    # Curriculum
    use_curriculum: bool = True
    curriculum_schedule: Dict[str, int] = None

    # Paths
    checkpoint_dir: str = None
    log_dir: str = None

    def __post_init__(self):
        if self.curriculum_schedule is None:
            self.curriculum_schedule = {
                "easy": 1,  # Epoch 1
                "medium": 2,  # Epoch 2
                "hard": 3   # Epoch 3+
            }

        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(__file__).parent / "checkpoints" / "sft"

        if self.log_dir is None:
            self.log_dir = Path(__file__).parent / "logs" / "sft"

class SFTDataset(Dataset):
    """PyTorch dataset for SFT training."""

    def __init__(
        self,
        examples: List[Tuple[str, str, str]],
        tokenizer,
        max_length: int = 2048
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, sql, reasoning = self.examples[idx]

        # Format with prompt template including schema context
        prompt_gen = SQLPromptTemplates()
        full_prompt = prompt_gen.generate_training_prompt(
            query=query,
            sql=sql,
            reasoning=reasoning,
            schema_context=None  # Will be loaded automatically by generate_training_prompt
        )

        # Tokenize
        encoding = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For language modeling
        }

class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for SQL generation.

    This trainer implements curriculum learning and reasoning-aware training
    for the Qwen model.
    """

    def __init__(
        self,
        config: Optional[SFTConfig] = None,
        model: Optional[QwenModelWrapper] = None,
        dataset_loader: Optional[DatasetLoader] = None
    ):
        """
        Initialize the SFT trainer.

        Args:
            config: Training configuration
            model: Qwen model wrapper (optional, will create if not provided)
            dataset_loader: Dataset loader (optional, will create if not provided)
        """
        self.config = config or SFTConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        if model is None:
            self.logger.info("Initializing Qwen model...")
            self.model_wrapper = QwenModelWrapper(
                model_name_or_path=self.config.model_name,
                use_lora=self.config.use_lora
                # Note: lora_rank, lora_alpha, lora_dropout are configured in QwenModelWrapper's config
            )
            self.model_wrapper.load_model()
        else:
            self.model_wrapper = model

        self.model = self.model_wrapper.model
        self.tokenizer = self.model_wrapper.tokenizer

        # Initialize dataset
        if dataset_loader is None:
            self.logger.info("Initializing dataset loader...")
            self.dataset_loader = DatasetLoader()
        else:
            self.dataset_loader = dataset_loader

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.current_epoch = 0

        # Metrics tracking
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": []
        }

    def setup_training(self):
        """Setup training components."""
        # Setup optimizer
        if self.config.use_lora:
            # Only optimize LoRA parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()

        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Calculate total steps
        train_size = len(self.dataset_loader.train_data)
        steps_per_epoch = train_size // (self.config.batch_size * self.config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.config.num_epochs

        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

        self.logger.info(f"Training setup complete. Total steps: {total_steps}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting SFT training...")
        self.setup_training()

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            current_difficulty = self._get_curriculum_difficulty(epoch)

            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            if self.config.use_curriculum:
                self.logger.info(f"Curriculum difficulty: {current_difficulty}")

            # Train epoch
            train_loss = self._train_epoch(current_difficulty)
            self.logger.info(f"Average training loss: {train_loss:.4f}")

            # Validation
            val_loss = self._validate()
            self.logger.info(f"Validation loss: {val_loss:.4f}")

            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                self.logger.info("Saved best checkpoint")

            # Regular checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")

            # Log metrics
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["epoch"].append(epoch + 1)

        # Save final model
        self._save_checkpoint("final")
        self._save_training_history()

        self.logger.info("Training complete!")
        return self.training_history

    def _train_epoch(self, current_difficulty: str = "medium") -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulation_counter = 0

        # Create progress bar
        epoch_iterator = self.dataset_loader.iterate_epochs(
            num_epochs=1,
            batch_size=self.config.batch_size,
            split="train",
            shuffle=True
        )

        with tqdm(epoch_iterator, desc="Training") as pbar:
            for _, batch in pbar:
                # Get curriculum batch if enabled
                if self.config.use_curriculum:
                    batch = self.dataset_loader.get_curriculum_batch(
                        batch_size=self.config.batch_size,
                        current_difficulty=current_difficulty,
                        split="train"
                    )

                if batch is None:
                    continue

                # Prepare batch
                loss = self._compute_batch_loss(batch)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                accumulation_counter += 1

                # Gradient accumulation
                if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Log learning rate
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.training_history["learning_rate"].append(current_lr)
                    self.training_history["step"].append(self.global_step)

                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # Validation check
                if self.global_step % self.config.val_check_interval == 0:
                    val_loss = self._validate()
                    self.logger.info(f"Step {self.global_step} - Val loss: {val_loss:.4f}")
                    self.model.train()

                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

        return total_loss / max(num_batches, 1)

    def _compute_batch_loss(self, batch: SFTBatch) -> torch.Tensor:
        """Compute loss for a batch."""
        total_loss = 0
        num_examples = len(batch.queries)

        for i in range(num_examples):
            # Format prompt with schema context
            prompt_gen = SQLPromptTemplates()
            full_prompt = prompt_gen.generate_training_prompt(
                query=batch.queries[i],
                sql=batch.sqls[i],
                reasoning=batch.reasonings[i],
                schema_context=None  # Will be loaded automatically by generate_training_prompt
            )

            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length
            ).to(self.model.device)

            # Forward pass (handle both GPU and CPU)
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
            else:
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            total_loss += loss

        return total_loss / num_examples

    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for _ in range(10):  # Validate on 10 batches
                batch = self.dataset_loader.get_sft_batch(
                    batch_size=self.config.batch_size,
                    split="val"
                )

                if batch is None:
                    break

                loss = self._compute_batch_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _get_curriculum_difficulty(self, epoch: int) -> str:
        """Get curriculum difficulty for current epoch."""
        if not self.config.use_curriculum:
            return "medium"

        for difficulty, threshold_epoch in self.config.curriculum_schedule.items():
            if epoch < threshold_epoch:
                return difficulty

        return "hard"

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}.pt"

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict() if not self.config.use_lora else None,
            "lora_state_dict": self.model_wrapper.get_lora_state_dict() if self.config.use_lora else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "training_history": self.training_history
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if self.config.use_lora and "lora_state_dict" in checkpoint:
            self.model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.training_history = checkpoint.get("training_history", self.training_history)

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = Path(self.config.log_dir) / "training_history.json"

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Saved training history: {history_path}")

    def evaluate(self, test_examples: Optional[List] = None) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            test_examples: Optional test examples, uses test split if not provided

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        results = {
            "total_examples": 0,
            "avg_loss": 0,
            "exact_match": 0,
            "sql_valid": 0,
            "examples": []
        }

        # Get test data
        if test_examples is None:
            test_batches = []
            for _ in range(5):  # Evaluate on 5 batches
                batch = self.dataset_loader.get_sft_batch(
                    batch_size=self.config.batch_size,
                    split="test"
                )
                if batch:
                    test_batches.append(batch)
        else:
            # Convert to batch format
            test_batches = [SFTBatch(
                queries=[ex["query"] for ex in test_examples],
                sqls=[ex["sql"] for ex in test_examples],
                reasonings=[ex.get("reasoning", "") for ex in test_examples],
                metadata=[ex.get("metadata", {}) for ex in test_examples]
            )]

        with torch.no_grad():
            for batch in test_batches:
                # Compute loss
                loss = self._compute_batch_loss(batch)
                results["avg_loss"] += loss.item()

                # Generate predictions
                for i in range(len(batch.queries)):
                    query = batch.queries[i]
                    expected_sql = batch.sqls[i]

                    # Generate SQL
                    generated = self.model_wrapper.generate(query)

                    # Extract SQL from generated text
                    sql = self.model_wrapper.extract_sql(generated)

                    # Check exact match
                    if sql.strip() == expected_sql.strip():
                        results["exact_match"] += 1

                    # Store example
                    results["examples"].append({
                        "query": query,
                        "expected": expected_sql,
                        "generated": sql,
                        "match": sql.strip() == expected_sql.strip()
                    })

                    results["total_examples"] += 1

        # Calculate metrics
        if results["total_examples"] > 0:
            results["avg_loss"] /= len(test_batches)
            results["exact_match_rate"] = results["exact_match"] / results["total_examples"]
        else:
            results["exact_match_rate"] = 0

        return results

    def generate_sql(self, query: str, schema_context: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Generate SQL for a query using the trained model.

        Args:
            query: Natural language query
            schema_context: Optional schema context

        Returns:
            Tuple of (sql, reasoning)
        """
        self.model.eval()

        with torch.no_grad():
            # Generate with model
            generated = self.model_wrapper.generate(query)

            # Extract SQL and reasoning
            sql = self.model_wrapper.extract_sql(generated)
            reasoning = self.model_wrapper.extract_reasoning(generated)

        return sql, reasoning