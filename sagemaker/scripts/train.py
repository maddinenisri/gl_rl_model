#!/usr/bin/env python3
"""
GL RL Model Training Script for SageMaker
Unified training script with automatic dependency handling
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Check if running in SageMaker training container
IS_SAGEMAKER_TRAINING = 'SM_MODEL_DIR' in os.environ

if IS_SAGEMAKER_TRAINING:
    # Install required packages for SageMaker training container
    import subprocess
    print("Installing required packages for SageMaker training...")

    # First, upgrade transformers to fix compatibility issues
    print("Upgrading transformers for compatibility...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'transformers>=4.36.0'])

    requirements_file = Path(__file__).parent.parent / 'requirements' / 'sagemaker-training.txt'
    if requirements_file.exists():
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)])
    else:
        # Fallback to essential packages
        packages = ['datasets>=2.14.0', 'peft>=0.6.0', 'accelerate>=0.24.0', 'sentencepiece>=0.1.99']
        for package in packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class GLRLTrainer:
    """Clean trainer class for GL RL Model"""

    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def setup_paths(self):
        """Setup SageMaker or local paths"""
        self.model_dir = os.environ.get('SM_MODEL_DIR', './model')
        self.output_dir = os.environ.get('SM_OUTPUT_DIR', './output')
        self.train_dir = os.environ.get('SM_CHANNEL_TRAINING', './data/training')

        # Create directories if they don't exist
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load training data from S3 or local"""
        logger.info(f"Loading data from {self.train_dir}")

        # Find JSONL files
        data_path = Path(self.train_dir)
        jsonl_files = list(data_path.glob("*.jsonl")) or list(data_path.rglob("*.jsonl"))

        if not jsonl_files:
            logger.warning("No training data found, creating sample data...")
            sample_data = self._create_sample_data()
            sample_file = data_path / "sample_data.jsonl"
            with open(sample_file, 'w') as f:
                for item in sample_data:
                    f.write(json.dumps(item) + '\n')
            jsonl_files = [sample_file]

        # Load dataset
        dataset = load_dataset('json', data_files=[str(f) for f in jsonl_files])
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)

        logger.info(f"Training examples: {len(dataset['train'])}")
        logger.info(f"Evaluation examples: {len(dataset['test'])}")

        return dataset

    def _create_sample_data(self):
        """Create sample training data"""
        return [
            {
                "query": "Show me all customers",
                "sql": "SELECT * FROM customers;",
                "context": "customers(id, name, email, created_at)"
            },
            {
                "query": "Get total sales by month",
                "sql": "SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) as total FROM sales GROUP BY month;",
                "context": "sales(id, date, amount, product_id)"
            },
            {
                "query": "Find top 5 products by revenue",
                "sql": "SELECT p.name, SUM(s.amount) as revenue FROM products p JOIN sales s ON p.id = s.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 5;",
                "context": "products(id, name, price), sales(id, product_id, amount)"
            },
            {
                "query": "List users who registered today",
                "sql": "SELECT * FROM users WHERE DATE(created_at) = CURDATE();",
                "context": "users(id, name, email, created_at)"
            },
            {
                "query": "Calculate average order value",
                "sql": "SELECT AVG(total_amount) as avg_order_value FROM orders;",
                "context": "orders(id, customer_id, total_amount, order_date)"
            }
        ]

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA"""
        logger.info(f"Loading model: {self.args.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            padding_side='left'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Configure and apply LoRA
        logger.info(f"Configuring LoRA: r={self.args.lora_r}, alpha={self.args.lora_alpha}")
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def preprocess_data(self, examples):
        """Preprocess examples for training"""
        prompts = []
        responses = []

        for query, sql, context in zip(examples['query'], examples['sql'], examples['context']):
            prompt = f"""<|im_start|>system
You are a SQL expert. Generate SQL queries based on natural language questions.
Schema: {context}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant"""

            response = f"{sql}<|im_end|>"

            prompts.append(prompt)
            responses.append(response)

        # Tokenize
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.args.max_length,
            truncation=True,
            padding=True
        )

        # Tokenize responses
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                responses,
                max_length=self.args.max_length,
                truncation=True,
                padding=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        """Compute training metrics"""
        predictions, labels = eval_pred
        loss = np.mean(predictions)
        perplexity = np.exp(loss)
        return {'perplexity': perplexity, 'loss': loss}

    def train(self):
        """Main training loop"""
        # Load data
        dataset = self.load_data()

        # Setup model and tokenizer
        self.setup_model_and_tokenizer()

        # Preprocess data
        logger.info("Preprocessing data...")
        tokenized_dataset = dataset.map(
            lambda x: self.preprocess_data(x),
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            learning_rate=self.args.learning_rate,
            fp16=self.args.fp16,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=self.args.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.args.eval_steps,
            save_strategy="steps",
            save_steps=self.args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"] if IS_SAGEMAKER_TRAINING else [],
            push_to_hub=False,
            gradient_checkpointing=self.args.gradient_checkpointing,
            optim="adamw_torch",
            remove_unused_columns=False
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save model
        logger.info(f"Saving model to {self.model_dir}")
        trainer.save_model(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        # Save metrics
        metrics = trainer.evaluate()
        with open(f"{self.output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete! Metrics: {metrics}")

        # Test the model
        self.test_model()

    def test_model(self):
        """Test the trained model"""
        test_query = "Show me all customers"
        test_input = f"""<|im_start|>user
{test_query}<|im_end|>
<|im_start|>assistant"""

        inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200, temperature=0.7)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Test query: {test_query}")
        logger.info(f"Generated SQL: {response}")


def parse_args():
    """Parse training arguments"""
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model_name', type=str,
                       default='Qwen/Qwen2.5-Coder-1.5B-Instruct',
                       help='Hugging Face model name')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Optimization parameters
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--gradient_checkpointing', type=bool, default=True)

    # Output parameters
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Log GPU information if available
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Initialize and run trainer
    trainer = GLRLTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()