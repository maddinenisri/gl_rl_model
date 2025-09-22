#!/usr/bin/env python3
"""
GL RL Model Training Script for SageMaker
Trains the Qwen2.5-Coder model for SQL generation using LoRA
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# SageMaker specific paths
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DIR = os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output')
SM_TRAIN_DIR = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')

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
    parser.add_argument('--output_dir', type=str, default=SM_MODEL_DIR)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)

    return parser.parse_args()

def load_training_data(data_path):
    """Load and prepare training data"""
    logger.info(f"Loading training data from {data_path}")

    # Find JSONL files
    jsonl_files = list(Path(data_path).glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {data_path}")

    logger.info(f"Found {len(jsonl_files)} data files")

    # Load dataset
    dataset = load_dataset('json', data_files=[str(f) for f in jsonl_files])

    # Split into train/eval
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)

    logger.info(f"Training examples: {len(dataset['train'])}")
    logger.info(f"Evaluation examples: {len(dataset['test'])}")

    return dataset

def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess examples for training"""

    # Format as instruction-following
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
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding=True
    )

    # Tokenize responses
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            responses,
            max_length=max_length,
            truncation=True,
            padding=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    """Compute training metrics"""
    predictions, labels = eval_pred

    # Calculate perplexity
    loss = np.mean(predictions)
    perplexity = np.exp(loss)

    return {
        'perplexity': perplexity,
        'loss': loss
    }

def main():
    """Main training function"""
    args = parse_args()

    logger.info("Starting GL RL Model training")
    logger.info(f"Arguments: {args}")

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side='left'
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto"
    )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configure LoRA
    logger.info(f"Configuring LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    dataset = load_training_data(SM_TRAIN_DIR)

    # Preprocess data
    logger.info("Preprocessing data...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_dir=f"{SM_OUTPUT_DIR}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        remove_unused_columns=False
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {SM_MODEL_DIR}")
    trainer.save_model(SM_MODEL_DIR)
    tokenizer.save_pretrained(SM_MODEL_DIR)

    # Save training metrics
    metrics = trainer.evaluate()
    with open(f"{SM_OUTPUT_DIR}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training complete! Metrics: {metrics}")

    # Test the model
    test_query = "Show me all customers"
    test_input = f"""<|im_start|>user
{test_query}<|im_end|>
<|im_start|>assistant"""

    inputs = tokenizer(test_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    logger.info(f"Test query: {test_query}")
    logger.info(f"Generated SQL: {response}")

if __name__ == "__main__":
    main()