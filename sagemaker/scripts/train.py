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

    # First, uninstall conflicting versions
    print("Removing conflicting package versions...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tokenizers', 'transformers', 'accelerate'],
                   capture_output=True)

    # Install specific compatible versions in correct order
    print("Installing compatible package versions...")

    # Install specific versions that are known to work together
    # Based on HuggingFace's compatibility matrix
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                          'transformers==4.35.0',
                          'tokenizers==0.14.1',
                          'accelerate==0.24.1'])
    print("✓ Installed transformers==4.35.0, tokenizers==0.14.1, accelerate==0.24.1")

    # Then install other packages with specific compatible versions
    packages = [
        'datasets==2.14.0',  # Specific version for compatibility
        'peft==0.6.0',       # Older PEFT version compatible with accelerate 0.24.1
        'sentencepiece>=0.1.99',
        'protobuf>=3.20.0,<5.0.0',
        'safetensors>=0.3.1',
        'huggingface-hub>=0.16.4',
        'filelock',
        'fsspec',
        'packaging',
        'pyyaml',
        'regex',
        'requests',
        'tqdm',
        'numpy'
    ]

    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            raise

    # Verify installations
    print("\nVerifying package versions...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if any(pkg in line.lower() for pkg in ['tokenizers', 'transformers', 'torch', 'accelerate', 'peft']):
            print(f"  {line.strip()}")

try:
    print("\n=== Starting imports ===")
    import torch
    import numpy as np
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\nImporting transformers...")
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq
    )
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")

    # Check tokenizers version
    import tokenizers
    print(f"✓ Tokenizers version: {tokenizers.__version__}")

    print("\nImporting datasets...")
    from datasets import load_dataset, Dataset
    import datasets
    print(f"✓ Datasets version: {datasets.__version__}")

    print("\nImporting PEFT...")
    from peft import LoraConfig, get_peft_model, TaskType
    import peft
    print(f"✓ PEFT version: {peft.__version__}")

    print("\n=== All imports successful ===\n")

except ImportError as e:
    print(f"\n✗ ERROR during imports: {str(e)}")
    print("\nDetailed import error information:")
    import traceback
    traceback.print_exc()

    # Try to provide helpful debugging info
    print("\n=== Debugging Information ===")
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print("\nInstalled packages:")
    subprocess.run([sys.executable, '-m', 'pip', 'list'], check=False)
    raise
except Exception as e:
    print(f"\n✗ Unexpected ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

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

        # Load all data manually to avoid schema issues
        all_data = []
        for jsonl_file in jsonl_files:
            logger.info(f"Loading {jsonl_file}")
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))

        logger.info(f"Loaded {len(all_data)} examples")

        # Check what fields we have
        if all_data:
            fields = list(all_data[0].keys())
            logger.info(f"Data fields: {fields}")

        # Convert to HuggingFace dataset format
        dataset = Dataset.from_list(all_data)

        # Split into train/test
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

        logger.info(f"Training examples: {len(dataset['train'])}")
        logger.info(f"Evaluation examples: {len(dataset['test'])}")

        return dataset

    def _create_sample_data(self):
        """Create sample training data matching actual data format"""
        # Use 'reasoning' field to match actual training data format
        return [
            {
                "query": "Show all active projects",
                "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active';",
                "reasoning": "Filter projects by active status"
            },
            {
                "query": "Find projects with budget over 100000",
                "sql": "SELECT Project_Code, Project_Name, Budget FROM PAC_MNT_PROJECTS WHERE Budget > 100000;",
                "reasoning": "Filter projects by budget threshold"
            },
            {
                "query": "List all companies",
                "sql": "SELECT Company_Code, Company_Name FROM SRM_COMPANIES;",
                "reasoning": "Select all companies from SRM_COMPANIES table"
            },
            {
                "query": "Show project contracts",
                "sql": "SELECT p.Project_Name, c.Contract_Value FROM PAC_MNT_PROJECTS p JOIN PROJCNTRTS c ON p.Project_Code = c.Project_Code;",
                "reasoning": "Join projects with contracts to show contract values"
            },
            {
                "query": "Get resource allocations",
                "sql": "SELECT Project_Code, Staff_Code, Allocation_Percent FROM PROJSTAFF WHERE Allocation_Percent > 0;",
                "reasoning": "Show staff allocations to projects"
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

        # Handle both 'reasoning' and 'context' fields dynamically
        # Check which field exists in the data
        context_field = None
        if 'reasoning' in examples:
            context_field = 'reasoning'
        elif 'context' in examples:
            context_field = 'context'

        for i in range(len(examples['query'])):
            query = examples['query'][i]
            sql = examples['sql'][i]

            # Get context/reasoning if available
            context = ""
            if context_field:
                context = examples[context_field][i]

            prompt = f"""<|im_start|>system
You are a SQL expert. Generate SQL queries based on natural language questions.
Context: {context}<|im_end|>
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
    parser.add_argument('--fp16', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                       default=True, help='Use fp16 training')
    parser.add_argument('--gradient_checkpointing', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                       default=True, help='Use gradient checkpointing')

    # Output parameters
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)

    return parser.parse_args()


def main():
    """Main entry point"""
    try:
        print("=" * 60)
        print("GL RL Model Training Starting...")
        print("=" * 60)

        args = parse_args()
        print(f"Arguments: {vars(args)}")

        # Log GPU information if available
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("No GPU available, using CPU")

        # Initialize and run trainer
        print("Initializing trainer...")
        trainer = GLRLTrainer(args)

        print("Starting training...")
        trainer.train()

        print("Training completed successfully!")

    except Exception as e:
        print(f"ERROR in main: {str(e)}")
        import traceback
        traceback.print_exc()
        # Re-raise to ensure SageMaker sees the error
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        import sys
        sys.exit(1)