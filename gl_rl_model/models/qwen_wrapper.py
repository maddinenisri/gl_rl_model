"""
Qwen2.5-Coder model wrapper for SQL generation with reasoning.
"""

import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from ..core.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class GenerationParams:
    """Parameters for text generation."""
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 1024
    num_return_sequences: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    early_stopping: bool = True

class QwenModelWrapper:
    """
    Wrapper for Qwen2.5-Coder model with LoRA support for SQL generation.

    This class handles model loading, tokenization, generation, and
    LoRA adapter management for efficient fine-tuning.
    """

    def __init__(self, model_name_or_path: Optional[str] = None,
                 use_lora: bool = True,
                 load_in_8bit: bool = False,
                 device_map: str = "auto"):
        """
        Initialize the Qwen model wrapper.

        Args:
            model_name_or_path: Model name or path to load from
            use_lora: Whether to use LoRA adapters
            load_in_8bit: Whether to load model in 8-bit quantization
            device_map: Device placement strategy
        """
        self.config = get_config()
        self.model_name_or_path = model_name_or_path or self.config.model.model_name
        self.use_lora = use_lora
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map

        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.lora_config = None

        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for the model wrapper."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def load_model(self, checkpoint_path: Optional[Path] = None) -> None:
        """
        Load the model and tokenizer.

        Args:
            checkpoint_path: Optional path to load fine-tuned checkpoint
        """
        logger.info(f"Loading model: {self.model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="left"  # Important for batch generation
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config if needed
        bnb_config = None
        if self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=bnb_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.load_in_8bit else None
        )

        # Apply LoRA if enabled
        if self.use_lora:
            if checkpoint_path and checkpoint_path.exists():
                # Load existing LoRA checkpoint
                logger.info(f"Loading LoRA checkpoint from {checkpoint_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    checkpoint_path,
                    is_trainable=True
                )
            else:
                # Create new LoRA config
                self._setup_lora()

        # Set generation config
        self._setup_generation_config()

        # Set model to evaluation mode by default
        self.model.eval()

        logger.info("Model loaded successfully")

    def _setup_lora(self) -> None:
        """Set up LoRA configuration and apply to model."""
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=self.config.model.lora_target_modules,
            bias="none",
            inference_mode=False
        )

        self.model = get_peft_model(self.model, self.lora_config)

        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(
            f"LoRA enabled - Trainable params: {trainable_params:,} / "
            f"Total params: {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def _setup_generation_config(self) -> None:
        """Set up generation configuration."""
        self.generation_config = GenerationConfig(
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            max_new_tokens=self.config.model.max_new_tokens,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def generate(self,
                prompt: str,
                params: Optional[GenerationParams] = None,
                return_tokens: bool = False) -> Union[str, List[str], Tuple[str, torch.Tensor]]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            params: Generation parameters
            return_tokens: Whether to return token IDs along with text

        Returns:
            Generated text or list of texts if num_return_sequences > 1
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = params or GenerationParams()

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_seq_length
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    max_new_tokens=params.max_new_tokens,
                    num_return_sequences=params.num_return_sequences,
                    do_sample=params.do_sample,
                    repetition_penalty=params.repetition_penalty,
                    length_penalty=params.length_penalty,
                    early_stopping=params.early_stopping,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            )

        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Skip input tokens
            generated_tokens = output[inputs['input_ids'].shape[-1]:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        if return_tokens:
            return generated_texts, outputs

        # Return single string if only one sequence
        if params.num_return_sequences == 1:
            return generated_texts[0]

        return generated_texts

    def batch_generate(self,
                      prompts: List[str],
                      params: Optional[GenerationParams] = None,
                      batch_size: int = 4) -> List[str]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts: List of input prompts
            params: Generation parameters
            batch_size: Batch size for generation

        Returns:
            List of generated texts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = params or GenerationParams()
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.model.max_seq_length
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=GenerationConfig(
                        temperature=params.temperature,
                        top_p=params.top_p,
                        max_new_tokens=params.max_new_tokens,
                        do_sample=params.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                )

            # Decode outputs
            for j, output in enumerate(outputs):
                # Skip input tokens
                input_length = inputs['input_ids'][j % len(batch_prompts)].shape[0]
                generated_tokens = output[input_length:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append(text)

        return results

    def format_prompt_for_sql(self,
                            query: str,
                            schema_context: str,
                            examples: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format a prompt for SQL generation with Qwen format.

        Args:
            query: Natural language query
            schema_context: Schema information
            examples: Optional few-shot examples

        Returns:
            Formatted prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are an expert SQL analyst. Generate accurate SQL queries with step-by-step reasoning.\n"
        prompt += "Always provide reasoning before the SQL query using <think> tags.\n"
        prompt += "<|im_end|>\n"

        # Add few-shot examples if provided
        if examples:
            for example in examples[:3]:  # Limit to 3 examples
                prompt += "<|im_start|>user\n"
                prompt += f"Schema: {example.get('schema', '')}\n"
                prompt += f"Query: {example['query']}\n"
                prompt += "<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"
                if 'reasoning' in example:
                    prompt += f"<think>\n{example['reasoning']}\n</think>\n\n"
                prompt += f"{example['sql']}\n"
                prompt += "<|im_end|>\n"

        # Add actual query
        prompt += "<|im_start|>user\n"
        prompt += f"Schema:\n{schema_context}\n\n"
        prompt += f"Query: {query}\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def extract_sql_and_reasoning(self, generated_text: str) -> Tuple[str, str]:
        """
        Extract SQL query and reasoning from generated text.

        Args:
            generated_text: Generated text containing reasoning and SQL

        Returns:
            Tuple of (sql_query, reasoning)
        """
        reasoning = ""
        sql = generated_text

        # Extract reasoning from <think> tags
        if "<think>" in generated_text and "</think>" in generated_text:
            start = generated_text.index("<think>") + 7
            end = generated_text.index("</think>")
            reasoning = generated_text[start:end].strip()
            # Remove reasoning from SQL
            sql = generated_text[end + 8:].strip()

        # Clean up SQL
        sql = sql.strip()

        # Remove any remaining tags or markers
        sql = sql.replace("<|im_end|>", "").strip()

        return sql, reasoning

    def extract_sql(self, generated_text: str) -> str:
        """
        Extract only the SQL query from generated text.

        Args:
            generated_text: Generated text containing SQL

        Returns:
            Extracted SQL query
        """
        sql, _ = self.extract_sql_and_reasoning(generated_text)

        # If no SQL found yet, try to extract from the full generated text
        if not sql and generated_text:
            sql = generated_text

        # Try to extract from markdown code block
        if "```sql" in sql.lower():
            # Extract from markdown code block
            start = sql.lower().index("```sql") + 6
            end = sql.index("```", start) if "```" in sql[start:] else len(sql)
            sql = sql[start:end].strip()
        # Check if there's SQL-like content
        elif any(kw in sql.upper() for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]):
            # Find the SQL statement
            lines = sql.split('\n')
            sql_lines = []
            in_sql = False
            in_code_block = False

            for line in lines:
                # Check for code block markers
                if "```sql" in line.lower():
                    in_code_block = True
                    continue
                elif "```" in line and in_code_block:
                    break

                upper_line = line.upper().strip()

                # Start capturing when we see SQL keywords
                if any(kw in upper_line for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]):
                    in_sql = True

                # Capture lines if we're in SQL mode or code block
                if in_sql or in_code_block:
                    sql_lines.append(line)
                    # Stop at semicolon if not in code block
                    if ';' in line and not in_code_block:
                        break
                    # Or if we hit a line that looks like end of SQL
                    if in_sql and not in_code_block and line.strip() == '':
                        # Check if next line doesn't look like SQL continuation
                        break

            if sql_lines:
                sql = '\n'.join(sql_lines)
        else:
            # No SQL found
            sql = ""

        return sql.strip()

    def extract_reasoning(self, generated_text: str) -> str:
        """
        Extract only the reasoning from generated text.

        Args:
            generated_text: Generated text containing reasoning

        Returns:
            Extracted reasoning
        """
        _, reasoning = self.extract_sql_and_reasoning(generated_text)
        return reasoning

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Save model checkpoint with LoRA weights.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        if self.use_lora:
            logger.info(f"Saving LoRA checkpoint to {checkpoint_path}")
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
        else:
            logger.warning("Model is not using LoRA. Skipping checkpoint save.")

    def set_training_mode(self, mode: bool = True) -> None:
        """
        Set model to training or evaluation mode.

        Args:
            mode: True for training, False for evaluation
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        if mode:
            self.model.train()
            if self.use_lora:
                # Enable LoRA gradients
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.peft_modules():
                    param.requires_grad = True
        else:
            self.model.eval()

    def compute_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute language modeling loss for training.

        Args:
            input_ids: Input token IDs
            labels: Target token IDs

        Returns:
            Loss tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.loss

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model info
        """
        if self.model is None:
            return {"status": "not_loaded"}

        info = {
            "model_name": self.model_name_or_path,
            "use_lora": self.use_lora,
            "load_in_8bit": self.load_in_8bit,
            "device_map": self.device_map,
            "status": "loaded"
        }

        if self.use_lora:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            info["trainable_parameters"] = trainable_params
            info["total_parameters"] = total_params
            info["trainable_percentage"] = 100 * trainable_params / total_params

        return info

    def get_lora_state_dict(self) -> Optional[Dict]:
        """
        Get LoRA adapter state dict.

        Returns:
            LoRA state dict if using LoRA, None otherwise
        """
        if not self.use_lora or self.model is None:
            return None

        return self.model.state_dict()

    def load_lora_state_dict(self, state_dict: Dict) -> None:
        """
        Load LoRA adapter state dict.

        Args:
            state_dict: LoRA state dict to load
        """
        if not self.use_lora or self.model is None:
            raise RuntimeError("Model not using LoRA or not loaded")

        self.model.load_state_dict(state_dict, strict=False)