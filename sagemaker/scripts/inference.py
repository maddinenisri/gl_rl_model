#!/usr/bin/env python3
"""
GL RL Model Inference Script for SageMaker Endpoints
Optimized for CPU inference with batch processing support
"""

import os
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """
    Load the model for inference
    Called once when the endpoint is created
    """
    logger.info(f"Loading model from {model_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        padding_side='left'
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True
    )

    # Check if this is a LoRA model
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        logger.info("Loading LoRA adapter")
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        model = base_model

    # Set to evaluation mode
    model.eval()

    return {
        'model': model,
        'tokenizer': tokenizer
    }


def input_fn(request_body, content_type='application/json'):
    """
    Parse input data
    Supports both single query and batch processing
    """
    if content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {content_type}")

    input_data = json.loads(request_body)

    # Support both single and batch inputs
    if isinstance(input_data, dict):
        # Single query
        queries = [input_data]
    elif isinstance(input_data, list):
        # Batch queries
        queries = input_data
    else:
        raise ValueError("Input must be a dict or list of dicts")

    return queries


def predict_fn(input_data, model_dict):
    """
    Generate SQL queries from natural language
    """
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']

    predictions = []

    for query_data in input_data:
        query = query_data.get('query', '')
        context = query_data.get('context', '')
        max_length = query_data.get('max_length', 200)
        temperature = query_data.get('temperature', 0.7)

        # Format prompt
        if context:
            prompt = f"""<|im_start|>system
You are a SQL expert. Generate SQL queries based on natural language questions.
Schema: {context}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant"""
        else:
            prompt = f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant"""

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract SQL from response
        sql = extract_sql(generated, prompt)

        predictions.append({
            'query': query,
            'sql': sql,
            'full_response': generated
        })

    return predictions


def output_fn(predictions, accept='application/json'):
    """
    Format output for response
    """
    if accept != 'application/json':
        raise ValueError(f"Unsupported accept type: {accept}")

    # Return single result if input was single
    if len(predictions) == 1:
        return json.dumps(predictions[0])
    else:
        return json.dumps(predictions)


def extract_sql(generated_text, prompt):
    """
    Extract SQL query from generated text
    """
    # Remove the prompt from the generated text
    response = generated_text[len(prompt):].strip()

    # Look for SQL patterns
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']

    # Find the SQL query in the response
    lines = response.split('\n')
    sql_lines = []
    in_sql = False

    for line in lines:
        line_upper = line.upper().strip()

        # Check if line starts with SQL keyword
        if any(line_upper.startswith(keyword) for keyword in sql_keywords):
            in_sql = True
            sql_lines.append(line)
        elif in_sql and line.strip().endswith(';'):
            sql_lines.append(line)
            break
        elif in_sql and line.strip():
            sql_lines.append(line)
        elif in_sql and not line.strip():
            # Empty line might indicate end of SQL
            break

    sql = '\n'.join(sql_lines).strip()

    # If no SQL found, return the entire response
    if not sql:
        sql = response.split('<|im_end|>')[0].strip()

    return sql


# For local testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "./model"

    # Load model
    print(f"Loading model from {model_dir}")
    model_dict = model_fn(model_dir)

    # Test query
    test_input = {
        "query": "Show me all customers",
        "context": "customers(id, name, email, created_at)"
    }

    # Parse input
    input_data = input_fn(json.dumps(test_input))

    # Generate prediction
    predictions = predict_fn(input_data, model_dict)

    # Format output
    output = output_fn(predictions)

    print("Test Result:")
    print(output)