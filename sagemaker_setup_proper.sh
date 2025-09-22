#!/bin/bash

# =============================================================================
# GL RL Model - Proper SageMaker Setup Script
# =============================================================================
# Uses best practices for SageMaker notebook instances
# Based on AWS documentation for package installation
# =============================================================================

set -e

echo "========================================="
echo "🚀 GL RL Model - Proper SageMaker Setup"
echo "========================================="
echo ""

# Check if we're in SageMaker
if [ ! -d "/home/ec2-user/SageMaker" ]; then
    echo "❌ Error: This script should be run in a SageMaker notebook instance"
    echo "Please run this from the SageMaker Jupyter terminal"
    exit 1
fi

# Navigate to SageMaker directory
cd /home/ec2-user/SageMaker

echo "📁 Current directory: $(pwd)"
echo ""

# Check if repo exists, if not clone it
if [ -d "gl_rl_model" ]; then
    echo "✅ Repository already exists"
    cd gl_rl_model
    echo "📥 Pulling latest changes..."
    git pull origin main || true
else
    echo "📥 Cloning repository..."
    git clone https://github.com/maddinenisri/gl_rl_model.git
    cd gl_rl_model
fi

echo ""
echo "🔍 Checking pip version..."
pip_version=$(pip --version | awk '{print $2}')
echo "Current pip version: $pip_version"

# Upgrade pip to ensure we get prebuilt wheels (need >= 19.0)
echo "⬆️ Upgrading pip to latest version..."
pip install --upgrade pip

echo ""
echo "📦 Installing packages using prebuilt wheels..."

# Install PyTorch CPU version first (prebuilt wheel)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install critical packages with prebuilt wheels
echo "Installing core packages with prebuilt wheels..."
pip install --no-cache-dir \
    numpy \
    pandas \
    pyarrow \
    transformers \
    tokenizers \
    huggingface-hub \
    accelerate \
    datasets \
    peft \
    trl \
    sentencepiece \
    protobuf \
    tqdm \
    fsspec \
    aiohttp

echo ""
echo "📥 Setting up training data..."
mkdir -p data/training

# Create sample training data if it doesn't exist
if [ ! -f "data/training/query_pairs.jsonl" ]; then
    cat > data/training/query_pairs.jsonl << 'EOF'
{"query": "Show me all customers", "sql": "SELECT * FROM customers;", "context": "customers(id, name, email, created_at)"}
{"query": "Get total sales by month", "sql": "SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) as total FROM sales GROUP BY month;", "context": "sales(id, date, amount, product_id)"}
{"query": "Find top 5 products by revenue", "sql": "SELECT p.name, SUM(s.amount) as revenue FROM products p JOIN sales s ON p.id = s.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 5;", "context": "products(id, name, price), sales(id, product_id, amount)"}
{"query": "List users who registered today", "sql": "SELECT * FROM users WHERE DATE(created_at) = CURDATE();", "context": "users(id, name, email, created_at)"}
{"query": "Calculate average order value", "sql": "SELECT AVG(total_amount) as avg_order_value FROM orders;", "context": "orders(id, customer_id, total_amount, order_date)"}
EOF
    echo "✅ Created sample training data"
else
    echo "✅ Training data already exists"
fi

# Try to download from S3 (may fail if no access)
aws s3 cp s3://gl-rl-model-sagemaker-340350204194-us-east-1/data/training/query_pairs.jsonl data/training/query_pairs_full.jsonl 2>/dev/null && \
    echo "✅ Downloaded full training data from S3" || \
    echo "ℹ️ Using local sample data"

echo ""
echo "📓 Creating test notebook..."

# Create a simple test notebook
cat > test_environment.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GL RL Model - Environment Test\n",
    "This notebook tests that all dependencies are properly installed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "print(f\"Python version: {sys.version}\")\n",
    "print(f\"Python executable: {sys.executable}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test imports\n",
    "import torch\n",
    "import transformers\n",
    "import datasets\n",
    "import peft\n",
    "import trl\n",
    "import pyarrow\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "print(f\"✅ PyTorch: {torch.__version__}\")\n",
    "print(f\"✅ Transformers: {transformers.__version__}\")\n",
    "print(f\"✅ Datasets: {datasets.__version__}\")\n",
    "print(f\"✅ PEFT: {peft.__version__}\")\n",
    "print(f\"✅ TRL: {trl.__version__}\")\n",
    "print(f\"✅ PyArrow: {pyarrow.__version__}\")\n",
    "print(f\"✅ Pandas: {pd.__version__}\")\n",
    "print(f\"✅ NumPy: {np.__version__}\")\n",
    "print(f\"\\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test data loading\n",
    "import json\n",
    "import os\n",
    "\n",
    "data_file = 'data/training/query_pairs.jsonl'\n",
    "if os.path.exists(data_file):\n",
    "    with open(data_file, 'r') as f:\n",
    "        data = [json.loads(line) for line in f]\n",
    "    print(f\"✅ Loaded {len(data)} training examples\")\n",
    "    print(f\"\\nSample data:\")\n",
    "    for i, example in enumerate(data[:2], 1):\n",
    "        print(f\"\\nExample {i}:\")\n",
    "        print(f\"  Query: {example['query']}\")\n",
    "        print(f\"  SQL: {example['sql']}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test tokenizer loading\n",
    "from transformers import AutoTokenizer\n",
    "\n",
    "model_name = \"Qwen/Qwen2.5-Coder-1.5B-Instruct\"\n",
    "print(f\"Loading tokenizer for {model_name}...\")\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n",
    "test_text = \"SELECT * FROM users WHERE age > 25;\"\n",
    "tokens = tokenizer.encode(test_text)\n",
    "decoded = tokenizer.decode(tokens)\n",
    "\n",
    "print(f\"✅ Tokenizer loaded successfully!\")\n",
    "print(f\"\\nTest text: '{test_text}'\")\n",
    "print(f\"Token count: {len(tokens)}\")\n",
    "print(f\"Decoded: '{decoded}'\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ✅ Environment Setup Complete!\n",
    "\n",
    "All dependencies are properly installed. You can now:\n",
    "1. Open `GL_RL_Model_Quick_Start.ipynb` for the full training pipeline\n",
    "2. Use this notebook instance for development\n",
    "3. Launch GPU training jobs using SageMaker Training"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✅ Created test notebook: test_environment.ipynb"

echo ""
echo "🧪 Running Python test..."

# Create and run a simple Python test
cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
"""Test that all required packages are installed"""

import sys
print(f"Python: {sys.version}")

packages = [
    "torch",
    "transformers",
    "datasets",
    "pyarrow",
    "pandas",
    "numpy",
    "peft",
    "trl",
    "tokenizers",
    "accelerate"
]

failed = []
for package in packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError as e:
        print(f"❌ {package}: {e}")
        failed.append(package)

if not failed:
    print("\n✅ All packages installed successfully!")
else:
    print(f"\n⚠️ Failed to import: {', '.join(failed)}")
    print("You may need to install these packages manually")
EOF

python test_imports.py

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Open test_environment.ipynb to verify the setup"
echo "2. Open GL_RL_Model_Quick_Start.ipynb to start training"
echo "3. Use the Python 3 kernel for best compatibility"
echo ""
echo "To reinstall packages from a notebook, use:"
echo "  %pip install <package_name>"
echo ""
echo "For GPU training, use SageMaker Training Jobs"
echo "with ml.g5.xlarge spot instances ($0.30/hour)"
echo ""
echo "========================================="