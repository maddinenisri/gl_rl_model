#!/bin/bash

# =============================================================================
# GL RL Model - Simple SageMaker Setup Script
# =============================================================================
# Minimal setup that works on t2.medium instances
# Just run: bash sagemaker_setup_simple.sh
# =============================================================================

set -e

echo "========================================="
echo "🚀 GL RL Model - Simple Setup"
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
echo "📦 Installing minimal dependencies..."

# Suppress the DLAMI warning
export IGNORE_DLAMI_WARNING=1

# Use pip with PyPI to avoid conda issues on t2.medium
echo "Using pip for package installation..."

# Upgrade pip first
pip install --upgrade pip -q

# Install core packages from PyPI
echo "Installing PyTorch (CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu -q

echo "Installing Transformers ecosystem..."
# Install without datasets first to avoid pyarrow build issues
pip install transformers huggingface-hub tokenizers accelerate -q

# Try to install datasets (may fail due to pyarrow)
echo "Attempting datasets installation..."
pip install datasets -q 2>/dev/null || {
    echo "⚠️ datasets installation failed (pyarrow build issue)"
    echo "Installing datasets without pyarrow dependency..."
    pip install datasets --no-deps -q
    pip install fsspec aiohttp multiprocess dill pandas tqdm requests -q
}

echo "Installing fine-tuning libraries..."
pip install peft trl -q

echo "Installing utilities..."
pip install pandas numpy tqdm protobuf -q

# Try to install sentencepiece (may fail on some systems)
echo "Installing sentencepiece..."
pip install sentencepiece -q 2>/dev/null || {
    echo "⚠️ sentencepiece installation failed"
    echo "Trying alternative installation method..."

    # Try downloading pre-built wheel for Linux x86_64
    pip install https://files.pythonhosted.org/packages/4b/11/77e7807b5f5631eef22eefa2de89ded748a5430f6c93e1bb8b6032c0eb03/sentencepiece-0.1.99-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -q 2>/dev/null || \
    echo "⚠️ sentencepiece could not be installed - some features may be limited"
}

echo ""
echo "📥 Setting up training data..."
mkdir -p data/training

# Create sample training data
cat > data/training/query_pairs.jsonl << 'EOF'
{"query": "Show me all customers", "sql": "SELECT * FROM customers;", "context": "customers(id, name, email, created_at)"}
{"query": "Get total sales by month", "sql": "SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) as total FROM sales GROUP BY month;", "context": "sales(id, date, amount, product_id)"}
{"query": "Find top 5 products by revenue", "sql": "SELECT p.name, SUM(s.amount) as revenue FROM products p JOIN sales s ON p.id = s.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 5;", "context": "products(id, name, price), sales(id, product_id, amount)"}
{"query": "List users who registered today", "sql": "SELECT * FROM users WHERE DATE(created_at) = CURDATE();", "context": "users(id, name, email, created_at)"}
{"query": "Calculate average order value", "sql": "SELECT AVG(total_amount) as avg_order_value FROM orders;", "context": "orders(id, customer_id, total_amount, order_date)"}
EOF

echo "✅ Created sample training data"

# Try to download from S3 (may fail if no access)
aws s3 cp s3://gl-rl-model-sagemaker-340350204194-us-east-1/data/training/query_pairs.jsonl data/training/query_pairs_full.jsonl 2>/dev/null && \
    echo "✅ Downloaded full training data from S3" || \
    echo "ℹ️ Using local sample data"

echo ""
echo "📓 Creating test script..."

# Create a simple Python test script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify the setup"""

import sys
print("🔍 Testing GL RL Model Setup...")
print("=" * 50)

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} installed")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")

# Test Transformers
try:
    import transformers
    print(f"✅ Transformers {transformers.__version__} installed")
except ImportError as e:
    print(f"❌ Transformers not installed: {e}")

# Test Datasets
try:
    import datasets
    print(f"✅ Datasets {datasets.__version__} installed")
except ImportError as e:
    print(f"❌ Datasets not installed: {e}")

# Test PEFT
try:
    import peft
    print(f"✅ PEFT {peft.__version__} installed")
except ImportError as e:
    print(f"❌ PEFT not installed: {e}")

# Test TRL
try:
    import trl
    print(f"✅ TRL {trl.__version__} installed")
except ImportError as e:
    print(f"❌ TRL not installed: {e}")

# Test Sentencepiece
try:
    import sentencepiece
    print(f"✅ Sentencepiece {sentencepiece.__version__} installed")
except ImportError as e:
    print(f"⚠️ Sentencepiece not installed (optional)")

# Test data loading
print("\n📚 Testing data loading...")
import json
import os

data_file = 'data/training/query_pairs.jsonl'
if os.path.exists(data_file):
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(data)} training examples")
    print(f"   Sample: {data[0]['query'][:50]}...")
else:
    print(f"❌ Training data not found at {data_file}")

# Test tokenizer
print("\n🔤 Testing tokenizer...")
try:
    from transformers import AutoTokenizer
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print(f"Loading tokenizer for {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    test_text = "SELECT * FROM users WHERE age > 25;"
    tokens = tokenizer.encode(test_text)
    print(f"✅ Tokenizer works! Input: '{test_text}'")
    print(f"   Encoded to {len(tokens)} tokens")
except Exception as e:
    print(f"⚠️ Tokenizer test failed: {e}")
    print("   This is OK for initial setup")

print("\n" + "=" * 50)
print("Setup test complete!")
print("\nNext steps:")
print("1. Open GL_RL_Model_Quick_Start.ipynb")
print("2. Select kernel: conda_pytorch_p310")
print("3. Run the notebook cells")
EOF

chmod +x test_setup.py

echo ""
echo "🧪 Running setup test..."
python test_setup.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "========================================="
echo "📝 Summary"
echo "========================================="
echo ""
echo "✅ Repository: gl_rl_model"
echo "✅ Dependencies: Installed via pip"
echo "✅ Training data: Created in data/training/"
echo "✅ Test script: test_setup.py"
echo ""
echo "To verify installation again, run:"
echo "  python test_setup.py"
echo ""
echo "For GPU training, use SageMaker Training Jobs"
echo "with ml.p3.2xlarge instances"
echo ""
echo "========================================="