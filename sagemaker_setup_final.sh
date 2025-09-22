#!/bin/bash

# =============================================================================
# GL RL Model - Final Working SageMaker Setup
# =============================================================================
# This setup uses conda for compiled packages and pip for the rest
# Tested and confirmed working on SageMaker ml.t2.medium instances
# =============================================================================

set -e

echo "========================================="
echo "🚀 GL RL Model - Final Working Setup"
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

# Clone or update repository
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
echo "🔧 Installing packages using the working method..."
echo ""

# Step 1: Use conda for compiled packages
echo "📦 Step 1: Installing compiled packages with conda..."
conda install -c conda-forge sentencepiece pyarrow -y

echo ""
echo "📦 Step 2: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "📦 Step 3: Installing Transformers ecosystem..."
pip install transformers datasets peft trl accelerate huggingface-hub tokenizers

echo ""
echo "📦 Step 4: Fixing dependency conflicts..."
pip install --upgrade multiprocess>=0.70.18

echo ""
echo "📦 Step 5: Installing remaining utilities..."
pip install numpy pandas protobuf tqdm fsspec aiohttp

echo ""
echo "📥 Setting up training data..."
mkdir -p data/training

# Create sample training data
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

echo ""
echo "🧪 Testing installation..."

cat > test_final_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test that all packages work correctly"""

import sys
print(f"Python: {sys.version}\n")
print("Testing imports...")
print("-" * 50)

success = True
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")
    success = False

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")
    success = False

try:
    import datasets
    print(f"✅ Datasets: {datasets.__version__}")
except ImportError as e:
    print(f"❌ Datasets: {e}")
    success = False

try:
    import sentencepiece
    print(f"✅ Sentencepiece: {sentencepiece.__version__}")
except ImportError as e:
    print(f"❌ Sentencepiece: {e}")
    success = False

try:
    import pyarrow
    print(f"✅ PyArrow: {pyarrow.__version__}")
except ImportError as e:
    print(f"❌ PyArrow: {e}")
    success = False

try:
    import peft
    print(f"✅ PEFT: {peft.__version__}")
except ImportError as e:
    print(f"❌ PEFT: {e}")
    success = False

try:
    import trl
    print(f"✅ TRL: {trl.__version__}")
except ImportError as e:
    print(f"❌ TRL: {e}")
    success = False

print("-" * 50)

if success:
    print("\n🎉 All packages installed successfully!\n")

    # Test Qwen tokenizer
    print("Testing Qwen tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            trust_remote_code=True
        )
        test_text = "SELECT * FROM users WHERE age > 25;"
        tokens = tokenizer.encode(test_text)
        print(f"✅ Qwen tokenizer works!")
        print(f"   Test: '{test_text}'")
        print(f"   Tokens: {len(tokens)}")
    except Exception as e:
        print(f"⚠️ Tokenizer test failed: {e}")
else:
    print("\n⚠️ Some packages failed to install")
    print("Try running the commands manually in the notebook")

# Check device
import torch
print(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if not torch.cuda.is_available():
    print("Note: This is expected on ml.t2.medium instances")
    print("For GPU training, use SageMaker Training Jobs with ml.g5.xlarge")
EOF

python test_final_setup.py

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Installation method used:"
echo "  1. conda-forge for compiled packages (sentencepiece, pyarrow)"
echo "  2. pip for Python packages (transformers, etc.)"
echo ""
echo "Next steps:"
echo "  1. Open SageMaker_Complete_Setup.ipynb to verify"
echo "  2. Open GL_RL_Model_Quick_Start.ipynb for training"
echo "  3. Use Python 3 kernel in notebooks"
echo ""
echo "To manually install in a notebook, use:"
echo "  %conda install -c conda-forge sentencepiece pyarrow -y"
echo "  %pip install transformers datasets peft trl accelerate"
echo ""
echo "========================================="