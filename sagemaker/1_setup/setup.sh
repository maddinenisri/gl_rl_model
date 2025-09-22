#!/bin/bash

# =============================================================================
# GL RL Model - SageMaker Setup Script
# =============================================================================
# Single, consolidated setup script that works on SageMaker instances
# Uses conda for compiled packages and pip for Python packages
# Tested on ml.t2.medium instances
# =============================================================================

set -e

echo "========================================="
echo "🚀 GL RL Model - SageMaker Setup"
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

echo "📁 Working directory: $(pwd)"
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
echo "📦 Installing dependencies..."
echo ""

# Step 1: Install compiled packages with conda (avoids CMake issues)
echo "Step 1/4: Installing compiled packages with conda..."
conda install -c conda-forge sentencepiece pyarrow -y -q

# Step 2: Install PyTorch
echo ""
echo "Step 2/4: Installing PyTorch..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install ML libraries
echo ""
echo "Step 3/4: Installing ML libraries..."
pip install -q transformers datasets peft trl accelerate huggingface-hub tokenizers

# Step 4: Install utilities and fix dependencies
echo ""
echo "Step 4/4: Installing utilities..."
pip install -q --upgrade numpy pandas protobuf tqdm fsspec aiohttp multiprocess>=0.70.18

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

# Quick test
echo ""
echo "🧪 Testing installation..."
python -c "
import torch, transformers, datasets, sentencepiece, pyarrow
print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ Datasets:', datasets.__version__)
print('✅ All core packages installed successfully!')
print('')
print('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU (use Training Jobs for GPU)')
" || echo "⚠️ Some packages may need manual installation"

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Open sagemaker/1_setup/Setup_Environment.ipynb to verify"
echo "2. Open sagemaker/2_training/GPU_Training.ipynb for training"
echo "3. Open sagemaker/3_inference/CPU_Inference.ipynb for inference"
echo ""
echo "To run this setup from a notebook, use:"
echo "  !bash sagemaker/1_setup/setup.sh"
echo ""
echo "========================================="