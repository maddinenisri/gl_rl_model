#!/bin/bash

# =============================================================================
# Quick Fix Script for SageMaker Dependency Issues
# =============================================================================
# Run this if setup.sh has conflicts
# =============================================================================

echo "🔧 Quick Fix for SageMaker Dependencies"
echo "======================================="

# Step 1: Clean all conflicting packages
echo "Step 1: Cleaning conflicting packages..."
pip uninstall -y dill multiprocess fsspec s3fs botocore boto3 aiobotocore 2>/dev/null

# Step 2: Install core dependencies in correct order
echo "Step 2: Installing filesystem packages..."
pip install --no-cache-dir fsspec==2024.6.1 s3fs==2024.6.1

echo "Step 3: Installing multiprocessing packages..."
pip install --no-cache-dir 'dill>=0.3.0,<0.3.9' multiprocess==0.70.16

echo "Step 4: Installing AWS packages..."
pip install --no-cache-dir 'botocore>=1.39.9,<1.40.19' 'boto3>=1.39.5,<2.0'

echo "Step 5: Installing PyTorch..."
pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

echo "Step 6: Installing ML packages..."
pip install --no-cache-dir transformers==4.56.2 datasets==2.21.0

echo "Step 7: Installing additional packages..."
pip install --no-cache-dir peft trl accelerate huggingface-hub pandas numpy pyarrow sqlparse scikit-learn

echo ""
echo "✅ Quick fix complete! Testing installation..."
echo ""

# Test imports
python -c "
import torch
import transformers
import datasets
print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ Datasets:', datasets.__version__)
print('✅ All core packages working!')
"

echo ""
echo "======================================="
echo "If you still have issues, try:"
echo "1. Restart the kernel"
echo "2. Run: python sagemaker/1_setup/verify_setup.py"
echo "======================================="