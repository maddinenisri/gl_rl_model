#!/bin/bash

# =============================================================================
# GL RL Model - SageMaker Conda-based Setup Script
# =============================================================================
# This script uses conda packages to avoid build issues
# Just run: bash sagemaker_setup_conda.sh
# =============================================================================

set -e

echo "========================================="
echo "🚀 GL RL Model - SageMaker Setup (Conda)"
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
echo "🐍 Setting up conda environment..."

# Activate conda environment
source /home/ec2-user/anaconda3/bin/activate

# Try to use pytorch environment first, otherwise use base
conda activate pytorch_p310 2>/dev/null || \
conda activate conda_pytorch_p310 2>/dev/null || \
echo "Using base environment"

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo ""

echo "📦 Installing dependencies via conda..."
echo "This approach avoids build issues..."

# Install PyTorch and core ML packages via conda
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch -q

# Install transformers ecosystem via conda-forge
conda install -y -c conda-forge \
    transformers \
    datasets \
    accelerate \
    sentencepiece \
    protobuf \
    -q

# Install additional packages via conda-forge
conda install -y -c conda-forge \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    jupyter \
    ipywidgets \
    -q

echo ""
echo "📦 Installing remaining packages via pip..."

# Install packages not available in conda
pip install -q peft trl 2>/dev/null || {
    echo "Installing peft and trl with minimal dependencies..."
    pip install -q --no-deps peft trl
    pip install -q huggingface-hub tokenizers
}

# Try bitsandbytes (optional for CPU)
pip install -q bitsandbytes 2>/dev/null || echo "⚠️ bitsandbytes not installed (OK for CPU)"

echo ""
echo "📥 Downloading training data from S3..."
mkdir -p data/training

# Download data files
aws s3 cp s3://gl-rl-model-sagemaker-340350204194-us-east-1/data/training/query_pairs.jsonl data/training/ 2>/dev/null || {
    echo "⚠️  Could not download query_pairs.jsonl from S3"
    echo "Creating sample data instead..."

    # Create sample data if S3 download fails
    cat > data/training/query_pairs.jsonl << 'EOF'
{"query": "Show me all customers", "sql": "SELECT * FROM customers;", "context": "customers(id, name, email)"}
{"query": "Get total sales by month", "sql": "SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) FROM sales GROUP BY month;", "context": "sales(id, date, amount)"}
{"query": "Find top products", "sql": "SELECT product_name, COUNT(*) as sales_count FROM orders GROUP BY product_name ORDER BY sales_count DESC LIMIT 10;", "context": "orders(id, product_name, quantity)"}
EOF
}

aws s3 cp s3://gl-rl-model-sagemaker-340350204194-us-east-1/data/training/query_pairs_expanded.jsonl data/training/ 2>/dev/null || {
    echo "⚠️  Could not download query_pairs_expanded.jsonl from S3"
    cp data/training/query_pairs.jsonl data/training/query_pairs_expanded.jsonl 2>/dev/null || true
}

echo ""
echo "📓 Creating Quick Start notebook..."

# Create the quick start notebook (same as before)
cat > GL_RL_Model_Quick_Start.ipynb << 'NOTEBOOK_EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GL RL Model - Quick Start Training\n",
    "\n",
    "This notebook provides a simple interface to test and train the GL RL Model on SageMaker.\n",
    "\n",
    "## 🔧 Environment Check"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import sys\n",
    "import os\n",
    "\n",
    "print(f\"Python: {sys.version}\")\n",
    "print(f\"PyTorch: {torch.__version__}\")\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
    "print(f\"Working Directory: {os.getcwd()}\")\n",
    "print(f\"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}\")\n",
    "\n",
    "if not torch.cuda.is_available():\n",
    "    print(\"\\n⚠️ No GPU detected - Running on CPU instance (ml.t2.medium)\")\n",
    "    print(\"💡 This is fine for testing. For actual training, use GPU instances.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📚 Load Training Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "\n",
    "# Load training data\n",
    "data_file = 'data/training/query_pairs.jsonl'\n",
    "\n",
    "if os.path.exists(data_file):\n",
    "    with open(data_file, 'r') as f:\n",
    "        data = [json.loads(line) for line in f]\n",
    "    \n",
    "    df = pd.DataFrame(data)\n",
    "    print(f\"✅ Loaded {len(df)} training examples\")\n",
    "    print(f\"\\nSample data:\")\n",
    "    print(df.head())\n",
    "else:\n",
    "    print(f\"❌ Training data not found at {data_file}\")\n",
    "    print(\"Run the setup script first: bash sagemaker_setup_conda.sh\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🤖 Test Model Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "\n",
    "# Model configuration\n",
    "model_name = \"Qwen/Qwen2.5-Coder-1.5B-Instruct\"\n",
    "\n",
    "print(f\"Testing tokenizer for: {model_name}\")\n",
    "\n",
    "try:\n",
    "    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n",
    "    print(\"✅ Tokenizer loaded successfully\")\n",
    "    \n",
    "    # Test tokenization\n",
    "    test_text = \"SELECT * FROM customers WHERE age > 25;\"\n",
    "    tokens = tokenizer.encode(test_text)\n",
    "    print(f\"\\nTest tokenization:\")\n",
    "    print(f\"Input: {test_text}\")\n",
    "    print(f\"Tokens: {len(tokens)} tokens\")\n",
    "    print(f\"Decoded: {tokenizer.decode(tokens)}\")\n",
    "except Exception as e:\n",
    "    print(f\"❌ Error loading tokenizer: {e}\")\n",
    "    print(\"Make sure sentencepiece is installed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 💻 Simple SQL Generation Test (CPU-friendly)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# For CPU testing, we'll just demonstrate the prompt format\n",
    "# Full model loading would be too slow on CPU\n",
    "\n",
    "def create_sql_prompt(query, context=\"\"):\n",
    "    \"\"\"Create a prompt for SQL generation\"\"\"\n",
    "    prompt = f\"\"\"You are a SQL expert. Convert the following natural language query to SQL.\n",
    "\n",
    "Context: {context if context else 'General database'}\n",
    "Query: {query}\n",
    "SQL: \"\"\"\n",
    "    return prompt\n",
    "\n",
    "# Test prompt creation\n",
    "test_queries = [\n",
    "    (\"Show all customers\", \"customers(id, name, email, age)\"),\n",
    "    (\"Get total sales by month\", \"sales(id, date, amount, product_id)\"),\n",
    "    (\"Find top 5 products\", \"products(id, name, price), orders(id, product_id, quantity)\")\n",
    "]\n",
    "\n",
    "print(\"Sample prompts for SQL generation:\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "for query, context in test_queries:\n",
    "    prompt = create_sql_prompt(query, context)\n",
    "    print(f\"\\nQuery: {query}\")\n",
    "    print(f\"Expected SQL format: SELECT ... FROM ... WHERE ...\")\n",
    "    print(\"-\" * 30)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🚀 For Full Model Training\n",
    "\n",
    "Since we're on a CPU instance (ml.t2.medium), full model training would be very slow.\n",
    "\n",
    "### Option 1: Launch GPU Training Job (Recommended)\n",
    "\n",
    "Use SageMaker Training Jobs with GPU instances for actual training:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime\n",
    "\n",
    "# Generate training job configuration\n",
    "job_name = f\"gl-rl-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}\"\n",
    "\n",
    "training_command = f\"\"\"# Run this command in the terminal to start GPU training:\n",
    "\n",
    "aws sagemaker create-training-job \\\\\n",
    "  --training-job-name {job_name} \\\\\n",
    "  --role-arn arn:aws:iam::340350204194:role/gl-rl-model-sagemaker-role \\\\\n",
    "  --algorithm-specification TrainingImage=763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.35.0-gpu-py310,TrainingInputMode=File \\\\\n",
    "  --resource-config InstanceType=ml.p3.2xlarge,InstanceCount=1,VolumeSizeInGB=100 \\\\\n",
    "  --enable-managed-spot-training \\\\\n",
    "  --input-data-config '[{{\"ChannelName\":\"training\",\"DataSource\":{{\"S3DataSource\":{{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"s3://gl-rl-model-sagemaker-340350204194-us-east-1/data/training/\",\"S3DataDistributionType\":\"FullyReplicated\"}}}}}}]' \\\\\n",
    "  --output-data-config S3OutputPath=s3://gl-rl-model-sagemaker-340350204194-us-east-1/models/ \\\\\n",
    "  --profile personal-yahoo\n",
    "\"\"\"\n",
    "\n",
    "print(training_command)\n",
    "print(\"\\n💰 Cost estimate: ~$3.06/hour with spot instances (~$0.92/hour)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Option 2: Check Package Versions\n",
    "\n",
    "Verify all required packages are installed:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pkg_resources\n",
    "\n",
    "required_packages = [\n",
    "    'torch',\n",
    "    'transformers', \n",
    "    'datasets',\n",
    "    'accelerate',\n",
    "    'peft',\n",
    "    'trl',\n",
    "    'sentencepiece',\n",
    "    'protobuf',\n",
    "    'pandas',\n",
    "    'numpy'\n",
    "]\n",
    "\n",
    "print(\"📦 Installed Package Versions:\")\n",
    "print(\"=\" * 40)\n",
    "\n",
    "for package in required_packages:\n",
    "    try:\n",
    "        version = pkg_resources.get_distribution(package).version\n",
    "        print(f\"✅ {package:15} : {version}\")\n",
    "    except:\n",
    "        print(f\"❌ {package:15} : Not installed\")\n",
    "\n",
    "print(\"\\n\" + \"=\" * 40)\n",
    "print(\"If any packages are missing, run:\")\n",
    "print(\"bash sagemaker_setup_conda.sh\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_pytorch_p310",
   "language": "python",
   "name": "conda_pytorch_p310"
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
NOTEBOOK_EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "========================================="
echo "📝 Next Steps:"
echo "========================================="
echo ""
echo "1. In JupyterLab, open: GL_RL_Model_Quick_Start.ipynb"
echo "2. Select kernel: conda_pytorch_p310"
echo "3. Run the cells to test the setup"
echo ""
echo "📊 Current Setup:"
echo "  - Instance: ml.t2.medium (CPU only)"
echo "  - Environment: $CONDA_DEFAULT_ENV"
echo "  - Data location: ./data/training/"
echo "  - Model: Qwen2.5-Coder-1.5B-Instruct"
echo ""
echo "🚀 For GPU Training:"
echo "  Use the commands in the notebook to launch"
echo "  a training job with ml.p3.2xlarge instance"
echo ""
echo "========================================="