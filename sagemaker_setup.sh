#!/bin/bash

# =============================================================================
# GL RL Model - SageMaker Automated Setup Script
# =============================================================================
# This script automatically sets up everything you need in SageMaker
# Just run: bash sagemaker_setup.sh
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
echo "📦 Installing Python dependencies..."
echo "This may take a few minutes..."

# Create a requirements file
cat > requirements_sagemaker.txt << 'EOF'
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0
sentencepiece>=0.1.99
protobuf>=3.20.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.65.0
jupyter>=1.0.0
ipywidgets>=8.0.0
EOF

# Install packages
pip install -q -r requirements_sagemaker.txt

echo ""
echo "📥 Downloading training data from S3..."
mkdir -p data/training

# Download data files
aws s3 cp s3://gl-rl-model-sagemaker-340350204194-us-east-1/data/training/query_pairs.jsonl data/training/ || {
    echo "⚠️  Could not download query_pairs.jsonl from S3"
    echo "Creating sample data instead..."

    # Create sample data if S3 download fails
    cat > data/training/query_pairs.jsonl << 'EOF'
{"query": "Show me all customers", "sql": "SELECT * FROM customers;", "context": "customers(id, name, email)"}
{"query": "Get total sales by month", "sql": "SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) FROM sales GROUP BY month;", "context": "sales(id, date, amount)"}
EOF
}

aws s3 cp s3://gl-rl-model-sagemaker-340350204194-us-east-1/data/training/query_pairs_expanded.jsonl data/training/ || {
    echo "⚠️  Could not download query_pairs_expanded.jsonl from S3"
    cp data/training/query_pairs.jsonl data/training/query_pairs_expanded.jsonl 2>/dev/null || true
}

echo ""
echo "📥 Downloading the optimized notebook..."

# Download the fixed notebook
aws s3 cp s3://gl-rl-model-sagemaker-340350204194-us-east-1/notebooks/gl_rl_model_sagemaker_fixed.ipynb . 2>/dev/null || {
    echo "Downloading from backup location..."
    wget -q https://raw.githubusercontent.com/maddinenisri/gl_rl_model/main/gl_rl_model_sagemaker.ipynb || true
}

# Create a simple starter notebook
cat > GL_RL_Model_Quick_Start.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GL RL Model - Quick Start Training\n",
    "\n",
    "This notebook provides a simple interface to train the GL RL Model on SageMaker.\n",
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
    "    print(\"Run the setup script first: bash sagemaker_setup.sh\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🤖 Initialize Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
    "import torch\n",
    "\n",
    "# Model configuration\n",
    "model_name = \"Qwen/Qwen2.5-Coder-1.5B-Instruct\"\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "print(f\"Loading model: {model_name}\")\n",
    "print(f\"Device: {device}\")\n",
    "\n",
    "# For CPU/testing only - load smaller version\n",
    "if not torch.cuda.is_available():\n",
    "    print(\"\\n📝 Loading model in CPU mode (this will be slow)...\")\n",
    "    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n",
    "    # Load with reduced precision for CPU\n",
    "    model = AutoModelForCausalLM.from_pretrained(\n",
    "        model_name,\n",
    "        torch_dtype=torch.float32,\n",
    "        device_map='cpu',\n",
    "        trust_remote_code=True\n",
    "    )\n",
    "    print(\"✅ Model loaded in CPU mode\")\n",
    "else:\n",
    "    print(\"\\n🚀 Loading model in GPU mode...\")\n",
    "    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n",
    "    model = AutoModelForCausalLM.from_pretrained(\n",
    "        model_name,\n",
    "        torch_dtype=torch.float16,\n",
    "        device_map='auto',\n",
    "        trust_remote_code=True\n",
    "    )\n",
    "    print(\"✅ Model loaded in GPU mode\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🧪 Test Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test the model with a simple query\n",
    "test_query = \"Show me all customers from the customers table\"\n",
    "\n",
    "prompt = f\"\"\"Convert this natural language query to SQL:\n",
    "Query: {test_query}\n",
    "SQL:\"\"\"\n",
    "\n",
    "inputs = tokenizer(prompt, return_tensors=\"pt\").to(device)\n",
    "\n",
    "print(\"Generating SQL...\")\n",
    "with torch.no_grad():\n",
    "    outputs = model.generate(\n",
    "        **inputs,\n",
    "        max_new_tokens=100,\n",
    "        temperature=0.1,\n",
    "        do_sample=True,\n",
    "        pad_token_id=tokenizer.eos_token_id\n",
    "    )\n",
    "\n",
    "response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
    "print(f\"\\n📝 Generated SQL:\")\n",
    "print(response.split(\"SQL:\")[-1].strip())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🚀 For Full Training\n",
    "\n",
    "Since we're on a CPU instance (ml.t2.medium), full training would be very slow.\n",
    "\n",
    "### Option 1: Use SageMaker Training Jobs (Recommended)\n",
    "\n",
    "Run this command in the terminal to start GPU training:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the training command\n",
    "training_command = \"\"\"aws sagemaker create-training-job \\\\\n",
    "  --training-job-name gl-rl-model-gpu-$(date +%Y%m%d-%H%M%S) \\\\\n",
    "  --role-arn arn:aws:iam::340350204194:role/gl-rl-model-sagemaker-role \\\\\n",
    "  --algorithm-specification TrainingImage=763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.35.0-gpu-py310-cu118,TrainingInputMode=File \\\\\n",
    "  --resource-config InstanceType=ml.p3.2xlarge,InstanceCount=1,VolumeSizeInGB=100 \\\\\n",
    "  --enable-managed-spot-training \\\\\n",
    "  --profile personal-yahoo\"\"\"\n",
    "\n",
    "print(\"Copy and run this command in the terminal:\")\n",
    "print(training_command)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Option 2: Continue with CPU Training (Not Recommended)\n",
    "\n",
    "If you want to test training on CPU (will be very slow):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Only run this if you really want to test on CPU\n",
    "if False:  # Change to True to enable\n",
    "    from trl import SFTTrainer\n",
    "    from transformers import TrainingArguments\n",
    "    \n",
    "    # Minimal training configuration for CPU\n",
    "    training_args = TrainingArguments(\n",
    "        output_dir=\"./results\",\n",
    "        num_train_epochs=1,\n",
    "        per_device_train_batch_size=1,\n",
    "        gradient_accumulation_steps=4,\n",
    "        warmup_steps=10,\n",
    "        max_steps=50,  # Just 50 steps for testing\n",
    "        logging_steps=10,\n",
    "        save_steps=25,\n",
    "        evaluation_strategy=\"no\",\n",
    "        fp16=False,  # No mixed precision on CPU\n",
    "    )\n",
    "    \n",
    "    print(\"Starting minimal CPU training (50 steps only)...\")\n",
    "    # Training code would go here"
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
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "========================================="
echo "📝 Next Steps:"
echo "========================================="
echo ""
echo "1. In JupyterLab, open: GL_RL_Model_Quick_Start.ipynb"
echo "2. Select kernel: conda_pytorch_p310"
echo "3. Run the cells to test the model"
echo ""
echo "📊 Current Setup:"
echo "  - Instance: ml.t2.medium (CPU only)"
echo "  - Data location: ./data/training/"
echo "  - Model: Qwen2.5-Coder-1.5B-Instruct"
echo ""
echo "🚀 For GPU Training:"
echo "  Use the commands in the notebook to launch"
echo "  a training job with ml.p3.2xlarge instance"
echo ""
echo "========================================="