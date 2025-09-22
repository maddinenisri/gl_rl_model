# 🚀 GL RL Model - SageMaker Quick Start

## ✅ WORKING Setup Method (Tested & Confirmed)

After opening your SageMaker notebook instance:

### Option 1: Use the Complete Setup Notebook (RECOMMENDED)
1. Navigate to `/home/ec2-user/SageMaker/gl_rl_model/`
2. Open `SageMaker_Complete_Setup.ipynb`
3. Run all cells to install dependencies
4. This uses conda for compiled packages and pip for the rest

### Option 2: Run the Final Working Setup Script
```bash
cd /home/ec2-user/SageMaker/gl_rl_model && bash sagemaker_setup_final.sh
```

### Key Installation Commands (if doing manually):
```python
# In a notebook cell:
%conda install -c conda-forge sentencepiece pyarrow -y
%pip install transformers datasets peft trl accelerate torch huggingface-hub
%pip install --upgrade multiprocess>=0.70.18
```

### Why this works:
- **conda-forge** provides precompiled binaries (no CMake needed)
- **pip** installs pure Python packages
- Avoids all compilation issues on SageMaker

The script will:
- ✅ Install all dependencies (using conda to avoid build issues)
- ✅ Download training data
- ✅ Create a ready-to-use notebook
- ✅ Set up the environment

## 📝 After Setup

1. **Open JupyterLab** (if not already open)
2. **Navigate to**: `/home/ec2-user/SageMaker/gl_rl_model/`
3. **Open**: `GL_RL_Model_Quick_Start.ipynb`
4. **Run the cells** to test the model

## 💡 Important Notes

- **Current Instance**: ml.t2.medium (CPU only, $0.05/hour)
- **For GPU Training**: The notebook includes commands to launch GPU training jobs
- **Cost-Optimized**: Uses spot instances for 70% savings

## 🆘 Troubleshooting

If the setup script fails, run these commands manually:

```bash
# 1. Go to the right directory
cd /home/ec2-user/SageMaker

# 2. Clone or update the repository
git clone https://github.com/maddinenisri/gl_rl_model.git
cd gl_rl_model

# 3. Install dependencies
pip install torch transformers datasets accelerate peft trl

# 4. You're ready to go!
```

## 📊 What You Get

- **Qwen2.5-Coder Model**: 1.5B parameter model for SQL generation
- **Training Data**: Query-SQL pairs for fine-tuning
- **Quick Start Notebook**: Simple interface to test and train
- **GPU Commands**: Ready-to-use commands for production training

## 🚀 Start Training on GPU

To run actual training with GPU (ml.p3.2xlarge), use:

```bash
aws sagemaker create-training-job \
  --training-job-name gl-rl-model-$(date +%Y%m%d-%H%M%S) \
  --role-arn arn:aws:iam::340350204194:role/gl-rl-model-sagemaker-role \
  --resource-config InstanceType=ml.p3.2xlarge,InstanceCount=1 \
  --enable-managed-spot-training \
  --profile personal-yahoo
```

---

**Need Help?** Check the full notebook: `gl_rl_model_sagemaker.ipynb`