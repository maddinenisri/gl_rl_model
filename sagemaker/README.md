# 🚀 GL RL Model - SageMaker Guide

Complete guide for running the GL RL Model on Amazon SageMaker.

## 📁 Directory Structure

```
sagemaker/
├── 1_setup/                    # Environment setup
│   ├── setup.sh                # One-command setup script
│   └── Setup_Environment.ipynb # Setup notebook
├── 2_training/                 # GPU training
│   ├── GPU_Training.ipynb      # Training guide
│   └── train.py                # Training script
└── 3_inference/                # CPU inference
    └── CPU_Inference.ipynb      # Inference guide
```

## 🎯 Quick Start (3 Steps)

### Step 1: Setup Environment

```bash
# From SageMaker terminal:
cd /home/ec2-user/SageMaker/gl_rl_model
bash sagemaker/1_setup/setup.sh
```

Or use the notebook: `sagemaker/1_setup/Setup_Environment.ipynb`

### Step 2: Train Model (GPU)

Open `sagemaker/2_training/GPU_Training.ipynb` to:
- Launch GPU training jobs
- Use spot instances (70% savings)
- Monitor progress
- Download trained models

### Step 3: Run Inference (CPU)

Open `sagemaker/3_inference/CPU_Inference.ipynb` to:
- Load trained models
- Generate SQL from queries
- Batch process
- Deploy as endpoint

## 💰 Cost Optimization

| Task | Instance | Cost | Duration | Total |
|------|----------|------|----------|-------|
| Development | ml.t2.medium | $0.05/hr | Ongoing | Variable |
| Training | ml.g5.xlarge spot | $0.30/hr | 2-4 hrs | $0.60-$1.20 |
| Inference | ml.t2.medium | $0.05/hr | As needed | Variable |

### Tips:
- ✅ Always use spot instances for training
- ✅ Stop notebook instances when not in use
- ✅ Use Training Jobs instead of GPU notebooks
- ✅ Set up budget alerts in AWS

## 🖥️ Instance Types

### CPU Instances (Development/Inference)
- **ml.t2.medium**: $0.05/hr - 4GB RAM (⚠️ Too small for setup, conda may fail)
- **ml.t3.xlarge**: $0.17/hr - 16GB RAM (✅ Recommended minimum)
- **ml.m5.xlarge**: $0.23/hr - 16GB RAM - Production inference
- **ml.t3.2xlarge**: $0.33/hr - 32GB RAM - Comfortable development

### GPU Instances (Training)
- **ml.g5.xlarge**: $1.00/hr ($0.30 spot) - Best value, A10G 24GB
- **ml.g4dn.xlarge**: $0.73/hr ($0.35 spot) - Budget option, T4 16GB
- **ml.p3.2xlarge**: $3.83/hr ($1.15 spot) - Fast training, V100 16GB

## 📊 Terraform Resources

The infrastructure is managed by Terraform in `/terraform`:
- S3 bucket for data and models
- IAM roles and policies
- SageMaker notebook instance
- Budget alerts and monitoring

## 🔧 Troubleshooting

### Installation Issues
- Use conda for compiled packages: `%conda install -c conda-forge sentencepiece pyarrow`
- Use pip for Python packages: `%pip install transformers datasets peft trl`

### GPU Training
- If spot capacity unavailable, try different region or instance type
- Enable checkpointing for spot interruption recovery

### Memory Issues
- Use gradient checkpointing: `gradient_checkpointing=True`
- Reduce batch size
- Use mixed precision: `fp16=True`

## 📚 Additional Resources

- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Hugging Face on SageMaker](https://huggingface.co/docs/sagemaker/)
- [AWS Cost Management](https://console.aws.amazon.com/cost-management/)

## 🎓 Model Details

- **Base Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Task**: SQL generation from natural language
- **Method**: LoRA fine-tuning (r=8, alpha=16)
- **Training**: 3 epochs on query-SQL pairs

## 🔧 Troubleshooting Common Issues

### Dependency Conflicts

If you encounter dependency conflicts during setup:

1. **Clean Installation**:
   ```bash
   # Remove conflicting packages
   pip uninstall -y dill multiprocess fsspec s3fs botocore boto3
   pip cache purge

   # Reinstall with fixed versions
   pip install -r sagemaker/1_setup/requirements-sagemaker.txt
   ```

2. **Verify Installation**:
   ```bash
   python sagemaker/1_setup/verify_setup.py
   ```

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `ImportError: libstdc++.so.6: version GLIBCXX_3.4.29 not found` | Run: `conda install -c conda-forge gcc_linux-64 libstdcxx-ng` |
| `botocore version conflict` | Install specific versions: `pip install botocore==1.40.21 boto3==1.40.21` |
| `fsspec/s3fs incompatibility` | Install matching versions: `pip install fsspec==2025.7.0 s3fs==2025.7.0` |
| `multiprocess version error` | Update: `pip install 'dill>=0.4.0' 'multiprocess>=0.70.18'` |
| `Out of memory during conda install` | Use pip instead: `pip install sentencepiece pyarrow` |

### Memory Issues on ml.t2.medium

For ml.t2.medium instances (4GB RAM):

1. **Skip conda installations**:
   ```bash
   # Use pip for everything
   pip install sentencepiece pyarrow
   pip install -r sagemaker/1_setup/requirements-sagemaker.txt
   ```

2. **Clear memory before installation**:
   ```bash
   # Stop Jupyter kernel
   pkill -f jupyter

   # Clear pip cache
   pip cache purge

   # Run setup
   bash sagemaker/1_setup/setup.sh
   ```

### AWS Connectivity Issues

If S3 access fails:

1. **Check IAM role**:
   - Ensure SageMaker execution role has S3 access
   - Add `AmazonS3FullAccess` policy if needed

2. **Verify credentials**:
   ```python
   import boto3
   s3 = boto3.client('s3')
   s3.list_buckets()  # Should list your buckets
   ```

### Clean Reinstallation

For a complete fresh start:

```bash
# Navigate to SageMaker directory
cd /home/ec2-user/SageMaker

# Remove existing repo
rm -rf gl_rl_model

# Clone fresh
git clone https://github.com/maddinenisri/gl_rl_model.git
cd gl_rl_model

# Run new setup
bash sagemaker/1_setup/setup.sh
```

### Logs and Debugging

Setup logs are saved to `/tmp/sagemaker_setup_[timestamp].log`

To view the latest log:
```bash
ls -lt /tmp/sagemaker_setup_*.log | head -1
tail -f /tmp/sagemaker_setup_*.log  # Follow log in real-time
```

---

**Remember**: Always stop your SageMaker notebook instance when not in use to avoid charges!