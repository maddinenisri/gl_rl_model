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
- **ml.t2.medium**: $0.05/hr - Basic development
- **ml.t3.xlarge**: $0.17/hr - Better performance
- **ml.m5.xlarge**: $0.23/hr - Production inference

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

---

**Remember**: Always stop your SageMaker notebook instance when not in use to avoid charges!