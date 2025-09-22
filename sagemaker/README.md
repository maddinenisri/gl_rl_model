# 🚀 GL RL Model - SageMaker Guide

Complete guide for training and deploying the GL RL Model on Amazon SageMaker.

## 📁 Directory Structure

```
sagemaker/
├── README.md                       # This file
├── requirements/                   # Dependency management
│   ├── conda.yaml                  # Conda environment specification
│   ├── pip-requirements.txt        # Python package requirements
│   └── sagemaker-training.txt      # Runtime requirements for training jobs
├── scripts/                        # Core scripts
│   ├── setup.sh                    # Environment setup (hybrid conda/pip)
│   ├── train.py                    # Unified training script
│   └── inference.py                # Inference endpoint script
└── notebooks/                      # Jupyter notebooks
    ├── 01_environment_setup.ipynb  # Setup and verification
    ├── 02_gpu_training.ipynb       # GPU training on SageMaker
    └── 03_cpu_inference.ipynb      # CPU inference deployment
```

## 🎯 Quick Start

### Step 1: Environment Setup

```bash
# In SageMaker terminal
cd /home/ec2-user/SageMaker
git clone https://github.com/maddinenisri/gl_rl_model.git
cd gl_rl_model
bash sagemaker/scripts/setup.sh
```

### Step 2: GPU Training

Open `notebooks/02_gpu_training.ipynb` to:
- Configure training job
- Launch on GPU instances (ml.g5.xlarge)
- Use spot instances for 70% cost savings
- Monitor training progress

### Step 3: CPU Inference

Open `notebooks/03_cpu_inference.ipynb` to:
- Deploy trained model
- Run batch inference
- Create REST API endpoint

## 💰 Cost Optimization

| Task | Instance Type | On-Demand | Spot | Recommended |
|------|--------------|-----------|------|-------------|
| Development | ml.t3.xlarge | $0.17/hr | N/A | ✅ |
| Training | ml.g5.xlarge | $1.00/hr | $0.30/hr | ✅ Spot |
| Inference | ml.m5.xlarge | $0.23/hr | N/A | ✅ |

### Tips for Cost Savings:
- ✅ Always use spot instances for training (70% savings)
- ✅ Stop notebook instances when not in use
- ✅ Use ml.t3.xlarge for development (sufficient memory)
- ✅ Set up AWS budget alerts

## 🔧 Training Configuration

### Basic Training Job

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='sagemaker/scripts',
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 3,
        'batch_size': 4,
        'learning_rate': 3e-5,
        'model_name': 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
    },
    use_spot_instances=True,  # Save costs
    max_wait=7200,
    max_run=3600
)

estimator.fit({'training': f's3://{bucket}/data/training'})
```

## 🚀 Deployment

### Deploy Inference Endpoint

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data=estimator.model_data,
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='sagemaker/scripts'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)
```

### Test Inference

```python
result = predictor.predict({
    'query': 'Show me all customers',
    'context': 'customers(id, name, email, created_at)'
})
print(result['sql'])
```

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: datasets` | Script auto-installs dependencies at runtime |
| `ResourceLimitExceeded` for spot | Request quota increase or use on-demand |
| `GLIBCXX` version error | Run setup.sh to install system libraries |
| Out of memory | Use larger instance or reduce batch size |
| Version conflicts | Use requirements in `requirements/` folder |

### Dependency Management

The project uses a **hybrid approach**:
- **Conda**: System libraries (gcc, libstdcxx-ng) and compiled packages (sentencepiece, pyarrow)
- **Pip**: Python packages with specific version constraints

This ensures compatibility while avoiding compilation issues on SageMaker.

### Memory Requirements

| Instance Type | Memory | Use Case |
|--------------|--------|----------|
| ml.t2.medium | 4GB | ❌ Too small for setup |
| ml.t3.xlarge | 16GB | ✅ Development |
| ml.g5.xlarge | 24GB | ✅ Training |
| ml.m5.xlarge | 16GB | ✅ Inference |

## 📊 Model Details

- **Base Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Task**: SQL generation from natural language
- **Method**: LoRA fine-tuning (r=8, alpha=16)
- **Training**: 3 epochs on query-SQL pairs
- **Parameters**: ~30M trainable (2% of total)

## 🛠️ Development Workflow

1. **Setup**: Run `setup.sh` once per instance
2. **Develop**: Use notebooks for experimentation
3. **Train**: Launch training jobs with `train.py`
4. **Deploy**: Use `inference.py` for endpoints
5. **Monitor**: Check CloudWatch logs and metrics

## 📈 Performance Metrics

Expected performance on test set:
- **Accuracy**: ~85% exact match
- **BLEU Score**: ~0.75
- **Inference Time**: <100ms per query
- **Training Time**: ~30 min for 3 epochs

## 🔐 Security Best Practices

- ✅ Use IAM roles, not credentials
- ✅ Encrypt S3 buckets
- ✅ Use VPC endpoints for private access
- ✅ Enable CloudTrail logging
- ✅ Regular model versioning

## 📚 Additional Resources

- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [PyTorch on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/)
- [Spot Instance Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)
- [Cost Management](https://console.aws.amazon.com/cost-management/)

---

**⚠️ Remember**: Always stop SageMaker notebook instances when not in use to avoid charges!