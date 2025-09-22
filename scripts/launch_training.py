#!/usr/bin/env python3
"""
Launch SageMaker Training Job with optimized settings

Usage:
    python launch_training.py
    python launch_training.py --instance-type ml.g4dn.xlarge
    python launch_training.py --epochs 5 --batch-size 8
"""

import boto3
import sagemaker
import argparse
from datetime import datetime
from pathlib import Path


def launch_training(args):
    """Launch SageMaker training job with specified configuration"""

    # Initialize SageMaker session
    if args.profile:
        boto3.setup_default_session(profile_name=args.profile)

    session = sagemaker.Session()
    role = sagemaker.get_execution_role() if args.role is None else args.role
    region = session.boto_region_name
    account_id = session.account_id()
    bucket = f"gl-rl-model-sagemaker-{account_id}-{region}"

    print(f"🚀 Launching SageMaker Training Job")
    print(f"📍 Region: {region}")
    print(f"📦 S3 Bucket: {bucket}")
    print(f"💻 Instance: {args.instance_type}")

    # Import PyTorch estimator
    from sagemaker.pytorch import PyTorch

    # Configure training job
    estimator = PyTorch(
        entry_point='train.py',
        source_dir=str(Path(__file__).parent.parent / 'sagemaker' / 'scripts'),
        role=role,

        # Instance configuration
        instance_type=args.instance_type,
        instance_count=1,

        # Framework version - PyTorch 2.1 for compatibility
        framework_version='2.1',
        py_version='py310',

        # Hyperparameters
        hyperparameters={
            'model_name': args.model_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'gradient_checkpointing': args.gradient_checkpointing,
            'fp16': args.fp16,
        },

        # Cost optimization - Spot instances
        use_spot_instances=args.use_spot,
        max_wait=86400,  # 24 hours
        max_run=args.max_runtime,

        # Output configuration
        output_path=f's3://{bucket}/output',
        base_job_name='gl-rl-gpu',

        # Checkpointing for spot interruption recovery
        checkpoint_s3_uri=f's3://{bucket}/checkpoints' if args.use_spot else None,
        checkpoint_local_path='/opt/ml/checkpoints' if args.use_spot else None,

        # Disable debugger to reduce overhead
        disable_profiler=True,
        debugger_hook_config=False,
    )

    # Generate job name
    job_name = f"gl-rl-gpu-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"\n📋 Job Configuration:")
    print(f"  Name: {job_name}")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  FP16: {args.fp16}")
    print(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
    print(f"  Spot Instances: {args.use_spot}")

    # Cost estimate
    spot_prices = {
        'ml.g5.xlarge': 0.30,
        'ml.g4dn.xlarge': 0.35,
        'ml.p3.2xlarge': 1.15,
        'ml.g5.2xlarge': 0.60,
    }

    if args.instance_type in spot_prices and args.use_spot:
        estimated_hours = 3  # Rough estimate
        estimated_cost = spot_prices[args.instance_type] * estimated_hours
        print(f"\n💰 Cost Estimate:")
        print(f"  Spot price: ${spot_prices[args.instance_type]:.2f}/hour")
        print(f"  Estimated time: {estimated_hours} hours")
        print(f"  Total cost: ${estimated_cost:.2f}")

    # Launch the job
    if not args.dry_run:
        print(f"\n🚀 Launching training job...")

        # Ensure training data exists
        s3_client = boto3.client('s3')
        try:
            s3_client.head_object(Bucket=bucket, Key='data/training/query_pairs.jsonl')
        except:
            print(f"⚠️  Warning: Training data not found in S3. Uploading sample data...")
            # Upload sample data if needed
            import json
            sample_data = [
                {"query": "Show all customers", "sql": "SELECT * FROM customers", "context": "customers(id, name, email)"},
                {"query": "Count orders", "sql": "SELECT COUNT(*) FROM orders", "context": "orders(id, customer_id, total)"}
            ]

            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for item in sample_data:
                    f.write(json.dumps(item) + '\n')
                temp_file = f.name

            s3_client.upload_file(temp_file, bucket, 'data/training/query_pairs.jsonl')
            Path(temp_file).unlink()

        # Start training
        estimator.fit(
            inputs={'training': f's3://{bucket}/data/training'},
            job_name=job_name,
            wait=False
        )

        print(f"\n✅ Job submitted successfully!")
        print(f"\n📊 Monitor progress:")
        print(f"  Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
        print(f"  Script:  python scripts/monitor_training.py {job_name}")

        return job_name
    else:
        print(f"\n🔍 Dry run - job not launched")
        return None


def main():
    parser = argparse.ArgumentParser(description='Launch SageMaker Training Job')

    # Instance configuration
    parser.add_argument('--instance-type', default='ml.g5.xlarge',
                       choices=['ml.g5.xlarge', 'ml.g4dn.xlarge', 'ml.p3.2xlarge', 'ml.g5.2xlarge'],
                       help='SageMaker instance type (default: ml.g5.xlarge)')

    # Model configuration
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-Coder-1.5B-Instruct',
                       help='Hugging Face model to fine-tune')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=3e-5,
                       help='Learning rate (default: 3e-5)')

    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA r parameter (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha parameter (default: 16)')

    # Optimization settings
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use FP16 mixed precision training')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                       help='Disable FP16 training')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True,
                       help='Enable gradient checkpointing')
    parser.add_argument('--no-gradient-checkpointing', dest='gradient_checkpointing', action='store_false',
                       help='Disable gradient checkpointing')

    # Spot instances
    parser.add_argument('--use-spot', action='store_true', default=True,
                       help='Use spot instances (default: True)')
    parser.add_argument('--no-spot', dest='use_spot', action='store_false',
                       help='Use on-demand instances')

    # Runtime configuration
    parser.add_argument('--max-runtime', type=int, default=14400,
                       help='Maximum runtime in seconds (default: 14400 = 4 hours)')

    # AWS configuration
    parser.add_argument('--profile', default='personal-yahoo',
                       help='AWS profile to use (default: personal-yahoo)')
    parser.add_argument('--role', default=None,
                       help='SageMaker execution role ARN (auto-detected if not specified)')

    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration without launching job')

    args = parser.parse_args()

    # Launch training
    job_name = launch_training(args)

    if job_name and not args.dry_run:
        print(f"\n💡 Tip: Monitor your job with:")
        print(f"   python scripts/monitor_training.py {job_name}")


if __name__ == '__main__':
    main()