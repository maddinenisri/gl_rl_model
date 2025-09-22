variable "aws_profile" {
  description = "AWS CLI profile to use"
  type        = string
  default     = "personal-yahoo"
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "gl-rl-model"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

# GPU Instance Configuration
variable "notebook_instance_type" {
  description = "SageMaker notebook instance type"
  type        = string
  default     = "ml.t3.xlarge"  # 16GB RAM, sufficient for setup
  # default   = "ml.t2.medium"  # 4GB RAM - too small for conda
}

variable "notebook_volume_size" {
  description = "EBS volume size in GB for notebook instance"
  type        = number
  default     = 30
}

variable "training_instance_type" {
  description = "Instance type for training jobs"
  type        = string
  default     = "ml.g5.xlarge"  # A10G GPU, 24GB VRAM - Better price/performance
}

variable "training_instance_count" {
  description = "Number of instances for distributed training"
  type        = number
  default     = 1
}

# Spot Instance Configuration
variable "use_spot_instances" {
  description = "Use spot instances for training"
  type        = bool
  default     = true
}

variable "notebook_spot_max_price" {
  description = "Maximum hourly price for notebook spot instance (USD)"
  type        = string
  default     = "0.40"  # ~70% of on-demand ml.g4dn.xlarge ($1.26/hr)
}

variable "training_spot_max_price" {
  description = "Maximum hourly price for training spot instance (USD)"
  type        = string
  default     = "1.20"  # ~70% of on-demand ml.p3.2xlarge ($3.83/hr)
}

variable "max_wait_time_seconds" {
  description = "Maximum wait time for spot instance fulfillment"
  type        = number
  default     = 86400  # 24 hours
}

variable "max_runtime_seconds" {
  description = "Maximum runtime for training jobs"
  type        = number
  default     = 86400  # 24 hours
}

# Auto-stop Configuration
variable "auto_stop_idle_minutes" {
  description = "Minutes of idle time before auto-stopping notebook"
  type        = number
  default     = 60
}

# Cost Management
variable "monthly_budget_amount" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 200
}

variable "budget_alert_threshold" {
  description = "Percentage of budget to trigger alert"
  type        = number
  default     = 80
}

variable "budget_alert_email" {
  description = "Email address for budget alerts"
  type        = string
  default     = ""  # Set in terraform.tfvars
}

# Training Configuration
variable "epochs" {
  description = "Number of training epochs"
  type        = number
  default     = 3
}

variable "batch_size" {
  description = "Training batch size"
  type        = number
  default     = 4
}

variable "learning_rate" {
  description = "Learning rate for training"
  type        = string
  default     = "3e-5"
}

# Model Configuration
variable "model_name" {
  description = "Hugging Face model name"
  type        = string
  default     = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
}

variable "use_mixed_precision" {
  description = "Use mixed precision training (fp16)"
  type        = bool
  default     = true
}

variable "gradient_checkpointing" {
  description = "Enable gradient checkpointing for memory efficiency"
  type        = bool
  default     = true
}

# Checkpointing for Spot Interruptions
variable "checkpoint_s3_uri" {
  description = "S3 URI for model checkpoints"
  type        = string
  default     = ""  # Will be set dynamically
}

variable "checkpoint_local_path" {
  description = "Local path for checkpoints"
  type        = string
  default     = "/opt/ml/checkpoints"
}

# GPU Instance Types and Pricing
variable "gpu_instance_specs" {
  description = "GPU instance specifications and spot pricing"
  type = map(object({
    gpu_type        = string
    gpu_memory      = string
    on_demand_price = number
    spot_price      = number
    use_for         = string
  }))
  default = {
    # EC2 instance types for spot price lookup
    "t2.medium" = {
      gpu_type        = "CPU"
      gpu_memory      = "N/A"
      on_demand_price = 0.05
      spot_price      = 0.02
      use_for         = "ec2-reference"
    }
    "g4dn.xlarge" = {
      gpu_type        = "T4"
      gpu_memory      = "16GB"
      on_demand_price = 0.53
      spot_price      = 0.16
      use_for         = "ec2-reference"
    }
    "p3.2xlarge" = {
      gpu_type        = "V100"
      gpu_memory      = "16GB"
      on_demand_price = 3.06
      spot_price      = 0.92
      use_for         = "ec2-reference"
    }
    # SageMaker instance types (ml. prefix)
    "ml.t2.medium" = {
      gpu_type        = "CPU"
      gpu_memory      = "N/A"
      on_demand_price = 0.09
      spot_price      = 0.03
      use_for         = "notebook"
    }
    "ml.t3.xlarge" = {
      gpu_type        = "CPU"
      gpu_memory      = "N/A"
      on_demand_price = 0.17
      spot_price      = 0.05
      use_for         = "notebook"
    }
    "ml.g4dn.xlarge" = {
      gpu_type        = "T4"
      gpu_memory      = "16GB"
      on_demand_price = 1.26
      spot_price      = 0.35
      use_for         = "small-training"
    }
    "ml.g4dn.2xlarge" = {
      gpu_type        = "T4"
      gpu_memory      = "16GB"
      on_demand_price = 1.80
      spot_price      = 0.54
      use_for         = "training"
    }
    "ml.g5.xlarge" = {
      gpu_type        = "A10G"
      gpu_memory      = "24GB"
      on_demand_price = 1.006
      spot_price      = 0.302
      use_for         = "recommended-training"
    }
    "ml.g5.2xlarge" = {
      gpu_type        = "A10G"
      gpu_memory      = "24GB"
      on_demand_price = 1.341
      spot_price      = 0.402
      use_for         = "fast-training"
    }
    "ml.p3.2xlarge" = {
      gpu_type        = "V100"
      gpu_memory      = "16GB"
      on_demand_price = 3.83
      spot_price      = 1.15
      use_for         = "training"
    }
    "ml.p3.8xlarge" = {
      gpu_type        = "4xV100"
      gpu_memory      = "64GB"
      on_demand_price = 14.69
      spot_price      = 4.40
      use_for         = "large-training"
    }
  }
}

# Tags
variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}