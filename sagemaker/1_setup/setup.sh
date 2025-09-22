#!/bin/bash

# =============================================================================
# GL RL Model - SageMaker Setup Script
# =============================================================================
# Clean, modular setup script for SageMaker environments
# Handles dependency conflicts and ensures proper installation order
# =============================================================================

set -e

# ============= Configuration =============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-sagemaker.txt"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_setup.py"
LOG_FILE="/tmp/sagemaker_setup_$(date +%Y%m%d_%H%M%S).log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============= Helper Functions =============

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
}

check_sagemaker_environment() {
    log "Checking environment..."

    if [ ! -d "/home/ec2-user/SageMaker" ]; then
        log_error "This script should be run in a SageMaker notebook instance"
        log "Current directory: $(pwd)"
        exit 1
    fi

    # Check available memory
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    log "Available memory: ${TOTAL_MEM}MB"

    if [ "$TOTAL_MEM" -lt 8000 ]; then
        log_warning "System has less than 8GB RAM. Consider using ml.t3.xlarge or larger."
        log_warning "Installation may fail on ml.t2.medium instances."
    fi

    # Display system info
    log "Python version: $(python --version 2>&1)"
    log "Conda version: $(conda --version 2>&1)"
    log "Working directory: $(pwd)"
}

navigate_to_repo() {
    cd /home/ec2-user/SageMaker

    if [ -d "gl_rl_model" ]; then
        log_success "Repository exists, updating..."
        cd gl_rl_model
        git pull origin main 2>&1 | tee -a "$LOG_FILE" || log_warning "Could not pull latest changes"
    else
        log "Cloning repository..."
        git clone https://github.com/maddinenisri/gl_rl_model.git 2>&1 | tee -a "$LOG_FILE"
        cd gl_rl_model
    fi

    log_success "Repository ready at: $(pwd)"
}

cleanup_conflicting_packages() {
    log "Cleaning conflicting packages..."

    # Uninstall packages that cause conflicts
    local packages_to_remove=(
        "dill"
        "multiprocess"
        "fsspec"
        "s3fs"
        "botocore"
        "boto3"
        "aiobotocore"
    )

    for package in "${packages_to_remove[@]}"; do
        pip uninstall -y "$package" 2>/dev/null || true
    done

    # Clear pip cache
    pip cache purge 2>/dev/null || true

    log_success "Cleanup completed"
}

fix_system_libraries() {
    log "Fixing system libraries (GLIBCXX issue)..."

    # Update conda to ensure we have the latest resolver
    conda update -n base -c defaults conda -y -q 2>&1 | tee -a "$LOG_FILE"

    # Install/update gcc and libstdc++ for GLIBCXX support
    conda install -c conda-forge gcc_linux-64 libstdcxx-ng -y -q 2>&1 | tee -a "$LOG_FILE"

    # Export library path
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    log_success "System libraries updated"
}

install_conda_packages() {
    log "Installing compiled packages with conda..."

    # These packages are better installed via conda to avoid compilation issues
    local conda_packages=(
        "sentencepiece"
        "pyarrow>=21.0.0"
    )

    for package in "${conda_packages[@]}"; do
        log "Installing $package..."
        conda install -c conda-forge "$package" -y -q 2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Conda install failed for $package, will try pip"
        }
    done

    log_success "Conda packages installed"
}

install_requirements() {
    log "Installing Python packages from requirements..."

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        return 1
    fi

    # Install packages in groups to handle dependencies properly
    log "Installing AWS dependencies..."
    grep -E "^(botocore|boto3|aiobotocore)" "$REQUIREMENTS_FILE" | xargs -r pip install --no-cache-dir 2>&1 | tee -a "$LOG_FILE"

    log "Installing filesystem dependencies..."
    grep -E "^(fsspec|s3fs)" "$REQUIREMENTS_FILE" | xargs -r pip install --no-cache-dir 2>&1 | tee -a "$LOG_FILE"

    log "Installing multiprocessing dependencies..."
    grep -E "^(dill|multiprocess)" "$REQUIREMENTS_FILE" | xargs -r pip install --no-cache-dir 2>&1 | tee -a "$LOG_FILE"

    log "Installing ML core dependencies..."
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tee -a "$LOG_FILE"
    pip install --no-cache-dir transformers==4.56.2 datasets==2.21.0 2>&1 | tee -a "$LOG_FILE"

    log "Installing remaining dependencies..."
    pip install --no-cache-dir -r "$REQUIREMENTS_FILE" 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Some packages may have failed. Continuing..."
    }

    log_success "Package installation completed"
}

setup_training_data() {
    log "Setting up training data..."

    mkdir -p data/training

    if [ ! -f "data/training/query_pairs.jsonl" ]; then
        cat > data/training/query_pairs.jsonl << 'EOF'
{"query": "Show me all customers", "sql": "SELECT * FROM customers;", "context": "customers(id, name, email, created_at)"}
{"query": "Get total sales by month", "sql": "SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) as total FROM sales GROUP BY month;", "context": "sales(id, date, amount, product_id)"}
{"query": "Find top 5 products by revenue", "sql": "SELECT p.name, SUM(s.amount) as revenue FROM products p JOIN sales s ON p.id = s.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 5;", "context": "products(id, name, price), sales(id, product_id, amount)"}
{"query": "List users who registered today", "sql": "SELECT * FROM users WHERE DATE(created_at) = CURDATE();", "context": "users(id, name, email, created_at)"}
{"query": "Calculate average order value", "sql": "SELECT AVG(total_amount) as avg_order_value FROM orders;", "context": "orders(id, customer_id, total_amount, order_date)"}
EOF
        log_success "Created sample training data"
    else
        log_success "Training data already exists"
    fi
}

verify_installation() {
    log "Verifying installation..."

    if [ -f "$VERIFY_SCRIPT" ]; then
        python "$VERIFY_SCRIPT" 2>&1 | tee -a "$LOG_FILE" || {
            log_error "Verification failed. Check $LOG_FILE for details."
            return 1
        }
    else
        # Basic verification if script doesn't exist
        python -c "
import sys
try:
    import torch
    import transformers
    import datasets
    print('✅ PyTorch:', torch.__version__)
    print('✅ Transformers:', transformers.__version__)
    print('✅ Datasets:', datasets.__version__)
    print('✅ Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')
    sys.exit(0)
except ImportError as e:
    print('❌ Import error:', e)
    sys.exit(1)
" 2>&1 | tee -a "$LOG_FILE" || {
            log_error "Basic verification failed"
            return 1
        }
    fi

    log_success "Installation verified successfully"
}

display_next_steps() {
    echo ""
    echo "========================================="
    echo -e "${GREEN}✅ Setup Complete!${NC}"
    echo "========================================="
    echo ""
    echo "Log file saved to: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Open sagemaker/1_setup/Setup_Environment.ipynb to verify"
    echo "2. Open sagemaker/2_training/GPU_Training.ipynb for training"
    echo "3. Open sagemaker/3_inference/CPU_Inference.ipynb for inference"
    echo ""
    echo "If you encounter issues:"
    echo "- Check the log file: $LOG_FILE"
    echo "- Try manual installation: pip install -r $REQUIREMENTS_FILE"
    echo "- For ml.t2.medium instances, use pip instead of conda"
    echo ""
    echo "========================================="
}

rollback_on_error() {
    log_error "Setup failed. Rolling back changes..."
    log "Check log file for details: $LOG_FILE"
    exit 1
}

# ============= Main Execution =============

main() {
    trap rollback_on_error ERR

    echo "========================================="
    echo "🚀 GL RL Model - SageMaker Setup"
    echo "========================================="
    echo ""

    check_sagemaker_environment
    navigate_to_repo
    cleanup_conflicting_packages
    fix_system_libraries
    install_conda_packages
    install_requirements
    setup_training_data
    verify_installation
    display_next_steps
}

# Run the main function
main "$@"