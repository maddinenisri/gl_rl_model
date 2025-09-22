#!/bin/bash

# =============================================================================
# GL RL Model - SageMaker Setup Script
# =============================================================================
# Hybrid approach: conda for system libraries, pip for Python packages
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAGEMAKER_DIR="$REPO_ROOT/sagemaker"
LOG_FILE="/tmp/gl_rl_setup_$(date +%Y%m%d_%H%M%S).log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "[$(date +'%H:%M:%S')] $1" | tee -a "$LOG_FILE"
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

# Check environment
check_environment() {
    log "Checking environment..."

    if [ ! -d "/home/ec2-user/SageMaker" ]; then
        log_error "Not in SageMaker environment"
        exit 1
    fi

    # Check memory
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    log "Available memory: ${TOTAL_MEM}MB"

    if [ "$TOTAL_MEM" -lt 8000 ]; then
        log_warning "Low memory detected. Consider using ml.t3.xlarge or larger."
    fi
}

# Setup conda libraries (required for GLIBCXX)
setup_conda_libraries() {
    log "Setting up system libraries with conda..."

    # Update conda
    conda update -n base -c defaults conda -y -q 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Conda update failed, continuing..."
    }

    # Install GCC and libstdc++ for GLIBCXX support
    log "Installing gcc_linux-64 and libstdcxx-ng (required for GLIBCXX)..."
    conda install -c conda-forge gcc_linux-64 libstdcxx-ng -y -q 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Failed to install system libraries"
        return 1
    }

    # Export library path
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    log_success "System libraries configured"
}

# Install compiled packages via conda
install_compiled_packages() {
    log "Installing compiled packages with conda..."

    # These packages are better installed via conda
    local packages=(
        "sentencepiece"
        "pyarrow>=21.0.0"
    )

    for package in "${packages[@]}"; do
        log "Installing $package..."
        conda install -c conda-forge "$package" -y -q 2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Conda install failed for $package, will try pip"
        }
    done

    log_success "Compiled packages installed"
}

# Install Python packages via pip
install_python_packages() {
    log "Installing Python packages with pip..."

    # Navigate to sagemaker directory
    cd "$SAGEMAKER_DIR"

    # Check if requirements file exists
    if [ -f "requirements/pip-requirements.txt" ]; then
        pip install --no-cache-dir -r requirements/pip-requirements.txt 2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Some packages failed to install"
        }
    else
        log_error "Requirements file not found"
        return 1
    fi

    log_success "Python packages installed"
}

# Setup training data
setup_training_data() {
    log "Setting up training data..."

    mkdir -p "$REPO_ROOT/data/training"

    if [ ! -f "$REPO_ROOT/data/training/query_pairs.jsonl" ]; then
        cat > "$REPO_ROOT/data/training/query_pairs.jsonl" << 'EOF'
{"query": "Show me all customers", "sql": "SELECT * FROM customers;", "context": "customers(id, name, email, created_at)"}
{"query": "Get total sales by month", "sql": "SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) as total FROM sales GROUP BY month;", "context": "sales(id, date, amount, product_id)"}
{"query": "Find top 5 products by revenue", "sql": "SELECT p.name, SUM(s.amount) as revenue FROM products p JOIN sales s ON p.id = s.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 5;", "context": "products(id, name, price), sales(id, product_id, amount)"}
{"query": "List users who registered today", "sql": "SELECT * FROM users WHERE DATE(created_at) = CURDATE();", "context": "users(id, name, email, created_at)"}
{"query": "Calculate average order value", "sql": "SELECT AVG(total_amount) as avg_order_value FROM orders;", "context": "orders(id, customer_id, total_amount, order_date)"}
EOF
        log_success "Sample training data created"
    else
        log_success "Training data exists"
    fi
}

# Verify installation
verify_installation() {
    log "Verifying installation..."

    python -c "
import sys
try:
    import torch
    import transformers
    import datasets
    import peft
    print('✅ PyTorch:', torch.__version__)
    print('✅ Transformers:', transformers.__version__)
    print('✅ Datasets:', datasets.__version__)
    print('✅ PEFT:', peft.__version__)
    print('✅ Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')
    sys.exit(0)
except ImportError as e:
    print('❌ Import error:', e)
    sys.exit(1)
" || {
        log_error "Verification failed"
        return 1
    }

    log_success "All packages verified"
}

# Main execution
main() {
    echo "========================================="
    echo "🚀 GL RL Model - SageMaker Setup"
    echo "========================================="
    echo ""

    check_environment

    # Navigate to repository
    cd /home/ec2-user/SageMaker
    if [ -d "gl_rl_model" ]; then
        cd gl_rl_model
        log "Updating repository..."
        git pull origin main 2>&1 | tee -a "$LOG_FILE" || log_warning "Could not pull latest"
    else
        log "Cloning repository..."
        git clone https://github.com/maddinenisri/gl_rl_model.git 2>&1 | tee -a "$LOG_FILE"
        cd gl_rl_model
    fi

    # Run setup steps
    setup_conda_libraries
    install_compiled_packages
    install_python_packages
    setup_training_data
    verify_installation

    echo ""
    echo "========================================="
    log_success "Setup Complete!"
    echo "========================================="
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Open notebooks/01_environment_setup.ipynb"
    echo "2. Run notebooks/02_gpu_training.ipynb for training"
    echo "3. Use notebooks/03_cpu_inference.ipynb for inference"
    echo ""
}

# Run main
main "$@"