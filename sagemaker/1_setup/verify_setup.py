#!/usr/bin/env python3
"""
GL RL Model - SageMaker Environment Verification Script

Verifies that all required packages are installed correctly and compatible.
Checks versions, imports, and AWS connectivity.
"""

import sys
import json
from typing import Dict, List, Tuple
from importlib import import_module
import warnings

# Suppress warnings during import checks
warnings.filterwarnings('ignore')

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def print_status(message: str, status: str = "info"):
    """Print colored status messages."""
    if status == "success":
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")
    elif status == "error":
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
    elif status == "info":
        print(f"{Colors.BLUE}ℹ️  {message}{Colors.ENDC}")
    else:
        print(message)

def check_package_import(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package can be imported and return its version."""
    import_name = import_name or package_name

    try:
        module = import_module(import_name)
        version = "unknown"

        # Try different version attributes
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = str(getattr(module, attr))
                break

        return True, version
    except ImportError as e:
        return False, str(e)

def check_core_packages() -> Dict[str, bool]:
    """Check core ML packages."""
    print("\n" + "="*50)
    print("Checking Core ML Packages")
    print("="*50)

    core_packages = {
        'torch': {'import': 'torch', 'min_version': '2.0.0'},
        'transformers': {'import': 'transformers', 'min_version': '4.0.0'},
        'datasets': {'import': 'datasets', 'min_version': '2.0.0'},
        'peft': {'import': 'peft', 'min_version': None},
        'trl': {'import': 'trl', 'min_version': None},
        'accelerate': {'import': 'accelerate', 'min_version': None},
        'tokenizers': {'import': 'tokenizers', 'min_version': None},
        'sentencepiece': {'import': 'sentencepiece', 'min_version': None},
    }

    results = {}

    for package, info in core_packages.items():
        success, version = check_package_import(package, info['import'])

        if success:
            print_status(f"{package}: {version}", "success")
            results[package] = True
        else:
            print_status(f"{package}: Failed - {version}", "error")
            results[package] = False

    return results

def check_aws_packages() -> Dict[str, bool]:
    """Check AWS-related packages."""
    print("\n" + "="*50)
    print("Checking AWS Packages")
    print("="*50)

    aws_packages = {
        'boto3': 'boto3',
        'botocore': 'botocore',
        's3fs': 's3fs',
        'fsspec': 'fsspec',
        'aiobotocore': 'aiobotocore',
    }

    results = {}

    for package, import_name in aws_packages.items():
        success, version = check_package_import(package, import_name)

        if success:
            print_status(f"{package}: {version}", "success")
            results[package] = True
        else:
            print_status(f"{package}: Failed - {version}", "warning")
            results[package] = False

    return results

def check_data_packages() -> Dict[str, bool]:
    """Check data processing packages."""
    print("\n" + "="*50)
    print("Checking Data Processing Packages")
    print("="*50)

    data_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pyarrow': 'pyarrow',
        'sklearn': 'sklearn',
        'sqlparse': 'sqlparse',
        'sympy': 'sympy',
    }

    results = {}

    for package, import_name in data_packages.items():
        success, version = check_package_import(package, import_name)

        if success:
            print_status(f"{package}: {version}", "success")
            results[package] = True
        else:
            print_status(f"{package}: Failed - {version}", "warning")
            results[package] = False

    return results

def check_multiprocessing_packages() -> Dict[str, bool]:
    """Check multiprocessing packages."""
    print("\n" + "="*50)
    print("Checking Multiprocessing Packages")
    print("="*50)

    mp_packages = {
        'dill': 'dill',
        'multiprocess': 'multiprocess',
    }

    results = {}

    for package, import_name in mp_packages.items():
        success, version = check_package_import(package, import_name)

        if success:
            print_status(f"{package}: {version}", "success")
            results[package] = True
        else:
            print_status(f"{package}: Failed - {version}", "warning")
            results[package] = False

    return results

def check_cuda_availability():
    """Check if CUDA is available."""
    print("\n" + "="*50)
    print("Checking GPU/CUDA Availability")
    print("="*50)

    try:
        import torch

        if torch.cuda.is_available():
            print_status(f"CUDA Available: {torch.cuda.get_device_name(0)}", "success")
            print_status(f"CUDA Version: {torch.version.cuda}", "info")
            print_status(f"Number of GPUs: {torch.cuda.device_count()}", "info")
            return True
        else:
            print_status("CUDA Not Available - CPU mode only", "warning")
            print_status("For GPU training, use SageMaker Training Jobs", "info")
            return False
    except Exception as e:
        print_status(f"Could not check CUDA: {e}", "error")
        return False

def check_aws_connectivity():
    """Check AWS S3 connectivity."""
    print("\n" + "="*50)
    print("Checking AWS Connectivity")
    print("="*50)

    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError

        # Try to list S3 buckets (minimal permission needed)
        s3 = boto3.client('s3')

        try:
            response = s3.list_buckets()
            print_status("AWS S3 connectivity successful", "success")
            print_status(f"Access to {len(response.get('Buckets', []))} bucket(s)", "info")
            return True
        except NoCredentialsError:
            print_status("AWS credentials not configured", "warning")
            print_status("Run 'aws configure' to set up credentials", "info")
            return False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDenied':
                print_status("AWS connected but limited permissions", "warning")
            else:
                print_status(f"AWS error: {error_code}", "error")
            return False

    except ImportError:
        print_status("boto3 not installed - AWS connectivity unavailable", "error")
        return False
    except Exception as e:
        print_status(f"Unexpected error checking AWS: {e}", "error")
        return False

def check_version_compatibility():
    """Check for known version compatibility issues."""
    print("\n" + "="*50)
    print("Checking Version Compatibility")
    print("="*50)

    issues = []

    try:
        import boto3
        import botocore
        import s3fs
        import fsspec

        # Check botocore/boto3 compatibility
        boto3_version = boto3.__version__
        botocore_version = botocore.__version__

        # Extract major.minor versions
        boto3_major_minor = '.'.join(boto3_version.split('.')[:2])
        botocore_major_minor = '.'.join(botocore_version.split('.')[:2])

        if boto3_major_minor != botocore_major_minor:
            issues.append(f"boto3 ({boto3_version}) and botocore ({botocore_version}) version mismatch")

        # Check fsspec/s3fs compatibility
        fsspec_version = fsspec.__version__
        s3fs_version = s3fs.__version__

        # These should typically have matching major versions
        if fsspec_version.split('.')[0] != s3fs_version.split('.')[0]:
            issues.append(f"fsspec ({fsspec_version}) and s3fs ({s3fs_version}) may be incompatible")

    except ImportError as e:
        issues.append(f"Could not check all versions: {e}")

    if issues:
        for issue in issues:
            print_status(issue, "warning")
    else:
        print_status("All checked versions are compatible", "success")

    return len(issues) == 0

def generate_summary(results: Dict[str, Dict[str, bool]]):
    """Generate and print a summary of all checks."""
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)

    total_checks = 0
    passed_checks = 0
    failed_critical = []
    failed_optional = []

    critical_packages = ['torch', 'transformers', 'datasets']

    for category, category_results in results.items():
        for package, success in category_results.items():
            total_checks += 1
            if success:
                passed_checks += 1
            elif package in critical_packages:
                failed_critical.append(package)
            else:
                failed_optional.append(package)

    # Print summary
    print(f"\nTotal checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")

    if failed_critical:
        print_status(f"\nCritical packages missing: {', '.join(failed_critical)}", "error")
        print_status("These packages are required for basic functionality", "error")

    if failed_optional:
        print_status(f"\nOptional packages missing: {', '.join(failed_optional)}", "warning")
        print_status("These packages may be needed for specific features", "info")

    # Overall status
    print("\n" + "="*50)
    if not failed_critical:
        print_status("✅ Environment is ready for GL RL Model!", "success")
        return True
    else:
        print_status("❌ Critical packages missing. Please run setup.sh again.", "error")
        return False

def main():
    """Main verification function."""
    print("\n" + "="*70)
    print("GL RL Model - SageMaker Environment Verification")
    print("="*70)

    results = {}

    # Run all checks
    results['core'] = check_core_packages()
    results['aws'] = check_aws_packages()
    results['data'] = check_data_packages()
    results['multiprocessing'] = check_multiprocessing_packages()

    # Additional checks
    cuda_available = check_cuda_availability()
    aws_connected = check_aws_connectivity()
    versions_compatible = check_version_compatibility()

    # Generate summary
    success = generate_summary(results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()