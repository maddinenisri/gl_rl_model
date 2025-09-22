#!/usr/bin/env python3
"""
Monitor SageMaker Training Job with CloudWatch Logs

Usage:
    python monitor_training.py <job-name>
    python monitor_training.py gl-rl-gpu-20250922-204753
"""

import sys
import time
import boto3
import argparse
from datetime import datetime
from typing import Dict, Any

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def format_time(seconds: int) -> str:
    """Format seconds into human-readable time"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_job_status(sagemaker_client: Any, job_name: str) -> Dict[str, Any]:
    """Get current job status and details"""
    try:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        return response
    except Exception as e:
        print(f"{Colors.RED}Error getting job status: {e}{Colors.END}")
        return None


def print_job_status(job_info: Dict[str, Any]) -> str:
    """Print formatted job status"""
    status = job_info['TrainingJobStatus']
    secondary_status = job_info.get('SecondaryStatus', 'Pending')

    # Color code based on status
    status_color = Colors.YELLOW
    if status == 'InProgress':
        status_color = Colors.CYAN
    elif status == 'Completed':
        status_color = Colors.GREEN
    elif status == 'Failed':
        status_color = Colors.RED
    elif status == 'Stopped':
        status_color = Colors.YELLOW

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")

    # Job name and status
    print(f"{status_color}[{datetime.now().strftime('%H:%M:%S')}] {status} - {secondary_status}{Colors.END}")

    # Runtime calculation
    if 'TrainingStartTime' in job_info:
        start_time = job_info['TrainingStartTime']
        if status == 'InProgress':
            runtime = int((datetime.now(start_time.tzinfo) - start_time).total_seconds())
            print(f"   ⏱️  Runtime: {format_time(runtime)}")
        elif status == 'Completed' and 'TrainingEndTime' in job_info:
            runtime = int((job_info['TrainingEndTime'] - start_time).total_seconds())
            print(f"   ✅ Total runtime: {format_time(runtime)}")

    # Secondary status messages
    if secondary_status == 'Downloading':
        print(f"   📥 Downloading data and model...")
    elif secondary_status == 'Training':
        if 'ResourceConfig' in job_info:
            instance_type = job_info['ResourceConfig']['InstanceType']
            print(f"   🔥 GPU Training in progress on {instance_type}!")
    elif secondary_status == 'Uploading':
        print(f"   📤 Uploading model artifacts...")

    return status


def stream_logs(logs_client: Any, job_name: str, start_time: int = None) -> int:
    """Stream CloudWatch logs for the training job"""
    log_group = '/aws/sagemaker/TrainingJobs'

    try:
        # Get log streams for this job
        response = logs_client.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name,
            orderBy='LastEventTime',
            descending=True,
            limit=5
        )

        if not response['logStreams']:
            return start_time

        # Stream logs from all available streams
        for stream in response['logStreams']:
            stream_name = stream['logStreamName']

            # Get log events
            kwargs = {
                'logGroupName': log_group,
                'logStreamName': stream_name,
                'startFromHead': False,
                'limit': 50
            }

            if start_time:
                kwargs['startTime'] = start_time

            try:
                events_response = logs_client.get_log_events(**kwargs)

                # Print new log events
                for event in events_response['events']:
                    message = event['message'].strip()

                    # Color code based on content
                    if 'ERROR' in message or 'Failed' in message:
                        print(f"{Colors.RED}{message}{Colors.END}")
                    elif 'WARNING' in message:
                        print(f"{Colors.YELLOW}{message}{Colors.END}")
                    elif 'SUCCESS' in message or '✓' in message:
                        print(f"{Colors.GREEN}{message}{Colors.END}")
                    elif 'epoch' in message.lower() or 'loss' in message.lower():
                        print(f"{Colors.CYAN}{message}{Colors.END}")
                    else:
                        print(message)

                    # Update start time for next iteration
                    if event['timestamp'] > (start_time or 0):
                        start_time = event['timestamp'] + 1

            except Exception as e:
                if 'ResourceNotFoundException' not in str(e):
                    print(f"{Colors.YELLOW}Warning: Could not read logs from {stream_name}{Colors.END}")

    except Exception as e:
        if 'ResourceNotFoundException' not in str(e):
            print(f"{Colors.RED}Error streaming logs: {e}{Colors.END}")

    return start_time


def monitor_job(job_name: str, region: str = 'us-east-1', profile: str = 'personal-yahoo'):
    """Main monitoring loop"""
    print(f"{Colors.BOLD}{Colors.CYAN}Monitoring SageMaker Training Job: {job_name}{Colors.END}")
    print(f"Region: {region}, Profile: {profile}")

    # Initialize AWS clients
    session = boto3.Session(profile_name=profile, region_name=region)
    sagemaker_client = session.client('sagemaker')
    logs_client = session.client('logs')

    last_log_time = None
    last_status = None

    try:
        while True:
            # Get job status
            job_info = get_job_status(sagemaker_client, job_name)
            if not job_info:
                break

            current_status = job_info['TrainingJobStatus']

            # Print status if changed
            if current_status != last_status:
                last_status = print_job_status(job_info)

            # Stream new logs
            if current_status == 'InProgress':
                last_log_time = stream_logs(logs_client, job_name, last_log_time)

            # Check if job completed
            if current_status in ['Completed', 'Failed', 'Stopped']:
                # Print final status
                print_job_status(job_info)

                # Stream any remaining logs
                stream_logs(logs_client, job_name, last_log_time)

                if current_status == 'Completed':
                    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Training completed successfully!{Colors.END}")
                    if 'ModelArtifacts' in job_info:
                        print(f"📦 Model artifacts: {job_info['ModelArtifacts']['S3ModelArtifacts']}")

                    # Calculate cost estimate
                    if 'TrainingStartTime' in job_info and 'TrainingEndTime' in job_info:
                        runtime_hours = (job_info['TrainingEndTime'] - job_info['TrainingStartTime']).total_seconds() / 3600
                        instance_type = job_info['ResourceConfig']['InstanceType']

                        # Spot price estimates
                        spot_prices = {
                            'ml.g5.xlarge': 0.30,
                            'ml.g4dn.xlarge': 0.35,
                            'ml.p3.2xlarge': 1.15
                        }

                        if instance_type in spot_prices:
                            cost = runtime_hours * spot_prices[instance_type]
                            print(f"💰 Estimated cost: ${cost:.2f}")

                elif current_status == 'Failed':
                    print(f"\n{Colors.RED}{Colors.BOLD}❌ Training failed!{Colors.END}")
                    if 'FailureReason' in job_info:
                        print(f"Reason: {job_info['FailureReason']}")
                    print(f"\nTo debug, check CloudWatch logs:")
                    print(f"https://console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs/log-events/{job_name}")

                break

            # Wait before next check
            time.sleep(10)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitoring stopped by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Monitoring error: {e}{Colors.END}")


def main():
    parser = argparse.ArgumentParser(description='Monitor SageMaker Training Job')
    parser.add_argument('job_name', help='Name of the training job to monitor')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    parser.add_argument('--profile', default='personal-yahoo', help='AWS profile (default: personal-yahoo)')

    args = parser.parse_args()

    monitor_job(args.job_name, args.region, args.profile)


if __name__ == '__main__':
    main()