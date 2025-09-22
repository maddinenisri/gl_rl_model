#!/usr/bin/env python3
"""
Fetch CloudWatch logs for a SageMaker training job

Usage:
    python fetch_cloudwatch_logs.py gl-rl-gpu-20250922-213443
"""

import boto3
import argparse
import sys
from datetime import datetime

def fetch_logs(job_name, profile='personal-yahoo', region='us-east-1'):
    """Fetch all CloudWatch logs for a training job"""

    # Initialize AWS session
    session = boto3.Session(profile_name=profile, region_name=region)
    logs_client = session.client('logs')

    log_group = '/aws/sagemaker/TrainingJobs'
    log_stream = f'{job_name}/algo-1-1758577012'

    print(f"Fetching logs from CloudWatch...")
    print(f"Log Group: {log_group}")
    print(f"Log Stream: {log_stream}")
    print("=" * 80)

    try:
        # Get all log events
        next_token = None
        all_events = []

        while True:
            kwargs = {
                'logGroupName': log_group,
                'logStreamName': log_stream,
                'startFromHead': True
            }

            if next_token:
                kwargs['nextToken'] = next_token

            response = logs_client.get_log_events(**kwargs)

            events = response.get('events', [])
            if not events:
                break

            all_events.extend(events)

            # Check if there are more events
            next_token = response.get('nextForwardToken')
            if next_token == kwargs.get('nextToken'):
                break

        # Print all events
        for event in all_events:
            timestamp = datetime.fromtimestamp(event['timestamp'] / 1000).strftime('%H:%M:%S')
            message = event['message'].rstrip()

            # Highlight important messages
            if 'ERROR' in message or 'Error' in message or 'error' in message:
                print(f"\033[91m[{timestamp}] {message}\033[0m")  # Red
            elif 'WARNING' in message or 'Warning' in message:
                print(f"\033[93m[{timestamp}] {message}\033[0m")  # Yellow
            elif 'Installing' in message or 'Installed' in message or '✓' in message:
                print(f"\033[92m[{timestamp}] {message}\033[0m")  # Green
            elif 'ImportError' in message or 'ModuleNotFoundError' in message or 'Traceback' in message:
                print(f"\033[91m[{timestamp}] {message}\033[0m")  # Red
            else:
                print(f"[{timestamp}] {message}")

        print("\n" + "=" * 80)
        print(f"Total log entries: {len(all_events)}")

        # Save to file
        output_file = f"{job_name}_logs.txt"
        with open(output_file, 'w') as f:
            for event in all_events:
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {event['message']}\n")

        print(f"Logs saved to: {output_file}")

    except Exception as e:
        print(f"Error fetching logs: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Fetch CloudWatch logs for SageMaker training job')
    parser.add_argument('job_name', help='Training job name')
    parser.add_argument('--profile', default='personal-yahoo', help='AWS profile')
    parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()
    fetch_logs(args.job_name, args.profile, args.region)

if __name__ == '__main__':
    main()