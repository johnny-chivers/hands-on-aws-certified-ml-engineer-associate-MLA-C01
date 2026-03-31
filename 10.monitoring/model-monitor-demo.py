"""
SageMaker Model Monitor Comprehensive Demonstration

Implements data drift detection, model quality monitoring, and constraint violations.
Maps to MLA-C01 Domain 3: Model Monitoring and Governance

Key Concepts:
- Data Capture Configuration: Captures input/output at prediction time
- Baselining: Creates baseline statistics and constraints from historical data
- Monitoring Schedule: Runs periodic jobs to detect drift
- Data Quality: Detects feature/output distribution changes
- Model Quality: Tracks prediction accuracy degradation
"""

import boto3
import json
from datetime import datetime, timedelta
from sagemaker.model_monitor import (
    DataCaptureConfig,
    DefaultModelMonitor,
    CronExpressionGenerator,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat

# Configuration placeholders for production
AWS_REGION = '<YOUR-REGION>'
ROLE_ARN = '<YOUR-ROLE-ARN>'
ENDPOINT_NAME = '<YOUR-ENDPOINT-NAME>'
BASELINE_DATA_S3_URI = 's3://<YOUR-BUCKET-NAME>/monitoring/baseline-data/'
BASELINE_STATISTICS_S3_URI = 's3://<YOUR-BUCKET-NAME>/monitoring/baseline-statistics/'
BASELINE_CONSTRAINTS_S3_URI = 's3://<YOUR-BUCKET-NAME>/monitoring/baseline-constraints/'
DATA_CAPTURE_S3_URI = 's3://<YOUR-BUCKET-NAME>/monitoring/data-capture/'
MONITORING_OUTPUT_S3_URI = 's3://<YOUR-BUCKET-NAME>/monitoring/results/'

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)


def setup_data_capture_config():
    """
    Configure data capture on an existing SageMaker endpoint.

    Data capture configuration enables Model Monitor to:
    - Record all input features sent to the endpoint
    - Record all predictions returned
    - Detect feature drift by comparing captured data to baseline

    MLA-C01 Exam Focus: Data capture is prerequisite for monitoring
    """
    print("[STEP 1] Setting up Data Capture Configuration...")

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,  # Capture 100% of requests (production: tune this)
        destination_s3_uri=DATA_CAPTURE_S3_URI,
        capture_options=[
            "Input",   # Capture request payload
            "Output",  # Capture prediction results
        ],
    )

    try:
        # Update endpoint to enable data capture
        sagemaker_client.update_endpoint_input(
            EndpointName=ENDPOINT_NAME,
            DataCaptureConfig=data_capture_config.to_dict(),
        )
        print(f"✓ Data capture enabled for endpoint: {ENDPOINT_NAME}")
        print(f"  Destination: {DATA_CAPTURE_S3_URI}")
        print(f"  Sampling: 100% (adjust for high-traffic endpoints)")
        return data_capture_config
    except Exception as e:
        print(f"✗ Error setting up data capture: {str(e)}")
        raise


def create_baseline_job(baseline_data_uri):
    """
    Create a baselining job to generate baseline statistics and constraints.

    Baselining process:
    1. Reads historical training/validation data
    2. Generates statistics (mean, std, quantiles, histograms)
    3. Generates constraints (feature value ranges, missing values)
    4. Stores output in S3 for later monitoring comparison

    MLA-C01 Exam Focus: Baselines are the "expected normal" for drift detection
    """
    print("\n[STEP 2] Creating Baseline Job...")

    monitor = DefaultModelMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=50,
        max_runtime_in_seconds=3600,
        sagemaker_session=None,
    )

    try:
        # suggest_baseline() automatically creates baseline statistics and constraints
        baseline_job_output = monitor.suggest_baseline(
            baseline_dataset=baseline_data_uri,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=BASELINE_STATISTICS_S3_URI,
            wait=True,  # Wait for job completion
            logs=True,  # Stream job logs
        )

        print(f"✓ Baseline job completed")
        print(f"  Baseline statistics URI: {baseline_job_output}")

        # Constraints are automatically generated alongside statistics
        print(f"  Baseline constraints URI: {BASELINE_CONSTRAINTS_S3_URI}")
        print(f"  These define acceptable ranges for each feature")

        return baseline_job_output
    except Exception as e:
        print(f"✗ Error creating baseline: {str(e)}")
        raise


def create_monitoring_schedule():
    """
    Schedule a continuous monitoring job.

    Monitoring job execution:
    - Runs on a schedule (hourly, daily, etc.)
    - Compares current data capture to baseline statistics
    - Detects drift when feature distributions deviate
    - Generates violations report
    - Can trigger CloudWatch alarms

    MLA-C01 Exam Focus: Production models must have continuous monitoring
    """
    print("\n[STEP 3] Creating Monitoring Schedule...")

    monitor = DefaultModelMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=50,
        max_runtime_in_seconds=3600,
        sagemaker_session=None,
    )

    # Define monitoring schedule: run every hour
    schedule_expression = CronExpressionGenerator.hourly()

    try:
        monitor.create_monitoring_schedule(
            monitor_schedule_name=f"model-monitor-{ENDPOINT_NAME}",
            endpoint_input=ENDPOINT_NAME,
            post_analytics_processor_script=None,  # Optional: custom processing
            output_s3_uri=MONITORING_OUTPUT_S3_URI,
            statistics=BASELINE_STATISTICS_S3_URI,
            constraints=BASELINE_CONSTRAINTS_S3_URI,
            schedule_cron_expression=schedule_expression,
            enable_cloudwatch_metrics=True,  # Publish metrics to CloudWatch
        )

        print(f"✓ Monitoring schedule created")
        print(f"  Schedule: Hourly (every 60 minutes)")
        print(f"  Baseline statistics: {BASELINE_STATISTICS_S3_URI}")
        print(f"  Baseline constraints: {BASELINE_CONSTRAINTS_S3_URI}")
        print(f"  Results output: {MONITORING_OUTPUT_S3_URI}")
        print(f"  CloudWatch metrics: Enabled")

        return f"model-monitor-{ENDPOINT_NAME}"
    except Exception as e:
        print(f"✗ Error creating monitoring schedule: {str(e)}")
        raise


def list_monitoring_executions(schedule_name):
    """
    List all executions of a monitoring schedule.

    Execution status indicates:
    - Pending: Waiting to run
    - InProgress: Currently running
    - Completed: Finished (check for violations)
    - Failed: Error occurred

    MLA-C01 Exam Focus: Monitor execution history for drift patterns
    """
    print(f"\n[STEP 4] Listing Monitoring Executions for {schedule_name}...")

    try:
        response = sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=schedule_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=10,
        )

        executions = response.get('MonitoringExecutionSummaries', [])

        if not executions:
            print("✓ No monitoring executions found yet")
            return

        print(f"✓ Found {len(executions)} monitoring executions:")
        for i, execution in enumerate(executions, 1):
            print(f"\n  [{i}] {execution['MonitoringExecutionName']}")
            print(f"      Status: {execution['MonitoringExecutionStatus']}")
            print(f"      Creation Time: {execution['CreationTime']}")

            # Check for violations
            if 'MonitoringJobDefinitionName' in execution:
                violations = execution.get('ViolationAttributes', {})
                if violations:
                    print(f"      ⚠ VIOLATIONS DETECTED: {violations}")

        return executions
    except Exception as e:
        print(f"✗ Error listing executions: {str(e)}")
        raise


def check_monitoring_violations(schedule_name):
    """
    Check for data drift violations in latest monitoring execution.

    Violation types:
    - FeatureBasedDrift: Feature distribution changed significantly
    - ModelOutputDrift: Prediction distribution changed
    - ConstraintViolations: Feature values outside acceptable ranges

    MLA-C01 Exam Focus: Drift detection triggers model retraining
    """
    print(f"\n[STEP 5] Checking for Monitoring Violations...")

    try:
        # Get latest execution
        response = sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=schedule_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1,
        )

        if not response.get('MonitoringExecutionSummaries'):
            print("✓ No executions available to check")
            return None

        latest_execution = response['MonitoringExecutionSummaries'][0]
        execution_name = latest_execution['MonitoringExecutionName']
        status = latest_execution['MonitoringExecutionStatus']

        print(f"  Latest execution: {execution_name}")
        print(f"  Status: {status}")

        if status == 'Completed':
            # Parse violation details from S3
            print("\n  Checking violation report in S3...")
            print(f"  Report location: {MONITORING_OUTPUT_S3_URI}{execution_name}/")

            # In production, parse violations_report.json from S3
            violations_detected = True  # Placeholder
            if violations_detected:
                print("  ⚠ DRIFT DETECTED: Feature distribution has changed")
                print("     Action: Consider retraining model with recent data")
                print("     Threshold: Use statistical tests (Kolmogorov-Smirnov)")
            else:
                print("  ✓ No drift detected - model behavior is stable")

        return latest_execution
    except Exception as e:
        print(f"✗ Error checking violations: {str(e)}")
        raise


def demonstrate_drift_detection_logic():
    """
    Educational demonstration of drift detection concepts.

    Drift Detection Methods:
    1. Statistical Tests:
       - Kolmogorov-Smirnov test: Compare distributions
       - Chi-square test: Categorical feature changes
       - Z-test: Mean shift detection

    2. Constraint-Based:
       - Value out of range
       - Missing values > threshold
       - New unseen categories

    3. Model Quality:
       - Prediction distribution changes
       - Confidence score degradation
       - Accuracy vs baseline

    MLA-C01 Exam Focus: Understanding drift triggers model intervention
    """
    print("\n[STEP 6] Drift Detection Logic Explanation...")

    print("""
    Drift Detection Mechanisms:

    [A] DATA QUALITY MONITORING:
        - Feature x1: mean shifted from 5.0 → 7.2 (drift!)
        - Feature x2: std dev increased 2x (variance drift!)
        - Feature x3: 5% missing values (quality issue!)

    [B] MODEL QUALITY MONITORING:
        - Prediction avg: 0.65 → 0.45 (output drift!)
        - Confidence: 0.95 → 0.72 (uncertainty increase!)
        - Error rate: 5% → 18% (performance degradation!)

    [C] STATISTICAL THRESHOLDS:
        - KS statistic > 0.2: Distribution changed significantly
        - p-value < 0.05: Statistically significant difference
        - Domain knowledge: Business metric thresholds

    RESPONSE ACTIONS:
        1. Alert: Notify ML team of drift
        2. Investigate: Analyze root cause
        3. Retrain: Use recent data to adapt
        4. Rollback: Switch to previous model version
        5. Validate: Run A/B test before production
    """)


def main():
    """
    Main orchestration function demonstrating Model Monitor workflow.

    MLA-C01 Task 3.3: Monitor ML models in production
    """
    print("=" * 70)
    print("SageMaker Model Monitor - Comprehensive Demonstration")
    print("=" * 70)
    print(f"Endpoint: {ENDPOINT_NAME}")
    print(f"Region: {AWS_REGION}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    try:
        # Step 1: Enable data capture on endpoint
        data_capture_config = setup_data_capture_config()

        # Step 2: Create baseline from historical data
        # NOTE: In production, baseline_data_uri should contain
        # representative training/validation data (CSV or Parquet format)
        baseline_output = create_baseline_job(BASELINE_DATA_S3_URI)

        # Step 3: Schedule monitoring job
        schedule_name = create_monitoring_schedule()

        # Step 4: List monitoring executions
        executions = list_monitoring_executions(schedule_name)

        # Step 5: Check for violations
        latest_execution = check_monitoring_violations(schedule_name)

        # Step 6: Educational drift detection concepts
        demonstrate_drift_detection_logic()

        print("\n" + "=" * 70)
        print("✓ Model Monitor Demonstration Complete")
        print("=" * 70)
        print("\nNext Steps in Production:")
        print("1. Review monitoring metrics in CloudWatch dashboard")
        print("2. Set up SNS alerts for drift violations")
        print("3. Establish SLA for model retraining (e.g., weekly)")
        print("4. Document acceptable drift thresholds")
        print("5. Implement automated retraining pipeline")

    except Exception as e:
        print(f"\n✗ Demonstration failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
