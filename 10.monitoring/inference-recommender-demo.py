"""
SageMaker Inference Recommender Comprehensive Demonstration

Automatically determines optimal instance types and configurations for model deployment.
Maps to MLA-C01 Domain 2: Building, Training, and Deploying ML Models

Key Concepts:
- Model Registry: Version control for ML models
- Inference Recommender: Tests different instance types
- Right-Sizing: Balance cost, latency, and throughput
- Load Testing: Measures performance under traffic
"""

import boto3
import json
from datetime import datetime
from sagemaker import Session
from sagemaker.model_monitor import ModelExplainability
import time

# Configuration placeholders
AWS_REGION = '<YOUR-REGION>'
ROLE_ARN = '<YOUR-ROLE-ARN>'
BUCKET_NAME = '<YOUR-BUCKET-NAME>'
MODEL_PACKAGE_ARN = 'arn:aws:sagemaker:<YOUR-REGION>:<ACCOUNT-ID>:model-package/<MODEL-PACKAGE-NAME>'
MODEL_NAME = '<YOUR-MODEL-NAME>'

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
session = Session(default_bucket=BUCKET_NAME, sagemaker_runtime_execution_role=ROLE_ARN)


def register_model_in_registry():
    """
    Register a model in SageMaker Model Registry.

    Model Registry provides:
    - Version control for trained models
    - Model lineage (data, training job, evaluation)
    - Approval workflow (Manual or Automatic)
    - Staging gates (Dev, Staging, Production)

    MLA-C01 Exam Focus: Model Registry is prerequisite for Inference Recommender
    """
    print("[STEP 1] Registering Model in Model Registry...")

    try:
        # Create model package (prerequisite for Inference Recommender)
        model_package_response = sagemaker_client.create_model_package(
            ModelPackageName=MODEL_NAME,
            ModelPackageDescription="Production ML model for inference optimization",
            InferenceSpecification={
                'Containers': [
                    {
                        'Image': f'{AWS_REGION}.dkr.ecr.amazonaws.com/xgboost:latest',
                        'ModelDataUrl': f's3://{BUCKET_NAME}/models/model.tar.gz',
                        'Framework': 'XGBOOST',
                        'FrameworkVersion': '1.5',
                        'NearestModelName': MODEL_NAME,
                    }
                ],
                'SupportedContentTypes': ['text/csv', 'application/json'],
                'SupportedResponseMIMETypes': ['application/json'],
            },
            ModelApprovalStatus='Approved',  # Mark approved for deployment
            Tags=[
                {'Key': 'Purpose', 'Value': 'InferenceOptimization'},
                {'Key': 'Team', 'Value': 'MLOps'},
            ],
        )

        model_package_arn = model_package_response['ModelPackageArn']
        print(f"✓ Model registered in Model Registry")
        print(f"  Model Package ARN: {model_package_arn}")
        print(f"  Status: Approved for inference optimization")

        return model_package_arn
    except Exception as e:
        print(f"✗ Error registering model: {str(e)}")
        # In practice, may already exist; use existing ARN
        return MODEL_PACKAGE_ARN


def create_default_inference_recommender_job():
    """
    Create a default Inference Recommender job.

    Default job automatically tests:
    - ml.t3.medium, ml.t3.large, ml.t3.xlarge (general purpose, lowest cost)
    - ml.m5.large, ml.m5.xlarge (compute optimized)
    - Measures: latency p50/p99, throughput, cost per inference

    MLA-C01 Exam Focus: Right-sizing reduces costs without sacrificing latency SLA
    """
    print("\n[STEP 2] Creating Default Inference Recommender Job...")

    try:
        job_name = f"inference-recommender-default-{int(datetime.now().timestamp())}"

        response = sagemaker_client.create_inference_recommendations_job(
            JobName=job_name,
            RoleArn=ROLE_ARN,
            InputConfig={
                'ModelPackageVersionArn': MODEL_PACKAGE_ARN,
                'JobDurationInSeconds': 3600,  # 1 hour test duration
                'TrafficPattern': {
                    'TrafficType': 'UNIFORM',  # Constant load
                    'Phases': [
                        {
                            'InitialNumberOfUsers': 1,
                            'SpawnRate': 1,
                            'DurationInSeconds': 300,  # 5 minutes per phase
                        }
                    ],
                },
            },
            JobType='DEFAULT',  # AWS selects instance types
            Endpoint={
                'EndpointName': f'{MODEL_NAME}-inference-test',
            },
            OutputConfig={
                'S3OutputLocation': f's3://{BUCKET_NAME}/inference-recommender/results/',
            },
            Tags=[
                {'Key': 'Type', 'Value': 'DefaultRecommendation'},
            ],
        )

        job_name_returned = response['JobName']
        print(f"✓ Default Inference Recommender job created")
        print(f"  Job Name: {job_name_returned}")
        print(f"  Duration: 1 hour of load testing")
        print(f"  Traffic Pattern: Uniform load starting at 1 user")
        print(f"  AWS will test optimal instance types automatically")

        return job_name_returned
    except Exception as e:
        print(f"✗ Error creating default job: {str(e)}")
        raise


def create_advanced_inference_recommender_job():
    """
    Create an advanced/custom Inference Recommender job.

    Custom job allows:
    - Test specific instance types (not just AWS defaults)
    - Define custom traffic patterns (ramp-up, spike, constant)
    - Specify performance objectives (latency targets, throughput)
    - Test different batch sizes and concurrency levels

    MLA-C01 Exam Focus: Custom jobs match your actual production workload
    """
    print("\n[STEP 3] Creating Advanced Custom Inference Recommender Job...")

    try:
        job_name = f"inference-recommender-advanced-{int(datetime.now().timestamp())}"

        # Define custom traffic pattern: ramp-up to peak then sustain
        response = sagemaker_client.create_inference_recommendations_job(
            JobName=job_name,
            RoleArn=ROLE_ARN,
            InputConfig={
                'ModelPackageVersionArn': MODEL_PACKAGE_ARN,
                'JobDurationInSeconds': 5400,  # 90 minutes
                'TrafficPattern': {
                    'TrafficType': 'PHASES',
                    'Phases': [
                        {
                            'InitialNumberOfUsers': 1,
                            'SpawnRate': 2,  # Add 2 users per second
                            'DurationInSeconds': 600,  # 10 min ramp-up
                        },
                        {
                            'InitialNumberOfUsers': 121,  # (1 + 2*600/1)
                            'SpawnRate': 0,  # Hold steady
                            'DurationInSeconds': 1200,  # 20 min sustain
                        },
                        {
                            'InitialNumberOfUsers': 121,
                            'SpawnRate': 5,  # Spike test
                            'DurationInSeconds': 300,  # 5 min spike
                        },
                    ],
                },
            },
            JobType='ADVANCED',  # Custom instance types
            Endpoint={
                'EndpointName': f'{MODEL_NAME}-inference-advanced',
            },
            OutputConfig={
                'S3OutputLocation': f's3://{BUCKET_NAME}/inference-recommender/advanced/',
            },
            # Specify instance types to test
            InstanceTypes=[
                'ml.m5.large',
                'ml.m5.xlarge',
                'ml.m5.2xlarge',
                'ml.c5.large',
                'ml.c5.xlarge',
                'ml.c5.2xlarge',
                'ml.g4dn.xlarge',  # GPU instance for acceleration
            ],
            Tags=[
                {'Key': 'Type', 'Value': 'AdvancedRecommendation'},
                {'Key': 'TrafficPattern', 'Value': 'RampUpAndSpike'},
            ],
        )

        job_name_returned = response['JobName']
        print(f"✓ Advanced Inference Recommender job created")
        print(f"  Job Name: {job_name_returned}")
        print(f"  Duration: 90 minutes (ramp-up + sustain + spike)")
        print(f"  Instance Types Tested:")
        instance_types = ['ml.m5.large', 'ml.m5.xlarge', 'ml.m5.2xlarge',
                         'ml.c5.large', 'ml.c5.xlarge', 'ml.c5.2xlarge',
                         'ml.g4dn.xlarge']
        for idx, itype in enumerate(instance_types, 1):
            print(f"    [{idx}] {itype}")
        print(f"  Traffic Pattern: Ramp-up (0-121 users) → Sustain → Spike")

        return job_name_returned
    except Exception as e:
        print(f"✗ Error creating advanced job: {str(e)}")
        raise


def retrieve_job_results(job_name):
    """
    Retrieve and compare results from Inference Recommender job.

    Metrics collected:
    - MaxInvocations: Peak throughput (req/sec)
    - Latency (P50, P99): Response time percentiles
    - Cost per inference: Instance hourly cost / throughput
    - CPUUtilization, MemoryUtilization: Resource consumption

    MLA-C01 Exam Focus: Use results to select optimal instance type
    """
    print(f"\n[STEP 4] Retrieving Results for Job: {job_name}...")

    try:
        # Get job status
        response = sagemaker_client.describe_inference_recommendations_job(
            JobName=job_name,
        )

        job_status = response['Status']
        creation_time = response['CreationTime']
        last_modified = response['LastModifiedTime']

        print(f"✓ Job Status: {job_status}")
        print(f"  Created: {creation_time}")
        print(f"  Last Modified: {last_modified}")

        if job_status in ['InProgress', 'Pending']:
            print(f"\n  ⏳ Job still running... Check back in a few minutes")
            print(f"  Recommendation: Set up CloudWatch alarm to notify when complete")
            return None

        if job_status == 'Completed':
            # Extract recommendations
            recommendations = response.get('InferenceRecommendations', [])

            print(f"\n  Found {len(recommendations)} instance type recommendations:")
            print("\n  Rank  Instance Type      Latency P99  Throughput   Cost/1K Invokes")
            print("  " + "-" * 75)

            for idx, rec in enumerate(recommendations[:5], 1):  # Top 5
                metrics = rec.get('Metrics', {})
                instance_type = rec.get('InstanceType', 'N/A')
                latency_p99 = metrics.get('P99Latency', 0) / 1000  # Convert to seconds
                throughput = metrics.get('MaxInvocations', 0)
                cost_per_invocation = metrics.get('CostPerInference', 0)

                print(f"  [{idx}]  {instance_type:15}  {latency_p99:>8.3f}s  {throughput:>10.1f}  ${cost_per_invocation*1000:>12.2f}")

            # Provide recommendation logic
            print("\n  RECOMMENDATION LOGIC:")
            print("  ✓ Rank 1: Balanced cost/performance")
            print("  ✓ Rank 2-3: Trade-offs (faster but pricier, cheaper but slower)")
            print("  ✓ Production Selection: Match your SLA requirements")
            print("    - Latency SLA < 50ms? → Pick ml.c5.xlarge")
            print("    - Cost sensitive? → Try ml.m5.large (might be sufficient)")
            print("    - High throughput (>1000 req/s)? → ml.m5.2xlarge or ml.g4dn.xlarge")

            return recommendations

        print(f"  Status: {job_status}")
        return None

    except Exception as e:
        print(f"✗ Error retrieving results: {str(e)}")
        raise


def demonstrate_instance_selection_best_practices():
    """
    Educational guide for instance type selection.

    Instance Family Trade-offs:

    [GENERAL PURPOSE - ml.t3]
    - Cost: Lowest ($$$)
    - Performance: Lowest (single vCPU burst)
    - Use: Dev/test, low-traffic endpoints (<10 req/s)
    - Example: ml.t3.medium

    [COMPUTE OPTIMIZED - ml.c5]
    - Cost: Mid ($$)
    - Performance: High (CPU-intensive, good memory)
    - Use: High throughput, low-latency ML models
    - Example: ml.c5.2xlarge (48 vCPU, 192 GB RAM)

    [MEMORY OPTIMIZED - ml.r5]
    - Cost: High ($$$)
    - Performance: High (large working set)
    - Use: Large models, batch processing
    - Example: ml.r5.4xlarge (128 GB RAM)

    [GPU ACCELERATED - ml.g4dn]
    - Cost: Very High ($$$$)
    - Performance: Highest (GPU compute)
    - Use: Deep learning inference, real-time video
    - Example: ml.g4dn.xlarge (1 GPU, excellent latency)

    BEST PRACTICES:

    1. START WITH SMALLEST VIABLE INSTANCE
       - Reduce costs, measure baseline performance
       - Right-size after understanding demand

    2. MATCH INSTANCE TO WORKLOAD
       - CPU-bound (XGBoost, LightGBM) → ml.c5.x
       - Memory-heavy (LLMs) → ml.r5.x
       - GPU-dependent (Neural Networks) → ml.g4dn.x

    3. MONITOR AND ADJUST
       - CloudWatch: CPU%, Memory%, Latency
       - Too high utilization (>80%) → Upgrade
       - Too low utilization (<20%) → Downsize

    4. COST OPTIMIZATION
       - Reserved Instances: 30% discount for 1-year commitment
       - Spot Instances: 70% savings (but interruptible)
       - Inference Recommender: Automates this analysis

    5. MULTI-VARIANT DEPLOYMENT
       - Route traffic: 80% stable variant, 20% new
       - A/B test: Measure impact of new instance type
       - Gradual shift when confident
    """
    print("\n[STEP 5] Instance Selection Best Practices...")

    print("""
    ═════════════════════════════════════════════════════════════
    INSTANCE TYPE SELECTION FOR ML INFERENCE
    ═════════════════════════════════════════════════════════════

    GENERAL PURPOSE (ml.t3.* / ml.t3a.*)
    ├─ Cost: $$$ (Lowest)
    ├─ vCPU: 2-8 cores, burstable
    ├─ RAM: 1-32 GB
    ├─ Use Cases: Dev/test, low traffic (<10 req/sec)
    └─ Example: ml.t3.medium (2 vCPU, 4 GB RAM) = $0.05/hour

    COMPUTE OPTIMIZED (ml.c5.* / ml.c6i.*)
    ├─ Cost: $$ (Medium)
    ├─ vCPU: 2-96 cores, high CPU efficiency
    ├─ RAM: 2-192 GB
    ├─ Use Cases: XGBoost, LightGBM, real-time scoring
    └─ Example: ml.c5.2xlarge (8 vCPU, 16 GB RAM) = $0.34/hour

    MEMORY OPTIMIZED (ml.r5.* / ml.r6i.*)
    ├─ Cost: $$$ (High)
    ├─ vCPU: 2-96 cores
    ├─ RAM: 16-768 GB
    ├─ Use Cases: Large models, transformer networks
    └─ Example: ml.r5.4xlarge (16 vCPU, 128 GB RAM) = $1.35/hour

    GPU ACCELERATED (ml.g4dn.* / ml.p3.*)
    ├─ Cost: $$$$ (Highest)
    ├─ GPU: 1-8 NVIDIA T4/V100 GPUs
    ├─ RAM: 12-244 GB + GPU memory
    ├─ Use Cases: Deep learning, LLMs, computer vision
    └─ Example: ml.g4dn.xlarge (1 GPU T4, 4 vCPU) = $0.526/hour

    DECISION FLOWCHART:
    ┌─ What's your model type?
    ├─ XGBoost/LightGBM? → ml.c5.large or ml.c5.xlarge
    ├─ TensorFlow/PyTorch (CPU)? → ml.c5.2xlarge or ml.c6i.2xlarge
    ├─ Large LLM Model (>10GB)? → ml.r5.2xlarge or ml.r5.4xlarge
    └─ Neural Network (GPU friendly)? → ml.g4dn.xlarge or ml.g4dn.2xlarge

    COST OPTIMIZATION STRATEGIES:
    1. Multi-model on single instance (if memory allows)
    2. Reserved Instances (commit 1-3 years → 30-40% savings)
    3. Spot Instances (70% cheaper, but can be interrupted)
    4. Scheduled scaling (downsize off-peak hours)
    5. Right-sizing review (quarterly update)

    MLA-C01 EXAM TIPS:
    ✓ Inference Recommender = Automated instance selection
    ✓ Match instance to model size and latency SLA
    ✓ Monitor CloudWatch metrics for under/over-provisioning
    ✓ Cost/performance trade-off is key optimization
    """)


def main():
    """
    Main orchestration for Inference Recommender demonstration.

    MLA-C01 Task 2.2: Right-size and optimize endpoints
    """
    print("=" * 80)
    print("SageMaker Inference Recommender - Comprehensive Demonstration")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Region: {AWS_REGION}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    try:
        # Step 1: Register model in Model Registry (prerequisite)
        model_arn = register_model_in_registry()

        # Step 2: Create default inference recommender job
        default_job = create_default_inference_recommender_job()

        # Step 3: Create advanced custom job with specific instance types
        advanced_job = create_advanced_inference_recommender_job()

        # Step 4: Retrieve and compare results
        print("\n[Retrieving Default Job Results...]")
        default_results = retrieve_job_results(default_job)

        print("\n[Retrieving Advanced Job Results...]")
        advanced_results = retrieve_job_results(advanced_job)

        # Step 5: Best practices and decision framework
        demonstrate_instance_selection_best_practices()

        print("\n" + "=" * 80)
        print("✓ Inference Recommender Demonstration Complete")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Review Inference Recommender results in SageMaker console")
        print("2. Create endpoint with recommended instance type")
        print("3. Monitor actual latency/throughput vs. recommendations")
        print("4. Re-run quarterly as traffic patterns change")

    except Exception as e:
        print(f"\n✗ Demonstration failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
