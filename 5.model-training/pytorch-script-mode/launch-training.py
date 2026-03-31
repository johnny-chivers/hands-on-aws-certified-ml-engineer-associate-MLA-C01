"""
AWS MLA-C01: PyTorch Script Mode Training Job Launcher

This script launches a PyTorch training job on SageMaker:
  1. Initialize SageMaker session and execution role
  2. Prepare training data and upload to S3
  3. Create PyTorch Estimator with training script
  4. Define hyperparameters and training configuration
  5. Submit training job to SageMaker
  6. Monitor training job progress

Key MLA-C01 Concepts:
  - Script Mode: Run custom training code with minimal modifications
  - Estimator: Encapsulates training configuration and lifecycle
  - Hyperparameters: Passed to training script via command-line args
  - Entry Point: Python script containing training logic
  - Source Directory: Additional code/libraries for training
  - Training Metrics: CloudWatch metrics logged during training
"""

import logging
import boto3
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sagemaker import Session, get_execution_role
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.utils import name_from_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing PyTorch Training Job Launcher...")

try:
    # Step 1: Initialize SageMaker Configuration
    # Session: Manages AWS credentials, region, S3 bucket
    # Role: IAM role for SageMaker to access S3, ECR, CloudWatch
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: INITIALIZE SAGEMAKER")
    logger.info("=" * 70)

    sagemaker_session = Session()
    role = get_execution_role()
    region = boto3.Session().region_name
    bucket = sagemaker_session.default_bucket()

    logger.info(f"AWS Region: {region}")
    logger.info(f"SageMaker Role: {role}")
    logger.info(f"S3 Bucket: {bucket}")

    # Step 2: Prepare Training Data
    # Create sample churn dataset for training
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: PREPARE TRAINING DATA")
    logger.info("=" * 70)

    logger.info("Creating sample churn dataset...")

    # Generate synthetic churn data
    np.random.seed(42)
    n_samples = 500

    data = {
        "Churn": np.random.randint(0, 2, n_samples),  # Target: 0 or 1
        "Age": np.random.randint(18, 80, n_samples),
        "Tenure": np.random.randint(0, 60, n_samples),
        "MonthlyCharges": np.random.uniform(20, 120, n_samples),
        "TotalCharges": np.random.uniform(100, 5000, n_samples),
        "NumServices": np.random.randint(1, 8, n_samples),
    }

    df = pd.DataFrame(data)
    logger.info(f"Dataset created: {df.shape[0]} samples × {df.shape[1]} features")
    logger.info(f"Features: {list(df.columns[1:])}")
    logger.info(f"Target: {df.columns[0]}")

    # Train/validation split
    split_index = int(0.8 * len(df))
    train_data = df.iloc[:split_index]
    val_data = df.iloc[split_index:]

    logger.info(f"Train: {len(train_data)} samples")
    logger.info(f"Validation: {len(val_data)} samples")

    # Upload data to S3
    # SageMaker expects CSV format: target first, no header
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_prefix = f"pytorch-churn-{timestamp}"

    logger.info(f"\nUploading data to S3...")

    # Save CSVs locally
    train_file = "/tmp/train.csv"
    val_file = "/tmp/validation.csv"

    train_data.to_csv(train_file, header=False, index=False)
    val_data.to_csv(val_file, header=False, index=False)

    # Upload to S3
    train_path = f"s3://{bucket}/{s3_prefix}/train"
    val_path = f"s3://{bucket}/{s3_prefix}/validation"

    sagemaker_session.upload_data(
        path=train_file,
        bucket=bucket,
        key_prefix=f"{s3_prefix}/train"
    )

    sagemaker_session.upload_data(
        path=val_file,
        bucket=bucket,
        key_prefix=f"{s3_prefix}/validation"
    )

    logger.info(f"✓ Training data uploaded to {train_path}")
    logger.info(f"✓ Validation data uploaded to {val_path}")

    # Step 3: Create PyTorch Estimator
    # Estimator: Configuration for SageMaker training job
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: CREATE PYTORCH ESTIMATOR")
    logger.info("=" * 70)

    # Paths to training script and supporting code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = current_dir  # Directory containing train.py
    entry_point = "train.py"  # Training script

    logger.info(f"Entry point: {entry_point}")
    logger.info(f"Source directory: {source_dir}")

    # Job name: unique identifier for this training job
    job_name = name_from_base("pytorch-churn")

    # Output path: where SageMaker saves model artifacts
    output_path = f"s3://{bucket}/{s3_prefix}/output"

    logger.info(f"Job name: {job_name}")
    logger.info(f"Output path: {output_path}")

    # Create PyTorch Estimator
    # PyTorch: Wrapper for PyTorch training framework
    # Handles container image selection, distributed training, etc.
    pytorch_estimator = PyTorch(
        # Training script configuration
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,

        # Framework and version
        framework_version="2.0",  # PyTorch version
        py_version="py310",  # Python version
        instance_type="ml.m5.xlarge",  # Instance type (CPU or GPU)
        instance_count=1,  # Number of instances (1 for single-machine training)

        # Job configuration
        output_path=output_path,
        base_job_name="pytorch-churn",
        sagemaker_session=sagemaker_session,

        # Hyperparameter optimization configuration
        use_spot_instances=False,  # Set True to use cheaper spot instances
        max_wait=3600,  # Max wait time for spot capacity (seconds)

        # Code location and container
        code_location=output_path,

        # Environment variables (optional)
        environment={
            "PYTHONUNBUFFERED": "True",
        },

        # Metric definitions: CloudWatch metrics to track
        metric_definitions=[
            {
                "Name": "train_loss",
                "Regex": "Train Loss: ([0-9\\.]+)"
            },
            {
                "Name": "val_loss",
                "Regex": "Val Loss: ([0-9\\.]+)"
            },
            {
                "Name": "val_accuracy",
                "Regex": "Val Acc: ([0-9\\.]+)"
            },
        ],
    )

    logger.info("✓ PyTorch Estimator created")

    # Step 4: Set Hyperparameters
    # Hyperparameters: Passed to training script via command-line arguments
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: CONFIGURE HYPERPARAMETERS")
    logger.info("=" * 70)

    hyperparameters = {
        # Training hyperparameters
        "epochs": 10,  # Number of training epochs
        "batch-size": 32,  # Batch size for training
        "learning-rate": 0.001,  # Adam learning rate
        "hidden-dim": 64,  # Hidden layer dimension
        "dropout-rate": 0.3,  # Dropout probability
    }

    pytorch_estimator.set_hyperparameters(**hyperparameters)

    logger.info("Hyperparameters set:")
    for param, value in hyperparameters.items():
        logger.info(f"  {param}: {value}")

    # Step 5: Configure Input Data Channels
    # Channels: Named inputs mounted into training container
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: CONFIGURE INPUT CHANNELS")
    logger.info("=" * 70)

    # Training channel: mounted at /opt/ml/input/data/training/
    training_input = TrainingInput(
        s3_data=train_path,
        content_type="text/csv",
        s3_data_type="S3Prefix",
        record_wrapper_type="None",
        compression="None",
    )

    # Validation channel: mounted at /opt/ml/input/data/validation/
    validation_input = TrainingInput(
        s3_data=val_path,
        content_type="text/csv",
        s3_data_type="S3Prefix",
        record_wrapper_type="None",
        compression="None",
    )

    # Input data mapping
    inputs = {
        "training": training_input,
        "validation": validation_input,
    }

    logger.info("Input channels configured:")
    logger.info(f"  training: {train_path}")
    logger.info(f"  validation: {val_path}")

    # Step 6: Submit Training Job
    # fit() submits the job and returns when complete (or timeout)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: SUBMIT AND MONITOR TRAINING JOB")
    logger.info("=" * 70)

    logger.info("Submitting training job to SageMaker...")
    logger.info("This will take 5-15 minutes depending on data size and instance type")

    pytorch_estimator.fit(
        inputs=inputs,
        job_name=job_name,
        wait=True,  # Wait for job completion
        logs="All",  # Show training logs
    )

    logger.info(f"✓ Training job completed: {job_name}")

    # Step 7: Retrieve Training Job Information
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: TRAINING JOB DETAILS")
    logger.info("=" * 70)

    sagemaker_client = boto3.client("sagemaker", region_name=region)

    job_details = sagemaker_client.describe_training_job(TrainingJobName=job_name)

    logger.info(f"Training Job: {job_details['TrainingJobName']}")
    logger.info(f"Status: {job_details['TrainingJobStatus']}")
    logger.info(f"Created: {job_details['CreationTime']}")
    logger.info(f"Completed: {job_details.get('TrainingEndTime', 'In Progress')}")
    logger.info(f"Duration: {job_details.get('TrainingTimeInSeconds', 'N/A')} seconds")

    # Model artifacts location
    model_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
    logger.info(f"Model artifacts: {model_uri}")

    # Step 8: Deploy Model (Optional)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: DEPLOY MODEL (OPTIONAL)")
    logger.info("=" * 70)

    logger.info("\nTo deploy the trained model to a real-time endpoint:")
    logger.info(f"\n  predictor = pytorch_estimator.deploy(")
    logger.info(f"      initial_instance_count=1,")
    logger.info(f"      instance_type='ml.m5.xlarge',")
    logger.info(f"  )")

    logger.info("\nThen make predictions:")
    logger.info(f"  import json")
    logger.info(f"  test_data = {{'features': [25, 12, 85.0, 1020.0, 4]}}")
    logger.info(f"  prediction = predictor.predict(json.dumps(test_data))")

    # Step 9: Best Practices and Next Steps
    logger.info("\n" + "=" * 70)
    logger.info("PYTORCH SCRIPT MODE BEST PRACTICES")
    logger.info("=" * 70)

    logger.info("\n1. Data Preparation:")
    logger.info("   - Ensure data is in correct format (CSV with target first)")
    logger.info("   - Normalize/scale features for neural networks")
    logger.info("   - Use train/validation/test splits")

    logger.info("\n2. Model Development:")
    logger.info("   - Implement model_fn, input_fn, predict_fn for hosting")
    logger.info("   - Track hyperparameters for reproducibility")
    logger.info("   - Log metrics for monitoring")

    logger.info("\n3. Hyperparameter Tuning:")
    logger.info("   - Use SageMaker Automatic Model Tuning")
    logger.info("   - Bayesian optimization for efficient search")
    logger.info("   - Early stopping to reduce cost")

    logger.info("\n4. Distributed Training:")
    logger.info("   - For multi-GPU: increase instance_count")
    logger.info("   - Use PyTorch DistributedDataParallel")
    logger.info("   - Data parallelism for large datasets")

    logger.info("\n5. Model Deployment:")
    logger.info("   - Deploy to real-time endpoints for serving")
    logger.info("   - Use batch transform for offline inference")
    logger.info("   - Monitor model performance in production")

    logger.info("\n6. Monitoring:")
    logger.info("   - Use CloudWatch metrics for training progress")
    logger.info("   - Enable model monitoring for data/prediction drift")
    logger.info("   - Track model performance over time")

    logger.info("=" * 70)
    logger.info("PyTorch training job launcher completed successfully!")

except Exception as e:
    logger.error(f"Error in PyTorch training launcher: {str(e)}", exc_info=True)
    raise
