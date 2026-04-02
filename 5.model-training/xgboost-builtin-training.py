"""
AWS MLA-C01: SageMaker XGBoost Built-in Algorithm Training

This script demonstrates production-grade XGBoost training on SageMaker:
  1. Set up SageMaker session, IAM role, S3 paths
  2. Load housing dataset and perform train/validation split
  3. Upload data to S3 in required XGBoost format (CSV, target first, no header)
  4. Retrieve XGBoost container URI for the region
  5. Create Estimator with hyperparameters and instance configuration
  6. Define train/validation input channels
  7. Execute training job and monitor progress

Key MLA-C01 Concepts:
  - SageMaker Built-in Algorithms: Pre-optimized, managed by AWS
  - Container Images: ECR URIs for algorithm implementations by region
  - Training Jobs: Managed training on specified instances
  - Input/Output Channels: S3 data mounting and model artifact storage
  - Hyperparameters:
    * max_depth: Tree depth (prevents overfitting)
    * eta: Learning rate (controls step size)
    * num_round: Number of boosting rounds
    * subsample: Row subsampling for robustness
    * objective: Loss function (reg:squarederror for regression)
"""

import logging
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.utils import name_from_base
import sagemaker.image_uris

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BUCKET_NAME = "<YOUR-BUCKET-NAME>"
REGION = "<YOUR-REGION>"  # e.g., us-east-1
ROLE_ARN = "<YOUR-ROLE-ARN>"  # SageMaker execution role ARN

logger.info("Initializing XGBoost Training Script...")

try:
    # Step 1: Initialize AWS and SageMaker Configuration
    # Use the course bucket from CloudFormation, not the SageMaker default bucket
    sagemaker_session = Session(
        boto_session=boto3.Session(region_name=REGION),
        default_bucket=BUCKET_NAME,
    )
    role = ROLE_ARN
    region = REGION
    bucket = BUCKET_NAME

    logger.info(f"AWS Region: {region}")
    logger.info(f"SageMaker Role: {role}")
    logger.info(f"S3 Bucket: {bucket}")

    # Step 2: Create sample housing dataset
    # In production: load from S3, CSV, Parquet, or database
    logger.info("\nLoading housing dataset...")

    # Create sample data (housing prices)
    np.random.seed(42)
    n_samples = 1000

    data = {
        "Price": np.random.uniform(100000, 500000, n_samples),  # Target variable
        "SquareFeet": np.random.uniform(800, 5000, n_samples),
        "Bedrooms": np.random.randint(1, 6, n_samples),
        "Bathrooms": np.random.uniform(1, 4, n_samples),
        "YearBuilt": np.random.randint(1950, 2023, n_samples),
        "Garage": np.random.randint(0, 4, n_samples),
    }

    df = pd.DataFrame(data)

    # Add some correlation: price increases with square feet and bedrooms
    df["Price"] = (
        df["Price"] +
        df["SquareFeet"] * 50 +
        df["Bedrooms"] * 25000 +
        df["Bathrooms"] * 30000
    )

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {list(df.columns[1:])}")
    logger.info(f"Target: {df.columns[0]}")

    # Step 3: Train/Validation Split
    # SageMaker XGBoost can use separate validation channel for early stopping
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)

    train_data = df.iloc[:split_index]
    validation_data = df.iloc[split_index:]

    logger.info(f"Training set: {len(train_data)} samples")
    logger.info(f"Validation set: {len(validation_data)} samples")

    # Step 4: Prepare data for XGBoost
    # XGBoost requires: CSV format, target variable FIRST column, NO header row
    # Feature order: [target, feature1, feature2, ...]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_prefix = f"xgboost-housing-{timestamp}"
    train_path = f"s3://{bucket}/{s3_prefix}/train/train.csv"
    validation_path = f"s3://{bucket}/{s3_prefix}/validation/validation.csv"

    logger.info("\nUploading training data to S3...")
    logger.info(f"Train data: {train_path}")
    logger.info(f"Validation data: {validation_path}")

    # Write to CSV without header (XGBoost requirement)
    # Reorder columns to put target first
    train_data_reordered = train_data[["Price"] + [col for col in train_data.columns if col != "Price"]]
    validation_data_reordered = validation_data[["Price"] + [col for col in validation_data.columns if col != "Price"]]

    # Upload to S3
    train_file = "/tmp/train.csv"
    validation_file = "/tmp/validation.csv"

    train_data_reordered.to_csv(train_file, header=False, index=False)
    validation_data_reordered.to_csv(validation_file, header=False, index=False)

    sagemaker_session.upload_data(
        path=train_file,
        bucket=bucket,
        key_prefix=f"{s3_prefix}/train"
    )

    sagemaker_session.upload_data(
        path=validation_file,
        bucket=bucket,
        key_prefix=f"{s3_prefix}/validation"
    )

    logger.info("✓ Data uploaded to S3")

    # Step 5: Get XGBoost Container Image URI
    # Container URIs are region-specific (managed by AWS)
    # Format: <account>.dkr.ecr.<region>.amazonaws.com/sagemaker-xgboost:<version>
    logger.info("\nRetrieving XGBoost container URI...")

    xgboost_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",  # XGBoost version
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    logger.info(f"XGBoost image: {xgboost_image_uri}")

    # Step 6: Create SageMaker Estimator
    # Estimator: Encapsulates training configuration and manages training job lifecycle
    logger.info("\nCreating XGBoost Estimator...")

    job_name = name_from_base("xgboost-housing")
    output_path = f"s3://{bucket}/{s3_prefix}/output"

    estimator = Estimator(
        image_uri=xgboost_image_uri,
        role=role,
        instance_count=1,  # Number of training instances
        instance_type="ml.m5.xlarge",  # Instance type (CPU for XGBoost)
        output_path=output_path,  # S3 path for model artifacts
        sagemaker_session=sagemaker_session,
        base_job_name="xgboost-housing",
    )

    logger.info(f"Job name: {job_name}")
    logger.info(f"Output path: {output_path}")

    # Step 7: Set XGBoost Hyperparameters
    # These parameters control model behavior and training
    logger.info("\nSetting hyperparameters...")

    # Hyperparameter explanations for MLA-C01:
    hyperparameters = {
        # ===== Objective Function =====
        "objective": "reg:squarederror",  # Regression: minimize squared error
        # For classification: "binary:logistic", "multi:softmax"

        # ===== Tree Growth Control =====
        "max_depth": "5",  # Maximum tree depth (prevents overfitting)
        # Higher values: more complex trees, risk of overfitting
        # Lower values: simpler trees, may underfit
        # Typical range: 3-10 for regression

        # ===== Learning Rate Control =====
        "eta": "0.2",  # Learning rate (shrinkage)
        # Controls step size of each boosting round
        # Lower eta: more conservative updates, needs more rounds
        # Higher eta: larger updates, risk of overshooting
        # Typical range: 0.01-0.3

        # ===== Boosting Rounds =====
        "num_round": "100",  # Number of gradient boosting rounds
        # Each round adds a tree to reduce residual error
        # More rounds improve training but risk overfitting
        # Use early stopping to find optimal rounds

        # ===== Regularization =====
        "subsample": "0.8",  # Fraction of rows sampled for each round
        # Reduces variance, prevents overfitting
        # Values < 1.0 enable stochastic boosting
        # Typical range: 0.5-1.0

        "colsample_bytree": "0.8",  # Fraction of features sampled per tree
        # Reduces variance, helps with feature importance
        # Typical range: 0.5-1.0

        "lambda": "1",  # L2 regularization weight
        # Penalizes model complexity
        # Higher values: more regularization (simpler model)

        "alpha": "0",  # L1 regularization weight
        # Encourages sparsity (feature selection)
        # 0 means no L1 regularization

        # ===== Evaluation & Stopping =====
        "eval_metric": "rmse",  # Evaluation metric
        # For regression: rmse, mae, mape
        # For classification: error, auc

        "scale_pos_weight": "1",  # Balance positive class (for imbalanced data)
        # For regression, set to 1

        "min_child_weight": "1",  # Minimum sum of weights per child node
        # Higher values: prevent overfitting by pruning small splits
        # Typical range: 1-10

        # ===== Training Control =====
        "verbosity": "1",  # Logging verbosity (0=quiet, 1=warning, 2=info, 3=debug)
    }

    estimator.set_hyperparameters(**hyperparameters)

    logger.info("Hyperparameters set:")
    for param, value in hyperparameters.items():
        logger.info(f"  {param}: {value}")

    # Step 8: Define Input/Output Channels
    # Channels mount S3 data into training container at specific paths
    logger.info("\nDefining training input channels...")

    train_input = TrainingInput(
        s3_data=train_path,
        content_type="text/csv",
        s3_data_type="S3Prefix",  # S3 prefix (directory) mode
        record_wrapper_type="None",  # No record wrapper for CSV
        compression="None",
    )

    validation_input = TrainingInput(
        s3_data=validation_path,
        content_type="text/csv",
        s3_data_type="S3Prefix",
        record_wrapper_type="None",
        compression="None",
    )

    # Data channels: maps S3 data to training container paths
    # training: /opt/ml/input/data/training/
    # validation: /opt/ml/input/data/validation/
    data_channels = {
        "training": train_input,
        "validation": validation_input,
    }

    logger.info("✓ Input channels configured")

    # Step 9: Start Training Job
    # fit() submits training job to SageMaker and waits for completion
    logger.info("\nStarting training job...")
    logger.info(f"This will take several minutes...")

    start_time = time.time()

    estimator.fit(
        inputs=data_channels,
        job_name=job_name,
        wait=True,  # Wait for training to complete
        logs="All",  # Show training logs
    )

    training_time = time.time() - start_time
    logger.info(f"✓ Training completed in {training_time:.1f} seconds")

    # Step 10: Retrieve Training Job Information
    # After training completes, retrieve job details and model location
    logger.info("\nTraining Summary:")
    logger.info(f"  Job name: {job_name}")
    logger.info(f"  Duration: {training_time:.0f} seconds")

    # Model artifacts stored at output_path/job_name/output/model.tar.gz
    model_path = f"{output_path}/{job_name}/output/model.tar.gz"
    logger.info(f"  Model artifacts: {model_path}")

    # Step 11: Training Insights
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING INSIGHTS & BEST PRACTICES")
    logger.info("=" * 70)

    logger.info("\nHyperparameter Tuning:")
    logger.info("  - Use SageMaker Automatic Model Tuning for optimal hyperparameters")
    logger.info("  - Bayesian optimization: efficient search of hyperparameter space")
    logger.info("  - Warm start: initialize tuning with previous best jobs")
    logger.info("  - Early stopping: stop poor-performing jobs early")

    logger.info("\nTraining Best Practices:")
    logger.info("  - Always use validation set for early stopping")
    logger.info("  - Monitor train/validation loss divergence (overfitting indicator)")
    logger.info("  - Use stratified splits for classification")
    logger.info("  - Normalize features if using distance-based algorithms")

    logger.info("\nModel Deployment:")
    logger.info("  - Use estimator.deploy() to create real-time endpoint")
    logger.info("  - Use Batch Transform for offline predictions")
    logger.info("  - Use Async Inference for large-scale predictions")
    logger.info("  - Monitor model performance with SageMaker Model Monitor")

    logger.info("\nModel Registry:")
    logger.info("  - Register trained model in SageMaker Model Registry")
    logger.info("  - Track model versions and approval status")
    logger.info("  - Automate model deployment pipelines with SageMaker Pipelines")

    logger.info("=" * 70)
    logger.info("XGBoost training completed successfully!")

except Exception as e:
    logger.error(f"Error in XGBoost training: {str(e)}", exc_info=True)
    raise
