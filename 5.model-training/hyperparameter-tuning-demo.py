"""
AWS MLA-C01: SageMaker Automatic Model Tuning (Hyperparameter Optimization)

This script demonstrates SageMaker's Automatic Model Tuning capabilities:
  1. Create base XGBoost estimator with container image and hyperparameters
  2. Define hyperparameter ranges (continuous, integer, categorical)
  3. Configure HyperparameterTuner with Bayesian optimization strategy
  4. Launch parallel tuning jobs to search hyperparameter space
  5. Retrieve best training job and optimal hyperparameters
  6. Analyze tuning results and training job metrics

Key MLA-C01 Concepts:
  - Hyperparameter Optimization: Systematic search for best model parameters
  - Bayesian Optimization: Smart search using past trial results
  - Random Search: Baseline approach, samples randomly
  - Grid Search: Exhaustive search of parameter combinations
  - Early Stopping: Terminate unpromising jobs to save time/cost
  - Warm Start: Initialize tuning with previous results
  - Objective Metric: Metric to optimize (e.g., validation:rmse)
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
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
)
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

logger.info("Initializing SageMaker Hyperparameter Tuning Script...")

try:
    # Step 1: Initialize AWS and SageMaker Configuration
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

    # Step 2: Create sample dataset and prepare data
    logger.info("\nPreparing training data...")

    np.random.seed(42)
    n_samples = 1000

    data = {
        "Price": np.random.uniform(100000, 500000, n_samples),
        "SquareFeet": np.random.uniform(800, 5000, n_samples),
        "Bedrooms": np.random.randint(1, 6, n_samples),
        "Bathrooms": np.random.uniform(1, 4, n_samples),
        "YearBuilt": np.random.randint(1950, 2023, n_samples),
        "Garage": np.random.randint(0, 4, n_samples),
    }

    df = pd.DataFrame(data)
    df["Price"] = (
        df["Price"] +
        df["SquareFeet"] * 50 +
        df["Bedrooms"] * 25000 +
        df["Bathrooms"] * 30000
    )

    # Train/validation split
    split_index = int(len(df) * 0.8)
    train_data = df.iloc[:split_index]
    validation_data = df.iloc[split_index:]

    # Prepare CSV files (target first, no header)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_prefix = f"xgboost-tuning-{timestamp}"
    train_path = f"s3://{bucket}/{s3_prefix}/train"
    validation_path = f"s3://{bucket}/{s3_prefix}/validation"

    train_data_reordered = train_data[["Price"] + [col for col in train_data.columns if col != "Price"]]
    validation_data_reordered = validation_data[["Price"] + [col for col in validation_data.columns if col != "Price"]]

    train_file = "/tmp/train.csv"
    validation_file = "/tmp/validation.csv"

    train_data_reordered.to_csv(train_file, header=False, index=False)
    validation_data_reordered.to_csv(validation_file, header=False, index=False)

    sagemaker_session.upload_data(path=train_file, bucket=bucket, key_prefix=f"{s3_prefix}/train")
    sagemaker_session.upload_data(path=validation_file, bucket=bucket, key_prefix=f"{s3_prefix}/validation")

    logger.info("✓ Training data prepared and uploaded to S3")

    # Step 3: Get XGBoost Container Image
    logger.info("\nRetrieving XGBoost container URI...")

    xgboost_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    logger.info(f"✓ XGBoost image: {xgboost_image_uri}")

    # Step 4: Create Base Estimator
    # Base estimator defines the algorithm and training infrastructure
    logger.info("\nCreating base XGBoost Estimator...")

    output_path = f"s3://{bucket}/{s3_prefix}/output"

    estimator = Estimator(
        image_uri=xgboost_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=output_path,
        sagemaker_session=sagemaker_session,
        base_job_name="xgboost-tuning",
    )

    # Set static hyperparameters (not tuned)
    static_hyperparameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": "1",
    }

    estimator.set_hyperparameters(**static_hyperparameters)

    logger.info("✓ Base estimator created with static hyperparameters")

    # Step 5: Define Hyperparameter Ranges
    # Tuner will search these ranges to find optimal values
    logger.info("\nDefining hyperparameter search space...")

    hyperparameter_ranges = {
        # ===== Tree Growth Control =====
        "max_depth": IntegerParameter(
            min_value=3,
            max_value=10,
            scaling_type="Linear",  # Linear scale: uniform sampling
            # Alternative: Logarithmic, ReverseLogarithmic
        ),
        # Deeper trees: more complex, risk overfitting
        # Shallower trees: simpler, may underfit

        # ===== Learning Rate =====
        "eta": ContinuousParameter(
            min_value=0.01,
            max_value=0.5,
            scaling_type="Logarithmic",  # Log scale: explores wider range efficiently
        ),
        # Controls step size of boosting updates
        # Lower values: slower learning, needs more rounds
        # Higher values: faster learning, risk overshooting

        # ===== Boosting Rounds =====
        "num_round": IntegerParameter(
            min_value=50,
            max_value=300,
            scaling_type="Linear",
        ),
        # More rounds: better training performance, slower, risk overfitting
        # Fewer rounds: faster, may underfit

        # ===== Row Subsampling =====
        "subsample": ContinuousParameter(
            min_value=0.5,
            max_value=1.0,
            scaling_type="Linear",
        ),
        # Fraction of rows sampled per round
        # Lower values: reduce variance, faster computation
        # Higher values: more data per round, better generalization

        # ===== Feature Subsampling =====
        "colsample_bytree": ContinuousParameter(
            min_value=0.5,
            max_value=1.0,
            scaling_type="Linear",
        ),
        # Fraction of features sampled per tree
        # Lower values: prevent overfitting, force diverse features
        # Higher values: use all features

        # ===== Tree-Specific Parameters =====
        "min_child_weight": IntegerParameter(
            min_value=1,
            max_value=10,
            scaling_type="Linear",
        ),
        # Minimum weight sum in child nodes
        # Higher values: prevent overfitting, may underfit
        # Lower values: allow small splits, risk overfitting

        # ===== Regularization =====
        "lambda": ContinuousParameter(
            min_value=0.0,
            max_value=10.0,
            scaling_type="Linear",
        ),
        # L2 regularization strength
        # Higher values: simpler model, better generalization
        # Lower values: more complex model, may overfit

        "alpha": ContinuousParameter(
            min_value=0.0,
            max_value=5.0,
            scaling_type="Linear",
        ),
        # L1 regularization strength
        # Encourages feature selection (sparse models)
    }

    logger.info("Hyperparameter ranges defined:")
    for param_name, param_range in hyperparameter_ranges.items():
        logger.info(f"  {param_name}: {param_range}")

    # Step 6: Create HyperparameterTuner
    # Tuner orchestrates multiple training jobs with different hyperparameters
    logger.info("\nConfiguring HyperparameterTuner...")

    tuner = HyperparameterTuner(
        # Base estimator to tune
        estimator=estimator,

        # ===== Objective Configuration =====
        # Metric to optimize: minimize or maximize
        objective_metric_name="validation:rmse",
        # SageMaker tracks metrics logged during training
        # XGBoost logs: validation:rmse, training:rmse, etc.

        hyperparameter_ranges=hyperparameter_ranges,

        # ===== Tuning Strategy =====
        # strategy: How to search hyperparameter space
        # Options: "Bayesian", "Random", "Grid"
        strategy="Bayesian",
        # Bayesian: Intelligent search using past results (recommended)
        # Random: Randomly sample from ranges (baseline)
        # Grid: Exhaustive grid search (expensive)

        # Maximum number of training jobs
        max_jobs=20,
        # Total jobs to run across all combinations
        # Higher = more thorough search, more expensive

        # Maximum parallel jobs
        max_parallel_jobs=4,
        # Concurrent training jobs
        # Higher = faster tuning, higher resource cost
        # AWS limits: typically 1-10 for most accounts

        # ===== Early Stopping =====
        # Stop jobs that won't reach best performance
        objective_type="Minimize",  # "Maximize" or "Minimize"
        # Minimize: best metric has lowest value
        # Maximize: best metric has highest value

        # ===== Tags & Configuration =====
        metric_definitions=[
            {
                "Name": "validation:rmse",
                "Regex": "validation.*rmse: ([0-9\\.]+)"
                # Regex to extract metric from training logs
            }
        ],
    )

    logger.info("✓ HyperparameterTuner configured")
    logger.info(f"  Strategy: Bayesian optimization")
    logger.info(f"  Max jobs: 20")
    logger.info(f"  Max parallel: 4")
    logger.info(f"  Objective: Minimize validation:rmse")

    # Step 7: Define Training Input Channels
    train_input = TrainingInput(
        s3_data=train_path,
        content_type="text/csv",
        s3_data_type="S3Prefix",
        record_wrapper_type="None",
        compression="None",
    )

    validation_input = TrainingInput(
        s3_data=validation_path,
        content_type="text/csv",
        s3_data_type="S3Prefix",
        record_wrapper_type="None",
        compression="None",
    )

    data_channels = {
        "train": train_input,
        "validation": validation_input,
    }

    # Step 8: Launch Hyperparameter Tuning Job
    logger.info("\nStarting hyperparameter tuning job...")
    logger.info("This will launch multiple training jobs in parallel.")
    logger.info("Total time: 30-60 minutes depending on instance type and dataset size")

    tuning_job_name = name_from_base("xgboost-tuning")
    start_time = time.time()

    tuner.fit(
        inputs=data_channels,
        job_name=tuning_job_name,
        wait=False,  # Don't wait (tuning takes time)
        logs="All",
    )

    logger.info(f"✓ Tuning job submitted: {tuning_job_name}")
    logger.info("Monitor progress in SageMaker console or with:")
    logger.info(f"  aws sagemaker describe-hyper-parameter-tuning-job --name {tuning_job_name}")

    # Step 9: Retrieve Tuning Results (after job completes)
    # In production: use async polling or SNS notifications
    logger.info("\n" + "=" * 70)
    logger.info("WAITING FOR TUNING TO COMPLETE...")
    logger.info("=" * 70)

    # For demo: show how to retrieve results
    logger.info("\nTo retrieve best job after tuning completes:")
    logger.info(f"  best_job_name = tuner.best_training_job()")
    logger.info(f"  best_hyperparameters = tuner.hyperparameter_tuning_job_result")

    # Step 10: Bayesian Optimization Explanation
    logger.info("\n" + "=" * 70)
    logger.info("BAYESIAN OPTIMIZATION STRATEGY EXPLANATION")
    logger.info("=" * 70)

    logger.info("\nHow Bayesian Optimization Works:")
    logger.info("1. Start: Run initial random training jobs (exploration)")
    logger.info("2. Build Model: Create surrogate model of hyperparameter-performance relationship")
    logger.info("3. Predict: Use model to predict performance of untested hyperparameter combinations")
    logger.info("4. Select: Choose next job that maximizes expected improvement")
    logger.info("5. Evaluate: Run job, update surrogate model with results")
    logger.info("6. Repeat: Continue until max_jobs or convergence")

    logger.info("\nAdvantages vs. Random Search:")
    logger.info("  - 3-5x faster convergence to optimal hyperparameters")
    logger.info("  - Smart exploration: balances exploration vs. exploitation")
    logger.info("  - Efficient: requires fewer total jobs")
    logger.info("  - Cost-effective: saves compute resources")

    logger.info("\nWarm Start (if repeating tuning):")
    logger.info("  - Initialize with previous best jobs")
    logger.info("  - Avoids re-exploring bad hyperparameter regions")
    logger.info("  - Faster convergence on subsequent runs")

    logger.info("\n" + "=" * 70)
    logger.info("HYPERPARAMETER TUNING BEST PRACTICES")
    logger.info("=" * 70)

    logger.info("\n1. Grid Search vs. Random vs. Bayesian:")
    logger.info("   - Grid Search: Small, discrete parameter spaces (slow)")
    logger.info("   - Random: Baseline approach, embarrassingly parallel")
    logger.info("   - Bayesian: Recommended, most efficient (default)")

    logger.info("\n2. Resource Management:")
    logger.info("   - max_parallel_jobs: Balance speed vs. cost")
    logger.info("   - instance_type: Use cheaper instances for tuning")
    logger.info("   - Early stopping: Terminate unpromising jobs")

    logger.info("\n3. Parameter Selection:")
    logger.info("   - Include parameters with highest sensitivity")
    logger.info("   - Use domain knowledge to constrain ranges")
    logger.info("   - Consider computational cost in parameter selection")

    logger.info("\n4. Metric Selection:")
    logger.info("   - Validation metric: Use hold-out validation set")
    logger.info("   - Single metric: Easier optimization (avoid multi-objective)")
    logger.info("   - Track multiple metrics: For post-hoc analysis")

    logger.info("\n5. Analysis & Deployment:")
    logger.info("   - Analyze feature importance and parameter sensitivity")
    logger.info("   - Register best model in Model Registry")
    logger.info("   - Compare against baseline models")
    logger.info("   - Document optimal hyperparameters for reproducibility")

    logger.info("=" * 70)

    logger.info("\nHyperparameter tuning job initiated successfully!")
    logger.info(f"Tuning job name: {tuning_job_name}")

except Exception as e:
    logger.error(f"Error in hyperparameter tuning: {str(e)}", exc_info=True)
    raise
