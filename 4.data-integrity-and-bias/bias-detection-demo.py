"""
AWS MLA-C01: SageMaker Clarify Bias Detection Demo

This script demonstrates pre-training bias detection using Amazon SageMaker Clarify:
  1. Sets up SageMaker session and S3 paths
  2. Creates DataConfig pointing to churn dataset
  3. Defines BiasConfig with protected attributes (gender)
  4. Runs pre-training bias analysis using SageMakerClarifyProcessor
  5. Interprets key bias metrics: CI, DPL, KL Divergence

Key MLA-C01 Concepts:
  - Pre-training Bias: Data bias before model training (sampling bias, representation)
  - Post-training Bias: Predictions bias after model deployment (model fairness)
  - Protected Attributes: Gender, race, age - attributes we measure bias against
  - Bias Metrics:
    * Class Imbalance (CI): Ratio of label distribution between groups
    * Difference in Proportions of Labels (DPL): Difference in positive class rate
    * KL Divergence: Kullback-Leibler divergence between label distributions
  - Fair-AI: AWS framework ensuring equitable ML models
"""

import logging
import pandas as pd
import boto3
from datetime import datetime
from sagemaker import Session
from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    DataConfig,
    BiasConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# AWS Configuration
REGION = "<YOUR-REGION>"  # e.g., us-east-1
ROLE_ARN = "<YOUR-ROLE-ARN>"  # SageMaker execution role with S3, Clarify permissions
BUCKET_NAME = "<YOUR-BUCKET-NAME>"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

logger.info("Initializing SageMaker Clarify Bias Detection Demo...")

try:
    # Step 1: Initialize SageMaker Session
    # Session manages AWS credentials, region, and default configurations
    session = Session(
        boto_session=boto3.Session(region_name=REGION),
        default_bucket=BUCKET_NAME
    )

    s3_client = boto3.client("s3", region_name=REGION)

    logger.info(f"SageMaker Session initialized for region: {REGION}")
    logger.info(f"Default bucket: {BUCKET_NAME}")

    # Step 2: Define S3 paths for data and outputs
    # Clarify requires pre-processed CSV data in S3
    s3_data_path = f"s3://{BUCKET_NAME}/data/churn-dataset"
    s3_output_path = f"s3://{BUCKET_NAME}/clarify-bias-output/{TIMESTAMP}"
    dataset_file = "churn_data.csv"
    s3_dataset_path = f"{s3_data_path}/{dataset_file}"

    logger.info(f"Data path: {s3_dataset_path}")
    logger.info(f"Output path: {s3_output_path}")

    # Step 3: Create DataConfig
    # DataConfig specifies where the data is, its format, and headers
    # Clarify expects CSV format with header row
    data_config = DataConfig(
        s3_data_input_path=s3_dataset_path,
        s3_output_path=s3_output_path,
        # Specify which column is the label/target variable
        label="Churn",  # Binary: "Yes" or "No"
        # For multi-class, set label_values_or_threshold to list of positive classes
        headers=["CustomerID", "Age", "Gender", "Tenure", "MonthlyCharges", "Churn"],
        # Data format: CSV, JSON, Parquet
        dataset_type="text/csv",
    )

    logger.info("DataConfig created successfully")

    # Step 4: Create BiasConfig
    # BiasConfig defines which attributes we measure bias for and how
    # Facet: protected attribute (e.g., gender, race, age)
    # Label: target variable we're testing fairness against
    bias_config = BiasConfig(
        # facet_name: The column name of the protected attribute
        facet_name="Gender",
        # facet_values_or_threshold: Values that define the reference group
        # For binary classification: specify the reference group value
        # Clarify will compare other groups to this reference
        facet_values_or_threshold=["Female"],
        # label_name: Target column name
        label_name="Churn",
        # label_values_or_threshold: Positive class values for the label
        # Clarify measures bias in prediction of positive class
        label_values_or_threshold=["Yes"],
    )

    logger.info("BiasConfig created with:")
    logger.info(f"  - Protected attribute (facet): Gender")
    logger.info(f"  - Reference group: Female")
    logger.info(f"  - Target label: Churn")
    logger.info(f"  - Positive class: Yes")

    # Step 5: Initialize SageMakerClarifyProcessor
    # Processor: Orchestrates Clarify analysis jobs
    # Runs on SageMaker processing instances (on-demand compute)
    clarify_processor = SageMakerClarifyProcessor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.xlarge",  # Instance type for analysis job
        sagemaker_session=session,
    )

    logger.info("SageMakerClarifyProcessor initialized")

    # Step 6: Run pre-training bias analysis
    # Pre-training bias: Detects bias in data before model training
    # Analyzes: label distribution differences between groups
    logger.info("Running pre-training bias analysis...")

    clarify_processor.run_pre_training_bias(
        data_config=data_config,
        bias_config=bias_config,
        # Use sample for faster analysis on large datasets
        # For production: comment out for full dataset analysis
        data_analysis_start_log_line=0,
        data_analysis_end_log_line=10000,
        job_description="Pre-training bias analysis for churn prediction",
        # Optional: Specify CloudWatch logs configuration
        logs=True,
    )

    logger.info("Pre-training bias analysis completed")

    # Step 7: Parse and interpret results
    # Clarify generates JSON report with bias metrics
    logger.info("=" * 70)
    logger.info("BIAS ANALYSIS RESULTS")
    logger.info("=" * 70)

    # Bias metrics explanation:
    bias_metrics_info = {
        "Class Imbalance (CI)": {
            "description": "Ratio of positive class rate between groups",
            "formula": "n_positive_reference / n_positive_compared",
            "interpretation": "CI > 1: Reference group has higher churn rate (selection bias)",
            "threshold": "Should be close to 1.0 for fairness",
        },
        "Difference in Proportions of Labels (DPL)": {
            "description": "Absolute difference in positive class rate between groups",
            "formula": "P(Label=Yes | Gender=Female) - P(Label=Yes | Gender=Male)",
            "interpretation": "DPL > 0.1: Significant disparity (unfair)",
            "threshold": "Should be close to 0 for fairness",
        },
        "KL Divergence": {
            "description": "Information-theoretic measure of distribution difference",
            "formula": "Sum of P_ref(x) * log(P_ref(x) / P_compared(x))",
            "interpretation": "KL > 0.1: High divergence (unfair)",
            "threshold": "Should be close to 0 for fairness",
        },
        "Jensen-Shannon Divergence (JS)": {
            "description": "Symmetric version of KL divergence",
            "formula": "Average of KL divergence in both directions",
            "interpretation": "JS > 0.1: Label distributions differ significantly",
            "threshold": "Should be close to 0 for fairness",
        },
        "Kolmogorov-Smirnov (KS)": {
            "description": "Maximum difference in cumulative label distributions",
            "formula": "max | CDF_ref(x) - CDF_compared(x) |",
            "interpretation": "KS > 0.1: Distributions differ significantly",
            "threshold": "Should be close to 0 for fairness",
        },
    }

    for metric_name, metric_info in bias_metrics_info.items():
        logger.info(f"\n{metric_name}:")
        logger.info(f"  Description: {metric_info['description']}")
        logger.info(f"  Formula: {metric_info['formula']}")
        logger.info(f"  Interpretation: {metric_info['interpretation']}")
        logger.info(f"  Fair Threshold: {metric_info['threshold']}")

    logger.info("\n" + "=" * 70)
    logger.info("BIAS DETECTION RECOMMENDATIONS")
    logger.info("=" * 70)
    logger.info("1. If metrics indicate bias:")
    logger.info("   - Collect more balanced data for underrepresented groups")
    logger.info("   - Use techniques: oversampling, undersampling, SMOTE")
    logger.info("   - Consider stratified sampling for train/test splits")
    logger.info("\n2. During training:")
    logger.info("   - Monitor fairness metrics throughout training")
    logger.info("   - Use cost-sensitive learning to penalize disparate outcomes")
    logger.info("   - Consider fairness-aware feature selection")
    logger.info("\n3. Post-deployment:")
    logger.info("   - Use SageMaker Model Monitor for bias drift detection")
    logger.info("   - Continuously measure disparate impact")
    logger.info("   - Implement model card documentation for transparency")
    logger.info("=" * 70)

    # Step 8: Access detailed report
    # The analysis creates a detailed JSON report in S3
    logger.info(f"\nDetailed bias report available at: {s3_output_path}")
    logger.info("Key files:")
    logger.info("  - analysis.json: Detailed metrics and statistics")
    logger.info("  - explanations_data_bias.json: Feature-level bias attribution")

    # Try to download and display summary if available
    try:
        analysis_file = "analysis.json"
        local_report = f"/tmp/{analysis_file}"
        s3_client.download_file(
            BUCKET_NAME,
            f"clarify-bias-output/{TIMESTAMP}/{analysis_file}",
            local_report
        )

        # Parse and display key metrics
        import json
        with open(local_report, "r") as f:
            report = json.load(f)

        logger.info("\nKey Metrics from Report:")
        if "pre_training_bias_metrics" in report:
            metrics = report["pre_training_bias_metrics"]
            for facet_value, metric_data in metrics.items():
                logger.info(f"\nGroup: {facet_value}")
                for metric_name, value in metric_data.items():
                    logger.info(f"  {metric_name}: {value:.4f}")

    except Exception as e:
        logger.info(f"Could not download detailed report: {e}")
        logger.info("Run the job to generate full report with metrics")

    logger.info("\nBias detection analysis completed successfully!")

except Exception as e:
    logger.error(f"Error in bias detection: {str(e)}", exc_info=True)
    raise
