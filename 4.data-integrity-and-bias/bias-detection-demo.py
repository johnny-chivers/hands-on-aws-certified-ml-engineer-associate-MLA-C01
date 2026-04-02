"""
AWS MLA-C01: SageMaker Clarify Bias Detection Demo

This script demonstrates pre-training bias detection using Amazon SageMaker Clarify:
  1. Generates synthetic telco churn data with intentional gender bias
  2. Uploads data to S3 for Clarify processing
  3. Creates DataConfig and BiasConfig for bias analysis
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
import numpy as np
import pandas as pd
import boto3
import os
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

    # Step 2: Generate synthetic telco churn data
    # We intentionally introduce gender bias so Clarify has something to detect
    # In real-world ML: you'd use your actual dataset - this is for demonstration
    logger.info("Generating synthetic telco churn dataset with intentional bias...")

    np.random.seed(42)
    n_samples = 1000

    # Generate customer attributes
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(18.0, 118.0, size=n_samples), 2)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples, p=[0.55, 0.25, 0.20]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples, p=[0.35, 0.45, 0.20]
    )

    # Generate churn with INTENTIONAL GENDER BIAS
    # Male customers churn at ~35%, Female customers churn at ~20%
    # This disparity is what Clarify should detect
    churn = []
    for i in range(n_samples):
        if gender[i] == "Male":
            churn_prob = 0.35  # Higher churn rate for males
        else:
            churn_prob = 0.20  # Lower churn rate for females

        # Adjust by contract type (month-to-month churns more)
        if contract[i] == "Month-to-month":
            churn_prob += 0.10
        elif contract[i] == "Two year":
            churn_prob -= 0.10

        churn.append("Yes" if np.random.random() < churn_prob else "No")

    # Build DataFrame
    df = pd.DataFrame({
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "Churn": churn,
    })

    logger.info(f"Generated {len(df)} samples")
    logger.info(f"Churn distribution:\n{df['Churn'].value_counts()}")
    logger.info(f"Churn by gender:\n{df.groupby('gender')['Churn'].value_counts()}")

    # Step 3: Upload data to S3
    # Clarify requires data in S3 - save as CSV with headers
    s3_prefix = "data/churn-dataset"
    dataset_file = "churn_data.csv"
    local_path = f"/tmp/{dataset_file}"

    df.to_csv(local_path, index=False)

    s3_client.upload_file(local_path, BUCKET_NAME, f"{s3_prefix}/{dataset_file}")
    s3_dataset_path = f"s3://{BUCKET_NAME}/{s3_prefix}/{dataset_file}"
    s3_output_path = f"s3://{BUCKET_NAME}/clarify-bias-output/{TIMESTAMP}"

    logger.info(f"Data uploaded to: {s3_dataset_path}")
    logger.info(f"Output path: {s3_output_path}")

    # Step 4: Create DataConfig
    # DataConfig specifies where the data is, its format, and headers
    # Clarify expects CSV format with header row
    data_config = DataConfig(
        s3_data_input_path=s3_dataset_path,
        s3_output_path=s3_output_path,
        # Specify which column is the label/target variable
        label="Churn",  # Binary: "Yes" or "No"
        # Headers must match the CSV columns exactly
        headers=["gender", "SeniorCitizen", "tenure", "MonthlyCharges",
                 "Contract", "InternetService", "Churn"],
        # Data format: CSV, JSON, Parquet
        dataset_type="text/csv",
    )

    logger.info("DataConfig created successfully")

    # Step 5: Create BiasConfig
    # BiasConfig defines which attributes we measure bias for and how
    # Facet: protected attribute (e.g., gender, race, age)
    # Label: target variable we're testing fairness against
    bias_config = BiasConfig(
        # label_values_or_threshold: Positive class values for the label
        # Clarify measures bias in prediction of positive class
        label_values_or_threshold=["Yes"],
        # facet_name: The column name of the protected attribute
        facet_name="gender",
        # facet_values_or_threshold: Values that define the reference group
        # For binary classification: specify the reference group value
        # Clarify will compare other groups to this reference
        facet_values_or_threshold=["Female"],
    )

    logger.info("BiasConfig created with:")
    logger.info(f"  - Protected attribute (facet): gender")
    logger.info(f"  - Reference group: Female")
    logger.info(f"  - Target label: Churn")
    logger.info(f"  - Positive class: Yes")

    # Step 6: Initialize SageMakerClarifyProcessor
    # Processor: Orchestrates Clarify analysis jobs
    # Runs on SageMaker processing instances (on-demand compute)
    clarify_processor = SageMakerClarifyProcessor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.xlarge",  # Instance type for analysis job
        sagemaker_session=session,
    )

    logger.info("SageMakerClarifyProcessor initialized")

    # Step 7: Run pre-training bias analysis
    # Pre-training bias: Detects bias in data before model training
    # Analyzes: label distribution differences between groups
    logger.info("Running pre-training bias analysis...")

    clarify_processor.run_pre_training_bias(
        data_config=data_config,
        data_bias_config=bias_config,
        # methods="all" computes all pre-training bias metrics:
        # CI, DPL, KL, JS, LP, TVD, KS, CDDL, CDDPL
        methods="all",
    )

    logger.info("Pre-training bias analysis completed")

    # Step 8: Parse and interpret results
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

    # Step 9: Access detailed report
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
