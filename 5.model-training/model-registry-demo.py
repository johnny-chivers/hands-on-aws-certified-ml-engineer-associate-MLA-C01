"""
AWS MLA-C01: SageMaker Model Registry - Model Governance and Lineage

This script demonstrates SageMaker Model Registry for production ML workflows:
  1. Create Model Package Group for organizing related models
  2. Register a model package with metadata and approval status
  3. Update model approval status (Pending -> Approved -> Rejected)
  4. List and describe model packages with lineage tracking
  5. Query model registry for model promotion to production

Key MLA-C01 Concepts:
  - Model Governance: Track model versions, owners, and approval workflows
  - Model Lineage: Track training job, data, hyperparameters, metrics
  - Approval Status: Pending Approval -> Approved -> Production
  - Model Package: Versioned model with metadata
  - MLOps: Automated model deployment pipelines based on approval
  - Model Cards: Document model performance, fairness, limitations
"""

import logging
import boto3
import json
from datetime import datetime
from sagemaker import Session, get_execution_role
from sagemaker.model_monitor import ModelExplainabilityMonitor
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model_package import (
    ModelPackageGroup,
    ModelPackage,
    ModelPackageStatusDetails,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing SageMaker Model Registry Demo...")

try:
    # Step 1: Initialize AWS and SageMaker
    sagemaker_session = Session()
    role = get_execution_role()
    region = boto3.Session().region_name
    bucket = sagemaker_session.default_bucket()
    account_id = boto3.client("sts").get_caller_identity()["Account"]

    logger.info(f"AWS Region: {region}")
    logger.info(f"Account ID: {account_id}")
    logger.info(f"S3 Bucket: {bucket}")

    # Initialize clients
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    # Step 2: Create Model Package Group
    # Model Package Group: Logical container for versioned models
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: CREATE MODEL PACKAGE GROUP")
    logger.info("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_package_group_name = f"housing-prediction-models-{timestamp}"

    logger.info(f"\nCreating Model Package Group: {model_package_group_name}")

    try:
        model_package_group_response = sagemaker_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription="Housing price prediction models - MLA-C01 Demo",
            Tags=[
                {"Key": "Environment", "Value": "Production"},
                {"Key": "Team", "Value": "ML-Engineering"},
                {"Key": "Exam", "Value": "MLA-C01"},
            ],
        )

        model_package_group_arn = model_package_group_response["ModelPackageGroupArn"]
        logger.info(f"✓ Model Package Group created")
        logger.info(f"  ARN: {model_package_group_arn}")

    except sagemaker_client.exceptions.ValidationException as e:
        if "already exists" in str(e):
            logger.info(f"Model Package Group already exists: {model_package_group_name}")
            model_package_group_arn = f"arn:aws:sagemaker:{region}:{account_id}:model-package-group/{model_package_group_name}"
        else:
            raise

    # Step 3: Create Model Package (Register Model)
    # Model Package: Versioned model with metadata, metrics, approval status
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: REGISTER MODEL PACKAGE")
    logger.info("=" * 70)

    model_version = "1.0.0"
    model_package_name = f"xgboost-housing-{model_version}-{timestamp}"

    logger.info(f"\nCreating Model Package: {model_package_name}")

    # In production: model_uri comes from training job output
    model_uri = f"s3://{bucket}/models/xgboost-housing/model.tar.gz"

    logger.info(f"Model artifacts location: {model_uri}")

    # Create metadata for model
    model_metrics = {
        "TrainingMetrics": {
            "RMSE": 45000.0,  # Root Mean Square Error
            "MAE": 35000.0,   # Mean Absolute Error
            "R2Score": 0.92,  # Coefficient of determination
        },
        "ValidationMetrics": {
            "RMSE": 47500.0,
            "MAE": 36000.0,
            "R2Score": 0.90,
        },
        "TestMetrics": {
            "RMSE": 48000.0,
            "MAE": 37000.0,
            "R2Score": 0.89,
        },
    }

    # Model quality metrics
    quality_metrics = {
        "metrics": {
            "training": model_metrics["TrainingMetrics"],
            "validation": model_metrics["ValidationMetrics"],
            "test": model_metrics["TestMetrics"],
        }
    }

    # Save metrics to S3 for model registry
    metrics_path = f"s3://{bucket}/models/xgboost-housing/metrics.json"

    # Approval status: PendingManualApproval, Approved, Rejected, PendingApproval
    model_approval_status = "PendingManualApproval"

    logger.info(f"Model metrics:")
    for metric_type, metrics in quality_metrics["metrics"].items():
        logger.info(f"  {metric_type}:")
        for metric_name, value in metrics.items():
            logger.info(f"    {metric_name}: {value}")

    # Inference specification: describes inputs/outputs for inference
    inference_specification = {
        "Containers": [
            {
                "Image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.5-1",
                "ImageDigest": "sha256:abc123def456",  # Container image digest
                "ModelDataUrl": model_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
                },
            }
        ],
        "SupportedTransformInstanceTypes": ["ml.m5.xlarge", "ml.m5.2xlarge"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge", "ml.m5.2xlarge"],
    }

    try:
        model_package_response = sagemaker_client.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageName=model_package_name,
            ModelPackageVersion=int(model_version.split(".")[0]),
            ModelPackageDescription="XGBoost housing price prediction model",
            InferenceSpecification=inference_specification,
            ModelApprovalStatus=model_approval_status,  # Initial status: Pending approval
            Tags=[
                {"Key": "Version", "Value": model_version},
                {"Key": "Algorithm", "Value": "XGBoost"},
                {"Key": "Domain", "Value": "RealEstate"},
            ],
        )

        model_package_arn = model_package_response["ModelPackageArn"]
        logger.info(f"✓ Model Package registered")
        logger.info(f"  ARN: {model_package_arn}")
        logger.info(f"  Approval Status: {model_approval_status}")

    except Exception as e:
        logger.warning(f"Could not create model package: {str(e)[:100]}")
        model_package_arn = f"arn:aws:sagemaker:{region}:{account_id}:model-package/{model_package_group_name}/{model_package_name}"

    # Step 4: List Model Packages in Group
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: LIST MODEL PACKAGES")
    logger.info("=" * 70)

    logger.info(f"\nListing models in group: {model_package_group_name}")

    try:
        list_response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=10,
        )

        model_packages = list_response.get("ModelPackageSummaryList", [])
        logger.info(f"Found {len(model_packages)} model packages")

        for i, package in enumerate(model_packages, 1):
            logger.info(f"\nModel {i}:")
            logger.info(f"  Name: {package['ModelPackageName']}")
            logger.info(f"  Status: {package['ModelPackageStatus']}")
            logger.info(f"  Version: {package.get('ModelPackageVersion', 'N/A')}")
            logger.info(f"  Created: {package['CreationTime']}")

    except Exception as e:
        logger.info(f"Could not list model packages: {str(e)[:100]}")

    # Step 5: Describe Model Package (Get Details & Lineage)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: DESCRIBE MODEL PACKAGE (LINEAGE)")
    logger.info("=" * 70)

    logger.info("\nModel Package Details & Lineage:")
    logger.info(f"  Package Name: {model_package_name}")
    logger.info(f"  Group: {model_package_group_name}")
    logger.info(f"  Model URI: {model_uri}")
    logger.info(f"  Metrics Path: {metrics_path}")
    logger.info(f"  Created: {datetime.now().isoformat()}")

    logger.info("\nLineage Information:")
    logger.info("  Source Data: s3://bucket/data/housing/training/")
    logger.info("  Training Job: xgboost-housing-20240115-120000")
    logger.info("  Training Algorithm: XGBoost")
    logger.info("  Framework: SageMaker Built-in Algorithm")
    logger.info("  Instance Type: ml.m5.xlarge")

    # Step 6: Update Model Approval Status
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: UPDATE MODEL APPROVAL STATUS")
    logger.info("=" * 70)

    logger.info("\nModel Approval Workflow:")
    logger.info("  1. Register model: Status = PendingManualApproval")
    logger.info("  2. Validate performance: Run tests, compare baselines")
    logger.info("  3. Approve for production: Status = Approved")
    logger.info("  4. Deploy to production: Create endpoint from approved model")
    logger.info("  5. Reject if needed: Status = Rejected (for non-conforming models)")

    # Simulate approval workflow
    new_approval_status = "Approved"
    approval_reason = "Model passed validation tests, performance exceeds baseline"

    logger.info(f"\nUpdating approval status to: {new_approval_status}")
    logger.info(f"Reason: {approval_reason}")

    try:
        sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus=new_approval_status,
            ApprovalDescription=approval_reason,
        )

        logger.info(f"✓ Model approval status updated to: {new_approval_status}")

    except Exception as e:
        logger.warning(f"Could not update approval status: {str(e)[:100]}")
        logger.info("(This is expected if model package doesn't exist)")

    # Step 7: Model Registry Best Practices
    logger.info("\n" + "=" * 70)
    logger.info("MODEL REGISTRY BEST PRACTICES (MLA-C01)")
    logger.info("=" * 70)

    best_practices = {
        "1. Versioning": [
            "Use semantic versioning (major.minor.patch)",
            "Track version history in Model Package Group",
            "Document changes between versions",
        ],
        "2. Metadata": [
            "Include training data location and date",
            "Document hyperparameters used",
            "Tag models with domain, purpose, owner",
        ],
        "3. Approval Workflow": [
            "Implement human review for production models",
            "Require validation test passage",
            "Use approval status to gate deployment",
        ],
        "4. Monitoring": [
            "Track model performance metrics over time",
            "Monitor for data drift",
            "Alert on prediction drift",
        ],
        "5. Documentation": [
            "Create model cards documenting performance, fairness, limitations",
            "Include responsible AI practices",
            "Document A/B testing results",
        ],
        "6. Governance": [
            "Define approval authority (data scientist, manager, etc.)",
            "Implement audit trails",
            "Enforce model lineage tracking",
        ],
    }

    for category, practices in best_practices.items():
        logger.info(f"\n{category}")
        for practice in practices:
            logger.info(f"  • {practice}")

    # Step 8: MLOps Integration
    logger.info("\n" + "=" * 70)
    logger.info("MLOPS INTEGRATION WITH MODEL REGISTRY")
    logger.info("=" * 70)

    logger.info("\nAutomated Deployment Pipeline:")
    logger.info("  1. Training Job Completes")
    logger.info("  2. Register Model → PendingManualApproval")
    logger.info("  3. Automated Tests Run (performance, fairness, etc.)")
    logger.info("  4. Approval Status Updated → Approved (if tests pass)")
    logger.info("  5. SNS Notification to ML Engineer")
    logger.info("  6. SageMaker Pipelines Triggered")
    logger.info("  7. Deploy Approved Model to Production Endpoint")
    logger.info("  8. Run A/B Tests (if configured)")
    logger.info("  9. Monitor Production Performance")
    logger.info("  10. Trigger Retraining if Drift Detected")

    logger.info("\nKey SageMaker Components:")
    logger.info("  - Model Registry: Centralized model versioning & governance")
    logger.info("  - Pipelines: Orchestrate training → testing → deployment")
    logger.info("  - Model Monitor: Detect data/prediction drift")
    logger.info("  - Clarify: Explainability & fairness monitoring")
    logger.info("  - Endpoints: Real-time serving with traffic splitting (A/B tests)")

    logger.info("\n" + "=" * 70)
    logger.info("MODEL PROMOTION EXAMPLE")
    logger.info("=" * 70)

    logger.info("\nProduction Promotion Criteria:")
    logger.info("  ✓ RMSE < baseline (previous model RMSE)")
    logger.info("  ✓ Validation F1-score > 0.85")
    logger.info("  ✓ Fairness: DPL < 0.1 (no significant bias)")
    logger.info("  ✓ Explainability: Top 5 features explain 70%+ variance")
    logger.info("  ✓ Latency: p99 latency < 100ms")
    logger.info("  ✓ Manual review passed")

    logger.info("\nIf All Criteria Met:")
    logger.info("  ✓ Set approval_status = 'Approved'")
    logger.info("  ✓ Deploy via SageMaker Pipelines")
    logger.info("  ✓ Create endpoint with traffic splitting (90/10 new/old)")
    logger.info("  ✓ Monitor for 1-2 weeks for performance stability")
    logger.info("  ✓ Promote to 100% traffic if metrics stable")

    logger.info("=" * 70)
    logger.info("Model Registry demo completed successfully!")

except Exception as e:
    logger.error(f"Error in Model Registry demo: {str(e)}", exc_info=True)
    raise
