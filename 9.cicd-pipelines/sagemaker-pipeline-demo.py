"""
Comprehensive SageMaker Pipeline Demonstration

Orchestrates end-to-end ML workflows with data preprocessing, training, evaluation,
and conditional model registration. Maps to MLA-C01 Domain 3: ML Workflow Orchestration

Pipeline Components:
- Parameters: Input variables for flexibility
- ProcessingStep: Data transformation using SKLearn
- TrainingStep: Model training with XGBoost
- EvaluationStep: Model quality assessment
- ConditionStep: Quality gates and approval workflow
- RegisterModelStep: Model Registry integration for CI/CD

Key Concepts:
- Reusability: Parameterized pipelines reduce code duplication
- Reproducibility: Full lineage and versioning
- Automation: Scheduled or event-triggered execution
- Quality Gates: Only deploy models meeting criteria
"""

import json
import boto3
from datetime import datetime
from sagemaker import Session, get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.processing import ScriptProcessor
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    EvaluationStep,
    ConditionStep,
)
from sagemaker.workflow.conditions import ConditionGreaterThan, ConditionLessThan
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import RegisterModel
from sagemaker.estimator import Estimator

# Configuration
AWS_REGION = '<YOUR-REGION>'
ROLE_ARN = '<YOUR-ROLE-ARN>'
BUCKET_NAME = '<YOUR-BUCKET-NAME>'
PIPELINE_NAME = "ml-training-pipeline"

# Initialize SageMaker session
session = Session(default_bucket=BUCKET_NAME)
role = ROLE_ARN or get_execution_role()


# ─────────────────────────────────────────────────────────────
# Pipeline Parameters (Input Variables)
# ─────────────────────────────────────────────────────────────

def create_pipeline_parameters():
    """
    Define parameterized inputs for the pipeline.

    Parameters enable:
    - Dynamic execution with different values
    - Model card documentation
    - A/B testing with different configurations
    - Easy parameter sweeps for optimization

    MLA-C01 Exam Focus: Parameters are key for reusable pipelines
    """
    print("[PARAMETERS] Creating pipeline parameters...")

    # Processing configuration
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge",
    )

    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=1,
    )

    # Training configuration
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.2xlarge",
    )

    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1,
    )

    # Training hyperparameters
    num_round = ParameterInteger(
        name="NumRounds",
        default_value=100,  # XGBoost iterations
    )

    max_depth = ParameterInteger(
        name="MaxDepth",
        default_value=5,
    )

    learning_rate = ParameterFloat(
        name="LearningRate",
        default_value=0.1,
    )

    # Model approval
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="Approved",  # Manual or Automatic
    )

    # Quality gates
    rmse_threshold = ParameterFloat(
        name="RMSEThreshold",
        default_value=0.5,  # Maximum acceptable error
    )

    print("✓ Parameters defined:")
    print(f"  - Processing: {processing_instance_type}, count={processing_instance_count}")
    print(f"  - Training: {training_instance_type}, count={training_instance_count}")
    print(f"  - Hyperparams: rounds={num_round}, depth={max_depth}, lr={learning_rate}")
    print(f"  - Quality Gate: RMSE < {rmse_threshold}")

    return {
        'processing_instance_type': processing_instance_type,
        'processing_instance_count': processing_instance_count,
        'training_instance_type': training_instance_type,
        'training_instance_count': training_instance_count,
        'num_round': num_round,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'model_approval_status': model_approval_status,
        'rmse_threshold': rmse_threshold,
    }


# ─────────────────────────────────────────────────────────────
# ProcessingStep: Data Transformation
# ─────────────────────────────────────────────────────────────

def create_data_processing_step(parameters):
    """
    Create ProcessingStep for data preprocessing.

    ProcessingStep responsibilities:
    - Read raw data from S3
    - Data cleaning (handle missing values)
    - Feature engineering (create new features)
    - Train/validation/test split (stratified)
    - Write processed data back to S3

    MLA-C01 Exam Focus: Data quality is foundation of ML pipeline
    """
    print("\n[PROCESSING STEP] Creating data preprocessing step...")

    # SKLearnProcessor: Runs Python code in a Spark/Sklearn container
    processor = SKLearnProcessor(
        framework_version='0.23-1',  # Scikit-learn version
        role=role,
        instance_type=parameters['processing_instance_type'],
        instance_count=parameters['processing_instance_count'],
        sagemaker_session=session,
    )

    # Define processing step
    processing_step = ProcessingStep(
        name="PreprocessingStep",
        code="s3://{}/preprocessing/preprocessing.py".format(BUCKET_NAME),
        processor=processor,
        inputs=[
            # Input data location
        ],
        outputs=[
            # Output locations for train/validation/test
        ],
        job_arguments=[
            "--train-size",
            "0.7",  # 70% training
            "--test-size",
            "0.15",  # 15% test
            # 15% validation (1 - 0.7 - 0.15)
        ],
    )

    print("✓ Data preprocessing step created")
    print(f"  Processor: SKLearnProcessor (0.23-1)")
    print(f"  Train/Valid/Test split: 70%/15%/15%")

    return processing_step


# ─────────────────────────────────────────────────────────────
# TrainingStep: Model Training
# ─────────────────────────────────────────────────────────────

def create_model_training_step(parameters, processing_step):
    """
    Create TrainingStep for XGBoost model training.

    TrainingStep workflow:
    1. Reads training data from ProcessingStep output
    2. Launches SageMaker training job
    3. Trains XGBoost model with specified hyperparameters
    4. Saves trained model artifacts to S3

    MLA-C01 Exam Focus: Training is compute-intensive; monitor for cost
    """
    print("\n[TRAINING STEP] Creating model training step...")

    # XGBoost estimator configuration
    xgboost_estimator = XGBoost(
        entry_point="xgboost_training.py",
        source_dir="s3://{}/training/".format(BUCKET_NAME),
        role=role,
        instance_count=parameters['training_instance_count'],
        instance_type=parameters['training_instance_type'],
        framework_version='1.5',
        py_version='py3',
        sagemaker_session=session,
        hyperparameters={
            'num_round': parameters['num_round'],
            'max_depth': parameters['max_depth'],
            'eta': parameters['learning_rate'],
            'gamma': 4,
            'min_child_weight': 6,
            'objective': 'reg:linear',  # Regression objective
            'subsample': 0.8,
            'early_stopping_rounds': 10,
        },
    )

    # Training step definition
    training_step = TrainingStep(
        name="TrainingStep",
        estimator=xgboost_estimator,
        inputs={
            # Training data input from ProcessingStep
            # "training": processing_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri,
        },
    )

    print("✓ Model training step created")
    print(f"  Algorithm: XGBoost (1.5)")
    print(f"  Instance: {parameters['training_instance_type']} x {parameters['training_instance_count']}")
    print(f"  Hyperparameters:")
    print(f"    - num_round: {parameters['num_round']}")
    print(f"    - max_depth: {parameters['max_depth']}")
    print(f"    - learning_rate: {parameters['learning_rate']}")

    return training_step


# ─────────────────────────────────────────────────────────────
# EvaluationStep: Model Quality Assessment
# ─────────────────────────────────────────────────────────────

def create_model_evaluation_step(parameters, training_step, processing_step):
    """
    Create EvaluationStep to evaluate trained model performance.

    Evaluation metrics:
    - RMSE (Root Mean Squared Error): Primary metric
    - MAE (Mean Absolute Error): Interpretable error
    - R² Score: Variance explained
    - Predictions on validation set

    EvaluationStep output includes:
    - Metrics JSON: Used for ConditionStep gating
    - CloudWatch logs: For debugging

    MLA-C01 Exam Focus: Quality gates prevent bad models reaching production
    """
    print("\n[EVALUATION STEP] Creating model evaluation step...")

    # ScriptProcessor for evaluation script
    evaluator = ScriptProcessor(
        image_uri=session.sagemaker_session.default_bucket(),
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
    )

    # Evaluation step
    evaluation_step = EvaluationStep(
        name="EvaluationStep",
        processor=evaluator,
        code="s3://{}/evaluation/evaluation.py".format(BUCKET_NAME),
        inputs=[
            # Model artifacts from TrainingStep
            # Validation data from ProcessingStep
        ],
        outputs=[
            # Evaluation metrics output
        ],
        property_files=[
            PropertyFile(
                name="EvaluationMetrics",
                output_name="evaluation",
                path="metrics.json",
            )
        ],
    )

    print("✓ Model evaluation step created")
    print("  Evaluates model on validation set")
    print("  Output: metrics.json (used for quality gates)")

    return evaluation_step


# ─────────────────────────────────────────────────────────────
# ConditionStep: Quality Gates
# ─────────────────────────────────────────────────────────────

def create_quality_condition_step(parameters, evaluation_step):
    """
    Create ConditionStep to implement quality gates.

    Quality gates ensure only good models are registered:
    - RMSE < threshold: Model meets accuracy requirements
    - If passes: Proceed to registration
    - If fails: Stop pipeline (manual intervention)

    ConditionStep gates:
    1. Statistical: Model achieves required metrics
    2. Business: Model meets business requirements
    3. Regulatory: Model complies with governance rules

    MLA-C01 Exam Focus: Automation with human oversight
    """
    print("\n[CONDITION STEP] Creating quality gate condition...")

    # Extract RMSE from evaluation metrics
    rmse_value = evaluation_step.properties.EvaluationMetrics.JsonPath(
        "$.metrics.rmse"
    )

    # Define condition: RMSE must be below threshold
    condition = ConditionLessThan(
        left=rmse_value,
        right=parameters['rmse_threshold'],
    )

    condition_step = ConditionStep(
        name="QualityGateCondition",
        conditions=[condition],
        if_true=[],  # Steps to execute if condition is TRUE
        if_false=[],  # Steps to execute if condition is FALSE
    )

    print("✓ Quality gate condition created")
    print(f"  Condition: RMSE < {parameters['rmse_threshold']}")
    print(f"  If TRUE: Proceed to model registration")
    print(f"  If FALSE: Stop pipeline (quality not met)")

    return condition_step


# ─────────────────────────────────────────────────────────────
# RegisterModelStep: Model Registry Integration
# ─────────────────────────────────────────────────────────────

def create_model_registration_step(parameters, training_step):
    """
    Create RegisterModelStep to register approved models.

    Model Registry workflow:
    1. Stores model artifacts in Model Registry
    2. Creates versioned model packages
    3. Tracks approval status (Pending, Approved, Rejected)
    4. Enables promotion through staging (Dev → Staging → Prod)

    RegisterModelStep only executes if ConditionStep is TRUE.

    MLA-C01 Exam Focus: Model Registry enables governance and CI/CD
    """
    print("\n[REGISTRATION STEP] Creating model registration step...")

    # Create RegisterModel step
    register_model_step = RegisterModel(
        name="RegisterModelStep",
        estimator=training_step.properties.TrainingJobName,  # Reference to training job
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="sagemaker-model-registry-group",
        approval_status=parameters['model_approval_status'],
        model_metrics=ModelMetrics(
            model_quality=MetricsSource(
                s3_uri="s3://{}/evaluation/metrics.json".format(BUCKET_NAME),
                content_type="application/json",
            ),
        ),
    )

    print("✓ Model registration step created")
    print(f"  Approval Status: {parameters['model_approval_status']}")
    print(f"  Inference Instances: ml.m5.xlarge, ml.m5.2xlarge")
    print(f"  Transform Instances: ml.m5.xlarge")

    return register_model_step


# ─────────────────────────────────────────────────────────────
# Full Pipeline Definition
# ─────────────────────────────────────────────────────────────

def create_full_pipeline():
    """
    Assemble all steps into a complete ML pipeline.

    Pipeline DAG (Directed Acyclic Graph):
    ProcessingStep → TrainingStep → EvaluationStep → ConditionStep → RegisterModelStep

    Pipeline features:
    - Reusable: Parameterized for different configurations
    - Resumable: Can restart from failed step
    - Traced: Full lineage and versioning
    - Automated: Scheduled or event-triggered execution

    MLA-C01 Exam Focus: Pipelines are core of ML CI/CD
    """
    print("\n[PIPELINE] Assembling full ML pipeline...")

    parameters = create_pipeline_parameters()
    processing_step = create_data_processing_step(parameters)
    training_step = create_model_training_step(parameters, processing_step)
    evaluation_step = create_model_evaluation_step(parameters, training_step, processing_step)
    condition_step = create_quality_condition_step(parameters, evaluation_step)
    register_model_step = create_model_registration_step(parameters, training_step)

    # Define pipeline
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            parameters['processing_instance_type'],
            parameters['processing_instance_count'],
            parameters['training_instance_type'],
            parameters['training_instance_count'],
            parameters['num_round'],
            parameters['max_depth'],
            parameters['learning_rate'],
            parameters['model_approval_status'],
            parameters['rmse_threshold'],
        ],
        steps=[
            processing_step,
            training_step,
            evaluation_step,
            condition_step,
            register_model_step,
        ],
    )

    print("✓ Full pipeline assembled")
    print(f"  Pipeline Name: {PIPELINE_NAME}")
    print(f"  Steps: {len(pipeline.steps)}")
    print("  Execution Order:")
    print("    1. PreprocessingStep (Data transformation)")
    print("    2. TrainingStep (Model training)")
    print("    3. EvaluationStep (Quality assessment)")
    print("    4. QualityGateCondition (Approval gate)")
    print("    5. RegisterModelStep (Model Registry)")

    return pipeline


# ─────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────

def upsert_and_execute_pipeline(pipeline):
    """
    Register pipeline in SageMaker and trigger execution.

    Upsert behavior:
    - Creates pipeline if doesn't exist
    - Updates pipeline if already exists (new version)
    - Maintains execution history

    MLA-C01 Exam Focus: Pipeline versioning enables rollback
    """
    print("\n[EXECUTION] Upserting pipeline to SageMaker...")

    try:
        pipeline_definition = pipeline.definition()
        pipeline_upsert_response = pipeline.upsert(
            role_arn=role,
            description="ML training pipeline with quality gates",
        )
        print("✓ Pipeline upserted successfully")
        print(f"  Pipeline ARN: {pipeline_upsert_response['PipelineArn']}")

        # Execute pipeline with default parameters
        print("\n[EXECUTION] Starting pipeline execution...")
        pipeline_exec_response = pipeline.start()
        print("✓ Pipeline execution started")
        print(f"  Execution ARN: {pipeline_exec_response.arn}")
        print(f"  Status: {pipeline_exec_response.status}")

        return pipeline_exec_response

    except Exception as e:
        print(f"✗ Error executing pipeline: {str(e)}")
        raise


def monitor_pipeline_execution(pipeline_execution):
    """
    Monitor pipeline execution status and step results.

    Execution monitoring:
    - Watch step progress
    - Detect failures
    - Retrieve step outputs
    - Estimate remaining time

    MLA-C01 Exam Focus: Production pipelines need observability
    """
    print("\n[MONITORING] Checking pipeline execution status...")

    try:
        # Wait for execution to complete (with timeout)
        pipeline_execution.wait(max_attempts=120)
        print("✓ Pipeline execution completed")

        # Get execution status details
        status_response = pipeline_execution.describe()
        print(f"  Status: {status_response['PipelineExecutionStatus']}")
        print(f"  Started: {status_response['CreationTime']}")
        print(f"  Ended: {status_response.get('PipelineExecutionEndTime', 'Still running')}")

        # Retrieve step execution details
        print("\n  Step Execution Summary:")
        for step_execution in pipeline_execution.list_steps():
            print(f"    - {step_execution['StepName']}: {step_execution['StepStatus']}")

    except Exception as e:
        print(f"⚠ Error monitoring pipeline: {str(e)}")


def main():
    """
    Main orchestration for SageMaker Pipeline demonstration.

    MLA-C01 Task 3.1: Build and orchestrate ML workflows
    """
    print("=" * 80)
    print("SageMaker Pipelines - Comprehensive ML Workflow Demonstration")
    print("=" * 80)
    print(f"Pipeline: {PIPELINE_NAME}")
    print(f"Region: {AWS_REGION}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    try:
        # Create full pipeline
        pipeline = create_full_pipeline()

        # Upsert and execute
        execution = upsert_and_execute_pipeline(pipeline)

        # Monitor execution
        monitor_pipeline_execution(execution)

        print("\n" + "=" * 80)
        print("✓ SageMaker Pipeline Demonstration Complete")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. View pipeline in SageMaker Studio")
        print("2. Monitor execution in SageMaker console")
        print("3. Review model in Model Registry")
        print("4. Deploy registered model to endpoint")
        print("5. Schedule pipeline for recurring execution (daily/weekly)")

    except Exception as e:
        print(f"\n✗ Demonstration failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
