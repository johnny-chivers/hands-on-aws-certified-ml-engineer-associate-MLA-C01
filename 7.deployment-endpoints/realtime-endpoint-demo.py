"""
Real-time SageMaker Endpoint Demonstration

MLA-C01 Exam Relevance:
- Task 3.1: Select deployment infrastructure based on existing architecture and requirements
- Task 3.2: Deploy and operationalize ML solutions
- Real-time endpoints for synchronous, low-latency predictions (< 1 second)
- Use cases: API services, fraud detection, recommendation engines

This demo shows how to:
1. Create a SageMaker Model from trained artifacts
2. Configure endpoint settings (instance type, variant weights)
3. Deploy a real-time endpoint
4. Invoke the endpoint with sample data
5. Update the endpoint with a new model version (production variant)
6. Clean up resources
"""

import boto3
import json
import time
import logging
from sagemaker import Session
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AWS_REGION = "<YOUR-AWS-REGION>"                      # e.g. "eu-west-1"
ROLE_ARN = "<YOUR-SAGEMAKER-ROLE-ARN>"                # SageMaker execution role
MODEL_DATA_S3_URI = "<YOUR-MODEL-S3-URI>"             # s3://bucket/path/model.tar.gz
CONTAINER_IMAGE_URI = "<YOUR-CONTAINER-IMAGE-URI>"    # e.g. XGBoost built-in image
ENDPOINT_NAME = "mla-c01-realtime-endpoint"
ENDPOINT_CONFIG_NAME = "mla-c01-realtime-config"
MODEL_NAME = "mla-c01-xgboost-model"

# ---------------------------------------------------------------------------
# AWS Clients
# ---------------------------------------------------------------------------
sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
sagemaker_session = Session(boto3_session=boto3.Session(region_name=AWS_REGION))


# ===================================================================
# STEP 1 — Create a SageMaker Model
# ===================================================================
def create_model():
    """Register model artifacts + container as a SageMaker Model."""
    logger.info("Creating SageMaker Model: %s", MODEL_NAME)

    try:
        sagemaker_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                "Image": CONTAINER_IMAGE_URI,
                "ModelDataUrl": MODEL_DATA_S3_URI,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": MODEL_DATA_S3_URI,
                },
            },
            ExecutionRoleArn=ROLE_ARN,
        )
        logger.info("Model created successfully.")
    except sagemaker_client.exceptions.ClientError as e:
        if "Cannot create already existing model" in str(e):
            logger.warning("Model %s already exists — skipping creation.", MODEL_NAME)
        else:
            raise


# ===================================================================
# STEP 2 — Create an Endpoint Configuration
# ===================================================================
def create_endpoint_config():
    """
    Define how the endpoint should be provisioned.

    MLA-C01 Exam Tip:
    - ProductionVariants let you do A/B testing by splitting traffic.
    - InitialVariantWeight controls the traffic split (1.0 = 100%).
    - InstanceType choice is a common exam question:
        ml.m5.xlarge  — general purpose / balanced
        ml.c5.xlarge  — compute-optimised (CPU inference)
        ml.g4dn.xlarge — GPU inference
    """
    logger.info("Creating Endpoint Config: %s", ENDPOINT_CONFIG_NAME)

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                "VariantName": "primary-variant",
                "ModelName": MODEL_NAME,
                "InstanceType": "ml.m5.xlarge",
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1.0,
            }
        ],
        # Optional — enable data capture for Model Monitor
        # DataCaptureConfig={
        #     "EnableCapture": True,
        #     "InitialSamplingPercentage": 100,
        #     "DestinationS3Uri": "s3://your-bucket/data-capture",
        #     "CaptureOptions": [
        #         {"CaptureMode": "Input"},
        #         {"CaptureMode": "Output"},
        #     ],
        # },
    )
    logger.info("Endpoint config created.")


# ===================================================================
# STEP 3 — Deploy the Endpoint
# ===================================================================
def deploy_endpoint():
    """
    Launch the real-time endpoint.  This provisions the instance(s)
    behind the scenes — typically takes 5-8 minutes.
    """
    logger.info("Deploying endpoint: %s", ENDPOINT_NAME)

    sagemaker_client.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )

    logger.info("Waiting for endpoint to become InService …")
    waiter = sagemaker_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )
    logger.info("Endpoint %s is now InService.", ENDPOINT_NAME)


# ===================================================================
# STEP 4 — Invoke the Endpoint (send a prediction request)
# ===================================================================
def invoke_endpoint(payload_csv: str):
    """
    Send a CSV-formatted row to the real-time endpoint and get a prediction.

    MLA-C01 Exam Tip:
    - ContentType must match what the model container expects.
    - For XGBoost built-in: 'text/csv' (no header row).
    - For custom containers: whatever your /invocations route accepts.
    """
    logger.info("Invoking endpoint with payload: %s", payload_csv[:80])

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="application/json",
        Body=payload_csv.encode("utf-8"),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    logger.info("Prediction result: %s", result)
    return result


# ===================================================================
# STEP 5 — Describe the Endpoint (status, variants, data capture)
# ===================================================================
def describe_endpoint():
    """Retrieve the current endpoint details — useful for debugging."""
    desc = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    logger.info("Endpoint Status : %s", desc["EndpointStatus"])
    logger.info("Creation Time   : %s", desc["CreationTime"])

    for pv in desc.get("ProductionVariants", []):
        logger.info(
            "  Variant %-20s | instances: %s | weight: %s",
            pv["VariantName"],
            pv.get("CurrentInstanceCount", "N/A"),
            pv.get("CurrentWeight", "N/A"),
        )
    return desc


# ===================================================================
# STEP 6 — Update the Endpoint (e.g. swap to a new model version)
# ===================================================================
def update_endpoint(new_config_name: str):
    """
    Blue/green-style update: create a new EndpointConfig pointing to a
    new model, then call update_endpoint.  SageMaker provisions the new
    fleet before draining traffic from the old one (zero-downtime).

    MLA-C01 Exam Tip:
    - RetainAllVariantProperties keeps auto-scaling settings.
    - You can also use update_endpoint_weights_and_capacities for
      traffic shifting without a full config swap.
    """
    logger.info("Updating endpoint to config: %s", new_config_name)

    sagemaker_client.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=new_config_name,
        RetainAllVariantProperties=True,
    )

    waiter = sagemaker_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    logger.info("Endpoint updated and InService.")


# ===================================================================
# STEP 7 — Clean Up
# ===================================================================
def cleanup():
    """Delete the endpoint, config, and model to avoid ongoing charges."""
    logger.info("Cleaning up resources …")
    try:
        sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        logger.info("Endpoint deleted.")
    except Exception as e:
        logger.warning("Could not delete endpoint: %s", e)

    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
        logger.info("Endpoint config deleted.")
    except Exception as e:
        logger.warning("Could not delete endpoint config: %s", e)

    try:
        sagemaker_client.delete_model(ModelName=MODEL_NAME)
        logger.info("Model deleted.")
    except Exception as e:
        logger.warning("Could not delete model: %s", e)

    logger.info("Cleanup complete.")


# ===================================================================
# Main — run all steps
# ===================================================================
def main():
    logger.info("=== MLA-C01: Real-time Endpoint Demo ===\n")

    # 1. Create model
    create_model()

    # 2. Create endpoint config
    create_endpoint_config()

    # 3. Deploy
    deploy_endpoint()

    # 4. Describe
    describe_endpoint()

    # 5. Invoke with sample data  (8 features — matches housing dataset)
    sample_row = "8.3252,41.0,6.984127,1.023810,322.0,2.555556,-122.23,37.88"
    invoke_endpoint(sample_row)

    # 6. Cleanup (uncomment when you're done recording)
    # cleanup()

    logger.info("\n=== Real-time Endpoint Demo Complete ===")


if __name__ == "__main__":
    main()
