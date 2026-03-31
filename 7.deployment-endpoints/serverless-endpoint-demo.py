"""
Serverless SageMaker Endpoint Demonstration

MLA-C01 Exam Relevance:
- Task 3.1: Select deployment infrastructure based on requirements
- Serverless endpoints scale to zero — ideal for unpredictable/intermittent traffic
- Cold-start latency trade-off (exam loves this scenario question)
- Cost-effective for dev/test, low-traffic production, and PoC workloads

This demo shows how to:
1. Create a SageMaker Model
2. Configure a serverless endpoint (MemorySizeInMB, MaxConcurrency)
3. Deploy and invoke the endpoint
4. Show the cold-start vs warm invocation difference
5. Clean up resources
"""

import boto3
import json
import time
import logging
from sagemaker import Session

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AWS_REGION = "<YOUR-AWS-REGION>"
ROLE_ARN = "<YOUR-SAGEMAKER-ROLE-ARN>"
MODEL_DATA_S3_URI = "<YOUR-MODEL-S3-URI>"             # s3://bucket/path/model.tar.gz
CONTAINER_IMAGE_URI = "<YOUR-CONTAINER-IMAGE-URI>"    # e.g. XGBoost built-in image URI
ENDPOINT_NAME = "mla-c01-serverless-endpoint"
ENDPOINT_CONFIG_NAME = "mla-c01-serverless-config"
MODEL_NAME = "mla-c01-serverless-model"

# ---------------------------------------------------------------------------
# AWS Clients
# ---------------------------------------------------------------------------
sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


# ===================================================================
# STEP 1 — Create a SageMaker Model
# ===================================================================
def create_model():
    """Register model artifacts with a container image."""
    logger.info("Creating SageMaker Model: %s", MODEL_NAME)
    try:
        sagemaker_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                "Image": CONTAINER_IMAGE_URI,
                "ModelDataUrl": MODEL_DATA_S3_URI,
            },
            ExecutionRoleArn=ROLE_ARN,
        )
        logger.info("Model created successfully.")
    except sagemaker_client.exceptions.ClientError as e:
        if "Cannot create already existing model" in str(e):
            logger.warning("Model already exists — skipping.")
        else:
            raise


# ===================================================================
# STEP 2 — Create a Serverless Endpoint Configuration
# ===================================================================
def create_serverless_endpoint_config():
    """
    Configure the serverless endpoint.

    MLA-C01 Exam Tip — key parameters:
    ───────────────────────────────────
    MemorySizeInMB  : 1024 | 2048 | 3072 | 4096 | 5120 | 6144
        Larger memory also gets proportionally more vCPU.
        Choose the smallest that meets your model's requirements.

    MaxConcurrency  : 1 – 200
        Max number of concurrent invocations before throttling.
        Keep low for dev/test, higher for production traffic.

    ProvisionedConcurrency (optional):
        Pre-warm a number of instances to eliminate cold starts.
        Exam may ask about this for latency-sensitive serverless use cases.
    """
    logger.info("Creating Serverless Endpoint Config: %s", ENDPOINT_CONFIG_NAME)

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                "VariantName": "serverless-variant",
                "ModelName": MODEL_NAME,
                "ServerlessConfig": {
                    "MemorySizeInMB": 2048,
                    "MaxConcurrency": 5,
                    # "ProvisionedConcurrency": 1,  # Uncomment to eliminate cold starts
                },
            }
        ],
    )
    logger.info("Serverless endpoint config created.")


# ===================================================================
# STEP 3 — Deploy the Serverless Endpoint
# ===================================================================
def deploy_endpoint():
    """Deploy the serverless endpoint (usually faster than real-time)."""
    logger.info("Deploying serverless endpoint: %s", ENDPOINT_NAME)

    sagemaker_client.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )

    logger.info("Waiting for endpoint to become InService …")
    waiter = sagemaker_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={"Delay": 15, "MaxAttempts": 40})
    logger.info("Serverless endpoint %s is InService.", ENDPOINT_NAME)


# ===================================================================
# STEP 4 — Invoke and measure cold-start vs warm latency
# ===================================================================
def invoke_and_measure(payload_csv: str, label: str = ""):
    """
    Invoke the serverless endpoint and log the round-trip time.

    MLA-C01 Exam Tip:
    - First invocation after idle → cold start (seconds).
    - Subsequent invocations while warm → much lower latency.
    - The exam often tests whether you know the cold-start trade-off
      and when serverless is NOT appropriate (latency-critical apps).
    """
    start = time.time()
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="application/json",
        Body=payload_csv.encode("utf-8"),
    )
    elapsed_ms = (time.time() - start) * 1000
    result = json.loads(response["Body"].read().decode("utf-8"))

    logger.info("[%s] Prediction: %s  |  Latency: %.0f ms", label, result, elapsed_ms)
    return result, elapsed_ms


def demonstrate_cold_vs_warm():
    """Show the cold-start penalty followed by warm invocations."""
    sample = "8.3252,41.0,6.984127,1.023810,322.0,2.555556,-122.23,37.88"

    logger.info("\n--- Cold-start invocation (first call after idle) ---")
    _, cold_ms = invoke_and_measure(sample, label="COLD")

    logger.info("\n--- Warm invocations (back-to-back) ---")
    warm_times = []
    for i in range(3):
        _, warm_ms = invoke_and_measure(sample, label=f"WARM-{i+1}")
        warm_times.append(warm_ms)

    avg_warm = sum(warm_times) / len(warm_times)
    logger.info(
        "\nCold-start: %.0f ms  |  Avg warm: %.0f ms  |  Difference: %.1fx",
        cold_ms,
        avg_warm,
        cold_ms / avg_warm if avg_warm > 0 else 0,
    )


# ===================================================================
# STEP 5 — Describe the Endpoint
# ===================================================================
def describe_endpoint():
    """Show serverless-specific configuration details."""
    desc = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    logger.info("Endpoint Status: %s", desc["EndpointStatus"])

    for pv in desc.get("ProductionVariants", []):
        sc = pv.get("CurrentServerlessConfig", {})
        logger.info(
            "  Variant: %s  |  Memory: %s MB  |  MaxConcurrency: %s",
            pv["VariantName"],
            sc.get("MemorySizeInMB", "N/A"),
            sc.get("MaxConcurrency", "N/A"),
        )


# ===================================================================
# STEP 6 — Clean Up
# ===================================================================
def cleanup():
    """Delete serverless endpoint resources."""
    logger.info("Cleaning up serverless endpoint resources …")
    for resource, delete_fn, kwargs in [
        ("Endpoint", sagemaker_client.delete_endpoint, {"EndpointName": ENDPOINT_NAME}),
        ("Config", sagemaker_client.delete_endpoint_config, {"EndpointConfigName": ENDPOINT_CONFIG_NAME}),
        ("Model", sagemaker_client.delete_model, {"ModelName": MODEL_NAME}),
    ]:
        try:
            delete_fn(**kwargs)
            logger.info("  %s deleted.", resource)
        except Exception as e:
            logger.warning("  Could not delete %s: %s", resource, e)

    logger.info("Cleanup complete.")


# ===================================================================
# Main
# ===================================================================
def main():
    logger.info("=== MLA-C01: Serverless Endpoint Demo ===\n")

    create_model()
    create_serverless_endpoint_config()
    deploy_endpoint()
    describe_endpoint()
    demonstrate_cold_vs_warm()

    # Uncomment when done recording
    # cleanup()

    logger.info("\n=== Serverless Endpoint Demo Complete ===")


if __name__ == "__main__":
    main()
