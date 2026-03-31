"""
Batch Transform Job Demonstration

MLA-C01 Exam Relevance:
- Task 3.2: Deploy and operationalize ML solutions
- Batch transform = asynchronous, high-throughput offline predictions
- No persistent endpoint — pay only for the duration of the job
- Use cases: nightly scoring, bulk inference, dataset enrichment

This demo shows how to:
1. Prepare input data in S3
2. Configure and launch a Batch Transform job
3. Monitor job progress
4. Retrieve and inspect the prediction results
5. Compare batch transform vs real-time vs serverless (exam decision tree)
"""

import boto3
import json
import time
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AWS_REGION = "<YOUR-AWS-REGION>"
ROLE_ARN = "<YOUR-SAGEMAKER-ROLE-ARN>"
MODEL_NAME = "<YOUR-MODEL-NAME>"                       # Must already exist in SageMaker

BUCKET_NAME = "<YOUR-BUCKET-NAME>"
INPUT_S3_PREFIX = "batch-transform/input/"
OUTPUT_S3_PREFIX = "batch-transform/output/"

JOB_NAME = f"mla-c01-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# ---------------------------------------------------------------------------
# AWS Clients
# ---------------------------------------------------------------------------
s3_client = boto3.client("s3", region_name=AWS_REGION)
sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)


# ===================================================================
# STEP 1 — Upload input data to S3
# ===================================================================
def prepare_input_data():
    """
    Upload a CSV file containing multiple rows for batch prediction.
    Each row is one inference request (no header row for XGBoost).
    """
    logger.info("Preparing batch input data in S3 …")

    # Sample rows from the housing dataset (8 features per row)
    batch_rows = [
        "8.3252,41.0,6.984127,1.023810,322.0,2.555556,-122.23,37.88",
        "8.3014,21.0,6.238137,0.971880,2401.0,2.109842,-122.22,37.86",
        "7.2574,52.0,8.288136,1.073446,496.0,2.802260,-122.24,37.85",
        "5.6431,52.0,5.817352,1.073059,558.0,2.547945,-122.25,37.85",
        "3.8462,52.0,6.281853,1.081081,565.0,2.181467,-122.25,37.85",
        "4.0368,52.0,4.761658,1.103627,413.0,2.139896,-122.25,37.84",
        "3.6591,52.0,4.931907,0.951362,1094.0,2.128405,-122.25,37.84",
        "3.1200,52.0,4.797527,1.061824,1157.0,1.788253,-122.26,37.84",
        "2.0804,42.0,4.294118,1.117647,1206.0,2.026891,-122.26,37.84",
        "3.6912,52.0,4.970588,0.990196,1551.0,2.172269,-122.26,37.84",
    ]

    body = "\n".join(batch_rows)
    key = f"{INPUT_S3_PREFIX}housing_batch.csv"

    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=body.encode("utf-8"))
    input_s3_uri = f"s3://{BUCKET_NAME}/{key}"
    logger.info("Uploaded %d rows to %s", len(batch_rows), input_s3_uri)
    return input_s3_uri


# ===================================================================
# STEP 2 — Launch the Batch Transform Job
# ===================================================================
def create_batch_transform_job(input_s3_uri: str):
    """
    Configure and start the batch transform job.

    MLA-C01 Exam Tip — key parameters to know:
    ─────────────────────────────────────────────
    SplitType        : "Line" splits input by newline (one record per line)
    AssembleWith     : "Line" assembles output predictions line-by-line
    BatchStrategy    : "MultiRecord" sends multiple rows per request for throughput
                       "SingleRecord" sends one row per request (safer for large records)
    MaxPayloadInMB   : Limits individual request payload size
    MaxConcurrentTransforms : Parallelism — increase for larger datasets
    JoinSource       : "Input" joins predictions with input (useful for auditing)
    """
    output_s3_uri = f"s3://{BUCKET_NAME}/{OUTPUT_S3_PREFIX}"

    logger.info("Creating batch transform job: %s", JOB_NAME)
    logger.info("  Input:  %s", input_s3_uri)
    logger.info("  Output: %s", output_s3_uri)

    sagemaker_client.create_transform_job(
        TransformJobName=JOB_NAME,
        ModelName=MODEL_NAME,
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_s3_uri,
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": output_s3_uri,
            "AssembleWith": "Line",
            "Accept": "text/csv",
        },
        TransformResources={
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1,
        },
        BatchStrategy="MultiRecord",
        MaxPayloadInMB=6,
        MaxConcurrentTransforms=1,
        # Optional: join input with output for traceability
        # DataProcessing={
        #     "JoinSource": "Input",
        # },
    )
    logger.info("Batch transform job submitted.")


# ===================================================================
# STEP 3 — Monitor Job Progress
# ===================================================================
def wait_for_job():
    """Poll the job status until it completes or fails."""
    logger.info("Monitoring batch transform job …")

    while True:
        desc = sagemaker_client.describe_transform_job(TransformJobName=JOB_NAME)
        status = desc["TransformJobStatus"]
        logger.info("  Status: %s", status)

        if status == "Completed":
            duration = (desc["TransformEndTime"] - desc["TransformStartTime"]).total_seconds()
            logger.info("Job completed in %.0f seconds.", duration)
            return desc
        elif status == "Failed":
            logger.error("Job FAILED: %s", desc.get("FailureReason", "Unknown"))
            return desc
        elif status == "Stopped":
            logger.warning("Job was stopped.")
            return desc

        time.sleep(15)


# ===================================================================
# STEP 4 — Retrieve and Inspect Results
# ===================================================================
def inspect_results():
    """Download and display the batch predictions from S3."""
    logger.info("Retrieving results from s3://%s/%s …", BUCKET_NAME, OUTPUT_S3_PREFIX)

    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=OUTPUT_S3_PREFIX)

    if "Contents" not in response:
        logger.warning("No output files found.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith(".out"):
            logger.info("Reading: %s", key)
            result = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            predictions = result["Body"].read().decode("utf-8").strip().split("\n")
            for i, pred in enumerate(predictions):
                logger.info("  Row %d prediction: %s", i + 1, pred)


# ===================================================================
# STEP 5 — Exam Decision Tree: When to Use Batch Transform
# ===================================================================
def print_decision_guide():
    """
    MLA-C01 Exam Tip — choosing the right inference type:
    ──────────────────────────────────────────────────────

    Real-time endpoint:
      - Need sub-second latency
      - Persistent, always-on traffic
      - Example: fraud detection API, recommendation engine

    Serverless endpoint:
      - Intermittent / unpredictable traffic
      - Acceptable cold-start latency (seconds)
      - Example: dev/test, PoC, low-traffic production

    Batch transform:
      - Large dataset, no real-time requirement
      - Scheduled scoring (nightly, weekly)
      - One-time bulk inference
      - Example: credit scoring run, customer segmentation

    Asynchronous endpoint:
      - Large payloads (up to 1 GB)
      - Long processing time (minutes)
      - Need results later (queued)
      - Example: video processing, document analysis
    """
    logger.info("\n--- Endpoint Type Decision Guide ---")
    logger.info("Real-time   → Low latency, persistent traffic")
    logger.info("Serverless  → Intermittent traffic, cost-optimised")
    logger.info("Batch       → Large datasets, no real-time need")
    logger.info("Async       → Large payloads, long processing time")


# ===================================================================
# STEP 6 — Clean Up
# ===================================================================
def cleanup():
    """Stop the job if running and remove output from S3."""
    logger.info("Cleaning up …")
    try:
        sagemaker_client.stop_transform_job(TransformJobName=JOB_NAME)
        logger.info("Transform job stopped.")
    except Exception:
        pass  # Job may already be complete

    # Optionally remove output files
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=OUTPUT_S3_PREFIX)
    if "Contents" in response:
        for obj in response["Contents"]:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=obj["Key"])
        logger.info("Output files cleaned up.")


# ===================================================================
# Main
# ===================================================================
def main():
    logger.info("=== MLA-C01: Batch Transform Demo ===\n")

    # 1. Prepare input data
    input_uri = prepare_input_data()

    # 2. Launch job
    create_batch_transform_job(input_uri)

    # 3. Monitor
    wait_for_job()

    # 4. Inspect results
    inspect_results()

    # 5. Show decision guide
    print_decision_guide()

    # 6. Cleanup (uncomment when done recording)
    # cleanup()

    logger.info("\n=== Batch Transform Demo Complete ===")


if __name__ == "__main__":
    main()
