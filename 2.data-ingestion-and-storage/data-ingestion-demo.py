"""
AWS MLA-C01 Certification: Data Ingestion and Storage Demo

MLA-C01 Exam Relevance:
- Task 1.1: Ingest and store data for ML workloads
- S3 as the primary ML data lake
- SageMaker Feature Store for feature management (online + offline)
- Understanding when to use each storage option

This demo shows how to:
1. Upload source data to S3 with proper folder structure
2. Read data from S3 into pandas
3. Organise data for SageMaker training channels
4. Create a SageMaker Feature Store feature group
5. Ingest records into the feature store
6. Query features from the online store (real-time)
7. Clean up resources
"""

import boto3
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = "<YOUR-BUCKET-NAME>"
REGION = "<YOUR-REGION>"
ROLE_ARN = "<YOUR-SAGEMAKER-ROLE-ARN>"

LOCAL_HOUSING_DATA = "0.source-data/housing/housing.csv"
LOCAL_CHURN_DATA = "0.source-data/customer-churn/churn.csv"

S3_DATA_PREFIX = "mla-c01/raw-data/"
FEATURE_GROUP_NAME = "housing-features-mla-c01"

# ---------------------------------------------------------------------------
# AWS Clients
# ---------------------------------------------------------------------------
s3_client = boto3.client("s3", region_name=REGION)
sagemaker_client = boto3.client("sagemaker", region_name=REGION)
featurestore_runtime = boto3.client("sagemaker-featurestore-runtime", region_name=REGION)
boto_session = boto3.Session(region_name=REGION)
sagemaker_session = Session(boto_session=boto_session)


# ===================================================================
# STEP 1 — Upload Data to S3
# ===================================================================
def upload_to_s3(local_path: str, s3_prefix: str) -> str:
    """
    Upload a local file to S3 and return the S3 URI.

    MLA-C01 Exam Tip:
    - S3 is the default data lake for all SageMaker operations.
    - Use prefixes to organise: raw/, processed/, train/, val/, test/
    - Enable versioning for data lineage and reproducibility.
    """
    file_name = local_path.split("/")[-1]
    s3_key = f"{s3_prefix}{file_name}"

    logger.info("Uploading %s → s3://%s/%s", local_path, BUCKET_NAME, s3_key)
    s3_client.upload_file(local_path, BUCKET_NAME, s3_key)

    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"
    logger.info("Upload complete: %s", s3_uri)
    return s3_uri


def upload_all_source_data():
    """Upload both datasets to S3 with proper folder structure."""
    logger.info("\n=== STEP 1: Upload Source Data to S3 ===")

    housing_uri = upload_to_s3(LOCAL_HOUSING_DATA, f"{S3_DATA_PREFIX}housing/")
    churn_uri = upload_to_s3(LOCAL_CHURN_DATA, f"{S3_DATA_PREFIX}customer-churn/")

    # Verify uploads
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_DATA_PREFIX)
    logger.info("\nFiles in S3 under %s:", S3_DATA_PREFIX)
    for obj in response.get("Contents", []):
        logger.info("  %s  (%d bytes)", obj["Key"], obj["Size"])

    return housing_uri, churn_uri


# ===================================================================
# STEP 2 — Read Data from S3
# ===================================================================
def read_from_s3(s3_uri: str) -> pd.DataFrame:
    """
    Read CSV data directly from S3 into a pandas DataFrame.

    MLA-C01 Exam Tip:
    - pandas can read S3 URIs directly (uses boto3 under the hood).
    - For large datasets, use SageMaker Processing jobs instead.
    - Parquet format is preferred for large datasets (columnar, compressed).
    """
    logger.info("\n=== STEP 2: Read Data from S3 ===")
    logger.info("Reading: %s", s3_uri)

    df = pd.read_csv(s3_uri)
    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])
    logger.info("Columns: %s", list(df.columns))
    logger.info("\nPreview:\n%s", df.head(3).to_string())
    return df


# ===================================================================
# STEP 3 — Organise Data for SageMaker Training Channels
# ===================================================================
def organise_training_channels(df: pd.DataFrame, dataset_name: str):
    """
    Split and upload data into the channel structure SageMaker expects.

    MLA-C01 Exam Tip:
    - SageMaker training jobs use "channels" mapped to S3 prefixes.
    - Default channels: "train", "validation", "test"
    - For XGBoost built-in: target column must be FIRST, no header.
    """
    logger.info("\n=== STEP 3: Organise Training Channels ===")

    from sklearn.model_selection import train_test_split
    import io

    # 70/15/15 split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    channels = {"train": train_df, "validation": val_df, "test": test_df}

    for channel, channel_df in channels.items():
        csv_buffer = io.StringIO()
        channel_df.to_csv(csv_buffer, index=False, header=False)

        s3_key = f"mla-c01/channels/{dataset_name}/{channel}/data.csv"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=csv_buffer.getvalue().encode("utf-8"),
        )
        logger.info(
            "  %s channel: %d rows → s3://%s/%s",
            channel, len(channel_df), BUCKET_NAME, s3_key,
        )


# ===================================================================
# STEP 4 — Create a SageMaker Feature Store Feature Group
# ===================================================================
def create_feature_group(df: pd.DataFrame):
    """
    Create and configure a Feature Store feature group.

    MLA-C01 Exam Tip:
    ──────────────────
    Online store:
      - Low-latency GetRecord (single-digit ms)
      - Used for real-time inference feature lookup
      - DynamoDB-backed

    Offline store:
      - S3-backed, Parquet format
      - Used for training data, batch feature retrieval
      - Queryable via Athena

    Record identifier:
      - Unique key for each record (like a primary key)

    Event time:
      - Timestamp for point-in-time correctness
      - Critical for avoiding data leakage in training
    """
    logger.info("\n=== STEP 4: Create Feature Store Feature Group ===")

    # Prepare the dataframe — Feature Store requires specific column setup
    fs_df = df.copy()

    # Add required columns
    fs_df["record_id"] = [str(i) for i in range(len(fs_df))]
    fs_df["event_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure all column names are Feature Store compatible (no special chars)
    fs_df.columns = [c.replace(" ", "_").replace(".", "_").lower() for c in fs_df.columns]

    logger.info("Feature group columns: %s", list(fs_df.columns))

    # Create the feature group
    feature_group = FeatureGroup(
        name=FEATURE_GROUP_NAME,
        sagemaker_session=sagemaker_session,
    )

    # Load the dataframe to define the feature definitions
    feature_group.load_feature_definitions(data_frame=fs_df)

    # Create with both online and offline stores
    try:
        feature_group.create(
            s3_uri=f"s3://{BUCKET_NAME}/feature-store/{FEATURE_GROUP_NAME}/",
            record_identifier_name="record_id",
            event_time_feature_name="event_time",
            role_arn=ROLE_ARN,
            enable_online_store=True,
        )
        logger.info("Feature group '%s' created. Waiting for it to become active …", FEATURE_GROUP_NAME)

        # Wait for feature group to be ready
        status = ""
        while status != "Created":
            desc = sagemaker_client.describe_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
            status = desc["FeatureGroupStatus"]
            logger.info("  Status: %s", status)
            if status == "CreateFailed":
                logger.error("Feature group creation failed: %s", desc.get("FailureReason"))
                return None, None
            time.sleep(5)

        logger.info("Feature group is active and ready.")
    except Exception as e:
        if "ResourceInUse" in str(e):
            logger.warning("Feature group already exists — using existing.")
        else:
            raise

    return feature_group, fs_df


# ===================================================================
# STEP 5 — Ingest Records into the Feature Store
# ===================================================================
def ingest_records(feature_group: FeatureGroup, df: pd.DataFrame):
    """
    Ingest DataFrame records into the feature group.

    Records are written to both online (DynamoDB) and offline (S3) stores.
    Offline store ingestion has a delay (minutes) for the Parquet file write.
    """
    logger.info("\n=== STEP 5: Ingest Records into Feature Store ===")

    logger.info("Ingesting %d records …", len(df))
    feature_group.ingest(data_frame=df, max_workers=3, wait=True)
    logger.info("Ingestion complete.")
    logger.info("  Online store: immediately available")
    logger.info("  Offline store: will appear in S3 within ~15 minutes")


# ===================================================================
# STEP 6 — Query Features from the Online Store
# ===================================================================
def query_online_store(record_ids: list):
    """
    Retrieve specific records from the online store (low-latency lookup).

    MLA-C01 Exam Tip:
    - Online store = GetRecord API = real-time serving
    - Use this in your inference pipeline to fetch features at prediction time.
    - Offline store = Athena queries = batch retrieval for training.
    """
    logger.info("\n=== STEP 6: Query Online Feature Store ===")

    for record_id in record_ids:
        try:
            response = featurestore_runtime.get_record(
                FeatureGroupName=FEATURE_GROUP_NAME,
                RecordIdentifierValueAsString=str(record_id),
            )
            record = response.get("Record", [])
            features = {item["FeatureName"]: item["ValueAsString"] for item in record}
            logger.info("Record %s: %s", record_id, json.dumps(features, indent=2)[:200])
        except Exception as e:
            logger.warning("Could not retrieve record %s: %s", record_id, e)


# ===================================================================
# STEP 7 — Data Format Comparison (exam knowledge)
# ===================================================================
def print_format_comparison():
    """
    MLA-C01 Exam Tip — Data format decision guide:
    ────────────────────────────────────────────────
    CSV      : Human-readable, no schema, slow for large data
    Parquet  : Columnar, compressed, schema-embedded — best for analytics/ML
    JSON     : Semi-structured, nested data — good for APIs, logs
    ORC      : Columnar (Hadoop ecosystem) — similar to Parquet
    Avro     : Row-based, schema evolution — good for streaming
    RecordIO : SageMaker's native format — efficient for built-in algorithms
    """
    logger.info("\n--- Data Format Comparison ---")
    formats = [
        ("CSV",      "Row-based", "Human-readable, simple",         "Small datasets, exploration"),
        ("Parquet",  "Columnar",  "Compressed, schema-embedded",    "ML training, analytics (PREFERRED)"),
        ("JSON",     "Document",  "Semi-structured, nested",        "APIs, logs, config files"),
        ("RecordIO", "Binary",    "Efficient serialisation",        "SageMaker built-in algorithms"),
        ("Avro",     "Row-based", "Schema evolution support",       "Streaming (Kafka/Kinesis)"),
    ]
    for fmt, layout, desc, use_case in formats:
        logger.info("  %-10s | %-10s | %-35s | %s", fmt, layout, desc, use_case)


# ===================================================================
# STEP 8 — Clean Up
# ===================================================================
def cleanup():
    """Delete feature group and uploaded S3 data."""
    logger.info("\nCleaning up …")
    try:
        sagemaker_client.delete_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
        logger.info("Feature group deleted.")
    except Exception as e:
        logger.warning("Could not delete feature group: %s", e)

    # Clean up S3 objects
    for prefix in [S3_DATA_PREFIX, f"feature-store/{FEATURE_GROUP_NAME}/"]:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        for obj in response.get("Contents", []):
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=obj["Key"])
        logger.info("Cleaned S3 prefix: %s", prefix)


# ===================================================================
# Main
# ===================================================================
def main():
    logger.info("=== MLA-C01: Data Ingestion & Storage Demo ===\n")

    # 1. Upload data to S3
    housing_uri, churn_uri = upload_all_source_data()

    # 2. Read data from S3
    housing_df = read_from_s3(housing_uri)

    # 3. Organise training channels
    organise_training_channels(housing_df, "housing")

    # 4. Create Feature Store group
    feature_group, fs_df = create_feature_group(housing_df)

    # 5. Ingest records (uncomment when running on AWS with Feature Store)
    # if feature_group and fs_df is not None:
    #     ingest_records(feature_group, fs_df)
    #     time.sleep(5)
    #     # 6. Query online store
    #     query_online_store(["0", "1", "2"])

    # 7. Data format comparison
    print_format_comparison()

    # 8. Cleanup (uncomment when done recording)
    # cleanup()

    logger.info("\n=== Data Ingestion & Storage Demo Complete ===")


if __name__ == "__main__":
    main()
