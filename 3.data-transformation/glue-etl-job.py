"""
AWS MLA-C01: AWS Glue ETL Job - Housing Data Transformation

This script demonstrates a production-grade AWS Glue ETL job that:
  1. Reads housing CSV data from S3 using DynamicFrame
  2. Applies schema mappings and column transformations
  3. Filters invalid rows (null bedrooms)
  4. Adds derived features using Spark SQL
  5. Writes output as Parquet with partitioning for optimal performance
  6. Implements proper error handling and logging

Key MLA-C01 Concepts:
  - AWS Glue DynamicFrame: Schema-on-read approach, handles semi-structured data
  - Spark Transformations: Efficient distributed processing
  - Data Partitioning: Improves query performance on large datasets
  - Schema Mapping: Renames and type-casts columns
  - Glue Catalog: Metadata management
"""

import sys
import logging
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import (
    col, when, coalesce, lower, round as spark_round,
    year, month, dayofmonth, unix_timestamp
)

# Configure logging for Glue job execution
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get job parameters from Glue job configuration
# These are passed via the Glue job definition or command-line arguments
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "S3_INPUT_PATH",
        "S3_OUTPUT_PATH",
    ]
)

job_name = args["JOB_NAME"]
s3_input_path = args["S3_INPUT_PATH"]  # e.g., s3://<YOUR-BUCKET-NAME>/data/housing-raw/
s3_output_path = args["S3_OUTPUT_PATH"]  # e.g., s3://<YOUR-BUCKET-NAME>/data/housing-transformed/

logger.info(f"Starting Glue ETL Job: {job_name}")
logger.info(f"Input path: {s3_input_path}")
logger.info(f"Output path: {s3_output_path}")

# Initialize Spark and Glue contexts
# SparkContext: Entry point for Spark functionality
# GlueContext: AWS Glue wrapper around SparkContext with additional features
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Initialize Glue Job: tracks job execution and auto-commits on success
job = Job(glueContext)
job.init(job_name, args)

try:
    # Step 1: Read CSV data from S3 into a DynamicFrame
    # DynamicFrame is Glue's distributed data frame with flexible schema
    # Advantages: handles schema drift, semi-structured data, and missing columns
    logger.info("Reading housing CSV from S3...")

    input_dyf = glueContext.create_dynamic_frame.from_options(
        format_options={"multiline": False, "withHeader": True},
        connection_type="s3",
        format="csv",
        connection_options={
            "paths": [s3_input_path],
            "recurse": True,
        },
        transformation_ctx="input_dyf",
    )

    logger.info(f"DynamicFrame created. Record count: {input_dyf.count()}")

    # Step 2: Convert DynamicFrame to Spark DataFrame for SQL transformations
    df = input_dyf.toDF()

    # Step 3: Apply schema mapping - rename columns and cast types
    # Mapping example: original column names to standardized names with proper types
    logger.info("Applying schema mappings and type casting...")

    df_mapped = (
        df
        .withColumnRenamed("price", "sale_price")
        .withColumnRenamed("bedrooms", "num_bedrooms")
        .withColumnRenamed("bathrooms", "num_bathrooms")
        .withColumnRenamed("sqft_living", "sq_ft_living")
        .withColumnRenamed("sqft_lot", "sq_ft_lot")
        .withColumnRenamed("yr_built", "year_built")
        .withColumnRenamed("yr_renovated", "year_renovated")
        .withColumnRenamed("zipcode", "zip_code")
    )

    # Cast numeric columns to appropriate types
    df_mapped = (
        df_mapped
        .withColumn("sale_price", col("sale_price").cast("double"))
        .withColumn("num_bedrooms", col("num_bedrooms").cast("int"))
        .withColumn("num_bathrooms", col("num_bathrooms").cast("double"))
        .withColumn("sq_ft_living", col("sq_ft_living").cast("int"))
        .withColumn("sq_ft_lot", col("sq_ft_lot").cast("int"))
        .withColumn("year_built", col("year_built").cast("int"))
        .withColumn("year_renovated", col("year_renovated").cast("int"))
        .withColumn("zip_code", col("zip_code").cast("string"))
    )

    # Step 4: Data Filtering - remove rows with invalid data
    # Filtering is critical for ML datasets: null values, invalid ranges impact model quality
    logger.info("Filtering rows with null or invalid values...")

    df_filtered = df_mapped.filter(
        (col("num_bedrooms").isNotNull()) &
        (col("num_bedrooms") > 0) &
        (col("num_bedrooms") <= 10) &
        (col("sale_price").isNotNull()) &
        (col("sale_price") > 0) &
        (col("sq_ft_living") > 0)
    )

    logger.info(f"Rows after filtering: {df_filtered.count()}")

    # Step 5: Derive new features for ML models
    # Feature engineering improves model performance and interpretability
    logger.info("Creating derived features...")

    df_featured = (
        df_filtered
        # Calculate price per square foot - key metric in real estate
        .withColumn(
            "price_per_sqft",
            spark_round(col("sale_price") / col("sq_ft_living"), 2)
        )
        # Calculate age of property (approximation using current year)
        .withColumn(
            "property_age",
            2024 - col("year_built")
        )
        # Classify property size (small, medium, large)
        .withColumn(
            "size_category",
            when(col("sq_ft_living") < 1500, "small")
            .when(col("sq_ft_living") < 3000, "medium")
            .otherwise("large")
        )
        # Classify property price (budget, mid-range, luxury)
        .withColumn(
            "price_category",
            when(col("sale_price") < 300000, "budget")
            .when(col("sale_price") < 700000, "mid_range")
            .otherwise("luxury")
        )
        # Indicate if property was renovated
        .withColumn(
            "was_renovated",
            when(col("year_renovated") > 0, 1).otherwise(0)
        )
        # Add processing timestamp for lineage tracking
        .withColumn(
            "processed_timestamp",
            unix_timestamp()
        )
    )

    # Step 6: Convert DataFrame back to DynamicFrame
    # DynamicFrame handles schema flexibility during write operations
    output_dyf = DynamicFrame.fromDF(
        df_featured,
        glueContext,
        "output_dyf"
    )

    # Step 7: Write output to S3 in Parquet format
    # Parquet: columnar format, highly compressible, faster for analytics
    # Partitioning: organize data by date/category for faster queries
    logger.info("Writing transformed data to S3 as Parquet...")

    glueContext.write_dynamic_frame.from_options(
        frame=output_dyf,
        connection_type="s3",
        format="parquet",
        connection_options={
            "path": s3_output_path,
            "partitionKeys": ["price_category", "size_category"],
        },
        transformation_ctx="output_write",
    )

    logger.info(f"Successfully wrote partitioned Parquet to {s3_output_path}")

    # Step 8: Print data quality metrics
    total_records = df_mapped.count()
    filtered_records = df_filtered.count()
    dropped_records = total_records - filtered_records

    logger.info("=" * 60)
    logger.info("ETL JOB SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records read: {total_records}")
    logger.info(f"Records after filtering: {filtered_records}")
    logger.info(f"Records dropped (invalid): {dropped_records}")
    logger.info(f"Data quality rate: {(filtered_records/total_records*100):.2f}%")
    logger.info(f"Output location: {s3_output_path}")
    logger.info("=" * 60)

    # Commit Glue job - marks successful completion
    job.commit()
    logger.info("Glue job completed successfully")

except Exception as e:
    logger.error(f"Error in Glue ETL job: {str(e)}", exc_info=True)
    job.commit()
    raise
