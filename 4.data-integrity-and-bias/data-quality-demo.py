"""
AWS MLA-C01: Data Quality and Validation Demo

This script demonstrates comprehensive data quality checks and validation:
  1. Load dataset and inspect basic statistics
  2. Validate schema (columns exist, correct dtypes)
  3. Analyze missing values and null patterns
  4. Detect duplicate records
  5. Range validation (numeric columns)
  6. Categorical value validation (allowed values)
  7. Outlier detection using IQR method
  8. Generate comprehensive data quality report

Key MLA-C01 Concepts:
  - Data Quality: Foundational for accurate ML models
  - AWS Glue DataBrew: Visual data preparation tool (PII masking, deduplication)
  - AWS Glue Data Quality: Rules engine for automated validation
  - Data Lineage: Track data transformations and quality checks
  - Garbage In, Garbage Out: Poor data leads to poor model performance
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing Data Quality Validation Demo...")

try:
    # Step 1: Load dataset
    # Using telco churn dataset as example
    logger.info("Loading dataset...")

    # In production, load from S3: pd.read_csv("s3://bucket/data.csv")
    # For demo, create sample dataset
    data = {
        "CustomerID": ["C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008"],
        "Age": [25, 45, 35, None, 55, 28, 42, 65],
        "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Unknown", "Female"],
        "Tenure": [12, 24, 6, 18, None, 3, 48, 120],
        "MonthlyCharges": [65.5, 89.0, 45.0, 75.5, 95.0, -5.0, 110.0, 150.5],
        "TotalCharges": [786, 2136, 270, None, 4750, 150, 5280, 18060],
        "Churn": ["No", "Yes", "No", "No", "Yes", "Yes", "No", "No"],
    }

    df = pd.DataFrame(data)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2: Define data quality rules
    # These rules define what constitutes "good" data
    quality_rules = {
        # Schema validation: column names and types
        "schema": {
            "CustomerID": "object",
            "Age": "float64",
            "Gender": "object",
            "Tenure": "float64",
            "MonthlyCharges": "float64",
            "TotalCharges": "float64",
            "Churn": "object",
        },
        # Range validation: acceptable value ranges for numeric columns
        "ranges": {
            "Age": (18, 100),  # Age between 18 and 100
            "Tenure": (0, 120),  # Tenure in months, 0 to 10 years
            "MonthlyCharges": (0, 1000),  # Monthly charges positive
            "TotalCharges": (0, 100000),  # Total charges positive
        },
        # Categorical validation: allowed values
        "categories": {
            "Gender": ["Male", "Female", "Other"],
            "Churn": ["Yes", "No"],
        },
    }

    # Step 3: Schema Validation
    # Verify expected columns exist with correct data types
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: SCHEMA VALIDATION")
    logger.info("=" * 70)

    schema_issues = []

    # Check if all required columns exist
    required_columns = set(quality_rules["schema"].keys())
    actual_columns = set(df.columns)

    missing_columns = required_columns - actual_columns
    extra_columns = actual_columns - required_columns

    if missing_columns:
        schema_issues.append(f"Missing columns: {missing_columns}")
        logger.warning(f"Missing columns: {missing_columns}")

    if extra_columns:
        logger.info(f"Extra columns found (may be OK): {extra_columns}")

    # Check data types
    for column, expected_type in quality_rules["schema"].items():
        if column in df.columns:
            # Skip type check for columns with nulls (pandas converts to float/object)
            actual_type = str(df[column].dtype)
            if expected_type in actual_type or actual_type in expected_type:
                logger.info(f"  ✓ {column}: {actual_type} (expected {expected_type})")
            else:
                logger.warning(f"  ✗ {column}: {actual_type} (expected {expected_type})")
                schema_issues.append(f"Type mismatch in {column}")

    if not schema_issues:
        logger.info("✓ Schema validation passed")
    else:
        logger.warning(f"✗ Schema issues found: {len(schema_issues)}")

    # Step 4: Null/Missing Value Analysis
    # Missing data can bias models and reduce training data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: NULL/MISSING VALUE ANALYSIS")
    logger.info("=" * 70)

    null_analysis = {}
    null_issues = []

    for column in df.columns:
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        null_analysis[column] = {
            "null_count": null_count,
            "null_percentage": null_percentage,
            "valid_count": len(df) - null_count,
        }

        if null_count > 0:
            logger.warning(f"  {column}: {null_count} nulls ({null_percentage:.1f}%)")
            if null_percentage > 20:
                null_issues.append(f"{column} has {null_percentage:.1f}% missing data")
        else:
            logger.info(f"  ✓ {column}: No nulls")

    # Print summary
    total_nulls = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    logger.info(f"\nTotal nulls: {total_nulls} / {total_cells} ({(total_nulls/total_cells)*100:.1f}%)")

    if null_issues:
        logger.warning("Recommendations for null handling:")
        for issue in null_issues:
            logger.warning(f"  - {issue}")
            logger.warning("    Options: Drop rows, Impute (mean/median/mode), Forward-fill, Drop column")

    # Step 5: Duplicate Detection
    # Duplicates reduce effective training data and can bias models
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: DUPLICATE DETECTION")
    logger.info("=" * 70)

    # Check for fully duplicate rows
    full_duplicates = df.duplicated().sum()
    logger.info(f"Full row duplicates: {full_duplicates}")

    # Check for duplicates on key column (CustomerID)
    id_duplicates = df.duplicated(subset=["CustomerID"]).sum()
    logger.info(f"Duplicates on CustomerID: {id_duplicates}")

    if full_duplicates > 0:
        logger.warning(f"Found {full_duplicates} duplicate rows - consider deduplication")
        logger.warning("In Glue: Use DropDuplicates transform or DataBrew deduplication rule")

    if id_duplicates == 0:
        logger.info("✓ No duplicate IDs found")
    else:
        logger.warning(f"✗ Found {id_duplicates} duplicate IDs")

    # Step 6: Range Validation
    # Numeric values outside expected ranges indicate data quality issues
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: RANGE VALIDATION")
    logger.info("=" * 70)

    range_issues = []

    for column, (min_val, max_val) in quality_rules["ranges"].items():
        if column in df.columns:
            valid_data = df[column].dropna()  # Exclude nulls

            if len(valid_data) > 0:
                out_of_range = ((valid_data < min_val) | (valid_data > max_val)).sum()

                if out_of_range == 0:
                    logger.info(f"  ✓ {column}: All values in range [{min_val}, {max_val}]")
                else:
                    logger.warning(
                        f"  ✗ {column}: {out_of_range} values outside [{min_val}, {max_val}]"
                    )
                    range_issues.append(
                        f"{column}: {out_of_range} values outside acceptable range"
                    )

                    # Show offending values
                    bad_values = valid_data[(valid_data < min_val) | (valid_data > max_val)]
                    logger.warning(f"    Invalid values: {bad_values.values}")

    if range_issues:
        logger.warning("Recommendations:")
        logger.warning("  - Investigate root cause of out-of-range values")
        logger.warning("  - Options: Remove rows, Clip to range, Flag for review")

    # Step 7: Categorical Value Validation
    # Ensures categorical columns only contain expected values
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: CATEGORICAL VALUE VALIDATION")
    logger.info("=" * 70)

    cat_issues = []

    for column, allowed_values in quality_rules["categories"].items():
        if column in df.columns:
            valid_data = df[column].dropna()
            invalid_values = ~valid_data.isin(allowed_values)
            invalid_count = invalid_values.sum()

            if invalid_count == 0:
                logger.info(f"  ✓ {column}: All values in {allowed_values}")
            else:
                logger.warning(
                    f"  ✗ {column}: {invalid_count} invalid values"
                )
                unique_invalid = valid_data[invalid_values].unique()
                logger.warning(f"    Invalid values: {unique_invalid}")
                cat_issues.append(f"{column}: {invalid_count} unexpected values")

    # Step 8: Outlier Detection using IQR method
    # Identifies extreme values that may be errors or legitimate anomalies
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: OUTLIER DETECTION (IQR Method)")
    logger.info("=" * 70)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outlier_analysis = {}

    for column in numeric_columns:
        valid_data = df[column].dropna()

        if len(valid_data) > 0:
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1

            # IQR method: outliers are 1.5*IQR beyond Q1/Q3
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (valid_data < lower_bound) | (valid_data > upper_bound)
            outlier_count = outliers.sum()

            outlier_analysis[column] = {
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percentage": (outlier_count / len(valid_data)) * 100,
            }

            if outlier_count > 0:
                logger.warning(f"  {column}: {outlier_count} outliers ({outlier_analysis[column]['outlier_percentage']:.1f}%)")
                logger.warning(f"    Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                logger.warning(f"    Outlier values: {valid_data[outliers].values}")
            else:
                logger.info(f"  ✓ {column}: No outliers detected")

    # Step 9: Generate Data Quality Report
    # Comprehensive summary of all quality checks
    logger.info("\n" + "=" * 70)
    logger.info("DATA QUALITY REPORT SUMMARY")
    logger.info("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_shape": df.shape,
        "schema_validation": {
            "passed": len(schema_issues) == 0,
            "issues": schema_issues,
        },
        "null_analysis": null_analysis,
        "duplicates": {
            "full_duplicates": int(full_duplicates),
            "id_duplicates": int(id_duplicates),
        },
        "range_validation": {
            "issues": range_issues,
        },
        "categorical_validation": {
            "issues": cat_issues,
        },
        "outlier_analysis": {
            col: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                  for k, v in details.items()}
            for col, details in outlier_analysis.items()
        },
    }

    # Calculate overall quality score (0-100)
    total_issues = (
        len(schema_issues) +
        len(null_issues) +
        len(range_issues) +
        len(cat_issues) +
        sum(1 for v in outlier_analysis.values() if v['outlier_count'] > 0)
    )
    quality_score = max(0, 100 - (total_issues * 10))

    logger.info(f"\n{'Dataset Statistics':.<50} {df.shape[0]} rows × {df.shape[1]} cols")
    logger.info(f"{'Schema Issues':.<50} {len(schema_issues)}")
    logger.info(f"{'Null Value Issues':.<50} {len(null_issues)}")
    logger.info(f"{'Duplicate Records':.<50} {full_duplicates}")
    logger.info(f"{'Range Validation Issues':.<50} {len(range_issues)}")
    logger.info(f"{'Categorical Issues':.<50} {len(cat_issues)}")
    logger.info(f"{'Outlier Detections':.<50} {sum(1 for v in outlier_analysis.values() if v['outlier_count'] > 0)}")
    logger.info(f"\n{'OVERALL DATA QUALITY SCORE':.<50} {quality_score:.0f}/100")
    logger.info("=" * 70)

    # Step 10: AWS Glue Data Quality equivalents
    logger.info("\nAWS Glue Data Quality Rules Equivalents:")
    logger.info("  - Schema validation: RuleSet.Completeness, ColumnLength")
    logger.info("  - Null detection: RuleSet.Completeness")
    logger.info("  - Duplicate detection: RuleSet.Uniqueness (for ID columns)")
    logger.info("  - Range validation: RuleSet.Range")
    logger.info("  - Categorical validation: RuleSet.ColumnValues")
    logger.info("  - Outlier detection: RuleSet.StandardDeviation, IQR")
    logger.info("\nAWS Glue DataBrew:")
    logger.info("  - Visual data preparation without code")
    logger.info("  - Automated data profiling and quality assessment")
    logger.info("  - PII detection and masking")
    logger.info("  - Standardization and formatting rules")

    # Save report to JSON
    report_path = "/tmp/data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nDetailed report saved to: {report_path}")

    # Recommendations based on quality score
    logger.info("\nRECOMMENDATIONS:")
    if quality_score >= 90:
        logger.info("✓ Data quality is excellent. Ready for model training.")
    elif quality_score >= 70:
        logger.info("⚠ Data quality is acceptable. Address major issues before training:")
        for issue in schema_issues + null_issues + range_issues:
            logger.info(f"  - {issue}")
    else:
        logger.info("✗ Data quality is poor. Significant work needed:")
        for issue in schema_issues + null_issues + range_issues:
            logger.info(f"  - {issue}")

    logger.info("\nData quality validation completed successfully!")

except Exception as e:
    logger.error(f"Error in data quality validation: {str(e)}", exc_info=True)
    raise
