"""
AWS MLA-C01 Certification: Data Transformation and Feature Engineering Demo

MLA-C01 Exam Relevance:
- Task 1.2: Transform data and perform feature engineering
- Cleaning, encoding, scaling, splitting — core exam topics
- Choosing the right technique for the data type and model

This demo shows how to:
1. Load and explore the customer churn dataset
2. Handle missing values (imputation strategies)
3. Detect and remove duplicates
4. Feature engineering — create derived features
5. Encode categorical variables (one-hot, label encoding)
6. Scale numerical features (standardization, normalization)
7. Split data into train / validation / test sets
8. Save processed data to S3 for SageMaker training
"""

import pandas as pd
import numpy as np
import boto3
import logging
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = "<YOUR-BUCKET-NAME>"
REGION = "<YOUR-REGION>"
LOCAL_CHURN_DATA = "0.source-data/customer-churn/churn.csv"
S3_OUTPUT_PREFIX = "processed-data/churn/"

s3_client = boto3.client("s3", region_name=REGION)


# ===================================================================
# STEP 1 — Load and Explore
# ===================================================================
def load_and_explore(file_path: str) -> pd.DataFrame:
    """Load the churn dataset and print exploratory stats."""
    logger.info("Loading data from %s", file_path)
    df = pd.read_csv(file_path)

    logger.info("Shape: %d rows x %d columns", df.shape[0], df.shape[1])
    logger.info("Columns: %s", list(df.columns))
    logger.info("\nData types:\n%s", df.dtypes.to_string())
    logger.info("\nFirst 5 rows:\n%s", df.head().to_string())

    # Summary statistics
    logger.info("\nNumerical summary:\n%s", df.describe().to_string())

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.info("\nMissing values:\n%s", missing[missing > 0].to_string())
    else:
        logger.info("\nNo missing values found.")

    # Target distribution
    if "Churn" in df.columns:
        dist = df["Churn"].value_counts()
        logger.info("\nTarget distribution (Churn):\n%s", dist.to_string())
        logger.info("Class imbalance ratio: %.2f", dist.min() / dist.max())

    return df


# ===================================================================
# STEP 2 — Handle Missing Values
# ===================================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrate multiple imputation strategies.

    MLA-C01 Exam Tip:
    - Mean/median imputation: numerical features
    - Mode imputation: categorical features
    - Forward/backward fill: time-series data
    - Drop rows: only when missing rate is very low and data is abundant
    - KNN imputation: when features are correlated
    """
    logger.info("\n=== STEP 2: Handle Missing Values ===")
    df = df.copy()

    # Inject some missing values for demonstration
    np.random.seed(42)
    missing_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[missing_indices[:3], "MonthlyCharges"] = np.nan
    df.loc[missing_indices[3:], "tenure"] = np.nan
    logger.info("Injected missing values for demo: %d NaN in MonthlyCharges, %d in tenure",
                df["MonthlyCharges"].isna().sum(), df["tenure"].isna().sum())

    # Strategy 1: Mean imputation for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            logger.info("  Filled %s with mean: %.2f", col, mean_val)

    # Strategy 2: Mode imputation for categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info("  Filled %s with mode: %s", col, mode_val)

    logger.info("Missing values after imputation: %d", df.isna().sum().sum())
    return df


# ===================================================================
# STEP 3 — Remove Duplicates
# ===================================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Check for and remove duplicate rows."""
    logger.info("\n=== STEP 3: Remove Duplicates ===")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info("Removed %d duplicate rows (%d → %d)", before - after, before, after)
    return df


# ===================================================================
# STEP 4 — Feature Engineering
# ===================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features to improve model performance.

    MLA-C01 Exam Tip:
    - Feature engineering can be more impactful than model selection.
    - Binning, log transforms, interaction features, ratios are all testable.
    - SageMaker Data Wrangler and Processing Jobs automate this at scale.
    """
    logger.info("\n=== STEP 4: Feature Engineering ===")
    df = df.copy()

    # 4a. Tenure bins (binning / discretisation)
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72, np.inf],
            labels=["0-12m", "13-24m", "25-48m", "49-72m", "72m+"],
        )
        logger.info("Created tenure_group (binned): %s", df["tenure_group"].value_counts().to_dict())

    # 4b. Charges ratio (interaction feature)
    if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["charges_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, np.nan)
        df["charges_ratio"].fillna(0, inplace=True)
        logger.info("Created charges_ratio feature (MonthlyCharges / TotalCharges)")

    # 4c. Log transform for skewed features
    if "TotalCharges" in df.columns:
        df["log_total_charges"] = np.log1p(df["TotalCharges"].fillna(0))
        logger.info("Created log_total_charges (log1p transform)")

    # 4d. Binary flag feature
    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        logger.info("Created is_new_customer (tenure <= 6 months)")

    logger.info("New columns added: tenure_group, charges_ratio, log_total_charges, is_new_customer")
    logger.info("Total features: %d", len(df.columns))
    return df


# ===================================================================
# STEP 5 — Encode Categorical Variables
# ===================================================================
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical features to numerical representation.

    MLA-C01 Exam Tip:
    ──────────────────
    Label encoding:
      - Ordinal categories (Low < Medium < High)
      - Tree-based models can handle this directly

    One-hot encoding:
      - Nominal categories (no order — gender, contract type)
      - Required for linear models, neural networks
      - Watch for high-cardinality → feature explosion

    Binary encoding:
      - Two-class categories (Yes/No, Male/Female)
      - Compact representation
    """
    logger.info("\n=== STEP 5: Encode Categorical Variables ===")
    df = df.copy()

    # 5a. Binary encoding for Yes/No columns
    binary_map = {"Yes": 1, "No": 0}
    binary_cols = [col for col in df.columns if set(df[col].dropna().unique()) <= {"Yes", "No"}]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)
        logger.info("  Binary encoded: %s", col)

    # 5b. Label encoding for the target (if string)
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        le = LabelEncoder()
        df["Churn"] = le.fit_transform(df["Churn"])
        logger.info("  Label encoded target: Churn → %s", list(le.classes_))

    # 5c. One-hot encoding for remaining categorical columns
    remaining_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if remaining_cat:
        logger.info("  One-hot encoding: %s", remaining_cat)
        df = pd.get_dummies(df, columns=remaining_cat, drop_first=True, dtype=int)
        logger.info("  Columns after one-hot: %d", len(df.columns))

    return df


# ===================================================================
# STEP 6 — Scale Numerical Features
# ===================================================================
def scale_features(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Apply feature scaling.

    MLA-C01 Exam Tip:
    ──────────────────
    StandardScaler (z-score):
      - Mean = 0, StdDev = 1
      - Best for: linear regression, logistic regression, SVMs, neural networks
      - Not affected by outliers as much as MinMaxScaler

    MinMaxScaler (normalization):
      - Range [0, 1]
      - Best for: neural networks, algorithms sensitive to magnitude
      - Sensitive to outliers

    Tree-based models (XGBoost, Random Forest):
      - Do NOT require scaling (they split on feature values)
      - Exam may test this: "Do you need to scale features for XGBoost?" → No
    """
    logger.info("\n=== STEP 6: Feature Scaling ===")
    df = df.copy()

    # Separate features and target
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    # Identify numerical columns to scale
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Scaling %d numerical features", len(num_cols))

    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

    logger.info("Post-scaling stats (first 3 features):")
    for col in num_cols[:3]:
        logger.info("  %s: mean=%.4f, std=%.4f", col, X_scaled[col].mean(), X_scaled[col].std())

    # Also demo MinMaxScaler for comparison
    minmax = MinMaxScaler()
    X_minmax = X.copy()
    X_minmax[num_cols] = minmax.fit_transform(X[num_cols])
    logger.info("\nMinMaxScaler range check (first feature): min=%.4f, max=%.4f",
                X_minmax[num_cols[0]].min(), X_minmax[num_cols[0]].max())

    return X_scaled, y, scaler


# ===================================================================
# STEP 7 — Train / Validation / Test Split
# ===================================================================
def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split into train (70%), validation (15%), test (15%).

    MLA-C01 Exam Tip:
    - Stratify on the target for imbalanced datasets.
    - Random seed for reproducibility.
    - SageMaker expects train/val/test as separate S3 prefixes.
    """
    logger.info("\n=== STEP 7: Train/Validation/Test Split ===")

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Second split: 50/50 of temp → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info("Train:      %d rows (%.0f%%)", len(X_train), 100 * len(X_train) / len(X))
    logger.info("Validation: %d rows (%.0f%%)", len(X_val), 100 * len(X_val) / len(X))
    logger.info("Test:       %d rows (%.0f%%)", len(X_test), 100 * len(X_test) / len(X))

    # Check stratification preserved class balance
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        ratio = y_split.mean()
        logger.info("  %s churn rate: %.2f%%", name, ratio * 100)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ===================================================================
# STEP 8 — Save to S3 for SageMaker Training
# ===================================================================
def save_to_s3(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save processed CSVs to S3 in the structure SageMaker expects.

    MLA-C01 Exam Tip:
    - XGBoost built-in expects the target column FIRST (no header).
    - SageMaker channels map to S3 prefixes: train/, validation/, test/
    """
    logger.info("\n=== STEP 8: Save Processed Data to S3 ===")

    datasets = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split_name, (X, y) in datasets.items():
        # XGBoost format: target first, no header
        combined = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        csv_buffer = io.StringIO()
        combined.to_csv(csv_buffer, index=False, header=False)

        s3_key = f"{S3_OUTPUT_PREFIX}{split_name}/data.csv"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=csv_buffer.getvalue().encode("utf-8"),
        )
        s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"
        logger.info("  Saved %s (%d rows) → %s", split_name, len(X), s3_uri)


# ===================================================================
# Main
# ===================================================================
def main():
    logger.info("=== MLA-C01: Data Transformation Demo ===\n")

    # 1. Load and explore
    df = load_and_explore(LOCAL_CHURN_DATA)

    # 2. Handle missing values
    df = handle_missing_values(df)

    # 3. Remove duplicates
    df = remove_duplicates(df)

    # 4. Feature engineering
    df = engineer_features(df)

    # 5. Encode categoricals
    df = encode_categoricals(df)

    # 6. Scale features
    X_scaled, y, scaler = scale_features(df, target_col="Churn")

    # 7. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)

    # 8. Save to S3 (uncomment when running on AWS)
    # save_to_s3(X_train, X_val, X_test, y_train, y_val, y_test)

    logger.info("\n=== Data Transformation Demo Complete ===")
    logger.info("Total features after transformation: %d", X_scaled.shape[1])


if __name__ == "__main__":
    main()
