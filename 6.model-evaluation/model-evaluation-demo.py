"""
AWS MLA-C01: Comprehensive Model Evaluation and Explainability

This script demonstrates end-to-end model evaluation:
  1. Load test predictions and ground truth labels
  2. Compute classification metrics (accuracy, precision, recall, F1, AUC-ROC)
  3. Compute regression metrics (RMSE, MAE, R-squared)
  4. Generate confusion matrix and interpret results
  5. Plot ROC curve for classification performance
  6. Set up SageMaker Clarify for model explainability (SHAP)
  7. Interpret model predictions and feature importance

Key MLA-C01 Concepts:
  - Classification Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
  - Regression Metrics: RMSE, MAE, R², MAPE
  - Confusion Matrix: True Positives, False Positives, etc.
  - Threshold Selection: Trade-off between precision and recall
  - ROC Curve: TPR vs FPR at different thresholds
  - Model Explainability: SHAP values for feature importance
  - SageMaker Clarify: Automated explainability and bias detection
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict
import boto3
from sagemaker import Session
from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    SHAPConfig,
    ModelConfig,
)

# Import scikit-learn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing Model Evaluation and Explainability Demo...")

try:
    # ========================================================================
    # STEP 1: Generate Synthetic Test Data and Predictions
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOAD TEST DATA AND PREDICTIONS")
    logger.info("=" * 70)

    # In production: load from test set or deployment logs
    np.random.seed(42)
    n_samples = 500

    # Generate synthetic test data
    y_true = np.random.randint(0, 2, n_samples)  # Binary classification (0 or 1)
    # Model predictions (probabilities)
    y_pred_proba = np.random.rand(n_samples)

    # Make predictions correlated with truth (simulate reasonable model)
    y_pred_proba = y_pred_proba * 0.3 + np.array(y_true) * 0.7 + np.random.normal(0, 0.1, n_samples)
    y_pred_proba = np.clip(y_pred_proba, 0, 1)  # Keep in [0, 1]

    # Binary predictions at threshold 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    logger.info(f"Test set size: {len(y_true)} samples")
    logger.info(f"Positive class rate: {y_true.mean():.1%}")
    logger.info(f"Predicted positive rate: {y_pred.mean():.1%}")

    # ========================================================================
    # STEP 2: Classification Metrics
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: CLASSIFICATION METRICS")
    logger.info("=" * 70)

    # ===== Accuracy =====
    # Percentage of correct predictions
    # Formula: (TP + TN) / Total
    # Range: [0, 1] (1 = perfect)
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"\nAccuracy: {accuracy:.4f}")
    logger.info("  Definition: Fraction of correct predictions")
    logger.info("  Formula: (TP + TN) / Total")
    logger.info("  Interpretation: Overall correctness")
    logger.info("  Limitation: Misleading for imbalanced datasets")

    # ===== Precision =====
    # Of positive predictions, how many are correct?
    # Formula: TP / (TP + FP)
    # Range: [0, 1] (1 = no false positives)
    precision = precision_score(y_true, y_pred, zero_division=0)
    logger.info(f"\nPrecision: {precision:.4f}")
    logger.info("  Definition: Fraction of positive predictions that are correct")
    logger.info("  Formula: TP / (TP + FP)")
    logger.info("  Interpretation: Model's confidence in positive predictions")
    logger.info("  Best for: Minimizing false positives (e.g., spam detection)")

    # ===== Recall (Sensitivity, True Positive Rate) =====
    # Of actual positives, how many did we find?
    # Formula: TP / (TP + FN)
    # Range: [0, 1] (1 = no false negatives)
    recall = recall_score(y_true, y_pred, zero_division=0)
    logger.info(f"\nRecall: {recall:.4f}")
    logger.info("  Definition: Fraction of actual positives that were predicted correctly")
    logger.info("  Formula: TP / (TP + FN)")
    logger.info("  Interpretation: Ability to find positive cases")
    logger.info("  Best for: Minimizing false negatives (e.g., disease detection)")

    # ===== F1-Score =====
    # Harmonic mean of precision and recall
    # Formula: 2 * (Precision * Recall) / (Precision + Recall)
    # Range: [0, 1] (1 = perfect)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    logger.info(f"\nF1-Score: {f1:.4f}")
    logger.info("  Definition: Harmonic mean of precision and recall")
    logger.info("  Formula: 2 * (Precision * Recall) / (Precision + Recall)")
    logger.info("  Interpretation: Balance between precision and recall")
    logger.info("  Best for: Imbalanced datasets, when precision AND recall matter")

    # ===== AUC-ROC =====
    # Area under Receiver Operating Characteristic curve
    # Range: [0, 1] (1 = perfect, 0.5 = random)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    logger.info(f"\nAUC-ROC: {auc_roc:.4f}")
    logger.info("  Definition: Area under ROC curve (TPR vs FPR)")
    logger.info("  Range: [0, 1]")
    logger.info("  Interpretation: Probability model ranks random positive higher than random negative")
    logger.info("  Best for: Evaluating ranking quality, threshold-independent")
    logger.info("  Advantage: Invariant to class imbalance")

    # ========================================================================
    # STEP 3: Regression Metrics (for continuous predictions)
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: REGRESSION METRICS (Continuous Predictions)")
    logger.info("=" * 70)

    # Generate synthetic regression data for demonstration
    y_reg_true = np.linspace(0, 100, 100)
    y_reg_pred = y_reg_true + np.random.normal(0, 5, 100)

    # ===== RMSE (Root Mean Squared Error) =====
    # Square root of average squared residuals
    # Formula: sqrt(mean((y_true - y_pred)^2))
    # Unit: Same as target variable
    rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
    logger.info(f"\nRMSE: {rmse:.4f}")
    logger.info("  Definition: Root mean squared error")
    logger.info("  Formula: sqrt(mean((y_true - y_pred)^2))")
    logger.info("  Interpretation: Average prediction error (in target units)")
    logger.info("  Sensitivity: High sensitivity to large errors")

    # ===== MAE (Mean Absolute Error) =====
    # Average absolute residuals
    # Formula: mean(|y_true - y_pred|)
    # Unit: Same as target variable
    mae = mean_absolute_error(y_reg_true, y_reg_pred)
    logger.info(f"\nMAE: {mae:.4f}")
    logger.info("  Definition: Mean absolute error")
    logger.info("  Formula: mean(|y_true - y_pred|)")
    logger.info("  Interpretation: Average absolute prediction error")
    logger.info("  Robustness: Less sensitive to outliers than RMSE")

    # ===== R-Squared =====
    # Proportion of variance explained by model
    # Formula: 1 - (SS_res / SS_tot)
    # Range: (-inf, 1] (1 = perfect, 0 = baseline)
    r2 = r2_score(y_reg_true, y_reg_pred)
    logger.info(f"\nR-Squared: {r2:.4f}")
    logger.info("  Definition: Coefficient of determination")
    logger.info("  Formula: 1 - (SS_residual / SS_total)")
    logger.info("  Range: (-inf, 1]")
    logger.info("  Interpretation: Fraction of variance explained by model")
    logger.info("  Baseline: R² = 0 means model equals mean predictor")

    # ===== MAPE (Mean Absolute Percentage Error) =====
    # Average percentage error
    # Formula: mean(|y_true - y_pred| / |y_true|) * 100
    # Unit: Percentage
    mape = mean_absolute_percentage_error(y_reg_true, y_reg_pred)
    logger.info(f"\nMAPE: {mape:.2f}%")
    logger.info("  Definition: Mean absolute percentage error")
    logger.info("  Formula: mean(|error| / |y_true|) * 100")
    logger.info("  Unit: Percentage")
    logger.info("  Best for: Scale-independent error measurement")

    # ========================================================================
    # STEP 4: Confusion Matrix
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: CONFUSION MATRIX")
    logger.info("=" * 70)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info("\nConfusion Matrix:")
    logger.info("               Predicted")
    logger.info("             Neg     Pos")
    logger.info(f"Actual Neg  {tn:5d}  {fp:5d}")
    logger.info(f"Actual Pos  {fn:5d}  {tp:5d}")

    logger.info("\nInterpretation:")
    logger.info(f"  True Negatives (TN): {tn} - Correctly predicted negative")
    logger.info(f"  False Positives (FP): {fp} - Incorrectly predicted positive (Type I error)")
    logger.info(f"  False Negatives (FN): {fn} - Incorrectly predicted negative (Type II error)")
    logger.info(f"  True Positives (TP): {tp} - Correctly predicted positive")

    logger.info("\nDerived Metrics:")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    logger.info(f"  Specificity (True Negative Rate): {specificity:.4f}")
    logger.info(f"    Formula: TN / (TN + FP)")
    logger.info(f"    Interpretation: Ability to identify negative cases")

    # ========================================================================
    # STEP 5: ROC Curve
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: ROC CURVE ANALYSIS")
    logger.info("=" * 70)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    logger.info(f"\nROC Curve generated with {len(thresholds)} threshold points")
    logger.info("  FPR (False Positive Rate): TP / (TN + FP) for each threshold")
    logger.info("  TPR (True Positive Rate): TP / (TP + FN) for each threshold")

    logger.info("\nROC Curve Interpretation:")
    logger.info("  - (0, 1): Perfect classifier")
    logger.info("  - (0, 0)-(1, 1): Random classifier (AUC = 0.5)")
    logger.info("  - Knee of curve: Good threshold (high TPR, low FPR)")

    # Plot ROC Curve
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Display plot inline in notebook
        plt.show()
        logger.info("✓ ROC Curve displayed")

    except Exception as e:
        logger.warning(f"Could not create ROC plot: {e}")

    # ========================================================================
    # STEP 6: Threshold Selection
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: THRESHOLD SELECTION")
    logger.info("=" * 70)

    logger.info("\nThreshold Analysis:")
    logger.info("  Default threshold: 0.5 (equal precision/recall weight)")
    logger.info("  High precision needed? Use higher threshold (e.g., 0.7)")
    logger.info("  High recall needed? Use lower threshold (e.g., 0.3)")

    # Find optimal threshold (Youden's J-statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    logger.info(f"\nOptimal Threshold (Youden's J): {optimal_threshold:.3f}")
    logger.info(f"  At this threshold:")
    logger.info(f"    TPR: {tpr[optimal_idx]:.4f}")
    logger.info(f"    FPR: {fpr[optimal_idx]:.4f}")

    # ========================================================================
    # STEP 7: Model Explainability Setup (SageMaker Clarify)
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: MODEL EXPLAINABILITY WITH SAGEMAKER CLARIFY")
    logger.info("=" * 70)

    logger.info("\nSHAP (SHapley Additive exPlanations) Values:")
    logger.info("  - Feature importance: Which features drive predictions?")
    logger.info("  - Local explanations: Why this specific prediction?")
    logger.info("  - Global explanations: Average feature impact")

    logger.info("\nClarify Explainability Configuration:")

    region = "<YOUR-REGION>"
    role_arn = "<YOUR-ROLE-ARN>"
    bucket = "<YOUR-BUCKET-NAME>"

    # SHAP Configuration
    shap_config = SHAPConfig(
        # num_clusters: Clarify computes SHAP baseline by clustering the dataset
        # Higher values = more accurate baseline, slower computation
        num_clusters=1,
        # agg_method: How to aggregate SHAP values across samples
        agg_method="mean_abs",  # mean_abs or median
    )

    logger.info(f"SHAP Configuration:")
    logger.info(f"  Num clusters for baseline: 1")
    logger.info(f"  Aggregation: mean_abs")

    # Model Configuration (for endpoint-based explanation)
    model_config = ModelConfig(
        model_name="churn-prediction-model",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept_type="text/csv",
        content_type="text/csv",
    )

    logger.info(f"\nModel Configuration:")
    logger.info(f"  Model: churn-prediction-model")
    logger.info(f"  Instance: ml.m5.xlarge")
    logger.info(f"  Content Type: text/csv")

    # ========================================================================
    # STEP 8: Feature Importance (Simulated)
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: FEATURE IMPORTANCE")
    logger.info("=" * 70)

    # Simulated feature importance from SHAP
    features = ["Age", "Tenure", "MonthlyCharges", "TotalCharges", "NumServices"]
    importances = np.array([0.25, 0.20, 0.30, 0.15, 0.10])  # Simulated
    sorted_idx = np.argsort(importances)[::-1]

    logger.info("\nFeature Importance (SHAP mean absolute values):")
    for rank, idx in enumerate(sorted_idx, 1):
        logger.info(f"  {rank}. {features[idx]}: {importances[idx]:.4f}")

    logger.info("\nInterpretation:")
    logger.info("  - Monthly Charges: Strongest predictor of churn (30% contribution)")
    logger.info("  - Age: Second most important (25% contribution)")
    logger.info("  - Tenure: Customer loyalty indicator (20% contribution)")

    # ========================================================================
    # STEP 9: Evaluation Summary
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION SUMMARY")
    logger.info("=" * 70)

    logger.info("\nClassification Performance:")
    logger.info(f"  Accuracy:  {accuracy:.4f} (overall correctness)")
    logger.info(f"  Precision: {precision:.4f} (confidence in positive predictions)")
    logger.info(f"  Recall:    {recall:.4f} (finding positive cases)")
    logger.info(f"  F1-Score:  {f1:.4f} (balanced performance)")
    logger.info(f"  AUC-ROC:   {auc_roc:.4f} (ranking quality)")

    logger.info("\nModel Fit Assessment:")
    if accuracy > 0.85 and recall > 0.80:
        logger.info("  ✓ Model shows strong performance")
    elif accuracy > 0.75:
        logger.info("  ✓ Model shows acceptable performance")
    else:
        logger.warning("  ✗ Model needs improvement - consider retraining")

    # ========================================================================
    # STEP 10: Recommendations
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    logger.info("=" * 70)

    logger.info("\n1. Address Class Imbalance (if applicable):")
    logger.info("   - Use stratified k-fold cross-validation")
    logger.info("   - Apply SMOTE or other resampling")
    logger.info("   - Adjust class weights in loss function")

    logger.info("\n2. Threshold Optimization:")
    logger.info(f"   - Current threshold: 0.5")
    logger.info(f"   - Consider optimal threshold: {optimal_threshold:.3f}")
    logger.info("   - Adjust based on business requirements")

    logger.info("\n3. Feature Engineering:")
    logger.info("   - Focus on high-importance features")
    logger.info("   - Create interaction features")
    logger.info("   - Remove low-importance features")

    logger.info("\n4. Model Deployment:")
    logger.info("   - Set up Model Monitor for performance tracking")
    logger.info("   - Implement automated retraining pipelines")
    logger.info("   - Use A/B testing for model updates")

    logger.info("\n5. Continuous Monitoring:")
    logger.info("   - Track metrics daily in production")
    logger.info("   - Alert on performance degradation")
    logger.info("   - Document model decisions via SHAP")

    logger.info("=" * 70)
    logger.info("Model evaluation completed successfully!")

except Exception as e:
    logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
    raise
