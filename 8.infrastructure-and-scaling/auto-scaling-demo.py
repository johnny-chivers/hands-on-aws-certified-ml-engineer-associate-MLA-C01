"""
Auto-Scaling for SageMaker Endpoints Demonstration

MLA-C01 Exam Relevance:
- Task 3.2: Create and manage SageMaker endpoints with scaling policies
- Task 4.2: Optimise cost and performance of deployed ML solutions
- Target tracking, step scaling, and scheduled scaling policies
- Choosing appropriate scaling metrics (InvocationsPerInstance, CPUUtilization)

This demo shows how to:
1. Register a SageMaker endpoint variant as a scalable target
2. Configure target-tracking scaling (automatic)
3. Configure step scaling (manual thresholds)
4. Configure scheduled scaling (time-based)
5. Test and inspect scaling behaviour
6. Clean up scaling resources
"""

import boto3
import json
import time
import logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AWS_REGION = "<YOUR-AWS-REGION>"
ENDPOINT_NAME = "<YOUR-ENDPOINT-NAME>"     # Must be an InService real-time endpoint
VARIANT_NAME = "primary-variant"           # Match the variant in your endpoint config

# Scaling limits
MIN_INSTANCE_COUNT = 1
MAX_INSTANCE_COUNT = 4

# Resource ID format required by Application Auto Scaling
RESOURCE_ID = f"endpoint/{ENDPOINT_NAME}/variant/{VARIANT_NAME}"

# ---------------------------------------------------------------------------
# AWS Client
# ---------------------------------------------------------------------------
autoscaling_client = boto3.client("application-autoscaling", region_name=AWS_REGION)
cloudwatch_client = boto3.client("cloudwatch", region_name=AWS_REGION)


# ===================================================================
# STEP 1 — Register the Endpoint Variant as a Scalable Target
# ===================================================================
def register_scalable_target():
    """
    Tell Application Auto Scaling about your SageMaker endpoint variant.

    MLA-C01 Exam Tip:
    - ServiceNamespace: "sagemaker"
    - ScalableDimension: "sagemaker:variant:DesiredInstanceCount"
    - MinCapacity / MaxCapacity define the scaling boundaries.
    """
    logger.info("Registering scalable target for %s …", RESOURCE_ID)

    autoscaling_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=MIN_INSTANCE_COUNT,
        MaxCapacity=MAX_INSTANCE_COUNT,
    )
    logger.info(
        "Registered: min=%d, max=%d instances.", MIN_INSTANCE_COUNT, MAX_INSTANCE_COUNT
    )


# ===================================================================
# STEP 2 — Target Tracking Scaling Policy
# ===================================================================
def create_target_tracking_policy():
    """
    Automatically adjust instance count to maintain a target metric value.

    MLA-C01 Exam Tip:
    ──────────────────
    SageMakerVariantInvocationsPerInstance is the most common metric.
    Target value = how many invocations/min per instance before scaling out.

    Other predefined metrics:
    - SageMakerVariantProvisionedConcurrencyUtilization (serverless)

    Custom metrics:
    - CPUUtilization, GPUUtilization, MemoryUtilization via CloudWatch

    ScaleInCooldown / ScaleOutCooldown:
    - Prevents thrashing. Exam may test default values (300s / 300s).
    """
    logger.info("Creating target-tracking scaling policy …")

    response = autoscaling_client.put_scaling_policy(
        PolicyName="mla-c01-target-tracking-policy",
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 70.0,  # Scale when invocations/instance exceed 70/min
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 300,   # Wait 5 min before scaling in
            "ScaleOutCooldown": 60,   # Wait 1 min before scaling out again
        },
    )
    logger.info("Target-tracking policy created: %s", response["PolicyARN"])
    return response


# ===================================================================
# STEP 3 — Step Scaling Policy
# ===================================================================
def create_step_scaling_policy():
    """
    Define explicit scaling steps based on alarm breach thresholds.

    MLA-C01 Exam Tip:
    - StepScaling gives you fine-grained control over how many instances
      to add/remove at specific breach levels.
    - You must create a CloudWatch alarm separately and link it.
    - Less common in exam than target-tracking, but tested in scenarios
      where custom thresholds are needed.
    """
    logger.info("Creating step scaling policy …")

    # First, create the step scaling policy
    response = autoscaling_client.put_scaling_policy(
        PolicyName="mla-c01-step-scaling-policy",
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "StepAdjustments": [
                {
                    # 0-50% above threshold → add 1 instance
                    "MetricIntervalLowerBound": 0,
                    "MetricIntervalUpperBound": 50,
                    "ScalingAdjustment": 1,
                },
                {
                    # >50% above threshold → add 2 instances
                    "MetricIntervalLowerBound": 50,
                    "ScalingAdjustment": 2,
                },
            ],
            "Cooldown": 120,
        },
    )
    policy_arn = response["PolicyARN"]
    logger.info("Step scaling policy created: %s", policy_arn)

    # Now create a CloudWatch alarm that triggers the step policy
    cloudwatch_client.put_metric_alarm(
        AlarmName="mla-c01-high-invocations-alarm",
        Namespace="AWS/SageMaker",
        MetricName="InvocationsPerInstance",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": VARIANT_NAME},
        ],
        Statistic="Average",
        Period=60,
        EvaluationPeriods=2,
        Threshold=100.0,
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=[policy_arn],
    )
    logger.info("CloudWatch alarm created and linked to step policy.")
    return response


# ===================================================================
# STEP 4 — Scheduled Scaling
# ===================================================================
def create_scheduled_scaling():
    """
    Pre-scale the endpoint at known high-traffic times.

    MLA-C01 Exam Tip:
    - Use scheduled scaling when you know traffic patterns in advance.
    - Combine with target-tracking for both predictable and unpredictable spikes.
    - Schedule uses cron or at() syntax.
    """
    logger.info("Creating scheduled scaling actions …")

    # Scale up to 3 instances at 8 AM UTC on weekdays
    autoscaling_client.put_scheduled_action(
        ServiceNamespace="sagemaker",
        ScheduledActionName="mla-c01-scale-up-morning",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        Schedule="cron(0 8 ? * MON-FRI *)",
        ScalableTargetAction={
            "MinCapacity": 3,
            "MaxCapacity": MAX_INSTANCE_COUNT,
        },
    )
    logger.info("  Morning scale-up scheduled (3 instances at 8 AM UTC, Mon-Fri)")

    # Scale down to 1 instance at 8 PM UTC on weekdays
    autoscaling_client.put_scheduled_action(
        ServiceNamespace="sagemaker",
        ScheduledActionName="mla-c01-scale-down-evening",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        Schedule="cron(0 20 ? * MON-FRI *)",
        ScalableTargetAction={
            "MinCapacity": MIN_INSTANCE_COUNT,
            "MaxCapacity": 2,
        },
    )
    logger.info("  Evening scale-down scheduled (1 instance at 8 PM UTC, Mon-Fri)")


# ===================================================================
# STEP 5 — Inspect Current Scaling Configuration
# ===================================================================
def describe_scaling_config():
    """List all policies and scheduled actions for this endpoint."""
    logger.info("\n--- Current Scaling Configuration ---")

    # Scalable target
    targets = autoscaling_client.describe_scalable_targets(
        ServiceNamespace="sagemaker",
        ResourceIds=[RESOURCE_ID],
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    )
    for t in targets.get("ScalableTargets", []):
        logger.info(
            "Target: min=%d, max=%d, role=%s",
            t["MinCapacity"],
            t["MaxCapacity"],
            t.get("RoleARN", "N/A"),
        )

    # Policies
    policies = autoscaling_client.describe_scaling_policies(
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    )
    for p in policies.get("ScalingPolicies", []):
        logger.info("  Policy: %-40s  Type: %s", p["PolicyName"], p["PolicyType"])

    # Scheduled actions
    actions = autoscaling_client.describe_scheduled_actions(
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    )
    for a in actions.get("ScheduledActions", []):
        logger.info(
            "  Scheduled: %-40s  Cron: %s",
            a["ScheduledActionName"],
            a.get("Schedule", "N/A"),
        )


# ===================================================================
# STEP 6 — Scaling Activity History
# ===================================================================
def show_scaling_activities():
    """Show recent scaling activities (useful for debugging)."""
    logger.info("\n--- Recent Scaling Activities ---")

    activities = autoscaling_client.describe_scaling_activities(
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MaxResults=5,
    )
    for act in activities.get("ScalingActivities", []):
        logger.info(
            "  %s | Status: %s | Cause: %s",
            act.get("StartTime", ""),
            act.get("StatusCode", ""),
            act.get("StatusMessage", "")[:100],
        )

    if not activities.get("ScalingActivities"):
        logger.info("  No scaling activities recorded yet.")


# ===================================================================
# STEP 7 — Clean Up
# ===================================================================
def cleanup():
    """Remove all scaling policies, scheduled actions, and deregister the target."""
    logger.info("\nCleaning up auto-scaling resources …")

    # Delete policies
    for policy_name in ["mla-c01-target-tracking-policy", "mla-c01-step-scaling-policy"]:
        try:
            autoscaling_client.delete_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace="sagemaker",
                ResourceId=RESOURCE_ID,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )
            logger.info("  Deleted policy: %s", policy_name)
        except Exception as e:
            logger.warning("  Could not delete policy %s: %s", policy_name, e)

    # Delete scheduled actions
    for action_name in ["mla-c01-scale-up-morning", "mla-c01-scale-down-evening"]:
        try:
            autoscaling_client.delete_scheduled_action(
                ServiceNamespace="sagemaker",
                ScheduledActionName=action_name,
                ResourceId=RESOURCE_ID,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )
            logger.info("  Deleted scheduled action: %s", action_name)
        except Exception as e:
            logger.warning("  Could not delete action %s: %s", action_name, e)

    # Delete CloudWatch alarm
    try:
        cloudwatch_client.delete_alarms(AlarmNames=["mla-c01-high-invocations-alarm"])
        logger.info("  Deleted CloudWatch alarm.")
    except Exception as e:
        logger.warning("  Could not delete alarm: %s", e)

    # Deregister target
    try:
        autoscaling_client.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=RESOURCE_ID,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
        logger.info("  Deregistered scalable target.")
    except Exception as e:
        logger.warning("  Could not deregister target: %s", e)

    logger.info("Auto-scaling cleanup complete.")


# ===================================================================
# Main
# ===================================================================
def main():
    logger.info("=== MLA-C01: Auto-Scaling Demo ===\n")

    # 1. Register the endpoint variant
    register_scalable_target()

    # 2. Target tracking (most common on exam)
    create_target_tracking_policy()

    # 3. Step scaling (for custom thresholds)
    create_step_scaling_policy()

    # 4. Scheduled scaling (for predictable traffic)
    create_scheduled_scaling()

    # 5. Inspect
    describe_scaling_config()
    show_scaling_activities()

    # 6. Cleanup (uncomment when done recording)
    # cleanup()

    logger.info("\n=== Auto-Scaling Demo Complete ===")


if __name__ == "__main__":
    main()
