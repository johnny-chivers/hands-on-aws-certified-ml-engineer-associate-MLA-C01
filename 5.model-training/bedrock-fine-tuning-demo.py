"""
AWS MLA-C01: Amazon Bedrock Foundation Model Fine-Tuning Demo

This script demonstrates working with Amazon Bedrock:
  1. Invoke foundation model using bedrock-runtime client
  2. Generate predictions with different prompts
  3. Prepare JSONL training data for model customization
  4. Create fine-tuning job using Bedrock API
  5. Monitor job status and retrieve customized model ARN
  6. Use fine-tuned model for inference

Key MLA-C01 Concepts:
  - Foundation Models: Large pre-trained models (Claude, Llama, Mistral)
  - Bedrock: Managed service for accessing multiple foundation models
  - Model Customization: Fine-tune on domain-specific data
  - Embedding Models: Generate vector embeddings for semantic search
  - RAG (Retrieval-Augmented Generation): Combine models with custom knowledge
  - When to Use Bedrock vs. SageMaker vs. Fine-tuning from scratch:
    * Bedrock: Quick prototyping, multiple model options
    * SageMaker JumpStart: More control, fine-tuning capabilities
    * Custom Training: Full control, expensive (GPUs)
"""

import logging
import boto3
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing Amazon Bedrock Fine-Tuning Demo...")

try:
    # Step 1: Initialize Bedrock Clients
    # bedrock: Manage models, jobs, customizations
    # bedrock-runtime: Invoke models for inference
    region = "us-east-1"  # Bedrock available in limited regions
    bedrock_client = boto3.client("bedrock", region_name=region)
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

    logger.info(f"Bedrock clients initialized for region: {region}")

    # Step 2: List Available Foundation Models
    # Bedrock provides access to multiple providers' models
    logger.info("\n" + "=" * 70)
    logger.info("AVAILABLE FOUNDATION MODELS IN BEDROCK")
    logger.info("=" * 70)

    try:
        models_response = bedrock_client.list_foundation_models()
        logger.info(f"Total available models: {len(models_response.get('modelSummaries', []))}")

        # Show sample models
        for model in models_response.get("modelSummaries", [])[:5]:
            logger.info(f"  - {model['modelId']}: {model['modelName']}")
    except Exception as e:
        logger.info(f"Could not list models: {e}")
        logger.info("Using default models for demonstration")

    # Step 3: Part 1 - Invoke Foundation Model (Inference)
    # Demonstrate basic model invocation
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: INVOKE FOUNDATION MODEL")
    logger.info("=" * 70)

    # Using Claude model as example (Anthropic's foundation model in Bedrock)
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    # Alternative models:
    # "anthropic.claude-3-opus-20240229-v1:0"
    # "meta.llama2-70b-chat-v1"
    # "mistral.mistral-7b-instruct-v0:2"

    logger.info(f"\nUsing model: {model_id}")

    # Example prompts for inference
    test_prompts = [
        "What is machine learning?",
        "Explain AWS SageMaker in 2 sentences.",
        "What are the benefits of cloud computing?",
    ]

    logger.info("\nRunning inference on test prompts:")

    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nPrompt {i}: {prompt}")

        try:
            # For Claude models, use specific message format
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-06-01",
                    "max_tokens": 256,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                }),
            )

            # Parse response
            response_body = json.loads(response["body"].read())
            generated_text = response_body["content"][0]["text"]

            logger.info(f"Response: {generated_text[:200]}...")

        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            logger.info("(This is expected if Bedrock access is not enabled)")

    # Step 4: Part 2 - Prepare Training Data for Fine-tuning
    # Fine-tuning data format: JSONL (JSON Lines)
    # Each line: {"prompt": "...", "completion": "..."}
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: PREPARE FINE-TUNING TRAINING DATA")
    logger.info("=" * 70)

    # Example: Domain-specific Q&A data (e.g., AWS ML certification Q&A)
    training_data = [
        {
            "prompt": "What is AWS SageMaker used for?",
            "completion": " AWS SageMaker is a fully managed machine learning service for building, training, and deploying ML models at scale."
        },
        {
            "prompt": "What are the main components of SageMaker?",
            "completion": " SageMaker components include: Notebooks for development, Training for distributed training, Processing for data processing, Endpoints for real-time inference, and Pipelines for MLOps."
        },
        {
            "prompt": "How does SageMaker handle hyperparameter tuning?",
            "completion": " SageMaker uses Automatic Model Tuning with Bayesian optimization to search hyperparameter space efficiently, testing multiple combinations in parallel."
        },
        {
            "prompt": "What is model bias in machine learning?",
            "completion": " Model bias occurs when a model makes systematically different predictions for different groups. SageMaker Clarify detects and measures bias using metrics like DPL and KL divergence."
        },
        {
            "prompt": "How do you deploy models on SageMaker?",
            "completion": " Models can be deployed via: Real-time endpoints for synchronous predictions, Batch Transform for offline processing, Async Inference for large-scale predictions, or Serverless endpoints for variable workloads."
        },
        {
            "prompt": "What is the purpose of SageMaker Data Wrangler?",
            "completion": " Data Wrangler provides a visual interface for data preparation, including imputation, encoding, scaling, and feature engineering without writing code."
        },
        {
            "prompt": "How does SageMaker Clarify explain model predictions?",
            "completion": " Clarify uses SHAP (SHapley Additive exPlanations) values to compute feature importance and explains individual predictions by showing which features contributed most."
        },
        {
            "prompt": "What is A/B testing in SageMaker?",
            "completion": " A/B testing (also called traffic splitting) allows deploying multiple model variants to endpoints and gradually shifting traffic between versions to validate improvements."
        },
    ]

    # Save training data to JSONL format
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    training_data_file = f"/tmp/bedrock-training-data-{timestamp}.jsonl"

    with open(training_data_file, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Created training data file: {training_data_file}")
    logger.info(f"Training examples: {len(training_data)}")

    # Show sample data
    logger.info("\nSample training data:")
    for i, example in enumerate(training_data[:2], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Prompt: {example['prompt']}")
        logger.info(f"  Completion: {example['completion']}")

    # Step 5: Part 3 - Create Fine-tuning Job
    # Upload training data to S3 first (required)
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: CREATE FINE-TUNING JOB")
    logger.info("=" * 70)

    # In production: upload training data to S3
    # For demo: show the structure
    bucket = "<YOUR-BUCKET-NAME>"
    s3_training_data_path = f"s3://{bucket}/bedrock-training-data/training-data-{timestamp}.jsonl"

    logger.info(f"\nIn production, upload training data to S3:")
    logger.info(f"  S3 path: {s3_training_data_path}")

    # Create fine-tuning job configuration
    fine_tuning_job_config = {
        "baseModelIdentifier": model_id,  # Base model to fine-tune
        "trainingDataConfig": {
            "s3Uri": s3_training_data_path,  # Training data S3 path
            "contentType": "application/jsonl",
        },
        "outputDataConfig": {
            "s3Uri": f"s3://{bucket}/bedrock-models/"  # Where to save fine-tuned model
        },
        # Optional: validation data for monitoring
        "validationDataConfig": {
            "validators": [
                {
                    "s3Uri": f"s3://{bucket}/bedrock-training-data/validation.jsonl"
                }
            ]
        },
        "roleArn": "<YOUR-ROLE-ARN>",  # IAM role with S3 and Bedrock permissions
        "trainingParameters": {
            "epochCount": "3",  # Number of training passes over data
            "batchSize": "4",  # Batch size for training
            "learningRate": "0.0001",  # Learning rate for fine-tuning
        },
    }

    logger.info("\nFine-tuning job configuration:")
    logger.info(f"  Base model: {fine_tuning_job_config['baseModelIdentifier']}")
    logger.info(f"  Training data: {fine_tuning_job_config['trainingDataConfig']['s3Uri']}")
    logger.info(f"  Output path: {fine_tuning_job_config['outputDataConfig']['s3Uri']}")
    logger.info(f"  Epochs: {fine_tuning_job_config['trainingParameters']['epochCount']}")
    logger.info(f"  Batch size: {fine_tuning_job_config['trainingParameters']['batchSize']}")

    # Create the fine-tuning job
    # Note: This requires Bedrock access and valid S3 paths
    job_name = f"bedrock-fine-tune-{timestamp}"

    logger.info(f"\nTo create fine-tuning job (requires enabled Bedrock access):")
    logger.info(f"  bedrock_client.create_model_customization_job(")
    logger.info(f"      jobName='{job_name}',")
    logger.info(f"      customizationType='FINE_TUNING',")
    logger.info(f"      roleArn='<YOUR-ROLE-ARN>',")
    logger.info(f"      baseModelIdentifier='{model_id}',")
    logger.info(f"      trainingDataConfig=...,")
    logger.info(f"      outputDataConfig=...,")
    logger.info(f"  )")

    # Try to create (will fail without proper setup, but shows the API)
    try:
        response = bedrock_client.create_model_customization_job(
            jobName=job_name,
            customizationType="FINE_TUNING",
            roleArn="<YOUR-ROLE-ARN>",  # Replace with actual role
            baseModelIdentifier=model_id,
            trainingDataConfig=fine_tuning_job_config["trainingDataConfig"],
            outputDataConfig=fine_tuning_job_config["outputDataConfig"],
            hyperParameters={
                "epochCount": fine_tuning_job_config["trainingParameters"]["epochCount"],
                "batchSize": fine_tuning_job_config["trainingParameters"]["batchSize"],
                "learningRate": fine_tuning_job_config["trainingParameters"]["learningRate"],
            },
        )

        job_arn = response["jobArn"]
        logger.info(f"✓ Fine-tuning job created: {job_arn}")

    except Exception as e:
        logger.info(f"Fine-tuning job creation requires proper setup: {str(e)[:100]}")

    # Step 6: Part 4 - Monitor Fine-tuning Job Status
    logger.info("\n" + "=" * 70)
    logger.info("PART 4: MONITOR FINE-TUNING JOB")
    logger.info("=" * 70)

    logger.info("\nTo check job status:")
    logger.info(f"  response = bedrock_client.get_model_customization_job(jobIdentifier='{job_name}')")
    logger.info(f"  status = response['status']  # InProgress, Completed, Failed")
    logger.info(f"  customized_model_arn = response['outputModelArn']  # ARN of fine-tuned model")

    # Step 7: Training Duration & Cost Estimation
    logger.info("\n" + "=" * 70)
    logger.info("FINE-TUNING DURATION & COST")
    logger.info("=" * 70)

    logger.info("\nEstimated Duration (depends on model size and data):")
    logger.info("  - Small models (7B): 30 minutes - 2 hours")
    logger.info("  - Medium models (70B): 2-8 hours")
    logger.info("  - Large models (405B+): 8-24 hours")

    logger.info("\nCost Factors:")
    logger.info("  - Model size: Larger models = higher cost")
    logger.info("  - Training data: More data = longer training")
    logger.info("  - Compute: GPU instances required (on-demand)")

    logger.info("\nCost Optimization:")
    logger.info("  - Use smaller models for domain adaptation")
    logger.info("  - Batch multiple training jobs")
    logger.info("  - Start with small dataset, expand if needed")

    # Step 8: When to Use Bedrock vs. Alternatives
    logger.info("\n" + "=" * 70)
    logger.info("BEDROCK VS. SAGEMAKER VS. CUSTOM TRAINING")
    logger.info("=" * 70)

    comparison = {
        "Amazon Bedrock": {
            "Best For": "Quick prototyping, multiple model options, RAG applications",
            "Pros": [
                "Multiple foundation models (Claude, Llama, Mistral)",
                "Managed service (no infrastructure management)",
                "Fast fine-tuning on pre-trained models",
                "Built-in safety & compliance features",
            ],
            "Cons": [
                "Limited customization options",
                "Higher per-inference cost vs. self-hosted",
                "Bounded to AWS region availability",
            ],
            "Cost": "Pay per inference (tokens) + fine-tuning compute"
        },
        "SageMaker JumpStart": {
            "Best For": "Transfer learning, quick model deployments",
            "Pros": [
                "Pre-trained models from Hugging Face",
                "Easy fine-tuning interface",
                "Deploy on SageMaker endpoints",
                "Lower cost for inference at scale",
            ],
            "Cons": [
                "Fewer proprietary models",
                "Need to manage training infrastructure",
            ],
            "Cost": "Training compute (instances) + endpoint hosting"
        },
        "Custom Training (SageMaker/DIY)": {
            "Best For": "Custom architectures, full control, proprietary algorithms",
            "Pros": [
                "Complete control over training process",
                "Use any framework (PyTorch, TensorFlow, etc.)",
                "Optimize for specific use cases",
                "No vendor lock-in",
            ],
            "Cons": [
                "Requires ML expertise",
                "High GPU costs",
                "Infrastructure management overhead",
                "Longer development time",
            ],
            "Cost": "GPU instances (expensive), storage, engineering time"
        }
    }

    for option, details in comparison.items():
        logger.info(f"\n{option}:")
        logger.info(f"  Best For: {details['Best For']}")
        logger.info(f"  Cost: {details['Cost']}")
        logger.info(f"  Pros:")
        for pro in details['Pros']:
            logger.info(f"    + {pro}")
        logger.info(f"  Cons:")
        for con in details['Cons']:
            logger.info(f"    - {con}")

    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 70)

    logger.info("\nChoose Bedrock if:")
    logger.info("  ✓ Need multiple model options")
    logger.info("  ✓ Want fully managed service")
    logger.info("  ✓ Building RAG or generative AI applications")
    logger.info("  ✓ Want minimal infrastructure overhead")

    logger.info("\nChoose SageMaker JumpStart if:")
    logger.info("  ✓ Need transfer learning on standard models")
    logger.info("  ✓ Want endpoint-based serving")
    logger.info("  ✓ Expect high prediction volume (lower cost)")

    logger.info("\nChoose Custom Training if:")
    logger.info("  ✓ Need proprietary algorithms")
    logger.info("  ✓ Have specific performance requirements")
    logger.info("  ✓ Building competitive differentiation")

    logger.info("=" * 70)
    logger.info("Bedrock fine-tuning demo completed!")

except Exception as e:
    logger.error(f"Error in Bedrock demo: {str(e)}", exc_info=True)
    raise
