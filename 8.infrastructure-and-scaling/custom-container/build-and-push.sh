#!/bin/bash

# Build and Push Custom SageMaker Container to ECR
# Usage: ./build-and-push.sh <IMAGE_NAME> <AWS_ACCOUNT_ID> <AWS_REGION>

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <IMAGE_NAME> <AWS_ACCOUNT_ID> <AWS_REGION>"
    exit 1
fi

IMAGE_NAME=$1
AWS_ACCOUNT_ID=$2
AWS_REGION=$3

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE_URI="${ECR_REGISTRY}/${IMAGE_NAME}:latest"

echo "Building and pushing Docker image to ECR"
echo "Image URI: ${FULL_IMAGE_URI}"

# Authenticate with ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Create ECR repository if needed
aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${AWS_REGION} 2>/dev/null || true

# Build image
docker build -t ${FULL_IMAGE_URI} .

# Push image
docker push ${FULL_IMAGE_URI}

echo "Image pushed successfully: ${FULL_IMAGE_URI}"
