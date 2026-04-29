# Hands On AWS Certified Machine Learning Engineer - Associate (MLA-C01) Full Course

## Disclaimer

This course, youtube video, revision/exam guide, and/or slide deck for the AWS Certified Machine Learning Engineer - Associate certification is intended as a supplementary resource to aid in your preparation for the certification exam. While it is designed to help you identify and address knowledge gaps, it does not guarantee a passing score. Success on the exam depends on your understanding of the material, practical experience, and familiarity with AWS services and best practices. We recommend using this guide alongside other study materials and hands-on practice to enhance your readiness for the certification exam.

## Introduction

This github contains the code for the [youtube video](https://www.youtube.com/watch?v=REPLACE_WITH_VIDEO_ID). The youtube video acts as a revision aid for the AWS Certified Machine Learning Engineer - Associate (MLA-C01) certification exam. The repo contains the code alongside a detailed README that gives definitions of the AWS services and other areas of knowledge required for the certification.

[AWS Machine Learning Engineer - Associate MLA-C01 Revision Guide](https://johnny-chivers.kit.com/67c523d155)

[Knowza.ai Practice Exams](https://knowza.ai)

## Setup

The `setup-code.yaml` contains code to be executed using Amazon CloudFormation. The code creates the base networking artefacts, S3 Bucket, SageMaker execution role, and SageMaker Studio Domain required to complete the remaining sections of the course.

The following artefacts are created by the code:

- VPC with two public subnets, Internet Gateway, and route table
- Security Groups (NFS for EFS, inter-app communication)
- S3 Bucket (versioned, encrypted)
- SageMaker Execution Role (with Feature Store, S3, Glue, ECR, CloudWatch policies)
- SageMaker Studio Domain (with `PublicInternetOnly` network access)
- SageMaker User Profile

After the stack has been created, note the following values from the **Outputs** tab in CloudFormation — you will need them throughout the course:

- **S3BucketName** — the S3 bucket for data and model artifacts
- **SageMakerExecutionRoleArn** — the IAM execution role ARN
- **SageMakerDomainId** — the Studio Domain ID
- **SageMakerUserProfileName** — the user profile name

Then open SageMaker Studio:

1. In the AWS Console, go to **Amazon SageMaker AI** → **Applications and IDEs** → **SageMaker Studio**
2. Select the domain and user profile created by the stack
3. Click **Open Studio**
4. Once in Studio, open a **JupyterLab** space (create one if prompted — use `ml.t3.medium`)

Clone this repo from the JupyterLab terminal:

```bash
git clone https://github.com/johnny-chivers/hands-on-aws-certified-ml-engineer-associate-MLA-C01.git
cd hands-on-aws-certified-ml-engineer-associate-MLA-C01
```

Update the placeholder values in the demo scripts with your actual AWS resource names. You can do this globally from the terminal:

```bash
find . -name "*.py" -exec sed -i 's/<YOUR-BUCKET-NAME>/YOUR_ACTUAL_BUCKET/g' {} +
find . -name "*.py" -exec sed -i 's/<YOUR-REGION>/YOUR_ACTUAL_REGION/g' {} +
find . -name "*.py" -exec sed -i 's|<YOUR-SAGEMAKER-ROLE-ARN>|YOUR_ACTUAL_ROLE_ARN|g' {} +
```

**Important:** SageMaker Studio JupyterLab defaults to `/home/sagemaker-user` as the working directory. Before running any demo scripts in a notebook, set the working directory to the repo root:

```python
import os
os.chdir('/home/sagemaker-user/hands-on-aws-certified-ml-engineer-associate-MLA-C01')
```
**Important:** Code to get model uris
```python
import sagemaker.image_uris
image = sagemaker.image_uris.retrieve("xgboost", "us-east-1", version="1.5-1")
print(image)
```

No manual S3 folder creation or data upload is needed — each demo script handles its own S3 uploads automatically.

To view Feature Store, Model Registry, Endpoints, and other resources visually in the console, use the Studio left navigation: **Data** → **Feature Store**, **Models** → **Model Registry**, etc.

## Data

Below are the datasets used throughout the course.

### Housing Dataset

Used for regression tasks (predicting median house value). Based on the California Housing dataset.

| Column | Type | Description |
|--------|------|-------------|
| longitude | float | Longitude coordinate |
| latitude | float | Latitude coordinate |
| housing_median_age | int | Median age of houses in block |
| total_rooms | int | Total number of rooms in block |
| total_bedrooms | int | Total number of bedrooms in block |
| population | int | Total population in block |
| households | int | Total households in block |
| median_income | float | Median income (tens of thousands) |
| median_house_value | float | Median house value ($) |
| ocean_proximity | string | Location relative to ocean |

### Customer Churn Dataset

Used for classification tasks (predicting customer churn) and bias detection demos.

| Column | Type | Description |
|--------|------|-------------|
| customerID | string | Unique customer identifier |
| gender | string | Male/Female |
| SeniorCitizen | int | 1 if senior citizen, 0 if not |
| Partner | string | Has partner (Yes/No) |
| Dependents | string | Has dependents (Yes/No) |
| tenure | int | Number of months as customer |
| PhoneService | string | Has phone service (Yes/No) |
| MultipleLines | string | Has multiple lines |
| InternetService | string | Internet service type |
| OnlineSecurity | string | Has online security (Yes/No) |
| TechSupport | string | Has tech support (Yes/No) |
| Contract | string | Contract type |
| PaperlessBilling | string | Paperless billing (Yes/No) |
| PaymentMethod | string | Payment method |
| MonthlyCharges | float | Monthly charge amount |
| TotalCharges | float | Total charges to date |
| Churn | string | Customer churned (Yes/No) |

## Exam Domains

The MLA-C01 exam has four domains. This repo is structured to cover all of them with hands-on code.

| Domain | Weight | Repo Folders |
|--------|--------|-------------|
| Domain 1: Data Preparation for Machine Learning | 28% | 2, 3, 4 |
| Domain 2: ML Model Development | 26% | 5, 6 |
| Domain 3: Deployment and Orchestration of ML Workflows | 22% | 7, 8, 9 |
| Domain 4: ML Solution Monitoring, Maintenance, and Security | 24% | 10, 11 |

## Repo Structure

```
hands-on-aws-certified-ml-engineer-associate-MLA-C01/
|
├── 0.source-data/                    # Sample datasets
│   ├── housing/                      # Housing regression dataset
│   └── customer-churn/               # Churn classification dataset
|
├── 1.setup-code/                     # CloudFormation setup
│   └── setup-code.yaml               # VPC, S3, SageMaker Studio Domain, IAM role
|
├── 2.data-ingestion-and-storage/     # Domain 1 - Task 1.1
│   └── data-ingestion-demo.py        # S3 upload/download, Feature Store
|
├── 3.data-transformation/            # Domain 1 - Task 1.2
│   ├── data-transformation-demo.py   # Cleaning, feature engineering, encoding
│   └── glue-etl-job.py              # AWS Glue ETL (CSV to Parquet)
|
├── 4.data-integrity-and-bias/        # Domain 1 - Task 1.3
│   ├── bias-detection-demo.py        # SageMaker Clarify bias analysis
│   └── data-quality-demo.py          # Data validation and quality checks
|
├── 5.model-training/                 # Domain 2 - Tasks 2.1, 2.2
│   ├── xgboost-builtin-training.py   # SageMaker built-in XGBoost
│   ├── pytorch-script-mode/          # Custom PyTorch training
│   │   ├── train.py                  # Training script
│   │   └── launch-training.py        # SageMaker launcher
│   ├── hyperparameter-tuning-demo.py # Automatic Model Tuning
│   ├── bedrock-fine-tuning-demo.py   # Amazon Bedrock fine-tuning
│   └── model-registry-demo.py        # Model versioning
|
├── 6.model-evaluation/               # Domain 2 - Task 2.3
│   └── model-evaluation-demo.py      # Metrics, Clarify explainability
|
├── 7.deployment-endpoints/           # Domain 3 - Task 3.1
│   ├── realtime-endpoint-demo.py     # Real-time inference
│   ├── serverless-endpoint-demo.py   # Serverless inference
│   └── batch-transform-demo.py       # Batch transform
|
├── 8.infrastructure-and-scaling/     # Domain 3 - Task 3.2
│   ├── auto-scaling-demo.py          # Endpoint auto scaling
│   ├── custom-container/             # BYOC for SageMaker
│   │   ├── Dockerfile
│   │   ├── serve.py
│   │   └── build-and-push.sh
│   └── vpc-endpoint-setup.yaml       # VPC endpoints for SageMaker
|
├── 9.cicd-pipelines/                 # Domain 3 - Task 3.3
│   ├── sagemaker-pipeline-demo.py    # Full SageMaker Pipeline
│   ├── blue-green-deployment.yaml    # Blue/green deployment CFN
│   └── eventbridge-trigger.yaml      # Automated pipeline triggers
|
├── 10.monitoring/                    # Domain 4 - Tasks 4.1, 4.2
│   ├── model-monitor-demo.py         # SageMaker Model Monitor
│   ├── cloudwatch-dashboard.yaml     # Metrics dashboard + alarms
│   └── inference-recommender-demo.py # Instance right-sizing
|
├── 11.security/                      # Domain 4 - Task 4.3
│   ├── iam-policies.yaml             # Least privilege IAM policies
│   └── vpc-isolation-demo.yaml       # Network isolation
|
├── images/                           # Architecture diagrams
├── LICENSE
└── README.md
```

## AWS Services Covered

### Amazon SageMaker
Amazon SageMaker is a fully managed machine learning service. It provides tools to build, train, and deploy ML models at scale. Key features covered in this course include: SageMaker Studio notebooks, built-in algorithms (XGBoost, Linear Learner), script mode training, Feature Store, Model Registry, Model Monitor, Clarify, Automatic Model Tuning, Inference Recommender, Pipelines, and endpoint types (real-time, serverless, batch transform).

### Amazon Bedrock
Amazon Bedrock is a fully managed service for building generative AI applications using foundation models. In this course we cover invoking foundation models and fine-tuning models with custom datasets using the Bedrock API.

### AWS Glue
AWS Glue is a serverless data integration service for analytics, ML, and application development. We use Glue for ETL jobs that transform raw CSV data into optimised Parquet format, and Glue Data Catalog for metadata management.

### Amazon S3
Amazon Simple Storage Service (S3) is object storage built to retrieve any amount of data from anywhere. S3 is the primary data store used throughout this course for raw data, processed data, model artifacts, and training outputs.

### AWS CloudFormation
AWS CloudFormation is an infrastructure as code service that allows you to model and provision AWS resources. Used throughout this course for reproducible environment setup and deployment configurations.

### Amazon CloudWatch
Amazon CloudWatch is a monitoring and observability service. We use CloudWatch for endpoint metrics dashboards, alarms, and logging for ML infrastructure monitoring.

### Amazon EventBridge
Amazon EventBridge is a serverless event bus. Used in this course for scheduling automated pipeline executions and triggering model retraining.

### AWS IAM
AWS Identity and Access Management (IAM) is used to manage access to AWS services. We cover least-privilege policies for ML workloads, SageMaker execution roles, and role-based access control patterns.

## Cleanup

To avoid incurring unnecessary AWS charges after completing the course:

1. Delete any running SageMaker endpoints
2. Delete any Feature Store feature groups created during demos: `aws sagemaker delete-feature-group --feature-group-name <NAME>`
3. Delete any JupyterLab spaces in the SageMaker Studio Domain
4. Empty the S3 bucket: `aws s3 rm s3://<YOUR-BUCKET-NAME> --recursive`
5. Delete the CloudFormation stacks in reverse order (monitoring/security first, then setup last)
6. Delete any ECR repositories created during the custom container demo

## Author

**Johnny Chivers**
- YouTube: [@JohnnyChivers](https://www.youtube.com/@JohnnyChivers)
- Website: [johnnychivers.co.uk](https://www.johnnychivers.co.uk)
- Practice Exams: [knowza.ai](https://knowza.ai)
- GitHub: [github.com/johnny-chivers](https://github.com/johnny-chivers)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
