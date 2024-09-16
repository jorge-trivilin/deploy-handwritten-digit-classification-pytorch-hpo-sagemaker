# run_training_pipeline.py

"""
SageMaker Training Pipeline Execution Script

This script sets up and executes a SageMaker training pipeline for machine learning workflows.
It serves as the main entry point for initiating the training process within a SageMaker
environment.

Key Components:
1. Environment setup and logging configuration
2. Retrieval of AWS and project-specific parameters from environment variables
3. Pipeline creation and execution using the SageMaker SDK
4. Error handling and status reporting for pipeline execution

Main Function:
--------------
main():
    Orchestrates the entire process of setting up and running the SageMaker pipeline.
    It handles environment variable retrieval, pipeline creation, execution, and status logging.

Dependencies:
-------------
- os: For environment variable handling
- logging: For logging setup and usage
- boto3: For AWS SDK functionality
- train.training_pipeline.get_pipeline: Custom module for pipeline definition

Usage:
------
This script is typically run as the main entry point in a SageMaker training job.
Ensure all required environment variables are set before execution.

To run the script:
$ python run_training_pipeline.py

Note:
-----
- The script assumes that the necessary AWS credentials and permissions are properly configured.
- It relies on a separate module (train.training_pipeline) for the actual pipeline definition.
- Logging is set up for both the script's own operations and boto3 interactions.

Error Handling:
---------------
The script includes basic error handling for pipeline execution status retrieval.
Any failure in retrieving the status is logged as an error.

"""

import os
import logging
from train.training_pipeline import get_pipeline  # type: ignore
import boto3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Boto3 logger
format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
boto3.set_stream_logger("boto3", logging.INFO, format_string)


def main():

    """
    Orchestrates the setup and execution of a SageMaker training pipeline.

    This function performs the following tasks:
    1. Retrieves environment variables for AWS configuration and project settings.
    2. Sets up S3 bucket paths for various stages of the ML workflow.
    3. Constructs pipeline and model package group names.
    4. Calls the get_pipeline function to create a SageMaker pipeline.
    5. Upserts (creates or updates) the pipeline in SageMaker.
    6. Starts the pipeline execution.
    7. Attempts to retrieve and log the execution status.

    Environment Variables:
    - AWS_DEFAULT_REGION: AWS region for SageMaker operations (default: 'us-east-1')
    - SAGEMAKER_ROLE_ARN: ARN of the IAM role for SageMaker
    - BRANCH_NAME: Git branch name for versioning
    - PROJECT_NAME: Name of the project
    - VERSION: Version identifier
    - IMAGE_URI: URI of the Docker image for SageMaker
    - BUCKET_PROCESSED: S3 bucket for processed data
    - BUCKET_MODELS: S3 bucket for model artifacts
    - BUCKET_OUTPUT: S3 bucket for output data
    - BUCKET_PIPELINE: S3 bucket for pipeline artifacts

    The function uses these variables to configure and run a SageMaker pipeline,
    which typically includes steps for data preprocessing, model training,
    evaluation, and registration.

    Raises:
    - AttributeError: If unable to retrieve the pipeline execution status

    Note:
    This function is designed to be the entry point for the pipeline execution
    script and should be called when the script is run directly.
    """

    # Definition of environment variables and parameters
    REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    ROLE = os.getenv("SAGEMAKER_ROLE_ARN", "role_arn")
    BRANCH_NAME = os.getenv("BRANCH_NAME")
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    VERSION = os.getenv("VERSION")
    IMAGE_URI = os.getenv("IMAGE_URI")

    # Defining buckets and paths
    BUCKET_PROCESSED = os.getenv("BUCKET_PROCESSED")
    BUCKET_MODELS = os.getenv("BUCKET_MODELS")
    BUCKET_OUTPUT = os.getenv("BUCKET_OUTPUT")
    BUCKET_PIPELINE = os.getenv("BUCKET_PIPELINE")

    # Defining additional parameters
    pipeline_name = f"{BRANCH_NAME}-pipeline"
    model_package_group_name = f"{BRANCH_NAME}-package-group"

    # Creating the pipeline using the get_pipeline function
    pipeline = get_pipeline(
        region=REGION,
        role=ROLE,
        pipeline_name=pipeline_name,
        model_package_group_name=model_package_group_name,
        image_uri=IMAGE_URI,
        project_name=PROJECT_NAME,
        branch_name=BRANCH_NAME,
        bucket_processed=BUCKET_PROCESSED,
        bucket_models=BUCKET_MODELS,
        bucket_output=BUCKET_OUTPUT,
        bucket_pipeline=BUCKET_PIPELINE,
        version=VERSION,
    )

    # Insert the pipeline (create or update) and start its execution
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print(f"Pipeline {pipeline_name} started with execution {execution}")

    # Capture the execution result and do something with it (log, send notification, etc.)
    try:
        status = execution.describe()["PipelineExecutionStatus"]  # type: ignore
        logger.info(f"Pipeline execution completed with status: {status}")
    except AttributeError:
        logger.error(
            "Failed to retrieve pipeline execution status. Check if 'execution' has a 'describe' method."
        )


if __name__ == "__main__":
    main()
