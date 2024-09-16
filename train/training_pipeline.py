# -- training_pipeline.py --

"""
SageMaker Pipeline for MNIST Digit Classification

This script defines a comprehensive SageMaker pipeline for training, hyperparameter tuning, 
evaluating, and registering a PyTorch model for MNIST digit classification. It demonstrates 
how to orchestrate various components of the SageMaker SDK to create an end-to-end machine 
learning workflow.

Main Components:
----------------
1. Environment Setup:
   - get_environment_variable: Retrieves or sets environment variables.
   - create_s3_path: Constructs formatted S3 paths for data storage.
   - get_session: Creates a SageMaker session with specified configurations.

2. Pipeline Definition (get_pipeline function):
   - Preprocessing: Prepares MNIST data for training and testing.
   - Hyperparameter Tuning: Optimizes model hyperparameters using random search.
   - Model Evaluation: Assesses the performance of the best model from tuning.
   - Model Registration: Registers the best model with calculated metrics.

Key Functions:
--------------
get_environment_variable(name, default_value=""):
    Retrieves an environment variable or sets it to a default value if not defined.

create_s3_path(bucket, project, branch, version, model, path):
    Constructs a formatted S3 path for various data storage needs.

get_session(region=None, bucket_pipeline=None):
    Initializes a SageMaker session with specified AWS region and bucket.

get_pipeline(region, role, pipeline_name, model_package_group_name, image_uri, 
             project_name, branch_name, bucket_processed, bucket_models, 
             bucket_output, bucket_pipeline, version):
    Configures and creates a complete SageMaker pipeline for MNIST classification.
    - Sets up pipeline parameters and steps.
    - Defines data flows between steps.
    - Configures model training, evaluation, and registration processes.

Usage Notes:
------------
- This scrip requires appropriate AWS permissions and SageMaker resources to be set up.
- The pipeline uses a PyTorch estimator and is optimized for MNIST digit classification.
- Hyperparameter tuning focuses on learning rate and batch size optimization.
- The final model is registered with a pending manual approval status by default.
"""

import boto3  # type: ignore
import os
import logging
from typing import List, Dict, Union

from sagemaker.session import Session  # type: ignore
from sagemaker.model_metrics import MetricsSource, ModelMetrics  # type: ignore

from sagemaker.workflow.pipeline_context import PipelineSession  # type: ignore
from sagemaker.workflow.steps import ProcessingStep, TuningStep  # type: ignore
from sagemaker.workflow.parameters import ParameterString, ParameterInteger  # type: ignore
from sagemaker.workflow.pipeline import Pipeline  # type: ignore
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.entities import PipelineVariable  # type: ignore
from sagemaker.workflow.model_step import ModelStep  # type: ignore
from sagemaker.model import Model  # type: ignore
from sagemaker.workflow.functions import Join  # type: ignore

from sagemaker.processing import ScriptProcessor  # type: ignore
from sagemaker.processing import ProcessingInput, ProcessingOutput  # type: ignore

from sagemaker.inputs import TrainingInput  # type: ignore
from sagemaker.estimator import Estimator  # type: ignore

from sagemaker.tuner import (  # type: ignore
    ContinuousParameter,
    HyperparameterTuner,
    CategoricalParameter,
)


# Log configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

cache_config = CacheConfig(enable_caching=True, expire_after="PT7H")


def get_environment_variable(name, default_value=""):
    """Gets or sets an environment variable."""
    return os.environ.setdefault(name, default_value)


def create_s3_path(bucket, project, branch, version, model, path):
    """Creates a formatted S3 path."""
    return f"s3://{bucket}/{project}/{branch}/{version}/{model}/{path}"


def get_session(region=None, bucket_pipeline=None):
    """Creates a SageMaker session based on the region and bucket."""
    if region is None:
        region = os.getenv("AWS_DEFAULT_REGION")

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=bucket_pipeline,
    )


def get_pipeline(
    region,
    role,
    pipeline_name,
    model_package_group_name,
    image_uri,
    project_name,
    branch_name,
    bucket_processed,
    bucket_models,
    bucket_output,
    bucket_pipeline,
    version,
):
    """
    Creates and configures a SageMaker pipeline for MNIST digit classification using PyTorch.

    This pipeline includes steps for data preprocessing, hyperparameter tuning, model evaluation,
    and model registration. It is designed to be flexible and configurable through various parameters.

    Parameters:
    -----------
    region : str
        The AWS region where the pipeline will be created and executed.
    role : str
        The IAM role ARN with necessary permissions for SageMaker operations.
    pipeline_name : str
        The name to be given to the SageMaker pipeline.
    model_package_group_name : str
        The name of the model package group where the final model will be registered.
    image_uri : str
        The URI of the Docker image containing the training and inference code.
    project_name : str
        The name of the project, used in S3 path construction.
    branch_name : str
        The name of the git branch, used in S3 path construction.
    bucket_processed : str
        The S3 bucket name for storing processed data.
    bucket_models : str
        The S3 bucket name for storing trained models.
    bucket_output : str
        The S3 bucket name for storing output data (e.g., evaluation results).
    bucket_pipeline : str
        The S3 bucket name for pipeline-related storage.
    version : str
        The version identifier, used in S3 path construction.

    Returns:
    --------
    Pipeline
        A configured SageMaker Pipeline object ready for execution.

    Pipeline Steps:
    ---------------
    1. Preprocessing: Prepares the MNIST dataset for training and testing.
    2. Hyperparameter Tuning: Optimizes model hyperparameters using random search.
    3. Model Evaluation: Assesses the performance of the best model from tuning.
    4. Model Registration: Registers the best model in the specified model package group.

    Notes:
    ------
    - The pipeline uses a PyTorch estimator for training.
    - Hyperparameter tuning optimizes learning rate and batch size.
    - The pipeline includes caching to improve efficiency in subsequent runs.
    - Model metrics (accuracy, precision, recall, F1 score) are calculated during evaluation.
    - The final model is registered with a pending manual approval status by default.

    Example:
    --------
    >>> pipeline = get_pipeline(
    ...     region='us-west-2',
    ...     role='arn:aws:iam::123456789012:role/SageMakerRole',
    ...     pipeline_name='MNISTPipeline',
    ...     model_package_group_name='MNISTModelPackageGroup',
    ...     image_uri='12345.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch:1.8.0-cpu-py3',
    ...     project_name='MNIST',
    ...     branch_name='main',
    ...     bucket_processed='sagemaker-processed-data',
    ...     bucket_models='sagemaker-model-artifacts',
    ...     bucket_output='sagemaker-output-data',
    ...     bucket_pipeline='sagemaker-pipeline-storage',
    ...     version='v1.0'
    ... )
    >>> pipeline.upsert(role_arn='arn:aws:iam::123456789012:role/SageMakerRole')
    """

    # Defining S3 paths based on provided parameters
    preprocessing_output_train_path = create_s3_path(
        bucket_processed,
        project_name,
        branch_name,
        version,
        "Pytorch",
        "preprocessing/train/",
    )

    preprocessing_output_test_path = create_s3_path(
        bucket_processed,
        project_name,
        branch_name,
        version,
        "Pytorch",
        "preprocessing/test/",
    )

    model_output_s3_path = create_s3_path(
        bucket_models, project_name, branch_name, version, "Pytorch", ""
    )

    evaluation_metrics_output_path = create_s3_path(
        bucket_output, project_name, branch_name, version, "Pytorch", "evaluation"
    )

    # Logger
    logger.info(
        "Training preprocessed data will be saved in: %s",
        preprocessing_output_train_path,
    )
    logger.info(
        "Test preprocessed data will be saved in: %s", preprocessing_output_test_path
    )
    logger.info("Model will be saved in: %s", model_output_s3_path)
    logger.info(
        "Evaluation metrics will be saved in: %s", evaluation_metrics_output_path
    )

    # Creating SageMaker session
    sagemaker_session = get_session(region, bucket_pipeline)

    pipeline_session = PipelineSession(
        boto_session=sagemaker_session.boto_session,
        sagemaker_client=sagemaker_session.sagemaker_client,
        default_bucket=bucket_pipeline,
    )

    # Pipeline Parameters
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.c5.2xlarge"
    )

    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.c5.2xlarge"
    )

    # Preprocessing step
    script_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        sagemaker_session=pipeline_session,
    )

    preprocessing_step = ProcessingStep(
        name="Preprocessing",
        processor=script_processor,
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/train",
                destination=preprocessing_output_train_path,
                output_name="processed_train_data",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/test",
                destination=preprocessing_output_test_path,
                output_name="processed_test_data",
            ),
        ],
        code="train/preprocessing.py",
        cache_config=cache_config,
    )

    # Training step
    pytorch_estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_output_s3_path,
        entry_point="train/train.py",
        sagemaker_session=pipeline_session,
    )

    hyperparameter_ranges = {
        "lr": ContinuousParameter(0.001, 0.1),
        "batch-size": CategoricalParameter([32, 64, 128, 256, 512]),
    }

    objective_metric_name = "average test loss"

    def get_metric_definitions() -> List[Dict[str, Union[str, PipelineVariable]]]:
        return [
            {
                "Name": "average test loss",
                "Regex": "Test set: Average loss: ([0-9\\.]+)",
            }
        ]

    metric_definitions = get_metric_definitions()

    tuner = HyperparameterTuner(
        estimator=pytorch_estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=2,
        max_parallel_jobs=2,
        objective_type="Minimize",
        metric_definitions=metric_definitions,
        strategy="Random",
        early_stopping_type="Auto",
    )

    step_tuning = TuningStep(
        name="HPTuning",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[  # type: ignore
                    "processed_train_data"
                ].S3Output.S3Uri
            ),
            "test": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[  # type: ignore
                    "processed_test_data"
                ].S3Output.S3Uri
            ),
        },
        cache_config=cache_config,
    )

    """
    training_step = TrainingStep(
        name="TrainingStep",
        estimator=pytorch_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[  # type: ignore
                    "processed_train_data"
                ].S3Output.S3Uri
            ),
            "test": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[  # type: ignore
                    "processed_test_data"
                ].S3Output.S3Uri
            ),
        },
    )
    """

    best_training_job_name = step_tuning.properties.BestTrainingJob.TrainingJobName

    # Build the complete model URI using Join
    model_data_uri_path = Join(
        on="",
        values=[model_output_s3_path, best_training_job_name, "/output/model.tar.gz"],
    )

    model = Model(
        image_uri=image_uri,
        model_data=model_data_uri_path,
        sagemaker_session=pipeline_session,
        role=role,
        # entry_point="inference.py",  # If applicable
        # source_dir="src",            # If applicable
    )

    """
    step_model = ModelStep(
        name="CreateBestModel",
        step_args=model.create(
            instance_type="ml.m5.xlarge",
        ),
    )
    """

    # Model evaluation step
    script_evaluator = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        sagemaker_session=pipeline_session,
    )

    evaluation_step = ProcessingStep(
        name="ModelEvaluation",
        processor=script_evaluator,
        inputs=[
            ProcessingInput(
                source=model_data_uri_path,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "processed_test_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/evaluation",
                output_name="evaluation_output",
                destination=evaluation_metrics_output_path,
            )
        ],
        code="train/evaluate.py",
        job_arguments=["--evaluation-output-dir", "/opt/ml/processing/evaluation"],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    evaluation_step.properties.ProcessingOutputConfig.Outputs[
                        "evaluation_output"
                    ].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    register_model_step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    register_model_step = ModelStep(
        name="RegisterBestModel",
        step_args=register_model_step_args,
    )

    # Define dependencies
    step_tuning.depends_on = [preprocessing_step]
    evaluation_step.depends_on = [step_tuning]
    register_model_step.depends_on = [evaluation_step]

    # Defining the pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_approval_status,
            training_instance_count,
            training_instance_type,
            processing_instance_count,
            processing_instance_type,
        ],
        steps=[preprocessing_step, step_tuning, evaluation_step, register_model_step],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=role)

    return pipeline
