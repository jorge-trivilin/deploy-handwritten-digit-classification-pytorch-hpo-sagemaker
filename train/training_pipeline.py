# -- training_pipeline.py --
import boto3  # type: ignore
import os
import logging
from sagemaker.session import Session  # type: ignore
from sagemaker.model_metrics import MetricsSource, ModelMetrics  # type: ignore
from sagemaker.workflow.step_collections import RegisterModel  # type: ignore
from sagemaker.workflow.pipeline_context import PipelineSession  # type: ignore
from sagemaker.processing import ScriptProcessor  # type: ignore
from sagemaker.workflow.steps import ProcessingStep, TrainingStep  # type: ignore
from sagemaker.workflow.parameters import ParameterString, ParameterInteger  # type: ignore
from sagemaker.processing import ProcessingInput, ProcessingOutput  # type: ignore
from sagemaker.inputs import TrainingInput  # type: ignore
from sagemaker.estimator import Estimator  # type: ignore
from sagemaker.workflow.pipeline import Pipeline  # type: ignore
from sagemaker.workflow.steps import CacheConfig

# Configuração de logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

cache_config = CacheConfig(enable_caching=True, expire_after="PT7H")


def get_environment_variable(name, default_value=""):
    """Obtém ou define uma variável de ambiente."""
    return os.environ.setdefault(name, default_value)


def create_s3_path(bucket, project, branch, version, model, path):
    """Cria um caminho S3 formatado."""
    return f"s3://{bucket}/{project}/{branch}/{version}/{model}/{path}"


def get_session(region=None, bucket_pipeline=None):
    """Cria uma sessão SageMaker baseada na região e bucket."""
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
    """Cria e configura o pipeline do SageMaker."""

    # Definindo os caminhos S3 com base nos parâmetros fornecidos
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

    # Criação da sessão do SageMaker
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

    # Passo de pré-processamento
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

    # Passo de treinamento
    pytorch_estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_output_s3_path,
        entry_point="train/train.py",
        sagemaker_session=pipeline_session,
    )

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

    # Passo de avaliação do modelo
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
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
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
            s3_uri=evaluation_step.properties.ProcessingOutputConfig.Outputs[
                "evaluation_output"
            ].S3Output.S3Uri,
            content_type="application/json",
        )
    )

    register_model_step = RegisterModel(
        name="RegisterModelStep",
        estimator=pytorch_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,  # type: ignore
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # Step dependencies
    training_step.depends_on = [preprocessing_step]
    evaluation_step.depends_on = [training_step]
    register_model_step.depends_on = [evaluation_step]

    # Definindo o pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_approval_status,
            training_instance_count,
            training_instance_type,
            processing_instance_count,
            processing_instance_type,
        ],
        steps=[preprocessing_step, training_step, evaluation_step, register_model_step],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=role)

    return pipeline
