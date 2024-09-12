# run_training_pipeline.py
import os
import logging
from train.training_pipeline import get_pipeline # type: ignore
import boto3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Boto3 logger
format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
boto3.set_stream_logger("boto3", logging.INFO, format_string)


def main():
    # Definição das variáveis de ambiente e parâmetros
    REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    ROLE = os.getenv("SAGEMAKER_ROLE_ARN", "role_arn")
    BRANCH_NAME = os.getenv("BRANCH_NAME")
    PROJECT_NAME = os.getenv("PROJECT_NAME", "cnn")
    VERSION = os.getenv("VERSION")
    IMAGE_URI = os.getenv("IMAGE_URI")

    # Definindo buckets e paths
    BUCKET_PROCESSED = os.getenv("BUCKET_PROCESSED")
    BUCKET_MODELS = os.getenv("BUCKET_MODELS")
    BUCKET_OUTPUT = os.getenv("BUCKET_OUTPUT")
    BUCKET_PIPELINE = os.getenv("BUCKET_PIPELINE")

    # Definindo parâmetros adicionais
    pipeline_name = f"{BRANCH_NAME}-pipeline"
    model_package_group_name = f"{BRANCH_NAME}-package-group"

    # Criação do pipeline usando a função get_pipeline
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
        version=VERSION
    )

    # Inserir o pipeline (criar ou atualizar) e iniciar sua execução
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print(f"Pipeline {pipeline_name} iniciado com execução {execution}")


    # Captura o resultado da execução e faz algo com ele (logar, enviar notificação, etc.)
    try:
        status = execution.describe()["PipelineExecutionStatus"]  # type: ignore
        logger.info(f"Pipeline execution completed with status: {status}")
    except AttributeError:
        logger.error(
            "Failed to retrieve pipeline execution status. Check if 'execution' has a 'describe' method."
        )


if __name__ == "__main__":
    main()
