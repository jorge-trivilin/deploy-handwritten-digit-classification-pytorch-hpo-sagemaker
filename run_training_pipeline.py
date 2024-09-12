# run_training_pipeline.py
import os
import logging
from train.train_pipeline import get_pipeline
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
    region = os.getenv("AWS_DEFAULT_REGION")
    role = os.getenv("SAGEMAKER_ROLE_ARN", "default-role-arn")
    branch_name = os.getenv("BRANCH_NAME")
    default_bucket = os.getenv("BUCKET_PIPELINE")
    project_name = os.getenv("PROJECT_NAME")

    # Definindo parâmetros adicionais
    pipeline_name = f"{branch_name}"
    model_package_group_name = f"{branch_name}"
    base_folder = f"{project_name}/{branch_name}"

    # Criação do pipeline usando a função get_pipeline definida no pipeline.py
    pipeline = get_pipeline(
        region=region,
        role=role,
        default_bucket=default_bucket,
        pipeline_name=pipeline_name,
        model_package_group_name=model_package_group_name,
        base_folder=base_folder,
    )

    # Iniciar a execução do pipeline
    pipeline.upsert(role_arn=role)

    # Executar o pipeline
    execution = pipeline.start()

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
