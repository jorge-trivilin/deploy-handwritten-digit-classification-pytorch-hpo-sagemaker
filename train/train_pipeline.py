# train_pipeline.py
import boto3
import os
import logging
from sagemaker.session import Session  # type: ignore
from sagemaker.model_metrics import MetricsSource, ModelMetrics  # type: ignore
from sagemaker.workflow.step_collections import RegisterModel  # type: ignore
from sagemaker.workflow.pipeline_context import PipelineSession  # type: ignore
from sagemaker.processing import ScriptProcessor  # type: ignore
from sagemaker.workflow.steps import ProcessingStep, TrainingStep  # type: ignore
from sagemaker.workflow.parameters import (  # type: ignore
    ParameterString,
    ParameterFloat,
    ParameterInteger,
)
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.functions import Join  # type: ignore
from sagemaker.workflow.pipeline import Pipeline  # type: ignore
from sagemaker.inputs import TrainingInput  # type: ignore
from sagemaker.estimator import Estimator  # type: ignore
from sagemaker.workflow.properties import PropertyFile  # type: ignore
from typing import cast

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Logging with timestamp
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def log_dag_info(**kwargs):
    logger.info("-- SageMaker processing logging --")
    logger.info(
        "SageMaker processing irá processar dados de treino de: %s", train_data_path
    )
    logger.info(
        "SageMaker processing irá processar dados de teste de: %s", test_data_path
    )
    logger.info("")  # Linha em branco para espaçamento
    logger.info(
        "SageMaker processing irá armazenar os dados de treino processados em: %s",
        preprocessing_output_train_path,
    )
    logger.info(
        "SageMaker processing irá armazenar os dados de teste processados em: %s",
        preprocessing_output_test_path,
    )
    logger.info(
        "SageMaker processing irá armazenar os dados de scaling em: %s", scalers_output
    )
    logger.info(
        "SageMaker processing irá armazenar os dados de treino pós processados em: %s",
        postprocessing_train_output_path,
    )
    logger.info(
        "SageMaker processing irá armazenar os dados de teste pós processados em: %s",
        postprocessing_test_output_path,
    )
    logger.info("")  # Linha em branco para espaçamento
    logger.info("-- PCA  logging --")
    logger.info("Nome do modelo PCA: %s", pca_model_name)
    logger.info("Modelo PCA será salvo em: %s", pca_model_output)
    logger.info(
        "PCA será treinando com os dados de: %s", preprocessing_output_train_path
    )
    logger.info("")  # Linha em branco para espaçamento
    logger.info("-- PCA  Batch Logging --")
    logger.info(
        "Resultados do Batch Transform nos dados de treino do PCA serão salvos em: %s",
        pca_batch_transform_train_data_output_path,
    )
    logger.info(
        "Resultados do Batch Transform nos dados de teste do PCA serão salvos em: %s",
        pca_batch_transform_test_data_output_path,
    )
    logger.info("")  # Linha em branco para espaçamento
    logger.info("-- KMEANS Logging --")
    logger.info("Nome do modelo KMEANS: %s", kmeans_model_name)
    logger.info("Modelo KMEANS será salvo em: %s", kmeans_model_output)
    logger.info(
        "KMEANS será treinando com os dados de: %s", postprocessing_train_output_path
    )
    logger.info(
        "KMEANS será testado com os dados de: %s", postprocessing_test_output_path
    )
    logger.info("")  # Linha em branco para espaçamento
    logger.info("-- KMEANS Batch Logging --")
    logger.info(
        "Resultados do Batch Transform nos dados de teste do KMEANS serão salvos em: %s",
        kmeans_batch_transform_test_output,
    )
    logger.info(
        "SageMaker processing irá armazenar os dados do KMEANS pós processados em: %s",
        rfm_clustering_data,
    )


def get_environment_variable(name, default_value=""):
    """Obtém ou define uma variável de ambiente."""
    return os.environ.setdefault(name, default_value)


def create_s3_path(bucket, project, branch, version, model, path):
    """Cria um caminho S3 formatado."""
    return f"s3://{bucket}/{project}/{branch}/{version}/{model}/{path}"


def get_session(region=None, default_bucket=None):
    """Gets the sagemaker sessions based on the region and default bucket.

    Args:
        region: the AWS region to start the sessions. If None, it uses the environment's region.
        default_bucket: the bucket to use for storing the artifacts. If none, it uses a default bucket.

    Returns:
        'sagemaker.session.Session' instance

    """
    # Use the environment variable if not provided
    if region is None:
        region = os.getenv("AWS_DEFAULT_REGION")

    # Create a boto3 session with the specified region
    boto_session = boto3.Session(region_name=region)

    # Create the necessary AWS clients for the SageMaker session
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")

    # Returns a configured SageMaker session instance
    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


# Variáveis de ambiente da Dag
AWS_DEFAULT_REGION = get_environment_variable(
    "AWS_DEFAULT_REGION", "${aws_default_region}"
)
SAGEMAKER_ROLE_ARN = get_environment_variable(
    "SAGEMAKER_ROLE_ARN", "${sagemaker_role_arn}"
)
BRANCH_NAME = get_environment_variable("BRANCH_NAME", "${branch_name}")
PROJECT_NAME = get_environment_variable("PROJECT_NAME", "${project_name}")
BUCKET_RAW = get_environment_variable("BUCKET_RAW", "${bucket_raw}")
BUCKET_PROCESSED = get_environment_variable("BUCKET_PROCESSED", "${bucket_processed}")
BUCKET_STAGING = get_environment_variable("BUCKET_STAGING", "${bucket_staging}")
BUCKET_VALIDATION = get_environment_variable(
    "BUCKET_VALIDATION", "${bucket_validation}"
)
BUCKET_OUTPUT = get_environment_variable("BUCKET_OUTPUT", "${bucket_output}")
BUCKET_MODELS = get_environment_variable("BUCKET_MODELS", "${bucket_models}")
BUCKET_AIRFLOW = get_environment_variable("BUCKET_AIRFLOW", "${bucket_airflow}")
VERSION = "${version}"
DAG_NAME = "${project_name}_${branch_name}_training_dag_v${version}"
IMAGE_URI = "${image_uri}"

# s3 paths
train_data_path = f"s3://{BUCKET_PROCESSED}/{PROJECT_NAME}/{BRANCH_NAME}/{VERSION}/train"
test_data_path = f"s3://{BUCKET_PROCESSED}/{PROJECT_NAME}/{BRANCH_NAME}/{VERSION}/test"

preprocessing_output_train_path = create_s3_path(
    BUCKET_STAGING,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "PCA",
    "preprocessing/train/processed_train_data.csv",
)
preprocessing_output_test_path = create_s3_path(
    BUCKET_STAGING,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "PCA",
    "preprocessing/test/processed_test_data.csv",
)
scalers_output = create_s3_path(
    BUCKET_STAGING, PROJECT_NAME, BRANCH_NAME, VERSION, "output", "scalers"
)

pca_model_name = f"{PROJECT_NAME}-{BRANCH_NAME}-{VERSION}-PCA"
pca_model_output = create_s3_path(
    BUCKET_MODELS, PROJECT_NAME, BRANCH_NAME, VERSION, "PCA", ""
)
preprocessing_job_name = f"{PROJECT_NAME}-{BRANCH_NAME}-{VERSION}-preprocessing"
pca_training_job_name = f"{PROJECT_NAME}-{BRANCH_NAME}-{VERSION}-pca-training"
postprocessing_job_name = f"{PROJECT_NAME}-{BRANCH_NAME}-{VERSION}-postprocessing-pca"
kmeans_training_job_name = f"{PROJECT_NAME}-{BRANCH_NAME}-{VERSION}-kmeans-training"
kmeans_postprocessing_job_name = (
    f"{PROJECT_NAME}-{BRANCH_NAME}-{VERSION}-postprocessing-kmeans"
)

pca_batch_transform_train_data_input = preprocessing_output_train_path
pca_batch_transform_test_data_input = preprocessing_output_test_path
pca_batch_transform_train_data_output_path = create_s3_path(
    BUCKET_OUTPUT, PROJECT_NAME, BRANCH_NAME, VERSION, "PCA", "output/results/train"
)
pca_batch_transform_test_data_output_path = create_s3_path(
    BUCKET_OUTPUT, PROJECT_NAME, BRANCH_NAME, VERSION, "PCA", "output/results/test"
)

postprocessing_train_input_path = create_s3_path(
    BUCKET_OUTPUT,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "PCA",
    "output/results/train/processed_train_data.csv.out",
)
postprocessing_test_input_path = create_s3_path(
    BUCKET_OUTPUT,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "PCA",
    "output/results/test/processed_test_data.csv.out",
)
postprocessing_train_output_path = create_s3_path(
    BUCKET_OUTPUT,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "PCA",
    "output/postprocessing/train/post_processed_train.csv",
)
postprocessing_test_output_path = create_s3_path(
    BUCKET_OUTPUT,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "PCA",
    "output/postprocessing/test/post_processed_test.csv",
)

kmeans_model_name = f"{PROJECT_NAME}-{BRANCH_NAME}-{VERSION}-KMEANS"
kmeans_model_output = create_s3_path(
    BUCKET_MODELS, PROJECT_NAME, BRANCH_NAME, VERSION, "KMEANS", ""
)
kmeans_batch_transform_test_data_input = postprocessing_test_output_path
kmeans_batch_transform_test_output = create_s3_path(
    BUCKET_OUTPUT, PROJECT_NAME, BRANCH_NAME, VERSION, "KMEANS", "output/results/test"
)
kmeans_postprocessing_input = create_s3_path(
    BUCKET_OUTPUT,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "KMEANS",
    "output/results/test/post_processed_test.csv.out",
)
rfm_clustering_data = create_s3_path(
    BUCKET_OUTPUT,
    PROJECT_NAME,
    BRANCH_NAME,
    VERSION,
    "KMEANS",
    "output/results/final/test/final_rfm_clustering.csv",
)

# Parâmetros adicionais
processor_instance_type = "ml.m5.4xlarge"
training_instance_type = "ml.c4.4xlarge"
batch_transform_instance_type = "ml.m5.4xlarge"
role = SAGEMAKER_ROLE_ARN
region = AWS_DEFAULT_REGION

def get_pipeline(
    region, role, default_bucket, pipeline_name, model_package_group_name, base_folder
):
    """Cria e configura um pipeline do SageMaker de forma genérica.

    Args:
        region: Região AWS.
        role: IAM role para execução do pipeline.
        default_bucket: Bucket padrão para armazenar artefatos.
        pipeline_name: Nome do pipeline.
        model_package_group_name: Nome do grupo de pacotes do modelo.

    Returns:
        Instância do SageMaker pipeline.
    """

    sagemaker_session = get_session(region, default_bucket)

    # Parâmetro de entrada dos dados
    #input_data = ParameterString(
        #name="InputData",
        #default_value=f"s3://{default_bucket}/{base_folder}/data/input_data.csv",)

    # Parâmetro da imagem de container (por exemplo, PyTorch ou outro)
    image_uri = ParameterString(
        name="ImageUri",
        default_value="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.0-cpu-py36-ubuntu18.04",
    )

    # Criação da sessão de pipeline
    pipeline_session = PipelineSession(
        boto_session=sagemaker_session.boto_session,
        sagemaker_client=sagemaker_session.sagemaker_client,
        default_bucket=default_bucket,
    )

    # Folders prefix
    preprocessed_prefix = f"{base_folder}/preprocessed"
    model_output_prefix = f"{base_folder}/model_output"
    evaluation_output_prefix = f"{base_folder}/evaluation_output"

    # Utilidade para criar caminhos S3
    def build_s3_path(bucket, *prefixes):
        return f"s3://{bucket}/" + "/".join(prefixes)

    # Caminhos S3
    preprocessed_train_data_s3_path = build_s3_path(default_bucket, preprocessed_prefix, "train")
    preprocessed_test_data_s3_path = build_s3_path(default_bucket, preprocessed_prefix, "test")
    model_output_s3_path = build_s3_path(default_bucket, model_output_prefix)
    evaluation_s3_path = build_s3_path(default_bucket, evaluation_output_prefix)

    # Parâmetros do pipeline
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.2xlarge")

    # Passo de pré-processamento #
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
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/preprocessed/train", destination=preprocessed_train_data_s3_path),
            ProcessingOutput(source="/opt/ml/processing/preprocessed/test", destination=preprocessed_test_data_s3_path),
        ],
        code="scripts/preprocessing.py",  # Script de preprocessamento genérico
        job_arguments=["--input-path", input_data]
    )

    # Passo de treinamento #
    pytorch_estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_output_s3_path,
        entry_point="scripts/train.py",  # Script de treinamento genérico
        sagemaker_session=pipeline_session,
    )

    training_step = TrainingStep(
        name="TrainingStep",
        estimator=pytorch_estimator,
        inputs={
            "train": TrainingInput(s3_data=preprocessed_train_data_s3_path, content_type="text/csv"),
            "test": TrainingInput(s3_data=preprocessed_test_data_s3_path, content_type="text/csv"),
        },
    )

    # Passo de avaliação #
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
            ProcessingInput(source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=preprocessed_test_data_s3_path, destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/evaluation", destination=evaluation_s3_path)],
        code="scripts/evaluate.py",  # Script de avaliação genérico
    )

    # Registro do modelo #
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{evaluation_s3_path}/evaluation.json", content_type="application/json"
        )
    )

    register_model_step = RegisterModel(
        name="RegisterModelStep",
        estimator=pytorch_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # Definição do pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_data, model_approval_status, training_instance_count, training_instance_type],
        steps=[preprocessing_step, training_step, evaluation_step, register_model_step],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=role)

    return pipeline
