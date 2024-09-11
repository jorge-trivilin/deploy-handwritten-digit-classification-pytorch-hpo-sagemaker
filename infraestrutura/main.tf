# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-east-1" 
}

# -- Variáveis --

variable "branch_name" {
  type        = string
  description = "Nome da branch"
}

variable "commit_version" {
  type        = string
  description = "Versão atual (SHA do commit)"
}

variable "aws_default_region" {
  type = string
  description = "Região padrão da AWS"
}

variable "bucket_glue" {
  type = string
  description = "Nome do bucket GLUE"
}

variable "bucket_pipeline" {
  type = string
  description = "Nome do bucket PIPELINE"
}

variable "bucket_processed" {
  type = string
  description = "Nome do bucket PROCESSED"
}

variable "bucket_raw" {
  type = string
  description = "Nome do bucket RAW"
}

variable "bucket_staging" {
  type = string
  description = "Nome do bucket STAGING"
}

variable "bucket_validation" {
  type = string
  description = "Nome do bucket VALIDATION"
}

variable "bucket_output" {
  type = string
  description = "Nome do bucket OUTPUT"
}

variable "bucket_models" {
  type = string
  description = "Nome do bucket MODELS"
}

variable "bucket_airflow" {
  type = string
  description = "Nome do bucket AIRFLOW"
}

variable "database_name" {
  type = string
  description = "Nome do banco de dados"
}

variable "glue_service_role" {
  type = string
  description = "ARN da role de serviço do Glue"
}

variable "project_name" {
  type = string
  description = "Nome do projeto"
}

variable "sagemaker_role_arn" {
  type = string
  description = "ARN da role do SageMaker"
}

variable "source_table_name" {
  type = string
  description = "Nome da tabela de origem"
}

variable "image_uri" {
  type = string
  description = "URI da imagem DOCKER para os jobs do SageMaker"
}

# -- Buscar informações dos buckets existentes -- 
data "aws_s3_bucket" "glue" {
  bucket = var.bucket_glue
}

data "aws_s3_bucket" "pipeline" {
  bucket = var.bucket_pipeline
}

data "aws_s3_bucket" "processed" {
  bucket = var.bucket_processed
}

data "aws_s3_bucket" "raw" {
  bucket = var.bucket_raw
}

data "aws_s3_bucket" "staging" {
  bucket = var.bucket_staging
}

data "aws_s3_bucket" "validation" {
  bucket = var.bucket_validation
}

data "aws_s3_bucket" "output" {
  bucket = var.bucket_output
}

data "aws_s3_bucket" "models" {
  bucket = var.bucket_models
}

data "aws_s3_bucket" "airflow" {
  bucket = var.bucket_airflow
}

# -- Commit_versioned Resources on s3 --

resource "aws_s3_object" "raw_folder" {
  bucket = var.bucket_raw
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = "" # Cria uma pasta vazia
}

resource "aws_s3_object" "processed_folder" {
  bucket = var.bucket_processed
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = "" 
}

resource "aws_s3_object" "staging_folder" {
  bucket = var.bucket_staging
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = "" 
}

resource "aws_s3_object" "glue_folder" {
  bucket = var.bucket_glue
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = ""
}

resource "aws_s3_object" "pipeline_folder" {
  bucket = var.bucket_pipeline
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = ""
}

resource "aws_s3_object" "validation_folder" {
  bucket = var.bucket_validation
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = ""
}

resource "aws_s3_object" "output_folder" {
  bucket = var.bucket_output
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = ""
}

resource "aws_s3_object" "models_folder" {
  bucket = var.bucket_models
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/"
  content = ""
}

# -- s3 uploadS --
resource "aws_s3_object" "glue_etl_script" {
  bucket = var.bucket_glue
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/etl.py"
  source = "../glue/scripts/etl.py"
}

resource "aws_s3_object" "glue_test_pre_processing_script" {
  bucket = var.bucket_glue
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/test_pre_processing.py"
  source = "../glue/scripts/test_pre_processing.py"
}

resource "aws_s3_object" "glue_train_pre_processing_script" {
  bucket = var.bucket_glue
  key    = "${var.project_name}/${var.branch_name}/${var.commit_version}/train_pre_processing.py"
  source = "../glue/scripts/train_pre_processing.py"
}

# Recurso para criar o arquivo da DAG
resource "local_file" "airflow_dag_script" {
  content  = templatefile("${path.module}/../airflow/code/training_pipeline_dag.py.tpl", {
    aws_default_region = var.aws_default_region,
    sagemaker_role_arn = var.sagemaker_role_arn,
    branch_name = var.branch_name,
    project_name = var.project_name, 
    bucket_raw = var.bucket_raw,
    bucket_processed = var.bucket_processed,
    bucket_staging = var.bucket_staging,
    bucket_validation = var.bucket_validation,
    bucket_output = var.bucket_output,
    bucket_models = var.bucket_models,
    bucket_airflow = var.bucket_airflow,
    version = var.commit_version,
    image_uri = var.image_uri  # Passa a versão do commit
  })
  filename = "${path.module}/../airflow/code/training_pipeline_dag.py"
}

# Recurso para enviar o arquivo da DAG para o S3
resource "aws_s3_object" "airflow_dag_script" {
  bucket = var.bucket_airflow
  key    = "dags/${var.project_name}_${var.branch_name}_${var.commit_version}_training_pipeline_dag.py"  # Nome único para o arquivo no S3
  source = "${path.module}/../airflow/code/training_pipeline_dag.py"  # Caminho para o arquivo original
}

# Json files

# Recurso para gerar o nome da DAG em JSON
resource "local_file" "training_dag_name" {
  filename = "training_dag_name.json"
  content  = jsonencode({
    training_dag_name = "${var.project_name}_${var.branch_name}_training_dag_v${var.commit_version}"
  })
}

resource "local_file" "uris" {
  filename = "s3_uris.json"
  content = jsonencode({
    RAW = "s3://${var.bucket_raw}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    PROCESSED = "s3://${var.bucket_processed}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    STAGING = "s3://${var.bucket_staging}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    VALIDATION = "s3://${var.bucket_validation}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    MODELS = "s3://${var.bucket_models}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    OUTPUT = "s3://${var.bucket_output}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    GLUE = "s3://${var.bucket_glue}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    PIPELINE = "s3://${var.bucket_pipeline}/${var.project_name}/${var.branch_name}/${var.commit_version}/",
    GLUE_ETL_SCRIPT = "s3://${var.bucket_glue}/${var.project_name}/${var.branch_name}/${var.commit_version}/etl.py",
    GLUE_TEST_PRE_PROCESSING_SCRIPT = "s3://${var.bucket_glue}/${var.project_name}/${var.branch_name}/${var.commit_version}/test_pre_processing.py",
    GLUE_TRAIN_PRE_PROCESSING_SCRIPT = "s3://${var.bucket_glue}/${var.project_name}/${var.branch_name}/${var.commit_version}/train_pre_processing.py",
    AIRFLOW_DAG_SCRIPT = "s3://${var.bucket_airflow}/dags/${var.project_name}_${var.branch_name}_${var.commit_version}_training_pipeline_dag.py", 
    VERSION = var.commit_version
  })
}