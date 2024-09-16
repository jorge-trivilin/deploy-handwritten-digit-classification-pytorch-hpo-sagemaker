# -- main.tf --
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

# -- Variables --

variable "branch_name" {
  type        = string
  description = "Branch name"
}

variable "commit_version" {
  type        = string
  description = "Current version (commit SHA)"
}

variable "environment" {
  type        = string
  description = "Github environment"
}

variable "aws_default_region" {
  type = string
  description = "Default AWS region"
}

variable "aws_account_id" {
  type = string
  description = "AWS account ID"
}

variable "bucket_glue" {
  type = string
  description = "GLUE bucket name"
}

variable "bucket_pipeline" {
  type = string
  description = "PIPELINE bucket name"
}

variable "bucket_processed" {
  type = string
  description = "PROCESSED bucket name"
}

variable "bucket_raw" {
  type = string
  description = "RAW bucket name"
}

variable "bucket_staging" {
  type = string
  description = "STAGING bucket name"
}

variable "bucket_validation" {
  type = string
  description = "VALIDATION bucket name"
}

variable "bucket_output" {
  type = string
  description = "OUTPUT bucket name"
}

variable "bucket_models" {
  type = string
  description = "MODELS bucket name"
}

variable "bucket_airflow" {
  type = string
  description = "AIRFLOW bucket name"
}

# variable "database_name" {
  # type = string
  # description = "Database name"
#}

variable "glue_service_role" {
  type = string
  description = "ARN of the Glue service role"
}

variable "project_name" {
  type = string
  description = "Project name"
}

variable "sagemaker_role_arn" {
  type = string
  description = "ARN of the SageMaker role"
}

# variable "source_table_name" {
  #type = string
  #description = "Source table name"
#}

variable "image_uri" {
  type = string
  description = "DOCKER image URI for SageMaker jobs"
}

# -- Search information about existing buckets -- 
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
  content = "" # Creates an empty folder
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

# -- Json files to be committed --

resource "local_file" "version" {
  filename = "${path.module}/commit_version.json"
  content = jsonencode({
    commit_version = var.commit_version
  })
}

resource "local_file" "image_uri" {
  filename = "${path.module}/image_uri.json"
  content = jsonencode({
    image_uri = var.image_uri
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
    # GLUE_ETL_SCRIPT = "s3://${var.bucket_glue}/${var.project_name}/${var.branch_name}/${var.commit_version}/etl.py",
    # GLUE_TEST_PRE_PROCESSING_SCRIPT = "s3://${var.bucket_glue}/${var.project_name}/${var.branch_name}/${var.commit_version}/test_pre_processing.py",
    # GLUE_TRAIN_PRE_PROCESSING_SCRIPT = "s3://${var.bucket_glue}/${var.project_name}/${var.branch_name}/${var.commit_version}/train_pre_processing.py",
    # AIRFLOW_DAG_SCRIPT = "s3://${var.bucket_airflow}/dags/${var.project_name}_${var.branch_name}_${var.commit_version}_training_pipeline_dag.py", 
    VERSION = var.commit_version
    ENVIRONMENT = var.environment
  })
}