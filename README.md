# MLOPS Pipeline using Sagemaker Pipelines and Pytorch for MNIST Digit Classification

This project implements an end-to-end machine learning workflow for MNIST digit classification using AWS SageMaker and Terraform. It includes infrastructure setup, data preprocessing, hyperparameter tuning, model training, evaluation, and deployment.

## Table of Contents

- [Overview](#overview)
- [Infrastructure Setup with Terraform](#infrastructure-setup-with-terraform)
- [SageMaker Pipeline](#sagemaker-pipeline)
  - [Preprocessing Step](#preprocessing-step)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Registration](#model-registration)
- [File Structure](#file-structure)
- [Running the Project Locally](#running-the-project-locally)
- [Setting up GitHub Actions](#setting-up-github-actions)
  - [Terraform Workflow](#terraform-workflow)
  - [Training Pipeline Workflow](#training-pipeline-workflow)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project aims to train and evaluate a PyTorch model for MNIST digit classification using SageMaker Pipelines. The pipeline covers data preprocessing, hyperparameter tuning, model evaluation, and model registration in the SageMaker Model Registry. The infrastructure is defined using Terraform, and the entire workflow is integrated with GitHub Actions for CI/CD automation.

## Infrastructure Setup with Terraform

The project uses Terraform to provision the required AWS infrastructure, including S3 buckets for data storage, IAM roles, and SageMaker resources. The Terraform script is located in the `infraestrutura/` directory.

**AWS Resources Provisioned:**
- S3 Buckets: For raw, processed, and output data, as well as model artifacts.
- SageMaker Resources: Roles and permissions needed for SageMaker jobs (training, tuning, evaluation).
- IAM Roles: IAM roles with appropriate permissions for SageMaker to access the S3 buckets and other resources.

The infrastructure setup is fully automated and can be triggered via a GitHub Actions workflow. 

## SageMaker Pipeline

The ML pipeline is defined in the `train/training_pipeline.py` file and consists of several stages:

### Preprocessing Step

The pipeline begins by preparing the MNIST dataset using the `preprocessing.py` script. This step ensures that the dataset is appropriately formatted for training and testing the model.

### Hyperparameter Tuning

The pipeline includes a hyperparameter tuning step using SageMaker’s built-in tuner, which searches for the optimal batch size and learning rate for training the model.

### Model Training

After preprocessing and tuning, the best hyperparameters are used to train a Convolutional Neural Network (CNN) model using PyTorch. The model architecture is defined in `train.py`.

### Model Evaluation

The trained model is evaluated on the test dataset to compute metrics such as accuracy, precision, and recall. These metrics are stored in an S3 bucket for further analysis.

### Model Registration

The final step registers the trained model in the SageMaker Model Registry. The model is registered with a pending manual approval status by default, allowing for human validation before deployment.

## File Structure

The project follows a clear structure for organizing code and infrastructure, making it easier to manage and maintain.

```
.github/
 └── workflows/                    # GitHub Actions workflows
     ├── linter.yml                # Linting workflow using Ruff and Black
     ├── run_pipeline.yml          # Workflow to run the SageMaker pipeline
     └── terraform.yml             # Workflow to run Terraform and build Docker images
infraestrutura/
 ├── commit_version.json           # JSON file to store the current commit version
 ├── image_uri.json                # JSON file containing the Docker image URI
 ├── main.tf                       # Terraform script for infrastructure provisioning
 └── s3_uris.json                  # JSON file storing S3 bucket URIs
train/
 ├── __init__.py                   # Init file for the `train` package
 ├── build-and-push-docker.sh      # Script to build and push Docker images to ECR
 ├── Dockerfile                    # Dockerfile for the SageMaker training image
 ├── entrypoint.py                 # Entry point for the Docker container
 ├── evaluate.py                   # Script for evaluating the model after training
 ├── preprocessing.py              # Script for preprocessing the MNIST dataset
 ├── requirements.txt              # Python dependencies for the project
 ├── train.py                      # PyTorch script to define and train the CNN model
 ├── training_pipeline.py          # Script defining the entire SageMaker pipeline
 └── commit.sh                     # Script for versioning and committing files to Git
pyproject.toml                     # Config file for Ruff and Black linters
README.md                          # This file
requirements.txt                   # Dependencies for the project
run_training_pipeline.py           # Script to run the SageMaker training pipeline
run_training.sh                    # Shell script to trigger the pipeline locally
terraform.sh                       # Shell script to run Terraform tasks
```

## Running the Project Locally

You can run parts of this project locally, such as preprocessing and training the model, by following these steps:

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Python dependencies**:
    Ensure you have `pip` installed and run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the preprocessing script**:
    ```bash
    python train/preprocessing.py --train-data-dir /path/to/train/data --test-data-dir /path/to/test/data
    ```

4. **Run the training script**:
    ```bash
    python train/train.py --epochs 10 --batch-size 64 --lr 0.01
    ```

## Setting up GitHub Actions

This project integrates GitHub Actions for Continuous Integration (CI) and Continuous Deployment (CD). The workflows are located in the `.github/workflows/` directory.

### Terraform Workflow

**File**: `terraform.yml`

- **Trigger**: Runs when there is a push to any branch with the commit message containing `-terraform`.
- **Actions**:
  - Checks out the code.
  - Runs Terraform to provision the required infrastructure.
  - Builds and pushes the Docker image for training.
  - Logs the environment variables for Terraform.

### Training Pipeline Workflow

**File**: `run_pipeline.yml`

- **Trigger**: Runs when there is a push to any branch with the commit message containing `-training`.
- **Actions**:
  - Checks out the code.
  - Configures AWS credentials.
  - Reads the IMAGE_URI from ECS.
  - Executes the SageMaker training pipeline defined in `training_pipeline.py`.

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
