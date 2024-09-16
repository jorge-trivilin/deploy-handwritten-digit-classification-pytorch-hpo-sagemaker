#!/bin/bash

# Arguments
account_id=$1
region=us-east-1
repo_name="iamdscli"
tag_name=$2

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $account_id.dkr.ecr.$region.amazonaws.com

# Create the ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names $repo_name --region $region || aws ecr create-repository --repository-name $repo_name --region $region

# Build the docker image
docker build -t $repo_name -f ./train/Dockerfile ./train

# Tag the docker image
docker tag $repo_name:latest $account_id.dkr.ecr.$region.amazonaws.com/$repo_name:$tag_name

# Define a variável ECR_URI 
ECR_URI="$account_id.dkr.ecr.$region.amazonaws.com/$repo_name:$tag_name"

# Push the docker image
docker push $ECR_URI

# Exporta a ECR_URI como uma variável de ambiente
echo "TF_VAR_image_uri=$ECR_URI" >> $GITHUB_ENV
