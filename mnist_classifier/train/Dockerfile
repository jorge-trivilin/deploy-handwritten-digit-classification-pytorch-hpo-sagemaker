FROM python:3.12

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /opt/ml/code/

# Install Python libraries listed in requirements.txt and sagemaker_training_toolkit
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copy train.py into the container
COPY train.py /opt/ml/code/train.py

# Set the working directory
WORKDIR /opt/ml/code
