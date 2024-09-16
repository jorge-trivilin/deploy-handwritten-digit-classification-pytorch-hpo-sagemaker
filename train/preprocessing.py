# preprocessing.py

"""
This script preprocesses the MNIST dataset by downloading it from torchvision, transforming it into tensors, and saving the processed data to specified directories.

Key Functions:
- **preprocess_mnist_data**: Downloads the MNIST dataset, applies transformations (normalization and tensor conversion), and saves the processed training and test datasets into separate files.

Usage:
- This script is designed to be executed directly. It handles downloading the MNIST dataset, preprocessing the images and labels, and saving them in a format suitable for training and testing machine learning models.

Function Details:
- `preprocess_mnist_data()`: 
  - **Local Directory**: Downloads MNIST dataset into `/opt/ml/processing/input/data`.
  - **Output Directory**: Saves the preprocessed data into `/opt/ml/processing`.
  - **Transformations**: Applies transformations including converting images to tensors and normalizing them.
  - **Saving Data**: Saves the processed training and test datasets as `.pt` files in separate directories within the output directory.

Error Handling:
- The script includes basic error handling to catch and report any issues encountered during the preprocessing.

Execution:
- The script is intended to be run as a standalone program. Upon execution, it will preprocess the MNIST data and store it in the specified output directories.

Directory Structure:
- Input Data Directory: `/opt/ml/processing/input/data`
- Output Directory: `/opt/ml/processing`
  - Training Data: `/opt/ml/processing/train/train.pt`
  - Test Data: `/opt/ml/processing/test/test.pt`
"""

import os
from torchvision.datasets import MNIST
from torchvision import transforms
import torch


def preprocess_mnist_data():
    try:
        local_dir = "/opt/ml/processing/input/data"  # Input directory
        output_dir = "/opt/ml/processing"  # Output directory
        print(f"Downloading MNIST dataset to directory {local_dir}...")

        # Download training and test datasets directly from PyTorch
        train_dataset = MNIST(
            local_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        test_dataset = MNIST(
            local_dir,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        # Output directories for preprocessed data
        train_output_dir = os.path.join(output_dir, "train")
        test_output_dir = os.path.join(output_dir, "test")
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        print("Processing and saving the training dataset...")
        # Load all training images and labels
        train_images = []
        train_labels = []
        for i, (img, label) in enumerate(train_dataset):
            train_images.append(img)
            train_labels.append(label)
            if i % 10000 == 0:
                print(f"{i} training images processed...")

        # Convert to tensors
        train_images = torch.stack(train_images)  # Dimension [N, C, H, W]
        train_labels = torch.tensor(train_labels)

        # Save data in a single file
        torch.save(
            (train_images, train_labels), os.path.join(train_output_dir, "train.pt")
        )

        print("Processing and saving the test dataset...")
        # Load all test images and labels
        test_images = []
        test_labels = []
        for i, (img, label) in enumerate(test_dataset):
            test_images.append(img)
            test_labels.append(label)
            if i % 1000 == 0:
                print(f"{i} test images processed...")

        # Convert to tensors
        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)

        # Save data in a single file
        torch.save((test_images, test_labels), os.path.join(test_output_dir, "test.pt"))

        print(
            f"Preprocessed MNIST data saved in directories {train_output_dir} and {test_output_dir}"
        )
    except Exception as e:
        print(f"An error occurred during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    preprocess_mnist_data()
