# evaluate.py

"""
This script evaluates a trained neural network model on a test dataset. It loads the model and test data, performs evaluation, and calculates various performance metrics.

Key Components:
- **Net**: Defines the architecture of the neural network, which is used for both training and evaluation. The network includes convolutional layers, dropout, and fully connected layers.
- **CustomMNISTDataset**: A custom PyTorch Dataset class for loading preprocessed MNIST data.
- **load_model**: Function to load a trained model from a tar.gz file.
- **evaluate**: Function to evaluate the model on test data, computing loss, accuracy, precision, recall, and F1 score.

Usage:
- This script is designed to be run directly. It takes command-line arguments to specify paths for the model, test data, and output directory for evaluation results.

Function Details:
- **load_model(model_dir, device)**:
  - **Inputs**: `model_dir` (directory containing the model file), `device` (PyTorch device).
  - **Output**: Loaded PyTorch model.
  - **Description**: Extracts the model from a tar.gz archive and loads its state_dict. It also handles the case where the state_dict keys have a 'module.' prefix.

- **evaluate(model, test_loader, device)**:
  - **Inputs**: `model` (PyTorch model), `test_loader` (DataLoader for test data), `device` (PyTorch device).
  - **Outputs**: Tuple containing test loss, accuracy, precision, recall, and F1 score.
  - **Description**: Evaluates the model on the test dataset and calculates performance metrics.

Command-Line Arguments:
- `--batch-size`: Batch size for evaluation (default: 1000).
- `--model-dir`: Directory containing the model files (default: "/opt/ml/processing/model").
- `--test-dir`: Directory containing the test data (default: "/opt/ml/processing/test").
- `--evaluation-output-dir`: Directory to save evaluation results (default: "/opt/ml/processing/evaluation").
- `--num-gpus`: Number of GPUs available (default: determined by the environment variable "SM_NUM_GPUS").

Error Handling:
- The script includes error handling for missing files and logging to report issues encountered during model loading and evaluation.

Execution:
- This script is intended to be executed as a standalone program. Upon execution, it will load the model, evaluate it on the test dataset, and save the evaluation metrics to a JSON file in the specified output directory.

Directory Structure:
- Model Directory: `/opt/ml/processing/model`
  - Model Archive: `model.tar.gz`
  - Model File: `model.pth`
- Test Data Directory: `/opt/ml/processing/test`
  - Test Data File: `test.pt`
- Evaluation Output Directory: `/opt/ml/processing/evaluation`
  - Evaluation Results: `evaluation.json`
"""

import argparse
import os
import json
import tarfile
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore

from typing import List, Tuple, cast

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Neural network architecture definition (same as training)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Custom class to load preprocessed data (.pt)
class CustomMNISTDataset(Dataset):
    def __init__(self, data_file: str):
        self.images, self.labels = torch.load(data_file)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def _get_test_data_loader(test_data_file: str, batch_size: int) -> DataLoader:
    dataset = CustomMNISTDataset(data_file=test_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Function to load the trained model
def load_model(model_dir: str, device: torch.device) -> nn.Module:
    logger.info(f"Loading model from directory {model_dir}")

    # Check if model.tar.gz exists and extract
    model_tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(model_tar_path):
        logger.info(f"Found model.tar.gz in {model_tar_path}, extracting...")
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
    else:
        logger.error(f"model.tar.gz not found in {model_tar_path}")
        raise FileNotFoundError(f"model.tar.gz not found in {model_tar_path}")

    # Now, load the model from the model.pth file
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found after extraction.")
        raise FileNotFoundError(f"Model file {model_path} not found.")

    model = Net().to(device)

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=device)

    # Remove the 'module.' prefix from keys, if necessary
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[7:]  # Remove 'module.' from the beginning of the key
        else:
            new_key = k
        new_state_dict[new_key] = v

    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)
    return model


def evaluate(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float, float, float, float]:
    model.eval()
    test_loss: float = 0.0
    correct: int = 0
    total: int = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Specify type of 'reduction' parameter as string literal
            loss = F.nll_loss(output, target, reduction="sum").item()
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            total += len(target)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    test_loss /= total
    accuracy = 100.0 * correct / total

    # Calculate additional metrics
    precision = cast(float, precision_score(all_targets, all_preds, average="weighted"))
    recall = cast(float, recall_score(all_targets, all_preds, average="weighted"))
    f1 = cast(float, f1_score(all_targets, all_preds, average="weighted"))

    logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    return test_loss, accuracy, precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()

    # Expected arguments from SageMaker
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Batch size for evaluation",
    )
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument(
        "--evaluation-output-dir", type=str, default="/opt/ml/processing/evaluation"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", "0"))
    )

    args = parser.parse_args()

    use_cuda: bool = args.num_gpus > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the trained model
    model = load_model(args.model_dir, device)

    # Build the full path to the test.pt file
    test_data_file = os.path.join(args.test_dir, "test.pt")

    # Load the test data
    test_loader = _get_test_data_loader(test_data_file, args.batch_size)

    # Evaluate the model
    test_loss, accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

    # Adapt the metrics dictionary
    metrics_data = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy / 100},
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1_score": {"value": f1},
            "loss": {"value": test_loss},
        }
    }

    # Save the results in the output directory
    os.makedirs(args.evaluation_output_dir, exist_ok=True)
    evaluation_output_path = os.path.join(args.evaluation_output_dir, "evaluation.json")
    logger.info(f"Saving evaluation results to {evaluation_output_path}")
    with open(evaluation_output_path, "w") as f:
        json.dump(metrics_data, f)


if __name__ == "__main__":
    main()
