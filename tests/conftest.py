import pytest
import os
import torch
from torch import Tensor
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)
@pytest.fixture(scope="module", autouse=True)
def setup_logging():
    logger.setLevel(logging.INFO)

@pytest.fixture
def local_mnist_data_dir(tmp_path):
    """
    Fixture that returns a temporary directory for storing MNIST dataset files.
    This directory is intended to hold the input data for MNIST digit classification tasks.
    Uses pytest's tmp_path fixture for automatic cleanup.
    """
    data_dir = tmp_path / "input" / "data"
    return data_dir

@pytest.fixture
def processed_data_output_dir(tmp_path):
    """
    Fixture that returns a temporary directory for processed data.
    Uses pytest's tmp_path fixture for automatic cleanup.
    """
    output_path = tmp_path / "processing"
    output_path.mkdir(parents=True)
    return output_path

@pytest.fixture
def mnist_transform():
    """
    Fixture that provides the same transformation pipeline used in preprocessing.
    """
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]
    )
    
@pytest.fixture
def mnist_training_data(local_mnist_data_dir,
                        mnist_transform):
    """
    Fixture that creates a small sample of MNIST-like data for training.
    Returns a tuple of (images_tensor, labels_tensor).
    """
    # Create a small sample of MNIST-like data
    # 10 Sample images (28x28 pixels) with corresponding labels
    images = torch.randn((10, 1, 28, 28))
    labels = torch.randint(low=0, high=10, size=(10,))
    return images, labels

@pytest.fixture
def mnist_test_data(local_mnist_data_dir,
                        mnist_transform):
    """
    Fixture that creates a small sample of MNIST-like data for testing.
    Returns a tuple of (images_tensor, labels_tensor).
    """
    images = torch.randn((10, 1, 28, 28))
    labels = torch.randint(low=0, high=10, size=(10,))
    return images, labels

@pytest.fixture
def expected_dirs(processed_data_output_dir):
    """
    Fixture that provides expected training and test directories.
    """
    train_dir = os.path.join(processed_data_output_dir, "train")
    test_dir = os.path.join(processed_data_output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, test_dir

@pytest.fixture
def model_output_dir(tmp_path):
    """
    Fixture that provides a temporary directory for saving trained models.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    return model_dir

@pytest.fixture
def hyperparameters():
    """
    Fixture that provides default hyperparameters for training.
    """
    return {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 1,
        "optimizer": "Adam"
    }

@pytest.fixture
def sample_batch(mnist_training_data):
    """Single batch for testing"""
    images, labels = mnist_training_data
    batch_images = images[:4]
    batch_labels = labels[:4]
    return batch_images, batch_labels

@pytest.fixture
def mock_metrics():
    """
    Fixture that provides simulated metrics for testing.
    """
    return {
        "train_loss": 0.5,
        "train_accuracy": 0.85,
        "test_loss": 0.6,
        "test_accuracy": 0.82
    }

@pytest.fixture
def model_config():
    """
    Fixture that provides the model configuration for testing.
    """
    return {
        "input_size": 784,  # 28x28
        "hidden_size": 128,
        "num_classes": 10
    }

@pytest.fixture(scope="session")
def test_device():
    """
    Fixture that determines the device for running tests (CPU/CUDA).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def mock_mnist_dataset(mnist_training_data):
    """Creates a mock dataset for testing"""
    images, labels = mnist_training_data
    return list(zip(images, labels))
