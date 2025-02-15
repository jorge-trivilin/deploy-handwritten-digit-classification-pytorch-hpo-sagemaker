# test_preprocessing.py

import os
import torch
from pathlib import Path
import pytest
from torchvision import transforms
import logging

from mnist_classifier.train.preprocessing import (
    get_mnist_transform,
    load_mnist_datasets,
    process_dataset,
    save_processed_data
)

def test_get_mnist_transform():
    """Test the transformation pipeline creation"""
    logger = logging.getLogger(__name__)
    logger.info("Testing MNIST transform creation...")
    
    transform = get_mnist_transform()
    
    assert isinstance(transform, transforms.Compose)
    assert len(transform.transforms) == 2
    assert isinstance(transform.transforms[0], transforms.ToTensor)
    assert isinstance(transform.transforms[1], transforms.Normalize)

def test_load_mnist_datasets(local_mnist_data_dir, mnist_transform):
    """Test dataset loading functionality"""
    logger = logging.getLogger(__name__)
    logger.info("Testing MNIST dataset loading...")
    
    train_dataset, test_dataset = load_mnist_datasets(
        str(local_mnist_data_dir),
        mnist_transform
    )
    
    assert train_dataset is not None
    assert test_dataset is not None
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0

def test_process_dataset(mnist_training_data):
    """Test dataset processing"""
    logger = logging.getLogger(__name__)
    logger.info("Testing dataset processing...")
    
    # Create a mock dataset with your synthetic data
    images, labels = mnist_training_data
    mock_dataset = list(zip(images, labels))
    
    processed_images, processed_labels = process_dataset(mock_dataset, batch_size=5)
    
    assert isinstance(processed_images, torch.Tensor)
    assert isinstance(processed_labels, torch.Tensor)
    assert processed_images.shape == (10, 1, 28, 28)
    assert processed_labels.shape == (10,)

def test_process_dataset_empty():
    """Test processing a completely empty dataset"""
    empty_dataset = []
    
    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        process_dataset(empty_dataset)

def test_process_dataset_no_valid_images(mnist_transform):
    """Test processing a dataset with no valid images"""
    # Create a dataset with empty tensors
    empty_tensors = [(torch.tensor([]), torch.tensor(0)) for _ in range(5)]
    
    with pytest.raises(ValueError, match="No valid images were processed from dataset"):
        process_dataset(empty_tensors)

def test_save_processed_data(processed_data_output_dir, mnist_training_data):
    """Test data saving functionality"""
    logger = logging.getLogger(__name__)
    logger.info("Testing data saving...")
    
    images, labels = mnist_training_data
    output_path = processed_data_output_dir / "test_save" / "data.pt"
    
    save_processed_data(images, labels, output_path)
    
    # Verify file exists
    assert output_path.exists()
    
    # Load and verify data
    loaded_images, loaded_labels = torch.load(output_path)
    assert torch.allclose(images, loaded_images)
    assert torch.allclose(labels, loaded_labels)

def test_save_processed_data_invalid_path(tmp_path):
    """Test saving data to invalid path"""
    images = torch.randn(10, 1, 28, 28)
    labels = torch.randint(0, 10, (10,))
    invalid_path = tmp_path / "nonexistent" / "deep" / "path" / "data.pt"
    
    save_processed_data(images, labels, invalid_path)
    assert invalid_path.exists()





