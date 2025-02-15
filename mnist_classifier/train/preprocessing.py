import os
from typing import List, Tuple, Optional
from pathlib import Path
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST

def get_mnist_transform() -> transforms.Compose:
    """Returns the transformation pipeline for MNIST data."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def load_mnist_datasets(data_dir: str, transform: Optional[transforms.Compose] = None) -> Tuple[MNIST, MNIST]:
    """
    Load MNIST training and test datasets.
    Args:
        data_dir: Directory to store/load MNIST data
        transform: Optional transform to apply to the data
    Returns:
        Tuple of (training_dataset, test_dataset)
    """
    if transform is None:
        transform = get_mnist_transform()
    
    train_dataset = MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_dir, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def process_dataset(dataset: MNIST, batch_size: int = 1000) -> Tuple[Tensor, Tensor]:
    """
    Process a dataset and convert to tensors.
    
    Args:
        dataset: MNIST dataset to process
        batch_size: Number of images to process before logging progress
    
    Returns:
        Tuple of (images_tensor, labels_tensor)
        
    Raises:
        ValueError: If dataset is empty or no images were processed
    """
    # Check for completely empty dataset
    if not dataset:
        raise ValueError("Dataset cannot be empty")
    
    images: List[Tensor] = []
    labels: List[int] = []
    
    for i, (img, label) in enumerate(dataset):
        # Validate individual images
        if img.numel() == 0:  # Check if image tensor is empty
            continue
            
        images.append(img)
        labels.append(label)
        if i % batch_size == 0 and i > 0:
            print(f"{i} images processed...")
    
    # Check if any valid images were processed
    if not images:
        raise ValueError("No valid images were processed from dataset")
            
    return torch.stack(images), torch.tensor(labels)

def save_processed_data(images: Tensor, labels: Tensor, output_path: Path) -> None:
    """
    Save processed images and labels to a file.
    Args:
        images: Tensor of processed images
        labels: Tensor of labels
        output_path: Path to save the data
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((images, labels), output_path)

def preprocess_mnist_data(input_dir: str = "/opt/ml/processing/input/data",
                         output_dir: str = "/opt/ml/processing") -> None:
    """Main preprocessing function."""
    try:
        print(f"Loading MNIST dataset from {input_dir}...")
        transform = get_mnist_transform()
        train_dataset, test_dataset = load_mnist_datasets(input_dir, transform)
        
        output_path = Path(output_dir)
        train_path = output_path / "train" / "train.pt"
        test_path = output_path / "test" / "test.pt"
        
        print("Processing training data...")
        train_images, train_labels = process_dataset(train_dataset)
        save_processed_data(train_images, train_labels, train_path)
        
        print("Processing test data...")
        test_images, test_labels = process_dataset(test_dataset)
        save_processed_data(test_images, test_labels, test_path)
        
        print(f"Preprocessed MNIST data saved in {output_dir}")
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_mnist_data()