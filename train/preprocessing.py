# preprocessing.py
import os
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import boto3

def preprocess_mnist_data():
    try:
        region = boto3.Session().region_name
        local_dir = "/opt/ml/processing/input/data"
        output_dir = "/opt/ml/processing"  # Mudado para corresponder à estrutura do pipeline
        
        print(f"Região AWS: {region}")
        print(f"Baixando dataset MNIST para o diretório {local_dir}...")
        
        MNIST.mirrors = [f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/"]
        
        # Download dos datasets de treinamento e teste separadamente
        train_dataset = MNIST(local_dir, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
        
        test_dataset = MNIST(local_dir, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
        
        os.makedirs(os.path.join(output_dir, 'preprocessed', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'preprocessed', 'test'), exist_ok=True)
        
        print("Processando e salvando o dataset de treinamento...")
        for i, (img, label) in enumerate(train_dataset):
            img_path = os.path.join(output_dir, 'preprocessed', 'train', f"img_{i}.pt")
            label_path = os.path.join(output_dir, 'preprocessed', 'train', f"label_{i}.pt")
            torch.save(img, img_path)
            torch.save(label, label_path)
            if i % 1000 == 0:
                print(f"{i} imagens de treinamento processadas...")
        
        print("Processando e salvando o dataset de teste...")
        for i, (img, label) in enumerate(test_dataset):
            img_path = os.path.join(output_dir, 'preprocessed', 'test', f"img_{i}.pt")
            label_path = os.path.join(output_dir, 'preprocessed', 'test', f"label_{i}.pt")
            torch.save(img, img_path)
            torch.save(label, label_path)
            if i % 1000 == 0:
                print(f"{i} imagens de teste processadas...")
        
        print(f"Dados MNIST pré-processados e salvos no diretório {output_dir}/preprocessed")
    
    except Exception as e:
        print(f"Ocorreu um erro durante o pré-processamento: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_mnist_data()