import os
from torchvision.datasets import MNIST
from torchvision import transforms
import torch

def preprocess_mnist_data():
    try:
        local_dir = "/opt/ml/processing/input/data"  # Diretório de entrada
        output_dir = "/opt/ml/processing"  # Diretório de saída

        print(f"Baixando dataset MNIST para o diretório {local_dir}...")

        # Download dos datasets de treinamento e teste diretamente do PyTorch
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

        # Diretórios de saída para os dados pré-processados
        train_output_dir = os.path.join(output_dir, 'train')
        test_output_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        print("Processando e salvando o dataset de treinamento...")
        # Carrega todas as imagens e labels de treinamento
        train_images = []
        train_labels = []
        for i, (img, label) in enumerate(train_dataset):
            train_images.append(img)
            train_labels.append(label)
            if i % 10000 == 0:
                print(f"{i} imagens de treinamento processadas...")

        # Converte para tensores
        train_images = torch.stack(train_images)  # Dimensão [N, C, H, W]
        train_labels = torch.tensor(train_labels)

        # Salva os dados em um único arquivo
        torch.save((train_images, train_labels), os.path.join(train_output_dir, 'train.pt'))

        print("Processando e salvando o dataset de teste...")
        # Carrega todas as imagens e labels de teste
        test_images = []
        test_labels = []
        for i, (img, label) in enumerate(test_dataset):
            test_images.append(img)
            test_labels.append(label)
            if i % 1000 == 0:
                print(f"{i} imagens de teste processadas...")

        # Converte para tensores
        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)

        # Salva os dados em um único arquivo
        torch.save((test_images, test_labels), os.path.join(test_output_dir, 'test.pt'))

        print(f"Dados MNIST pré-processados e salvos nos diretórios {train_output_dir} e {test_output_dir}")

    except Exception as e:
        print(f"Ocorreu um erro durante o pré-processamento: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_mnist_data()
