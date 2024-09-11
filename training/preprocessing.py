import os
from torchvision.datasets import MNIST
from torchvision import transforms
import sagemaker
import boto3

# Função para pré-processar os dados
def preprocess_mnist_data():
    region = boto3.Session().region_name
    local_dir = "/opt/ml/processing/input/data"  # Diretório padrão para os Processing Jobs no SageMaker
    output_dir = "/opt/ml/processing/output"  # Diretório de saída para salvar os dados pré-processados

    # Download do dataset MNIST
    MNIST.mirrors = [
        f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/"
    ]
    dataset = MNIST(
        local_dir,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    
    # Certifique-se de que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar o dataset pré-processado em formato apropriado (ex: em numpy, CSV, etc.)
    for i, (img, label) in enumerate(dataset):
        img_path = os.path.join(output_dir, f"img_{i}.pt")
        label_path = os.path.join(output_dir, f"label_{i}.pt")
        
        # Salva cada imagem e label
        torch.save(img, img_path)
        torch.save(label, label_path)
    
    print(f"Dados MNIST pré-processados e salvos no diretório {output_dir}")

# Chama a função de pré-processamento
if __name__ == "__main__":
    preprocess_mnist_data()
