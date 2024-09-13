import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
from torch.utils.data import DataLoader, Dataset
import json

# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Definição da arquitetura da rede neural (mesmo que o treinamento)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Classe personalizada para carregar os dados pré-processados (.pt)
class CustomMNISTDataset(Dataset):
    def __init__(self, data_file):
        self.images, self.labels = torch.load(data_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def _get_test_data_loader(test_data_file, batch_size):
    dataset = CustomMNISTDataset(data_file=test_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Função de avaliação
def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # Pega o índice da maior log-probabilidade
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)

    test_loss /= total
    accuracy = 100.0 * correct / total

    logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy

# Função para carregar o modelo treinado
def load_model(model_dir, device):
    logger.info(f"Carregando o modelo do diretório {model_dir}")
    model = Net().to(device)
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Argumentos esperados do SageMaker
    parser.add_argument("--batch-size", type=int, default=1000, metavar="N", help="Tamanho do batch para avaliação")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--evaluation-output-dir", type=str, default="/opt/ml/processing/evaluation")


    args = parser.parse_args()

    use_cuda = int(args.num_gpus) > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Carregar o modelo treinado
    model = load_model(args.model_dir, device)

    # Construir o caminho completo para o arquivo test.pt
    test_data_file = os.path.join(args.test_dir, 'test.pt')

    # Carregar os dados de teste
    test_loader = _get_test_data_loader(test_data_file, args.batch_size)

    # Avaliar o modelo
    test_loss, accuracy = evaluate(model, test_loader, device)

    # Salvar os resultados no diretório de saída
    evaluation_output_path = os.path.join(args.evaluation_output_dir, "evaluation.json")
    logger.info(f"Salvando os resultados da avaliação em {evaluation_output_path}")
    with open(evaluation_output_path, "w") as f:
        json.dump({"test_loss": test_loss, "accuracy": accuracy}, f)
