import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torch.utils.data.distributed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class CustomMNISTDataset(Dataset):
    def __init__(self, data_file):
        logger.info(f"Carregando dados de {data_file}")
        self.images, self.labels = torch.load(data_file)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# Modelo baseado em https://github.com/pytorch/examples/blob/master/mnist/main.py
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

def _get_train_data_loader(batch_size, data_file, is_distributed, **kwargs):
    logger.info("Get train data loader")
    dataset = CustomMNISTDataset(data_file=data_file)
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if is_distributed
        else None
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **kwargs
    )

def _get_test_data_loader(test_batch_size, data_file, **kwargs):
    logger.info("Get test data loader")
    dataset = CustomMNISTDataset(data_file=data_file)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug(f"Distributed training - {is_distributed}")
    use_cuda = args.num_gpus > 0
    logger.debug(f"Number of GPUs available - {args.num_gpus}")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(
            backend=args.backend, rank=host_rank, world_size=world_size
        )
        logger.info(
            f"Initialized the distributed environment: '{args.backend}' backend on {dist.get_world_size()} nodes. "
            f"Current host rank is {dist.get_rank()}. Number of GPUs: {args.num_gpus}"
        )

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Ajuste nos caminhos dos arquivos de dados
    train_data_file = os.path.join(args.train_data_dir, 'train.pt')
    test_data_file = os.path.join(args.test_data_dir, 'test.pt')

    train_loader = _get_train_data_loader(
        args.batch_size, train_data_file, is_distributed, **kwargs
    )
    test_loader = _get_test_data_loader(
        args.test_batch_size, test_data_file, **kwargs
    )

    model = Net().to(device)
    if is_distributed and use_cuda:
        # Multi-machine multi-GPU case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # Single-machine multi-GPU case or single-machine or multi-machine CPU case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if is_distributed and not use_cuda:
                # Average gradients manually for multi-machine CPU case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
        test(model, test_loader, device)
    save_model(model, args.model_dir)

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    return model.to(device)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logger.info(f"SM_CHANNEL_TRAIN: {os.environ.get('SM_CHANNEL_TRAIN')}")
    logger.info(f"SM_CHANNEL_TEST: {os.environ.get('SM_CHANNEL_TEST')}")

    # Hyperparameters and other arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=str, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--train_data_dir", type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--test_data_dir", type=str, default=os.environ['SM_CHANNEL_TESTING'])
    parser.add_argument("--num_gpus", type=int, default=os.environ.get("SM_NUM_GPUS", 1))

    args = parser.parse_args()

     # Adicionando logs para depuração
    logger.info(f"Argumentos de entrada: {args}")
    logger.info(f"Arquivos em {args.train_data_dir}: {os.listdir(args.train_data_dir)}")
    logger.info(f"Arquivos em {args.test_data_dir}: {os.listdir(args.test_data_dir)}")

    train(args)
