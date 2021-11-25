import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from demo import FashionMNIST
from demo import NeuralNetwork, adjust_learning_rate
import argparse
from demo import utils


device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    # Download training data from open datasets.
    training_data = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )


    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    # for X, y in test_dataloader:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     break

    
    print(f"Using {device} device")


    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.adamw(model.parameters(), lr=args.base_lr)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, args, t)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    

def train(dataloader, model, loss_fn, optimizer, args, epoch):
    size = len(dataloader.dataset)
    model.train()
    if args.lr_scheduler == 'step':
        adjust_learning_rate(optimizer, epoch+1 , args )
    elif args.lr_scheduler is None: 
        pass  
    else:
        raise NotImplementedError()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} lr: {optimizer.param_groups[0]['lr']} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
def get_args_parser():
    parser = argparse.ArgumentParser('argument for torch demo ')
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--base_lr', default=1e-2, type=float)
    parser.add_argument('--lrschduler', default=False, type=bool)
    parser.add_argument('--batch_size', default=64, type=int)
    
    args = parser.parse_args()
    
    return args
    
if __name__ == "__main__":
    args = get_args_parser()
    main(args)

