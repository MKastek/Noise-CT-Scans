from pathlib import Path

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import DnCNN
from data_loader import TrainDataset


def init_dncnn_parser():
    parser = argparse.ArgumentParser(description="Training DnCNN")
    parser.add_argument(
        "--num_scans", help="Number of CT scans", type=int, default=10000,
    )
    parser.add_argument(
        "--num_epochs", help="Number of epochs", type=int, default=10,
    )
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--data_path", type=str, default="data/train_dataset")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="model/pretrained/DnCNN/dncnn_25.pth",
    )
    return parser


def train_DnCNN(
    model,
    train_loader,
    criterion,
    optimizer,
    num_scans: int,
    num_epochs: int,
    output_path: str = "model/saved",
):
    """
    Train denoising CNN
    """
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}],  loss:{epoch_loss / num_scans}")
    torch.save(
        model,
        Path(output_path)
        / f"DnCNN_model_{str(num_epochs)}_epoch_{str(num_scans)}_scans.pt",
    )


if __name__ == "__main__":
    dncnn_parser = init_dncnn_parser()
    args = dncnn_parser.parse_args()
    dataset = TrainDataset(data_path=Path(args.data_path), num_scans=args.num_scans)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DnCNN()
    model.load_state_dict(torch.load(args.pretrained_model_path), strict=True)
    model = model.to("cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_DnCNN(
        model,
        train_loader,
        criterion,
        optimizer,
        num_scans=args.num_scans,
        num_epochs=args.num_epochs,
    )
