from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader.data_loaders import TrainDataset
from model import DnCNN
from utils import get_max, get_data


def train_CNN(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs: int = 10,
    output_filename: str = "CNN_model",
):
    """
    Train denoising CNN
    """
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], step:{i}")
    torch.save(
        model, Path().resolve().parents[0] / "model" / "saved" / f"{output_filename}.pt"
    )


if __name__ == "__main__":
    path = (
        Path().resolve().parents[2]
        / "dane"
        / "KARDIO ZAMKNIETE"
        / "A001"
        / "DICOM"
        / "P1"
        / "E1"
        / "S1"
    )
    normalize = get_max(path)
    dataset = TrainDataset(data_path=path, normalize=normalize)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DnCNN()
    criterion = nn.MSELoss()
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_CNN(model, train_loader, criterion, optimizer, num_epochs=100)
