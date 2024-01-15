from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader.data_loaders import TrainDataset
from model import DnCNN
from utils import get_max


def train(model, train_loader, criterion, optimizer, num_epochs=10, output_filename="CNN_model"):
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    torch.save(model, Path().resolve().parents[0] / 'model' / f"{output_filename}.pt")


if __name__ == '__main__':
    path = Path().resolve().parents[2] / "dane" / "KARDIO ZAMKNIETE" / "A001" / "DICOM" / "P1" / "E1" / "S1"
    normalize = get_max(path)
    dataset = TrainDataset(data_path=path, normalize=normalize)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DnCNN()
    criterion = nn.MSELoss()
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, criterion, optimizer, num_epochs=10)