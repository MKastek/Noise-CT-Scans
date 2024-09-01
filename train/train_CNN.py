from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader.data_loaders import TrainDataset
from model import DnCNN
from utils import get_max, get_data


def train_DnCNN(
    model,
    train_loader,
    criterion,
    optimizer,
        num_scans: int,
    num_epochs: int = 10,
    output_filename: str = "DnCNN_model",
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
    print(f"{output_filename}_{str(num_epochs)}_epoch_{str(num_scans)}_scans.pt")
    torch.save(
        model, Path().resolve().parents[0] / "model" / "saved" / f"{output_filename}_{str(num_epochs)}_epoch_{str(num_scans)}_scans.pt"
    )

if __name__ == "__main__":
    dataset = TrainDataset(data_path=Path().resolve().parents[0] / "data" / "train_dataset")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DnCNN()
    model.load_state_dict(torch.load("../model/pretrained/DnCNN/dncnn_25.pth"), strict=True)
    #model.eval()
    # for k, v in model.named_parameters():
    #     v.requires_grad = False
    model = model.to('cpu')
    print('Model path: {:s}'.format("dncnn_25.pth"))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('Params number: {}'.format(number_parameters))
    criterion = nn.MSELoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_DnCNN(model, train_loader, criterion, optimizer,len(dataset), num_epochs=100)


