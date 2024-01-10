import torch.nn as nn
import torch
from base import BaseModel
from torchsummary import summary


class CNN_DIP(BaseModel):
    def __init__(self, n_layers=16, n_channels=100, k_size=3):
        super(CNN_DIP, self).__init__()
        pd = int(k_size / 2)
        layers = [nn.Conv2d(1, n_channels, (k_size,k_size), padding=pd), nn.PReLU()]
        for _ in range(n_layers):
            layers.append(nn.Conv2d(n_channels, n_channels, (k_size, k_size), padding=pd))
            layers.append(nn.PReLU())
        layers.append(nn.Conv2d(n_channels, 1, (k_size, k_size), padding=pd))
        layers.append(nn.PReLU())

        self.deep_net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.squeeze(self.deep_net(x.unsqueeze(0).unsqueeze(0)))


class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        # in layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        # hidden layers
        hidden_layers = []
        for i in range(18):
          hidden_layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1, bias=False))
          hidden_layers.append(nn.BatchNorm2d(64))
          hidden_layers.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*hidden_layers)
        # out layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3,3), padding=1, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.mid_layer(out)
        out = self.conv3(out)
        return out
