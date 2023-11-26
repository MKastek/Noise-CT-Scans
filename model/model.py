import torch.nn as nn
import torch
from base import BaseModel


class CNNConfigurable(BaseModel):
    def __init__(self, n_layers, n_channels, k_size):
        super(CNNConfigurable, self).__init__()
        pd = int(k_size / 2)
        layers = [nn.Conv2d(1, n_channels, k_size, padding=pd), nn.PReLU()]
        for _ in range(n_layers):
            layers.append(nn.Conv2d(n_channels, n_channels, k_size, padding=pd))
            layers.append(nn.PReLU())
        layers.append(nn.Conv2d(n_channels, 1, k_size, padding=pd))
        layers.append(nn.PReLU())

        self.deep_net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.squeeze(self.deep_net(x.unsqueeze(0).unsqueeze(0)))


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Decoder
        x = self.decoder(x1)

        return x

