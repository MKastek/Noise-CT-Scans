from pathlib import Path

import torch
import torch.nn as nn
from base import BaseModel
from collections import OrderedDict

from utils import get_data


class CNN_DIP(BaseModel):
    def __init__(self, n_layers=16, n_channels=100, k_size=3):
        super(CNN_DIP, self).__init__()
        head = conv(in_channels=1, out_channels=n_channels, kernel_size=k_size, padding=1, mode='CPR')
        body = [conv( n_channels,  n_channels, mode='CPR') for _ in range(n_layers)]
        tail =  conv(in_channels=n_channels, out_channels=1, kernel_size=k_size, padding=1, mode='CPR')
        self.model = sequential(head, *body, tail)

    def forward(self, x):
        return self.model(x.unsqueeze(0).unsqueeze(0))


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    match mode:
        case 'C':
            return sequential(*[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                                         padding=padding, bias=bias)])
        case 'CR':
            return sequential(*[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                                         padding=padding, bias=bias), nn.ReLU(inplace=True)])
        case 'CPR':
            return sequential(*[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                                          padding=padding, bias=bias), nn.PReLU()])


class PDnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(PDnCNN, self).__init__()

        m_head = conv(in_nc, nc, mode='CR', bias=True)
        m_body = [conv(nc, nc, mode='CR', bias=True) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode='C', bias=True)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x - n


class DnCNN(BaseModel):
    def __init__(self):
        super(DnCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1, bias=False
        )
        self.relu1 = nn.ReLU(inplace=True)
        hidden_layers = []
        for i in range(17):
            hidden_layers.append(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )
            )
            hidden_layers.append(nn.BatchNorm2d(64))
            hidden_layers.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*hidden_layers)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=(3, 3), padding=1, bias=False
        )

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.mid_layer(out)
        out = self.conv3(out)
        return out
