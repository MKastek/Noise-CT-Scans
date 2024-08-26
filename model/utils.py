from collections import OrderedDict

import torch.nn as nn


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
