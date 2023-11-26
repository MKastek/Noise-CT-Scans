import torch


def nrmse(recon, reference):
    n = (reference - recon) ** 2
    den = reference ** 2
    return 100.0 * torch.mean(n) ** 0.5 / torch.mean(den) ** 0.5
