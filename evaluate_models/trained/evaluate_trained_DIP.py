from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from analyze_denoise import make_plot
from model import DnCNN

from utils import (
    get_data,
    torch_to_np,
    calculate_psnr,
    get_NPS_2D,
    get_rect_ROI,
    get_NPS_1D,
)

plt.rcParams["image.cmap"] = "gray"


def compare_epochs():
    epochs = [1000, 5000, 10000]
    for (idx, epoch), color in zip(enumerate(epochs), ["red", "green", "blue"]):
        loss = np.load(data_path / "images" / f"loss_dip_{epoch}_epochs_0.0001_lr.npy")
        plt.plot(loss, alpha=(1 / ((idx + 1))), label=f"{epoch} epochs", color=color)
    plt.title(
        "Epoch normalized Mean Squared Error loss with different number of epochs"
    )
    plt.ylabel("Epoch Normalized Mean Squared Error loss [%]")
    plt.xlabel("epochs")
    plt.grid(which="major", alpha=0.7)
    plt.grid(which="minor", alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()
    for epoch in epochs:
        test_image = np.load(
            data_path / "images" / f"orginal_image_{epoch}_epochs_0.0001_lr.npy"
        )
        denoised_image = np.load(
            data_path / "images" / f"denoised_image_{epoch}_epochs_0.0001_lr.npy"
        )
        make_plot(
            test_image,
            denoised_image,
            f"Test of model Deep Image Prior - {epoch} epochs",
            "Test image",
            "Denoised image",
        )
        plt.show()


def compare_lr():
    lr = [0.001, 0.0001, 1e-5]
    for (idx, lr), color in zip(enumerate(lr), ["red", "green", "blue"]):
        loss = np.load(data_path / "images" / f"loss_dip_1000_epochs_{lr}_lr.npy")
        plt.plot(loss, alpha=(1 / ((idx + 1))), label=f"{lr:.1e}", color=color)
    plt.title("Epoch normalized Mean Squared Error loss with different learning rate")
    plt.ylabel("Epoch Normalized Mean Squared Error loss [%]")
    plt.xlabel("epochs")
    plt.grid(which="major", alpha=0.7)
    plt.grid(which="minor", alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = Path().resolve().parents[1]
    for name in range(10):
        test_image = np.load(
            data_path / "images" / f"orginal_image_5000_epochs_0.0001_lr_exp_{name}.npy"
        )
        denoised_image = np.load(
            data_path
            / "images"
            / f"denoised_image_5000_epochs_0.0001_lr_exp_{name}.npy"
        )
        make_plot(
            test_image,
            denoised_image,
            f"Test of model Deep Image Prior - 6000 epochs",
            "Test image",
            "Denoised image",
        )
        plt.show()
    # compare_epochs()
    # compare_lr()
