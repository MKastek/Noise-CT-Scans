import argparse

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re


folder_path = Path("model") / "saved"

if __name__ == "__main__":

    loss_15 = np.load(
        folder_path
        / "loss_DnCNN_model_10_epoch_5000_scans_64_batch_size_0.0001_lr_15_noise_level.npy"
    )
    loss_25 = np.load(
        folder_path
        / "loss_DnCNN_model_10_epoch_5000_scans_64_batch_size_0.0001_lr_25_noise_level.npy"
    )
    loss_50 = np.load(
        folder_path
        / "loss_DnCNN_model_10_epoch_5000_scans_64_batch_size_0.0001_lr_50_noise_level.npy"
    )

    for loss, noise in zip([loss_15, loss_25, loss_50], ["15", "25", "50"]):
        plt.scatter(np.arange(1, len(loss) + 1, 1), loss)
        plt.plot(
            np.arange(1, len(loss) + 1, 1),
            loss,
            linestyle="-",
            label=fr"DnCNN $\sigma$={noise}",
        )
    plt.title("Epoch Mean Squared Error loss with different DnCNN pretrained model")
    plt.ylabel("Epoch Mean Squared Error loss")
    plt.xlabel("epochs")
    plt.grid(which="major", alpha=0.7)
    plt.grid(which="minor", alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()
