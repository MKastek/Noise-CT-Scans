import re
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from model import DnCNN

#
# plt.rcParams['image.cmap'] = 'gray'
from utils import (
    get_data,
    torch_to_np,
    calculate_psnr,
    get_NPS_2D,
    get_rect_ROI,
    get_NPS_1D,
)


def evaluate_trained_model(
    model,
    model_path: Path,
    data_path: Path,
    data_index: int,
    roi_row: slice,
    roi_column: slice,
):
    images = get_data(data_path)
    image_test = images[data_index][roi_row, roi_column]
    image_test_pytorch = torch.from_numpy(image_test).float().unsqueeze(0)
    plt.suptitle(fr"Test of trained model DnCNN")
    plt.subplot(121)
    plt.title("Test image")
    plt.imshow(image_test)
    plt.xticks([])
    plt.yticks([])

    image_denoised_pytorch = model(image_test_pytorch)
    image_denoised_numpy = torch_to_np(image_denoised_pytorch)

    plt.subplot(122)
    plt.title("Denoised image")
    plt.imshow(image_denoised_numpy)
    plt.xlabel(f"PSNR = {calculate_psnr(image_test, image_denoised_numpy):.2f} dB")
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(
        top=1.0, bottom=0.031, left=0.031, right=0.977, hspace=0.2, wspace=0.068
    )

    plt.show()


if __name__ == "__main__":
    model = DnCNN()
    pretrained_model_path = (
        Path().resolve().parents[1] / "model" / "pretrained" / "DnCNN"
    )
    trained_model_path = Path().resolve().parents[1] / "model" / "saved"
    data_path = (
        Path().resolve().parents[3]
        / "dane"
        / "KARDIO ZAMKNIETE"
        / "A001"
        / "DICOM"
        / "P1"
        / "E1"
        / "S3"
    )
    model_trained = torch.load(
        Path().resolve().parents[1] / "model" / "saved" / "DnCNN_model.pt"
    )

    evaluate_trained_model(
        model=model_trained,
        model_path=trained_model_path / "DnCNN_model.pt",
        data_path=data_path,
        data_index=200,
        roi_row=slice(30, 130),
        roi_column=slice(250, 350),
    )
