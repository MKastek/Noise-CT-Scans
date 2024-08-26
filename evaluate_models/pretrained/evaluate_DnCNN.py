import re
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from model import DnCNN

# plt.rcParams['image.cmap'] = 'gray'
from utils import (
    get_data,
    torch_to_np,
    calculate_psnr,
    get_NPS_2D,
    get_rect_ROI,
    get_NPS_1D,
)


def evaluate_pretrained_model(
    model,
    model_path: Path,
    data_path: Path,
    data_index: int,
    roi_row: slice,
    roi_column: slice,
):
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    images = get_data(data_path)
    image_test = images[data_index][roi_row, roi_column]
    image_test_pytorch = torch.from_numpy(image_test).float().unsqueeze(0)
    sigma = int(re.findall(r"\d+", model_path.name)[0])
    plt.suptitle(fr"Test of model DnCNN $ \sigma={sigma}$")
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
        top=0.98, bottom=0.152, left=0.031, right=0.97, hspace=0.122, wspace=0.063
    )

    plt.show()


def evaluate_pretrained_models_nps(
    model,
    models_path: list[Path],
    colors: list[str],
    data_path: Path,
    data_index: int,
    roi_row: slice,
    roi_column: slice,
):
    images = get_data(data_path)
    image_test = images[data_index][roi_row, roi_column]
    image_test_pytorch = torch.from_numpy(image_test).float().unsqueeze(0)
    for model_path, color in zip(models_path, colors):
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        sigma = int(re.findall(r"\d+", model_path.name)[0])
        image_denoised_pytorch = model(image_test_pytorch)
        image_denoised_numpy = torch_to_np(image_denoised_pytorch)
        ROI_array_rectangle_noise = get_rect_ROI(
            image=np.flipud(image_denoised_numpy), y=75, x=55, size=8, num=9, plot=False
        )
        NPS_2D_rectangle_noise = get_NPS_2D(ROI_array_rectangle_noise)
        rad_noise, intensity_noise = get_NPS_1D(NPS_2D_rectangle_noise)

        x_interpolate = np.linspace(0, 1.6, 64)
        cubic_spline = CubicSpline(rad_noise, intensity_noise)
        plt.plot(rad_noise, intensity_noise, ".", color=color)
        plt.plot(
            x_interpolate,
            cubic_spline(x_interpolate),
            label=f"model DnCNN $\sigma=$ {sigma}",
            color=color,
        )
    ROI_array_rectangle_noise = get_rect_ROI(
        image=np.flipud(image_test), y=75, x=55, size=8, num=9, plot=False
    )
    NPS_2D_rectangle_noise = get_NPS_2D(ROI_array_rectangle_noise)
    rad_noise, intensity_noise = get_NPS_1D(NPS_2D_rectangle_noise)

    x_interpolate = np.linspace(0, 1.6, 64)
    cubic_spline = CubicSpline(rad_noise, intensity_noise)
    plt.plot(rad_noise, intensity_noise, ".", color=colors[-1])
    plt.plot(
        x_interpolate, cubic_spline(x_interpolate), label="test image", color=colors[-1]
    )
    plt.title("Noise Power Spectrum ")
    plt.ylabel(r"NPS $ \left[ HU^{2}mm^{2} \right]$")
    plt.xlabel("$f_{r} [mm^{-1}]$")
    plt.xlim(0, 1.6)
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.7)
    plt.legend()
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

    evaluate_pretrained_model(
        model=DnCNN(),
        model_path=pretrained_model_path / "dncnn_15.pth",
        data_path=data_path,
        data_index=200,
        roi_row=slice(30, 130),
        roi_column=slice(250, 350),
    )
    evaluate_pretrained_model(
        model=DnCNN(),
        model_path=pretrained_model_path / "dncnn_25.pth",
        data_path=data_path,
        data_index=200,
        roi_row=slice(30, 130),
        roi_column=slice(250, 350),
    )
    evaluate_pretrained_model(
        model=DnCNN(),
        model_path=pretrained_model_path / "dncnn_50.pth",
        data_path=data_path,
        data_index=200,
        roi_row=slice(30, 130),
        roi_column=slice(250, 350),
    )
    evaluate_pretrained_models_nps(
        model=DnCNN(),
        models_path=[
            pretrained_model_path / "dncnn_15.pth",
            pretrained_model_path / "dncnn_25.pth",
            pretrained_model_path / "dncnn_50.pth",
        ],
        colors=["red", "green", "blue", "gray"],
        data_path=data_path,
        data_index=200,
        roi_row=slice(30, 130),
        roi_column=slice(250, 350),
    )
