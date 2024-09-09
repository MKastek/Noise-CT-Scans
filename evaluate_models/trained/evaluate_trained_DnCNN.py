import re
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from evaluate_models.pretrained.evaluate_pretrained_DnCNN import evaluate_pretrained_model
from model import DnCNN

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
    pretrained_model_path: Path,
    data_path: Path,
    data_index: int,
    roi_row: slice,
    roi_column: slice,
):
    _,_,images = get_data(data_path)

    image_test = images[data_index][roi_row, roi_column]
    image_test_pytorch = torch.from_numpy(image_test).float().unsqueeze(0)
    plt.suptitle(fr"Test of trained model DnCNN")
    plt.subplot(131)
    plt.title("Test image")
    plt.imshow(image_test)
    plt.xticks([])
    plt.yticks([])

    image_denoised_pytorch = model(image_test_pytorch)
    image_denoised_numpy = torch_to_np(image_denoised_pytorch)

    model.load_state_dict(torch.load(pretrained_model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    image_denoised_pretrained_pytorch = model(image_test_pytorch)
    image_denoised_pretrained_numpy = torch_to_np(image_denoised_pretrained_pytorch)

    plt.subplot(132)
    plt.title("Denoised image - pretrained DnCNN")
    plt.imshow(image_denoised_pretrained_numpy)
    plt.xlabel(f"PSNR = {calculate_psnr(image_test, image_denoised_pretrained_numpy):.2f} dB")
    plt.xticks([])
    plt.yticks([])


    plt.subplot(133)
    plt.title("Denoised image - fine-tuned DnCNN")
    plt.imshow(image_denoised_numpy)
    plt.xlabel(f"PSNR = {calculate_psnr(image_test, image_denoised_numpy):.2f} dB")
    plt.xticks([])
    plt.yticks([])

    plt.subplots_adjust(
        top=1.0, bottom=0.031, left=0.031, right=0.977, hspace=0.2, wspace=0.068
    )
    plt.show()

    ROI_trained = get_rect_ROI(
        image=np.flipud(image_denoised_numpy), y=5, x=5, size=8, num=9, plot=True
    )
    plt.show()
    NPS_2D_trained = get_NPS_2D(  ROI_trained )
    rad_trained, intensity_trained = get_NPS_1D(NPS_2D_trained)

    ROI_pretrained = get_rect_ROI(
        image=np.flipud(image_denoised_pretrained_numpy), y=5, x=5, size=8, num=9, plot=True
    )
    plt.show()
    NPS_2D_pretrained  = get_NPS_2D(  ROI_pretrained )
    rad_pretrained , intensity_pretrained  = get_NPS_1D(NPS_2D_pretrained)

    x_interpolate = np.linspace(0, 1.6, 64)



    cubic_spline_trained = CubicSpline(rad_trained, intensity_trained)
    plt.plot(rad_trained,intensity_trained / max(intensity_trained) , ".", color='red')


    plt.plot(
        x_interpolate,
        cubic_spline_trained(x_interpolate) /max(intensity_trained),
        label=f"model fine-tuned DnCNN ",
        color='red',
    )

    cubic_spline_pretrained = CubicSpline(rad_pretrained, intensity_pretrained)
    plt.plot(rad_pretrained, intensity_pretrained / max(intensity_pretrained), ".", color='blue')

    plt.plot(
        x_interpolate,
        cubic_spline_pretrained(x_interpolate) /  max(intensity_pretrained),
        label=f"model pretrained DnCNN ",
        color='blue',
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
    trained_model_path = Path().resolve().parents[1] / "model" / "saved"
    data_path = Path().resolve().parents[1] / "data" / "test_dataset"

    pretrained_model_path = (
            Path().resolve().parents[1] / "model" / "pretrained" / "DnCNN"
    )

    model_trained = torch.load(
        Path().resolve().parents[1] / "model" / "saved" / "DnCNN_model_10_epoch_5000_scans_64_batch_size_0.0001_lr_25_noise_level.pt"
    )

    evaluate_trained_model(
        model=model_trained,
        pretrained_model_path = pretrained_model_path / "dncnn_25.pth",
        data_path=data_path,
        data_index=4,
        roi_row=slice(30, 130),
        roi_column=slice(250, 350),
    )
