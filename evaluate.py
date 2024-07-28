from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import (
    get_data,
    torch_to_np,
    np_to_torch,
    make_two_plots,
    make_evaluate_plot,
    load_array,
)


def evaluate_CNN(model, img_noise, title="NPS 1D comparison after 100 epochs"):
    img_noise_torch = np_to_torch(img_noise).reshape(1, 1, 100, 100)
    img_denoised = torch_to_np(model(img_noise_torch))
    rad_noise, intensity_noise, rad_denoise, intensity_denoised = make_evaluate_plot(
        img_noise, img_denoised
    )
    make_two_plots(
        rad_noise,
        intensity_noise,
        rad_denoise,
        intensity_denoised,
        title,
        "orginal",
        "denoised",
        min_x=0,
        max_x=1.0,
        num=64,
    )
    plt.show()


def evaluate_DIP(
    img_noise_path, img_denoised_path, title="NPS 1D comparison after 5000 epochs"
):
    img_noise = load_array(img_noise_path)
    img_denoised = load_array(img_denoised_path)
    rad_noise, intensity_noise, rad_denoise, intensity_denoised = make_evaluate_plot(
        img_noise, img_denoised, 65, 55
    )
    make_two_plots(
        rad_noise,
        intensity_noise,
        rad_denoise,
        intensity_denoised,
        title,
        "orginal",
        "denoised",
        min_x=0,
        max_x=1.0,
        num=64,
    )
    plt.show()




if __name__ == "__main__":
    model = torch.load(Path().cwd() / "model" / "saved" / "CNN_model.pt")
    path = (
        Path().resolve().parents[1]
        / "dane"
        / "KARDIO ZAMKNIETE"
        / "A001"
        / "DICOM"
        / "P1"
        / "E1"
        / "S3"
    )
    images = get_data(path)
    roi_row = slice(30, 130)
    roi_column = slice(250, 350)
    img_noise = images[55][roi_row, roi_column] / 1350
    evaluate_CNN(model, img_noise)
    evaluate_DIP("orginal_image_5000.npy", "denoised_image_5000.npy")





