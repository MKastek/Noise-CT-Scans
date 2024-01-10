from pathlib import Path

import torch
from matplotlib import pyplot as plt

from utils import get_data, torch_to_np, np_to_torch, make_two_plots, \
    make_evaluate_plot


def evaluate(model, roi_row, roi_column):
    pass


if __name__ == '__main__':
    model = torch.load(Path().cwd() / 'model' / 'CNN_model.pt')
    path = Path().resolve().parents[1] / "dane" / "KARDIO ZAMKNIETE" / "A001" / "DICOM" / "P1" / "E1" / "S3"
    images = get_data(path)
    roi_row = slice(30, 130)
    roi_column = slice(250, 350)
    img_noise = images[55][roi_row, roi_column] / 1350
    img_denoised = model(np_to_torch(img_noise).reshape(1, 1, 100, 100))

    rad_noise, intensity_noise, rad_denoise, intensity_denoised = make_evaluate_plot(img_noise,
                                                                                     torch_to_np(img_denoised))

    make_two_plots(rad_noise, intensity_noise, rad_denoise, intensity_denoised, "NPS 1D comparison after 10000 epochs",
                   "orginal", "denoised", min_x=0, max_x=1.0, num=64)
    plt.show()
