from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import get_data, calculate_psnr

plt.rcParams["image.cmap"] = "gray"


def apply_mean_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image.astype("float32"), kernel_size)


def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def analyze_psnr(images):
    def print_array(str, arr):
        print(
            f"{str} min:{np.min(arr):.2f}({np.argmin(arr)}), max:{np.max(arr):.2f}({np.argmax(arr)}), avg:{np.average(arr):.2f}"
        )

    psnr_mean_3 = np.array([])
    psnr_mean_5 = np.array([])
    psnr_median_3 = np.array([])
    psnr_median_5 = np.array([])
    psnr_gaussian_3 = np.array([])
    psnr_gaussian_5 = np.array([])
    for image in images:
        np.append(psnr_mean_3, calculate_psnr(image, apply_mean_filter(image, 3)))
        np.append(psnr_mean_5, calculate_psnr(image, apply_mean_filter(image, 5)))
        np.append(psnr_median_3, calculate_psnr(image, apply_median_filter(image, 3)))
        np.append(psnr_median_5, calculate_psnr(image, apply_median_filter(image, 5)))
        np.append(
            psnr_gaussian_3, calculate_psnr(image, apply_gaussian_filter(image, 3, 0))
        )
        np.append(
            psnr_gaussian_5, calculate_psnr(image, apply_gaussian_filter(image, 5, 0))
        )
    print_array("Mean filter(3x3)", psnr_mean_3)
    print_array("Mean filter(5x5)", psnr_mean_5)
    print_array("Median filter(3x3)", psnr_median_3)
    print_array("Median filter(5x5)", psnr_median_5)
    print_array("Gaussian filter(3x3)", (psnr_gaussian_3))
    print_array("Gaussian filter(5x5)", psnr_gaussian_5)


def make_plot(image, denoised_image, title, title_image, title_denoised_image):
    plt.suptitle(title)
    plt.subplot(121)
    plt.title(title_image)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.title(title_denoised_image)
    plt.imshow(denoised_image)
    plt.xlabel(f"PSNR = {calculate_psnr(image, denoised_image):.2f} dB")
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(
        top=0.98, bottom=0.152, left=0.031, right=0.97, hspace=0.122, wspace=0.063
    )
    plt.savefig(
        f"ch05_{ title.replace(' ','_') +title_denoised_image.replace(' ','_')}.png"
    )


if __name__ == "__main__":
    data_path = Path() / "data" / "train_dataset"
    _, _, images = get_data(data_path, num_scans=10)
    test_image = images[2]

    make_plot(
        test_image,
        apply_mean_filter(test_image, 3),
        "Denoised image with mean filter",
        "Test image",
        "Denoised image with (3x3) fiter",
    )

    make_plot(
        test_image,
        apply_mean_filter(test_image, 5),
        "Denoised image with mean filter",
        "Test image",
        "Denoised image with (5x5) fiter",
    )

    make_plot(
        test_image,
        apply_median_filter(test_image, 3),
        "Denoised image with median filter",
        "Test image",
        "Denoised image with (3x3) fiter",
    )

    make_plot(
        test_image,
        apply_median_filter(test_image, 5),
        "Denoised image with median filter",
        "Test image",
        "Denoised image with (5x5) fiter",
    )

    make_plot(
        test_image,
        apply_gaussian_filter(test_image, 3, 0),
        "Denoised image with gaussian filter",
        "Test image",
        "Denoised image with (3x3) fiter",
    )

    make_plot(
        test_image,
        apply_gaussian_filter(test_image, 5, 0),
        "Denoised image with gaussian filter",
        "Test image",
        "Denoised image with (5x5) fiter",
    )
    analyze_psnr(images[:1000])
