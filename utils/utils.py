from pathlib import Path
import numpy as np
import torch
from numpy import ndarray
from pydicom import dcmread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline
from torch.autograd import Variable

path = (
    Path().resolve().parents[2]
    / "dane"
    / "KARDIO ZAMKNIETE"
    / "A001"
    / "DICOM"
    / "P1"
    / "E1"
    / "S1"
)


def get_data(
    data_path: Path,
    skip_index: np.ndarray = np.concatenate(
        (
            np.arange(61, 67),
            np.arange(68, 78),
            np.arange(79, 89),
            np.arange(90, 100),
            np.arange(101, 111),
            np.arange(113, 123),
            np.arange(124, 132),
        ),
        axis=0,
    ),
) -> np.ndarray:
    """
    Return array with CT image data
    """
    return np.stack(
        [
            np.flip(dcmread(file).pixel_array)
            for idx, file in enumerate(data_path.iterdir())
            if idx not in skip_index
        ],
        axis=0,
    )


def get_rect_ROI(
    image: np.ndarray, x: int, y: int, size: int, num: int, plot: bool = True, ax=None
) -> np.ndarray:
    """
    Return array rectangular region of interest (ROIs)
    """
    rect_size = int(np.sqrt(num))
    ROI_arr = np.empty((rect_size * rect_size, size, size))
    if plot:
        if ax is None:
            plt.pcolormesh(image)
            plt.colorbar()
            plt.title("Selected ROIs")
        else:
            ax.pcolormesh(image)
    for i in range(num):
        yy = y + size * (i % rect_size)
        if i % rect_size == 0:
            xx = x + size * (i // rect_size)
        if plot:
            if ax is None:
                plt.gca().add_patch(
                    Rectangle(
                        (xx - size // 2, yy - size // 2),
                        size,
                        size,
                        edgecolor="red",
                        facecolor="none",
                        lw=1,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (xx - size // 2, yy - size // 2),
                        size,
                        size,
                        edgecolor="red",
                        facecolor="none",
                        lw=1,
                    )
                )

        ROI_arr[i] = image[
            yy - size // 2 : yy + size // 2, xx - size // 2 : xx + size // 2
        ]
    return ROI_arr


def get_NPS_2D(
    ROI_arr: np.ndarray, pixel_size_x: float = 0.402, pixel_size_y: float = 0.402
):
    """
    Return array with 2D Noise Power Spectrum
    """
    NPS_array = np.empty_like(ROI_arr)
    for i, roi in enumerate(ROI_arr):
        rows, cols = roi.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x, y = np.meshgrid(x, y)
        z = np.polyfit(x.ravel(), y.ravel(), 2)
        poly_fit = np.polyval(z, x)
        # Subtract the polynomial fit from the image
        roi = roi - poly_fit
        # calculation od DFT 2D
        dft = np.fft.fft2(roi - np.mean(roi))
        # shift of FT
        shifted_dft = np.fft.fftshift(dft)
        # calcation of absolute value
        NPS_array[i] = np.abs(shifted_dft) ** 2
    N = len(NPS_array)
    Ly = NPS_array[0].shape[0]
    Lx = NPS_array[0].shape[1]
    return (
        (1 / N)
        * (1 / (Lx * Ly))
        * (np.sum(NPS_array, axis=0) * pixel_size_x * pixel_size_y)
    )


def get_NPS_1D(
    NPS_2D: np.ndarray, size_of_pixel_in_spatial_domain: float = 0.402
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return array with radial frequency and Noise Power Spectrum 1D values
    """
    cen_x = NPS_2D.shape[0] // 2
    cen_y = NPS_2D.shape[1] // 2

    [X, Y] = np.meshgrid(
        np.arange(NPS_2D.shape[1]) - cen_x, np.arange(NPS_2D.shape[1]) - cen_y
    )
    R = np.sqrt(np.square(X) + np.square(Y))

    rad = np.arange(0, np.max(R), 1)
    intensity = np.zeros(len(rad))
    index = 0
    bin_size = 1

    for i in rad:
        mask = np.greater(R, i - bin_size) & np.less(R, i + bin_size)
        rad_values = NPS_2D[mask]
        intensity[index] = np.mean(rad_values)
        index += 1

    f_radial = rad * 1.0 / (size_of_pixel_in_spatial_domain * NPS_2D.shape[1])
    NPS_1D = intensity
    return f_radial, NPS_1D


def get_noise(NPS_2D: np.ndarray):
    """
    Return noise calculated as 2D integral of 2D Noise Power Spectrum
    """
    return np.sqrt(np.trapz(np.trapz(NPS_2D, axis=0), axis=0))


def make_plot(
    x_points: np.ndarray,
    y_points: np.ndarray,
    title: str,
    legend: str,
    min_x: float = 0,
    max_x: float = 1.0,
    num: int = 64,
    ax=None,
):
    """
    Make plot of points with interpolated function with cubic splines
    """
    x_interpolate = np.linspace(min_x, max_x, num)
    cubic_spline = CubicSpline(x_points, y_points)
    if ax is None:
        plt.plot(x_points, y_points, ".")
        plt.plot(x_interpolate, cubic_spline(x_interpolate), color="blue", label=legend)
        plt.title(title)
        plt.legend()
        plt.xlabel("$f_{r} [mm^{-1}]$")
        plt.xlim(min_x, max_x)
        plt.grid(which="minor", alpha=0.3)
        plt.grid(which="major", alpha=0.7)
    else:
        ax.plot(x_points, y_points, ".")
        ax.plot(x_interpolate, cubic_spline(x_interpolate), color="blue", label=legend)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel("$f_{r} [mm^{-1}]$")
        ax.set_xlim(min_x, max_x)
        ax.grid(which="minor", alpha=0.3)
        ax.grid(which="major", alpha=0.7)


def make_two_plots(
    x_points_1: np.ndarray,
    y_points_1: np.ndarray,
    x_points_2: np.ndarray,
    y_points_2: np.ndarray,
    title: str,
    legend_1: str,
    legend_2: str,
    min_x: float = 0,
    max_x: float = 1.0,
    num: int = 64,
):
    """
    Make plot of two sets of points with interpolated functions with cubic splines
    """
    x_interpolate = np.linspace(min_x, max_x, num)
    cubic_spline_1 = CubicSpline(x_points_1, y_points_1)
    cubic_spline_2 = CubicSpline(x_points_2, y_points_2)
    plt.plot(x_points_1, y_points_1, ".")
    plt.plot(x_points_2, y_points_2, ".")
    plt.plot(x_interpolate, cubic_spline_1(x_interpolate), label=legend_1)
    plt.plot(x_interpolate, cubic_spline_2(x_interpolate), label=legend_2)
    plt.legend()
    plt.title(title)
    plt.xlabel("$f_{r} [mm^{-1}]$")
    plt.xlim(min_x, max_x)
    plt.grid(which="major", alpha=0.7)


def make_evaluate_plot(
    img_noise: np.ndarray, img_denoised: np.ndarray, y: int = 55, x: int = 55
):
    """
    Make evaluated plot of denoised image
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ROI_array_rectangle_noise = get_rect_ROI(
        image=np.flipud(img_noise), y=y, x=x, size=8, num=9, plot=True, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Noise image")

    ROI_array_rectangle_denoised = get_rect_ROI(
        image=np.flipud(img_denoised),
        y=y,
        x=x,
        size=8,
        num=9,
        plot=True,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Denoised image")

    NPS_2D_rectangle_noise = get_NPS_2D(ROI_array_rectangle_noise)
    rad_noise, intensity_noise = get_NPS_1D(NPS_2D_rectangle_noise)

    make_plot(
        rad_noise,
        intensity_noise,
        title="CT Scan - rectangle ROI of original image",
        legend=" NPS 1D",
        min_x=0,
        max_x=1.0,
        num=64,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("NPS 1D noised image")
    NPS_2D_rectangle_denoised = get_NPS_2D(ROI_array_rectangle_denoised)
    rad_denoise, intensity_denoised = get_NPS_1D(NPS_2D_rectangle_denoised)
    make_plot(
        rad_denoise,
        intensity_denoised,
        title="CT Scan - rectangle ROI of original image",
        legend=" NPS 1D",
        min_x=0,
        max_x=1.0,
        num=64,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("NPS 1 denoised image")
    plt.tight_layout()
    plt.show()
    return rad_noise, intensity_noise, rad_denoise, intensity_denoised


def np_to_torch(np_array):
    """
    Convert numpy array to torch tensor
    """
    return torch.from_numpy(np_array).float()


def torch_to_np(torch_array):
    """
     Convert torch tensor to numpy array
     """
    return np.squeeze(torch_array.detach().cpu().numpy())


def save_array(
    np_array: np.ndarray,
    file_name: str,
    file_path: Path = Path().resolve().parents[0] / "output" / "data",
):
    with open(file_path / file_name, "wb") as f:
        np.save(f, np_array)


def load_array(file_name):
    return np.load(Path("output/data") / file_name)


def add_noise(image, noise_level=0.05):
    """
    Add random Gaussian noise to the input image.
    """
    noise = Variable(image.data.new(image.size()).normal_(0, noise_level) * 0.5)
    noisy_image = image + noise

    return noisy_image, noise


def get_max(data_path: Path):
    """
    Get maximum value from all images in given folders
    """
    images = get_data(data_path)
    return np.max(images)


def nrmse(recon_img, reference_img):
    """
    Normalized Root Mean Square Error
    """
    n = (reference_img - recon_img) ** 2
    den = reference_img ** 2
    return 100.0 * torch.mean(n) ** 0.5 / torch.mean(den) ** 0.5
