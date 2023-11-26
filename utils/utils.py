from pathlib import Path
import numpy as np
import torch
from pydicom import dcmread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline

path = Path().resolve().parents[2] / "dane" / "KARDIO ZAMKNIETE" / "A001" / "DICOM" / "P1" / "E1" / "S1"


def get_data(data_path: Path) -> np.ndarray:
    """
    Return array with CT image data

    Parameters
    ----------
    data_path : Path
        Path to folder with data

    Returns
    -------
    `np.ndarray`

    """
    return np.stack([np.flip(dcmread(file).pixel_array) for file in data_path.iterdir()], axis=0)


def get_rect_ROI(image: np.ndarray, x: int, y: int, size: int, num: int, plot: bool = True) -> np.ndarray:
    """

    Parameters
    ----------
    image
    x
    y
    size
    num
    plot

    Returns
    -------

    """
    rect_size = int(np.sqrt(num))
    ROI_arr = np.empty((rect_size * rect_size, size, size))
    if plot:
        plt.pcolormesh(image)
        plt.colorbar()
        plt.title("Selected ROIs")
    for i in range(num):
        yy = y + size * (i % rect_size)
        if i % rect_size == 0:
            xx = x + size * (i // rect_size)
        if plot:
            plt.gca().add_patch(Rectangle((xx - size // 2, yy - size // 2), size, size,
                                          edgecolor='red',
                                          facecolor='none',
                                          lw=1))
        ROI_arr[i] = image[yy - size // 2:yy + size // 2, xx - size // 2:xx + size // 2]
    return ROI_arr


def get_NPS_2D(ROI_arr: np.ndarray, pixel_size_x: float = 0.402, pixel_size_y: float = 0.402):
    """

    Parameters
    ----------
    ROI_arr
    pixel_size_x
    pixel_size_y

    Returns
    -------

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
    return (1 / N) * (1 / (Lx * Ly)) * (np.sum(NPS_array, axis=0) * pixel_size_x * pixel_size_y)


def get_NPS_1D(NPS_2D: np.ndarray, size_of_pixel_in_spatial_domain: float = 0.402) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    NPS_2D
    size_of_pixel_in_spatial_domain

    Returns
    -------

    """
    cen_x = NPS_2D.shape[1] // 2
    cen_y = NPS_2D.shape[1] // 2

    # Find radial distances
    [X, Y] = np.meshgrid(np.arange(NPS_2D.shape[1]) - cen_x, np.arange(NPS_2D.shape[1]) - cen_y)
    R = np.sqrt(np.square(X) + np.square(Y))

    rad = np.arange(0, np.max(R), 1)
    intensity = np.zeros(len(rad))
    index = 0
    bin_size = 1

    for i in rad:
        mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
        rad_values = NPS_2D[mask]
        intensity[index] = np.mean(rad_values)
        index += 1
        # Plot data
    x = rad * 1.0 / (size_of_pixel_in_spatial_domain * NPS_2D.shape[1])
    y = intensity
    return x, y


def make_plot(x_points: np.ndarray, y_points: np.ndarray, title: str, legend: str,
              min_x: float = 0, max_x: float = 1.0, num: int = 64):
    """

    Parameters
    ----------
    x_points
    y_points
    title
    legend
    min_x
    max_x
    num

    Returns
    -------

    """
    x_interpolate = np.linspace(min_x, max_x, num)
    cubic_spline = CubicSpline(x_points, y_points)
    plt.plot(x_points, y_points, '.')
    plt.plot(x_interpolate, cubic_spline(x_interpolate), color='blue', label=legend)
    plt.title(title)
    plt.legend()
    plt.xlabel("$f_{r} [mm^{-1}]$")
    plt.xlim(min_x, max_x)
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.7)


def make_two_plots(x_points_1: np.ndarray, y_points_1: np.ndarray, x_points_2: np.ndarray, y_points_2: np.ndarray,
                   title: str, legend_1: str, legend_2: str, min_x: float = 0, max_x: float = 1.0, num: int = 64):
    """

    Parameters
    ----------
    x_points_1
    y_points_1
    x_points_2
    y_points_2
    title
    legend_1
    legend_2
    min_x
    max_x
    num

    Returns
    -------

    """
    x_interpolate = np.linspace(min_x, max_x, num)
    cubic_spline_1 = CubicSpline(x_points_1, y_points_1)
    cubic_spline_2 = CubicSpline(x_points_2, y_points_2)
    plt.plot(x_points_1, y_points_1, '.')
    plt.plot(x_points_2, y_points_2, '.')
    plt.plot(x_interpolate, cubic_spline_1(x_interpolate), label=legend_1)
    plt.plot(x_interpolate, cubic_spline_2(x_interpolate), label=legend_2)
    plt.legend()
    plt.title(title)
    plt.xlabel("$f_{r} [mm^{-1}]$")
    plt.xlim(min_x, max_x)
    plt.grid(which="major", alpha=0.7)


def np_to_torch(np_array):
    return torch.from_numpy(np_array).float()


def torch_to_np(torch_array):
    return np.squeeze(torch_array.detach().cpu().numpy())


def save_array(np_array: np.ndarray, file_name: str, file_path: Path = Path().resolve().parents[0] / "output" / "data"):
    with open(file_path / file_name, 'wb') as f:
        np.save(f, np_array)

def load_array(file_name):
    return np.load(Path('output') / file_name)



if __name__ == '__main__':
    image = get_data(path)[200]
    ROI_array_rectangle = get_rect_ROI(image=image, y=188, x=60, size=16, num=9, plot=True)
    plt.show()
    NPS_2D_rectangle = get_NPS_2D(ROI_array_rectangle)
    radius, intensity = get_NPS_1D(NPS_2D_rectangle)
    make_plot(radius, intensity, title="CT Scan - rectangle ROI", legend=" NPS 1D", min_x=0, max_x=1.0, num=64)
    plt.show()
