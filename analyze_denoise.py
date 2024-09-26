import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from utils import get_data, calculate_psnr

plt.rcParams["image.cmap"] = "gray"


class Filter:
    """Class to apply different types of filters."""

    @staticmethod
    def apply_mean(image, kernel_size):
        """Apply mean filter to the image."""
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def apply_median(image, kernel_size):
        """Apply median filter to the image."""
        return cv2.medianBlur(image.astype("float32"), kernel_size)

    @staticmethod
    def apply_gaussian(image, kernel_size, sigma):
        """Apply Gaussian filter to the image."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


class PSNRAnalyzer:
    """Class to analyze PSNR values with different filters."""

    def __init__(self, images):
        self.images = images
        self.psnr_results = {
            "mean_3": np.array([]),
            "mean_5": np.array([]),
            "median_3": np.array([]),
            "median_5": np.array([]),
            "gaussian_3": np.array([]),
            "gaussian_5": np.array([]),
        }

    def _calculate_psnr_for_filters(self):
        for image in self.images:
            self.psnr_results["mean_3"] = np.append(
                self.psnr_results["mean_3"], calculate_psnr(image, Filter.apply_mean(image, 3))
            )
            self.psnr_results["mean_5"] = np.append(
                self.psnr_results["mean_5"], calculate_psnr(image, Filter.apply_mean(image, 5))
            )
            self.psnr_results["median_3"] = np.append(
                self.psnr_results["median_3"], calculate_psnr(image, Filter.apply_median(image, 3))
            )
            self.psnr_results["median_5"] = np.append(
                self.psnr_results["median_5"], calculate_psnr(image, Filter.apply_median(image, 5))
            )
            self.psnr_results["gaussian_3"] = np.append(
                self.psnr_results["gaussian_3"], calculate_psnr(image, Filter.apply_gaussian(image, 3, 0))
            )
            self.psnr_results["gaussian_5"] = np.append(
                self.psnr_results["gaussian_5"], calculate_psnr(image, Filter.apply_gaussian(image, 5, 0))
            )

    def _print_array(self, name, arr):
        print(
            f"{name} min:{np.min(arr):.2f}({np.argmin(arr)}), max:{np.max(arr):.2f}({np.argmax(arr)}), avg:{np.average(arr):.2f}")

    def analyze(self):
        """Analyze and print PSNR values for each filter."""
        self._calculate_psnr_for_filters()

        # Print PSNR values for each filter
        self._print_array("Mean filter (3x3)", self.psnr_results["mean_3"])
        self._print_array("Mean filter (5x5)", self.psnr_results["mean_5"])
        self._print_array("Median filter (3x3)", self.psnr_results["median_3"])
        self._print_array("Median filter (5x5)", self.psnr_results["median_5"])
        self._print_array("Gaussian filter (3x3)", self.psnr_results["gaussian_3"])
        self._print_array("Gaussian filter (5x5)", self.psnr_results["gaussian_5"])

    def plot_psnr_results(self):
        """Plot boxplot for PSNR values of different filters."""
        medianprops = dict(linewidth=2.5)
        data = [
            self.psnr_results["mean_3"],
            self.psnr_results["mean_5"],
            self.psnr_results["median_3"],
            self.psnr_results["median_5"],
            self.psnr_results["gaussian_3"],
            self.psnr_results["gaussian_5"]
        ]
        labels = [
            "Mean (3x3)",
            "Mean (5x5)",
            "Median (3x3)",
            "Median (5x5)",
            "Gaussian (3x3)",
            "Gaussian (5x5)"
        ]
        plt.figure(figsize=(8, 6))
        plt.boxplot(data, medianprops=medianprops)
        plt.xticks(ticks=range(1, 7), labels=labels, fontsize=16)
        plt.grid(which="minor", alpha=0.3)
        plt.grid(which="major", alpha=0.7)

        plt.title("Filters PSNR values", fontsize=20)
        plt.show()


class Plotter:
    """Class to handle plotting images and results."""

    @staticmethod
    def make_plot(image, denoised_image, title, title_image, title_denoised_image, save=False):
        """Plot the original and denoised images side by side."""
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
        if save:
            plt.savefig(f"ch05_{title.replace(' ', '_') + title_denoised_image.replace(' ', '_')}.png")
        else:
            plt.show()


if __name__ == "__main__":
    # Load the data
    data_path = Path() / "data" / "train_dataset"
    _, _, images = get_data(data_path, num_scans=1000)

    # Analyze PSNR values for different filters
    psnr_analyzer = PSNRAnalyzer(images[:1000])
    psnr_analyzer.analyze()
    psnr_analyzer.plot_psnr_results()

    # Test image for visual comparisons
    test_image = images[2]

    # Plot results for each filter
    filters = {
        "Mean (3x3)": Filter.apply_mean(test_image, 3),
        "Mean (5x5)": Filter.apply_mean(test_image, 5),
        "Median (3x3)": Filter.apply_median(test_image, 3),
        "Median (5x5)": Filter.apply_median(test_image, 5),
        "Gaussian (3x3)": Filter.apply_gaussian(test_image, 3, 0),
        "Gaussian (5x5)": Filter.apply_gaussian(test_image, 5, 0)
    }

    for title, denoised_image in filters.items():
        Plotter.make_plot(
            test_image,
            denoised_image,
            f"Denoised image with {title} filter",
            "Test image",
            f"Denoised image with {title} filter"
        )