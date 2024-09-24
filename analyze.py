from pathlib import Path

from matplotlib import pyplot as plt

from utils import get_NPS_2D, get_data, get_rect_ROI, get_NPS_1D, make_plot

if __name__ == "__main__":
    data_path = Path() / "data" / "train_dataset"
    _, _, image = get_data(data_path)
    image = image[2]
    plt.imshow(image)
    plt.show()
    ROI_array_rectangle = get_rect_ROI(
        image=image, y=40, x=40, size=8, num=16, plot=True
    )
    plt.show()
    NPS_2D_rectangle = get_NPS_2D(ROI_array_rectangle)
    plt.imshow(NPS_2D_rectangle)
    plt.title("NPS 2D")
    plt.show()
    radius, intensity = get_NPS_1D(NPS_2D_rectangle)
    make_plot(
        radius,
        intensity,
        title="CT Scan - rectangle ROI",
        legend=" NPS 1D",
        min_x=0,
        max_x=1.0,
        num=64,
    )
    plt.show()
