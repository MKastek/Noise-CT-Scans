from pathlib import Path

import numpy as np
import torch
from model import DIP
from utils import np_to_torch, torch_to_np, save_array, get_data, get_noise, nrmse
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

def get_denoised_image_DIP(
    reference_image: np.array,
    input_random_image: np.array,
    model,
    epochs: int,
    plot: bool,
    output_filename: str = "DIP_model",
) -> np.ndarray:
    """
    Train DIP, return denoised image
    """
    device = torch.device("cpu")
    image_torch = np_to_torch(reference_image).to(device)
    model_dip = model.to(device)
    input_image = input_random_image.to(device)
    optimiser = torch.optim.Adam(model_dip.parameters(), lr=1e-4)
    train_loss, train_noise = list(), list()
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    for ep in range(epochs):
        optimiser.zero_grad()
        output_image = model_dip(input_image)
        loss = nrmse(output_image, image_torch)
        train_loss.append(loss.item())
        train_noise.append(get_noise(torch_to_np(output_image)))
        loss.backward()
        optimiser.step()
        print(f"Epoch: {ep}")
        if ep % 4 == 0 and plot == True:
            axs[0, 0].set_title("Input image")
            axs[0, 0].imshow(input_image)
            axs[0, 1].set_title("Denoised image")
            axs[0, 1].imshow(torch_to_np(output_image))
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])


            axs[1, 0].cla()
            axs[1, 0].set_title(f"Loss epoch:{ep}")
            axs[1, 0].plot(train_loss)
            axs[1, 0].grid()
            axs[1, 1].set_title("Orginal image")
            axs[1, 1].imshow(image_torch)
            plt.pause(0.005)

    output_image_np = torch_to_np(output_image)
    save_array(output_image_np, f"denoised_image_{epochs}.npy")
    save_array(reference_image, f"orginal_image_{epochs}.npy")
    torch.save(
        model, Path().resolve().parents[0] / "model" / "saved" / f"{output_filename}.pt"
    )
    return output_image_np


if __name__ == "__main__":
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
    images = get_data(path)
    roi_images = np.array([image[30:130, 250:350] for image in images])
    roi_image = roi_images[220]
    input_random_image = torch.rand(100, 100)
    model = DIP(1, 1, 10)
    epochs = 5000
    output_image = get_denoised_image_DIP(
        roi_image, input_random_image, model, epochs, True
    )
