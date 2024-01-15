from pathlib import Path

import numpy as np
import torch
from loss import nrmse
from model import CNN_DIP
from utils import np_to_torch, torch_to_np, save_array, get_data
import matplotlib.pyplot as plt


def get_denoised_image_DIP(reference_image, input_random_image, model, epochs, plot, output_filename="DIP_model"):
    device = torch.device('cpu')
    image_torch = np_to_torch(reference_image).to(device)
    model_dip = model.to(device)
    input_image = input_random_image.to(device)
    optimiser = torch.optim.Adam(model_dip.parameters(), lr=1e-4)
    train_loss = list()
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    for ep in range(epochs):
        optimiser.zero_grad()
        output_image = model_dip(input_image)

        loss = nrmse(output_image, image_torch)

        train_loss.append(loss.item())
        loss.backward()
        optimiser.step()
        if ep % 4 == 0 and plot == True:
            axs[0, 0].set_title("Input image")
            axs[0, 0].imshow(input_image)
            axs[0, 1].set_title("Denoised image")
            axs[0, 1].imshow(torch_to_np(output_image))
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
    torch.save(model, Path().resolve().parents[0] / 'model' / f"{output_filename}.pt")
    return output_image_np


if __name__ == '__main__':
    print( Path().resolve().parents[0])
    path = Path().resolve().parents[2] / "dane" / "KARDIO ZAMKNIETE" / "A001" / "DICOM" / "P1" / "E1" / "S1"
    images = get_data(path)
    roi_images = np.array([image[30:130, 250:350] for image in images])
    roi_image = roi_images[200]
    input_random_image = torch.rand(100, 100)
    model = CNN_DIP(16, 100, 3)
    epochs = 20
    output_image = get_denoised_image_DIP(roi_image, input_random_image, model, epochs, True)