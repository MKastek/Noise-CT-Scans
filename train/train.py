from pathlib import Path

import torch

from model.model import CNNConfigurable
from loss import nrmse
from utils import np_to_torch, torch_to_np, save_array, get_data
import matplotlib.pyplot as plt


def get_denoised_image_DIP(reference_image, input_random_image, model, epochs, plot):

    nxd = reference_image.shape[0]
    device = torch.device('cpu')
    image_torch = np_to_torch(reference_image).to(device)
    cnn = model.to(device)
    input_image = input_random_image.to(device)
    optimiser = torch.optim.Adam(cnn.parameters(),lr=1e-4)
    train_loss = list()
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    for ep in range(epochs):
        optimiser.zero_grad()
        output_image = cnn(input_image)

        loss = nrmse(output_image,image_torch )

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
    plt.show()
    output_image_np = torch_to_np(output_image)
    save_array(output_image_np, f"denoised_image_{epochs}.npy")
    save_array(reference_image, f"orginal_image_{epochs}.npy")
    return output_image_np


if __name__ == '__main__':
    path = Path().resolve().parents[2] / "dane" / "KARDIO ZAMKNIETE" / "A001" / "DICOM" / "P1" / "E1" / "S1"
    image = get_data(path)[200]
    torch.manual_seed(0)
    reference_image = image[30:130, 250:350]
    input_random_image = torch.rand(100, 100)
    model = CNNConfigurable(16, 100, 3)
    epochs = 10
    output_image = get_denoised_image_DIP(reference_image, input_random_image, model, epochs, False)
