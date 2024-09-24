from pathlib import Path

import argparse
import numpy as np
import torch
from torch import optim
from model import DIP
from utils import np_to_torch, torch_to_np, save_array, get_data, get_noise, nrmse
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "gray"


def init_dip_parser():
    parser = argparse.ArgumentParser(description="Training DnCNN")
    parser.add_argument(
        "--num_epochs", help="Number of epochs", type=int, default=5000,
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--data_path", type=str, default="data/train_dataset")
    return parser


def train_DIP(
    model,
    optimizer,
    reference_image: np.array,
    input_random_image: np.array,
    epochs: int,
    lr: float,
    plot: bool,
    output_filename: str = "DIP_model",
    name: str = "1",
):
    image_torch = np_to_torch(reference_image)
    train_loss, train_noise = list(), list()
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    for ep in range(epochs):
        optimizer.zero_grad()
        output_image = model(input_random_image)
        loss = nrmse(output_image, image_torch)
        train_loss.append(loss.item())
        train_noise.append(get_noise(torch_to_np(output_image)))
        loss.backward()
        optimizer.step()
        if ep % 4 == 0 and plot == True:
            axs[0, 0].set_title("Input image")
            axs[0, 0].imshow(input_random_image)
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
    save_array(
        output_image_np,
        f"denoised_image_{epochs}_epochs_{lr}_lr_exp_{name}.npy",
        file_path=Path("images"),
    )
    save_array(
        reference_image,
        f"orginal_image_{epochs}_epochs_{lr}_lr_exp_{name}.npy",
        file_path=Path("images"),
    )
    save_array(
        np.array(train_loss),
        f"loss_dip_{epochs}_epochs_{lr}_lr_exp_{name}.npy",
        file_path=Path("images"),
    )
    torch.save(
        model, Path() / "model" / "saved" / f"{output_filename}_{epochs}_name{name}.pt"
    )
    return output_image_np


if __name__ == "__main__":
    dip_parser = init_dip_parser()
    args = dip_parser.parse_args()

    model = DIP(1, 1, 100)
    model = model.to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    data_path = Path() / "data" / "train_dataset"
    _, _, images = get_data(data_path, num_scans=10)
    input_random_image = torch.rand(100, 100)
    name = "0"
    output_image = train_DIP(
        model,
        optimizer,
        images[name],
        input_random_image,
        args.num_epochs,
        args.lr,
        False,
        name=str(name),
    )
