import argparse

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re


def init_analyze_loss_parser():
    parser = argparse.ArgumentParser(description="Training DnCNN")
    parser.add_argument(
        "--parameter", help="Check parameter", type=str,
    )
    parser.add_argument(
        "--model", help="Check parameter", type=str, default="DnCNN"
    )
    parser.add_argument(
        "--num_scans", help="Number of CT scans", type=int, default=5000,
    )
    parser.add_argument(
        "--num_epochs", help="Number of epochs", type=int, default=20,
    )
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)

    return parser

def get_parameters(args):
    match args.parameter:
        case "batch_size":
            pattern = f"loss_{args.model}_model_{args.num_epochs}_epoch_{args.num_scans}_scans_*_batch_size_{args.lr}_lr.npy"
            regex_pattern = re.compile(r'(\d+)_batch_size')
            str2num = lambda str: int(str)
            title = "Epoch Mean Squared Error loss with different batch size"
            return pattern, regex_pattern, str2num, title
        case "lr":
            pattern = f"loss_{args.model}_model_10_epoch_1000_scans_32_batch_size_*_lr.npy"
            regex_pattern = re.compile(r'(0\.\d+|1e-\d+)_lr')
            str2num = lambda str: format(float(str), ".1e")
            title = "Epoch Mean Squared Error loss with different learning rate"
            return pattern, regex_pattern, str2num, title



folder_path = Path('model') / "saved"

if __name__ == "__main__":
    analyze_loss_parser = init_analyze_loss_parser()
    args = analyze_loss_parser.parse_args()
    pattern, regex_pattern, str2num, title = get_parameters(args)
    file_batch_pairs = [(file, str2num(regex_pattern.search(file.name).group(1))) for file in list(folder_path.glob(pattern))]

    file_batch_pairs.sort(key=lambda pair: pair[1])

    for file, batch_size in file_batch_pairs:
        loss = np.load(file)
        plt.scatter(np.arange(1, len(loss) + 1, 1), loss)
        plt.plot(np.arange(1, len(loss) + 1, 1), loss, label=batch_size, linestyle="-")
    plt.title(title)
    plt.ylabel("Epoch Mean Squared Error loss")
    plt.xlabel("epochs")
    plt.grid(which="major", alpha=0.7)
    plt.grid(which="minor", alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()

