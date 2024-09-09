from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_data, add_noise


class TrainDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        num_scans: int = 10000,
        roi_row=slice(30, 130),
        roi_column=slice(250, 350),
        noise_level=25,
    ):
        self.roi_row = roi_row
        self.roi_column = roi_column
        self.noise_level = noise_level
        self.min, self.max, self.data = get_data(data_path, num_scans)
        self.data = np.array([(image[roi_row, roi_column]) for image in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(
            self.data[idx].reshape(
                1,
                (self.roi_row.stop - self.roi_row.start),
                (self.roi_column.stop - self.roi_column.start),
            ),
            dtype=torch.float32,
        )
        noise_image, noise = add_noise(image, self.min, self.max)
        return noise_image, image
