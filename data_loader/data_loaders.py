from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_data, add_noise


class TrainDataset(Dataset):

    def __init__(self, data_path: Path, roi_row=slice(30, 130), roi_column=slice(250, 350), normalize=1350):
        self.roi_row = roi_row
        self.roi_column = roi_column
        self.data = np.array([(image[roi_row, roi_column]) / normalize for image in get_data(data_path)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx].reshape(1, (self.roi_row.stop - self.roi_row.start),
                                                    (self.roi_column.stop - self.roi_column.start)),
                             dtype=torch.float32)
        noise_image, noise = add_noise(image)
        return noise_image, image

