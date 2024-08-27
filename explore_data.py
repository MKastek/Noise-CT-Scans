import shutil
from pathlib import Path
from pydicom import dcmread
from collections import Counter

from torch import nn

from constants import raw_data_path, WRONG_FILE_NAME
import pandas as pd
import numpy as np

from model import BaseModel, DnCNN, DIP


def count_patient_scans(path: Path, output_path: Path):
    ids = []
    for file in path.rglob("*"):
        if file.is_file() and file.name != WRONG_FILE_NAME:
            dcm_file = dcmread(file)
            try:
                ids.append(dcm_file[(0x0010, 0x1000)].value)
            except KeyError:
                ids.append(dcm_file[(0x0010, 0x0030)].value)
    df = pd.DataFrame.from_dict(Counter(ids), orient='index', columns=["scans"])
    df.index.name = 'patient'
    df = df.reset_index()
    df.to_csv(output_path / "count_patient_scans.csv")
    return df


def get_sum_scans(path: Path):
    return pd.read_csv(path)["scans"].sum()

def get_layers_count(model: BaseModel):
    trainable_layer_count = sum(
        1 for layer in model.modules() if isinstance(layer, nn.Conv2d) and len(list(layer.parameters())) > 0)
    print(f'Total number of trainable layers: {trainable_layer_count}')



if __name__ == "__main__":
    get_layers_count(DIP())
    # generate_train_test_dataset(Path("data"))
    # counter = count_patient_scans(raw_data_path, Path("output") / "statistics")
    # counter_df = pd.DataFrame.from_dict(counter, orient='index').reset_index()

    # print(get_sum_scans(Path("output")/"statistics"/"count_patient_scans.csv"))



