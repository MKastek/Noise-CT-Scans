from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from pydicom import dcmread
import numpy as np

from constants import BLACK_PIXEL_HU, WRONG_FILE_NAME, NUM_OF_BLACK_PIXELS, raw_data_path

files_to_process = list(raw_data_path.rglob("*"))

data_path = Path("data")


def process_file(file: Path):
    if file.is_file() and file.name != WRONG_FILE_NAME:
        dcm_file = dcmread(file)
        if np.count_nonzero(dcm_file.pixel_array == BLACK_PIXEL_HU) == NUM_OF_BLACK_PIXELS:
            with open(data_path / f"{str(file.parents[4].name)}_{str(file.parents[0].name)}_{file.name}.npy", 'wb') as f:
                np.save(f, np.flip(dcm_file.pixel_array))


if __name__ == "__main__":
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        executor.map(process_file, files_to_process)
    end = time.perf_counter()
    print(f"Processing files in {end - start:0.4f} seconds")