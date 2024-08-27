import shutil
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


def generate_train_test_dataset(data_path: Path, test_size = 1000):
    npy_files = list(map(lambda f: str(f),list(data_path.glob('*.npy'))))
    np.random.shuffle(npy_files)

    test_files = map(lambda f: Path(f), npy_files[:test_size])
    train_files = map(lambda f: Path(f), npy_files[test_size:])

    # Move the test files
    for file in test_files:
        shutil.move(file, data_path/ "test_dataset" / file.name)

    # Move the train files
    for file in train_files:
        shutil.move(Path(file), data_path / "train_dataset" / file.name)


if __name__ == "__main__":
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        executor.map(process_file, files_to_process)
    end = time.perf_counter()
    print(f"Processing files in {end - start:0.4f} seconds")

    start = time.perf_counter()
    generate_train_test_dataset(data_path, 1000)
    end = time.perf_counter()
    print(f"Generated test,train dataset in {end - start:0.4f} seconds")

