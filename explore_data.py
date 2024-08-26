from pathlib import Path
from pydicom import dcmread
from collections import Counter
from constants import raw_data_path, WRONG_FILE_NAME
import pandas as pd


def count_patient_scans(path: Path):
    ids = []
    for file in path.rglob("*"):
        if file.is_file() and file.name != WRONG_FILE_NAME:
            dcm_file = dcmread(file)
            try:
                ids.append( dcm_file[(0x0010, 0x1000)].value)
            except KeyError:
                ids.append(dcm_file[(0x0010, 0x0030)].value)
    return Counter(ids)


if __name__ == "__main__":
    counter = count_patient_scans(raw_data_path)
    counter_df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    print(counter_df)
    print(counter_df.columns)