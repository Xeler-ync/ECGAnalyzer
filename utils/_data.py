import pandas as pd
import numpy as np
import wfdb
import ast

from utils._config import PATH, SAMPLING_RATE, ECG_INDEX


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + df.filename_lr.values[ECG_INDEX])]
    else:
        data = [wfdb.rdsamp(path + df.filename_hr.values[ECG_INDEX])]
    data = np.array([sig for sig, meta in data])
    return data


# Load and convert annotation data
Y = pd.read_csv(PATH + "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, SAMPLING_RATE, PATH)
