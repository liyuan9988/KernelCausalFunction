from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd

from src.ate.data.data_class import ATETrainDataSet, ATETestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")


def generate_train_jobcorp():
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        data = pd.read_csv(DATA_PATH.joinpath("job_corps/JCdata.csv"), sep=" ")

    sub = data
    sub = data.loc[data["m"] > 0, :]
    sub = sub.loc[sub["d"] >= 40, :]
    outcome = sub["m"].to_numpy()
    treatment = sub["d"].to_numpy()
    backdoor = sub.iloc[:, 3:].to_numpy()
    return ATETrainDataSet(backdoor=backdoor,
                           outcome=outcome[:,np.newaxis],
                           treatment=treatment[:,np.newaxis])


def generate_test_jobcorp():
    return ATETestDataSet(treatment=np.linspace(40, 2500, 1000)[:, np.newaxis],
                          structural=None)
