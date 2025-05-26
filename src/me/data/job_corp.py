from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd
from itertools import product

from src.me.data.data_class import METrainDataSet, METestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")


def generate_train_jobcorp():
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        data = pd.read_csv(DATA_PATH.joinpath("job_corps/JCdata.csv"), sep=" ")

    data = data.loc[data["m"] > 0, :]
    data = data.loc[data["d"] >= 40, :]
    outcome = data["y"].to_numpy()
    treatment = data["d"].to_numpy()
    mediate = data["m"].to_numpy()
    backdoor = data.iloc[:, 3:].to_numpy()

    return METrainDataSet(outcome=outcome[:, np.newaxis],
                          treatment=treatment[:, np.newaxis],
                          mediate=mediate[:, np.newaxis],
                          backdoor=backdoor)


def generate_test_jobcorp():
    xv, yv = np.meshgrid(np.linspace(0.0, 2000.0, 51), np.linspace(0.0, 2000.0, 51))
    treatment = xv.ravel()
    new_treatment = yv.ravel()
    return METestDataSet(treatment=treatment[:, np.newaxis],
                         new_treatment=new_treatment[:, np.newaxis],
                         structural=None)
