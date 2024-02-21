from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd
from itertools import product

from src.cate.data.data_class import CATETrainDataSet, CATETestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")


def generate_train_jobcorp():
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        data = pd.read_csv(DATA_PATH.joinpath("job_corps/JCdata.csv"), sep=" ")

    sub = data.loc[data["m"] > 0, :]
    sub = sub.loc[sub["d"] >= 40, :]
    outcome = sub["m"].to_numpy()
    treatment = sub["d"].to_numpy()
    covariate = sub["age"].to_numpy()
    backdoor = sub.iloc[:, 3:].drop("age", axis=1).to_numpy()
    return CATETrainDataSet(backdoor=backdoor,
                            outcome=outcome[:, np.newaxis],
                            treatment=treatment[:, np.newaxis],
                            covariate=covariate[:, np.newaxis])


def generate_test_jobcorp():
    treatment = np.linspace(40, 2500, 1000)
    covariate = np.arange(16, 25)
    data = np.array(list(product(treatment, covariate)))
    return CATETestDataSet(treatment=data[:, [0]],
                           covariate=data[:, [1]],
                           structural=None)
