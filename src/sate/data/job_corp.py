from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd
from itertools import product

from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")


def generate_train_jobcorp():
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        X1 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/X1.csv"), sep=",").to_numpy()
        X2 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/X2.csv"), sep=",").to_numpy()
        D1 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/D1.csv"), sep=",").to_numpy()
        D2 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/D2.csv"), sep=",").to_numpy()
        Y = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/Y1.csv"), sep=",").to_numpy()

    flg = (D1[:, 0] + D2[:, 0]) > 40
    flg = np.logical_and(flg, Y[:, 0] > 0)
    return SATETrainDataSet(backdoor_1st=X1[flg],
                            backdoor_2nd=X2[flg],
                            treatment_1st=D1[flg],
                            treatment_2nd=D2[flg],
                            outcome=Y[flg])


def generate_test_jobcorp():
    treatment = np.linspace(40.0, 2500.0, 35)
    data = np.array(list(product(treatment, treatment)))
    treatment_1st = data[:, [0]]
    treatment_2nd = data[:, [1]]

    return SATETestDataSet(treatment_1st=treatment_1st,
                           treatment_2nd=treatment_2nd,
                           structural=None)
