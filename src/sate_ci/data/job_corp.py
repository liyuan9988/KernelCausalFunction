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
    flg = np.logical_and(flg, D1[:, 0] < 2000)
    flg = np.logical_and(flg, D2[:, 0] < 2000)

    tmp = pd.get_dummies(pd.cut(D1[flg, 0], [-1.0, 1000, 2000]))
    treatment_1st = tmp.to_numpy()

    tmp = pd.get_dummies(pd.cut(D2[flg, 0], [-1.0, 1000, 2000]))
    treatment_2nd = tmp.to_numpy()

    return SATETrainDataSet(backdoor_1st=X1[flg],
                            backdoor_2nd=X2[flg],
                            treatment_1st=treatment_1st,
                            treatment_2nd=treatment_2nd,
                            outcome=Y[flg])


def generate_test_jobcorp():
    tmp = []
    for i in range(2):
        for j in range(2):
            tmp.append([i,j])
    tmp = np.array(tmp, dtype=int)

    return SATETestDataSet(treatment_1st=np.eye(2)[tmp[:, 0]],
                           treatment_2nd=np.eye(2)[tmp[:, 1]],
                           structural=None)
