from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd

from src.ate.data.data_class import ATETrainDataSet, ATETestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")


def generate_train_jobcorp(split_type: int = 0):
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        data = pd.read_csv(DATA_PATH.joinpath("job_corps/JCdata.csv"), sep=" ")

    sub = data
    #sub = data.loc[data["m"] > 0, :]
    sub = sub.loc[sub["d"] >= 40, :]
    global NBINS
    if split_type == 0:
        tmp = pd.get_dummies(pd.qcut(sub["d"], 5))
        print(tmp.columns)
        treatment = tmp.to_numpy()
    elif split_type == 1:
        tmp = pd.get_dummies(pd.cut(sub["d"], [39.999, 500, 1000, 1500, 2000, 5143]))
        treatment = tmp.to_numpy()
    elif split_type == 2:
        tmp = pd.get_dummies(pd.cut(sub["d"], [39.999, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 5143]))
        treatment = tmp.to_numpy()
    elif split_type == 3:
        tmp = pd.get_dummies(pd.cut(sub["d"], [39.999, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 5143]))
        treatment = tmp.to_numpy()
        mid_points = np.array([145, 375, 625, 875, 1125, 1375, 1625, 1875, 3465])
        treatment = treatment @ mid_points[:, np.newaxis]
    else:
        raise ValueError(f"unknown split type {split_type}")

    outcome = sub["m"].to_numpy()
    backdoor = sub.iloc[:, 3:].to_numpy()
    print(backdoor.shape)
    return ATETrainDataSet(backdoor=backdoor,
                           outcome=outcome[:, np.newaxis],
                           treatment=treatment.astype(np.float64))


def generate_test_jobcorp(split_type: int = 0):
    if split_type == 0 or split_type == 1:
        test_treatment = np.eye(5)
    elif split_type == 2:
        test_treatment = np.eye(9)
    elif split_type == 3:
        test_treatment = np.array([145, 375, 625, 875, 1125, 1375, 1625, 1875, 3465])
        test_treatment = test_treatment[:, np.newaxis]
    else:
        raise ValueError(f"unknown split type {split_type}")

    return ATETestDataSet(treatment=test_treatment,
                          structural=None)
