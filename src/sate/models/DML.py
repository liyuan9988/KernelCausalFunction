from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from src.sate.data import generate_train_data, generate_test_data
from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, WarfarinBackdoorKernel
from src.utils import cal_loocv, cal_loocv_emb


class SequentialATEDML:
    theta_0: np.ndarray
    theta_1: np.ndarray

    def __init__(self, **kwargs):
        pass

    def split_and_fit(self, feature: np.ndarray, target_raw: np.ndarray):
        idx = np.arange(feature.shape[0])
        target = np.copy(target_raw)
        if target.shape[1] == 1:
            target = target[:, 0]
        pred = np.empty(target.shape)
        idx_1st, idx_2nd = train_test_split(idx, test_size=0.5)
        # fit for 1st idx
        mdl = RandomForestRegressor()
        mdl.fit(feature[idx_1st], target[idx_1st])

        # fit for 2nd idx
        mdl = RandomForestRegressor()
        mdl.fit(feature[idx_2nd], target[idx_2nd])
        pred[idx_1st] = mdl.predict(feature[idx_1st])
        res = target - pred
        if res.ndim == 1:
            res = res[:, np.newaxis]
        return res

    def fit(self, train_data: SATETrainDataSet, data_name: str):
        tilde_Y11 = self.split_and_fit(train_data.backdoor_2nd, train_data.outcome)
        tilde_T11 = self.split_and_fit(train_data.backdoor_2nd, train_data.treatment_2nd)
        tilde_Y10 = self.split_and_fit(train_data.backdoor_1st, train_data.outcome)
        tilde_T10 = self.split_and_fit(train_data.backdoor_1st, train_data.treatment_2nd)
        tilde_T00 = self.split_and_fit(train_data.backdoor_1st, train_data.treatment_1st)

        reg = 0.0001 * np.eye(tilde_T11.shape[1]) #for numerical stability
        bar_Y10 = tilde_Y11
        self.theta_0 = np.linalg.solve(tilde_T11.T.dot(tilde_T11) + reg, tilde_T11.T.dot(bar_Y10))
        bar_Y11 = tilde_Y10 - tilde_T10.dot(self.theta_0)
        self.theta_1 = np.linalg.solve(tilde_T00.T.dot(tilde_T00) + reg, tilde_T00.T.dot(bar_Y11))


    def predict(self, treatment_1st: np.ndarray, treatment_2nd: np.ndarray) -> np.ndarray:
        return np.dot(treatment_2nd, self.theta_0) + np.dot(treatment_1st, self.theta_1)

    def evaluate(self, test_data: SATETestDataSet):
        pred = self.predict(treatment_1st=test_data.treatment_1st,
                            treatment_2nd=test_data.treatment_2nd)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_DML(data_config: Dict[str, Any], model_param: Dict[str, Any],
                 random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = SequentialATEDML(**model_param)
    model.fit(train_data, data_config["name"])
    if test_data.structural is not None:
        return model.evaluate(test_data)
    else:
        return model.predict(test_data.treatment_1st,
                             test_data.treatment_2nd)[:, 0]
