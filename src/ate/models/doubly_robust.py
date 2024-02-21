from typing import Optional, Dict, Any
import numpy as np
from copy import deepcopy

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from src.ate.data import generate_train_data, generate_test_data
from src.ate.data.data_class import ATETrainDataSet, ATETestDataSet


class BackDoorDoublyRobust:
    reg_model: RegressorMixin
    ipw_model: RegressorMixin
    train_data: ATETrainDataSet

    def __init__(self, mdl_name: str,
                 density_bandwidth: float,
                 smoother_bandwidth: float,
                 mdl_param: Optional[Dict[str, Any]] = None,
                 **kwargs):
        if mdl_name == "RF":
            self.reg_model = RandomForestRegressor(**mdl_param)
            self.ipw_model = RandomForestRegressor(**mdl_param)
        elif mdl_name == "NN":
            self.reg_model = MLPRegressor(**mdl_param)
            self.ipw_model = MLPRegressor(**mdl_param)
        else:
            raise ValueError(f"model {mdl_name} is not known")
        self.density_bandwidth = density_bandwidth
        self.smoother_bandwidth = smoother_bandwidth

    def fit(self, train_data: ATETrainDataSet):
        fit_data = np.concatenate([train_data.treatment, train_data.backdoor], axis=1)
        if train_data.outcome.shape[1] == 1:
            self.reg_model.fit(fit_data, train_data.outcome.ravel())
        else:
            self.reg_model.fit(fit_data, train_data.outcome)
        self.train_data = deepcopy(train_data)

    def predict_one(self, treatment_one: np.ndarray):
        """

        Parameters
        ----------
        treatment_one : array (1, treatment_dim)
            one row of treatment

        Returns
        -------
        predict: array(1, outcome_dim)
            one row of prediction

        """
        # get reg prediction
        n_backdoor = self.train_data.backdoor.shape[0]
        aug_treatment_one = np.concatenate([treatment_one for i in range(n_backdoor)], axis=0)
        data = np.concatenate([aug_treatment_one, self.train_data.backdoor], axis=1)
        reg_predict = self.reg_model.predict(data)
        if reg_predict.ndim == 1:
            reg_predict = reg_predict[:, np.newaxis]

        # get smoothing ratio
        dist = np.sum((self.train_data.treatment - treatment_one) ** 2, axis=1)
        density_cond_target = np.exp(- dist / self.density_bandwidth / 2)
        density_cond_target = density_cond_target / np.sqrt(2 * np.pi * self.density_bandwidth)
        smoother = np.exp(- dist / self.smoother_bandwidth / 2)
        smoother = smoother / np.sqrt(2 * np.pi * self.smoother_bandwidth)
        self.ipw_model.fit(self.train_data.backdoor, density_cond_target)
        density_cond_pred = self.ipw_model.predict(self.train_data.backdoor)
        ratio = (smoother / density_cond_pred)[:, np.newaxis]

        pred_tmp = reg_predict + ratio * (self.train_data.outcome - reg_predict)
        pred = np.mean(pred_tmp, axis=0, keepdims=True)
        return pred

    def predict(self, treatment: np.ndarray) -> np.ndarray:
        return np.concatenate([self.predict_one(treatment[[i]]) for i in range(treatment.shape[0])], axis=0)

    def evaluate(self, test_data: ATETestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_doubly_robust(data_config: Dict[str, Any], model_param: Dict[str, Any],
                           random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = BackDoorDoublyRobust(**model_param)
    model.fit(train_data)
    return model.evaluate(test_data)
