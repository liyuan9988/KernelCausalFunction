from typing import Optional, Dict, Any
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

from src.ate.data import generate_train_data, generate_test_data
from src.ate.data.data_class import ATETrainDataSet, ATETestDataSet


class EdwardBackDoorDoublyRobust:
    xi: np.ndarray

    def __init__(self, mdl_name: str,
                 sigma: float,
                 smoother_bandwidth: float,
                 mdl_param: Optional[Dict[str, Any]] = None,
                 **kwargs):
        if mdl_name == "RF":
            self.reg_model = RandomForestRegressor(**mdl_param)
            self.treatment_model = RandomForestRegressor(**mdl_param)
            self.treatment_var_model = RandomForestRegressor(**mdl_param)
        elif mdl_name == "NN":
            self.reg_model = MLPRegressor(**mdl_param)
            self.treatment_model = MLPRegressor(**mdl_param)
            self.treatment_var_model = RandomForestRegressor(**mdl_param)
        else:
            raise ValueError(f"model {mdl_name} is not known")
        self.smoother_bandwidth = smoother_bandwidth
        params = {'bandwidth': np.logspace(-3, 3, 20)}
        self.density_model = GridSearchCV(KernelDensity(), params)

    def fit(self, train_data: ATETrainDataSet):
        fit_data = np.concatenate([train_data.treatment, train_data.backdoor], axis=1)
        assert train_data.outcome.shape[1] == 1
        assert train_data.treatment.shape[1] == 1

        self.reg_model.fit(fit_data, train_data.outcome.ravel())
        self.treatment_model.fit(train_data.backdoor, train_data.treatment.ravel())
        pred = self.treatment_model.predict(train_data.backdoor)
        self.treatment_var_model.fit(train_data.backdoor, (train_data.treatment.ravel() - pred) ** 2)
        var_pred = self.treatment_var_model.predict(train_data.backdoor)
        print(var_pred[:100])
        print((train_data.treatment.ravel() - pred)[:100])
        print(np.max((train_data.treatment.ravel() - pred) ** 2))
        print(np.min(var_pred))
        epsilon = (train_data.treatment.ravel() - pred) / (np.sqrt(np.maximum(var_pred, 0.0)) + 1.0e-3)
        self.density_model.fit(epsilon[:, np.newaxis])

        self.train_data = deepcopy(train_data)
        reg_pred = self.mean_reg(train_data)
        treatment_pred = self.density_estimate(train_data)
        self.xi = reg_pred + treatment_pred
        #print(np.mean((self.xi - 1.2 * train_data.treatment - train_data.treatment* train_data.treatment) ** 2))

    def density_estimate(self, train_data: ATETrainDataSet):
        n_data = train_data.backdoor.shape[0]
        idx = np.arange(n_data)
        treatment_id, backdoor_id = np.meshgrid(idx, idx)
        backdoor_id = backdoor_id.ravel()
        treatment_id = treatment_id.ravel()

        mean = self.treatment_model.predict(train_data.backdoor[backdoor_id])
        sigma = self.treatment_var_model.predict(train_data.backdoor[backdoor_id])
        epsilon = (train_data.treatment.ravel()[treatment_id] - mean) / (np.sqrt(np.maximum(sigma, 0.0)) + 1.0e-3)
        density = self.density_model.best_estimator_.score_samples(epsilon[:, np.newaxis]).reshape((n_data, n_data))
        density = np.exp(density)
        pred = self.reg_model.predict(np.concatenate([train_data.treatment, train_data.backdoor], axis=1))
        result = (train_data.outcome[:, 0] - pred) / np.diag(density) * np.mean(density, axis=0)
        return result

    def mean_reg(self, train_data: ATETrainDataSet):
        n_data = train_data.backdoor.shape[0]
        idx = np.arange(n_data)
        treatment_id, backdoor_id = np.meshgrid(idx, idx)
        backdoor_id = backdoor_id.ravel()
        treatment_id = treatment_id.ravel()
        data = np.concatenate([train_data.treatment[treatment_id], train_data.backdoor[backdoor_id]], axis=1)
        pred = self.reg_model.predict(data)
        pred = pred.reshape((n_data, n_data)).mean(axis=0)
        return pred

    def predict_one(self, treatment_one: float):
        """

        Parameters
        ----------
        treatment_one : float
            one row of single treatment

        Returns
        -------
        predict: float
            one row of prediction

        """
        feature = (self.train_data.treatment - treatment_one) / self.smoother_bandwidth
        weight = np.exp(-(feature ** 2)) / self.smoother_bandwidth / np.sqrt(2 * np.pi)
        mdl = LinearRegression()
        mdl.fit(feature, self.xi, sample_weight=weight[:, 0])
        pred = mdl.predict(np.array([[0.0]]))
        return pred[0]

    def predict(self, treatment: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(treatment[i]) for i in range(treatment.shape[0])])

    def evaluate(self, test_data: ATETestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_edward_doubly_robust(data_config: Dict[str, Any], model_param: Dict[str, Any],
                                  random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = EdwardBackDoorDoublyRobust(**model_param)
    model.fit(train_data)
    return model.evaluate(test_data)
