from typing import Dict, Any
import numpy as np

from src.cate.data.data_class import CATETrainDataSet, CATETestDataSet
from src.cate.data import generate_train_data, generate_test_data
from src.utils.kernel_func import FourthOrderGaussianKernel, SixthOrderGaussianKernel, GaussianKernel


class AbrevayaCATEEstimator():
    pseudo_outcome: np.ndarray
    train_covariate: np.ndarray

    def __init__(self, propensity_bandwidth_scale: float, cate_bandwidth_scale: float,
                 **model_param):
        self.propensity_bandwidth_scale = propensity_bandwidth_scale
        self.cate_bandwidth_scale = cate_bandwidth_scale
        self.propensity_kernel = FourthOrderGaussianKernel()
        self.cate_kernel = SixthOrderGaussianKernel()

    def fit(self, train_data: CATETrainDataSet, data_name: str):
        assert data_name == "synthetic"
        n_data = train_data.treatment.shape[0]
        self.propensity_kernel.bandwidth = (self.propensity_bandwidth_scale * (n_data ** (-1.0 / 8.0)))
        self.cate_kernel.bandwidth = (self.cate_bandwidth_scale * (n_data ** (-1.0 / 13.0)))

        # calculate propensity score
        all_backdoor = np.concatenate([train_data.covariate, train_data.backdoor], axis=1)
        kernel_mat = self.propensity_kernel.cal_kernel_mat(all_backdoor, all_backdoor)
        denominator = np.sum(kernel_mat, axis=0) - np.diag(kernel_mat)
        treatment_kernel_mat = train_data.treatment * kernel_mat
        numerator = np.sum(treatment_kernel_mat, axis=0) - np.diag(treatment_kernel_mat)
        propensity = (numerator / denominator)
        propensity = np.maximum(propensity, 0.005)
        propensity = np.minimum(propensity, 0.995)
        propensity = propensity[:, np.newaxis]


        self.pseudo_outcome = train_data.treatment * train_data.outcome / propensity
        self.pseudo_outcome -= (1.0 - train_data.treatment) * train_data.outcome / (1.0 - propensity)
        self.train_covariate = train_data.covariate

    def predict(self, test_covariate):
        covariate_kernel = self.cate_kernel.cal_kernel_mat(self.train_covariate, test_covariate)
        pred = np.sum(covariate_kernel * self.pseudo_outcome, axis=0)
        pred /= np.maximum(np.sum(covariate_kernel, axis=0), 0.001)
        return pred[:, np.newaxis]

    def evaluate(self, test_data: CATETestDataSet):
        pred = self.predict(test_covariate=test_data.covariate)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_abrevaya(data_config: Dict[str, Any], model_param: Dict[str, Any],
                      random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = AbrevayaCATEEstimator(**model_param)
    model.fit(train_data, data_config["name"])
    if test_data.structural is not None:
        return model.evaluate(test_data)
    else:
        return model.predict(test_data.covariate)[:, 0]
