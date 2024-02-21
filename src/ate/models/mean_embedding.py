from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from src.ate.data import generate_train_data, generate_test_data
from src.ate.data.data_class import ATETrainDataSet, ATETestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, BinaryKernel
from src.utils import cal_loocv


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel]:
    return GaussianKernel(), GaussianKernel()


class BackDoorMeanEmbedding:
    treatment_kernel_func: AbsKernel
    train_treatment: np.ndarray
    kernel_mat: np.ndarray
    mean_backdoor_kernel: np.ndarray
    outcome: np.ndarray

    def __init__(self, lam, **kwargs):
        self.lam = lam

    def fit(self, train_data: ATETrainDataSet, data_name: str):
        self.train_treatment = np.array(train_data.treatment, copy=True)
        self.outcome = np.array(train_data.outcome, copy=True)

        backdoor_kernel_func, self.treatment_kernel_func = get_kernel_func(data_name)
        backdoor_kernel_func.fit(train_data.backdoor, )
        self.treatment_kernel_func.fit(train_data.treatment, )

        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
        backdoor_kernel = backdoor_kernel_func.cal_kernel_mat(train_data.backdoor, train_data.backdoor)
        n_data = treatment_kernel.shape[0]
        if isinstance(self.lam, list):
            lam_score = [cal_loocv(treatment_kernel * backdoor_kernel, self.outcome, lam) for lam in self.lam]
            self.lam = self.lam[np.argmin(lam_score)]

        self.kernel_mat = treatment_kernel * backdoor_kernel + n_data * self.lam * np.eye(n_data)
        self.mean_backdoor_kernel = np.mean(backdoor_kernel, axis=0)

    def predict(self, treatment: np.ndarray) -> np.ndarray:
        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        test_kernel = test_kernel * self.mean_backdoor_kernel[:, np.newaxis]
        mat = np.linalg.solve(self.kernel_mat, test_kernel)
        return np.dot(mat.T, self.outcome)

    def predict_incremental(self, treatment: np.ndarray) -> np.ndarray:
        assert treatment.shape[1] == 1
        test_kernel = self.treatment_kernel_func.cal_kernel_mat_grad(self.train_treatment, treatment)
        test_kernel = test_kernel[:, :, 0]
        test_kernel = test_kernel * self.mean_backdoor_kernel[:, np.newaxis]
        mat = np.linalg.solve(self.kernel_mat, test_kernel)
        return np.dot(mat.T, self.outcome)

    def evaluate(self, test_data: ATETestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_mean_embedding(data_config: Dict[str, Any], model_param: Dict[str, Any],
                            random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = BackDoorMeanEmbedding(**model_param)
    model.fit(train_data, data_config["name"])
    output = model_param.get("output", "ate")
    if output == "incremental":
        return model.predict_incremental(test_data.treatment)[:, 0]
    else:
        if test_data.structural is not None:
            return model.evaluate(test_data)
        else:
            return model.predict(test_data.treatment)[:, 0]
