from typing import Optional, Dict, Any
import numpy as np

from src.cate.data import generate_train_data, generate_test_data
from src.cate.data.data_class import CATETrainDataSet, CATETestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, BinaryKernel
from src.utils import cal_loocv, cal_loocv_emb, cal_loocv_with_cluster


def get_kernel_func(data_name: str) -> [AbsKernel, AbsKernel, AbsKernel]:
    if data_name == "job_corp":
        return GaussianKernel(), GaussianKernel(), GaussianKernel()
    elif data_name == "synthetic":
        return GaussianKernel(), BinaryKernel(), GaussianKernel()
    else:
        return GaussianKernel(), GaussianKernel(), GaussianKernel()


class BackDoorMeanEmbedding:
    treatment_kernel_func: AbsKernel
    covariate_kernel_func: AbsKernel
    train_treatment: np.ndarray
    train_covariate: np.ndarray
    kernel_mat: np.ndarray
    cov_kernel_mat: np.ndarray
    backdoor_kernel: np.ndarray
    outcome: np.ndarray

    def __init__(self, lam1, lam2, **kwargs):
        self.lam1 = lam1
        self.lam2 = lam2

    def fit(self, train_data: CATETrainDataSet, data_name: str):
        self.train_treatment = np.array(train_data.treatment, copy=True)
        self.outcome = np.array(train_data.outcome, copy=True)
        self.train_covariate = np.array(train_data.covariate, copy=True)

        funcs = get_kernel_func(data_name)
        backdoor_kernel_func = funcs[0]
        self.treatment_kernel_func = funcs[1]
        self.covariate_kernel_func = funcs[2]

        backdoor_kernel_func.fit(train_data.backdoor, )
        self.treatment_kernel_func.fit(train_data.treatment, )
        self.covariate_kernel_func.fit(train_data.covariate, )

        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
        self.backdoor_kernel = backdoor_kernel_func.cal_kernel_mat(train_data.backdoor, train_data.backdoor)
        covariate_kernel = self.covariate_kernel_func.cal_kernel_mat(train_data.covariate, train_data.covariate)

        all_kernel_mat = covariate_kernel * treatment_kernel * self.backdoor_kernel
        if isinstance(self.lam1, list):
            score = [cal_loocv(all_kernel_mat, self.outcome, reg) for reg in self.lam1]
            self.lam1 = self.lam1[np.argmin(score)]
            print(self.lam1)

        if isinstance(self.lam2, list):
            score = [cal_loocv_emb(covariate_kernel, self.backdoor_kernel, reg) for reg in self.lam2]
            self.lam2 = self.lam2[np.argmin(score)]
            print(self.lam2)

        n_data = treatment_kernel.shape[0]
        self.kernel_mat = (all_kernel_mat + n_data * self.lam1 * np.eye(n_data))
        self.cov_kernel_mat = covariate_kernel + n_data * self.lam2 * np.eye(n_data)

    def predict(self, treatment: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        cov_kernel = self.covariate_kernel_func.cal_kernel_mat(self.train_covariate, covariate)
        test_kernel = np.linalg.solve(self.cov_kernel_mat, cov_kernel)
        test_kernel = np.dot(self.backdoor_kernel, test_kernel)
        test_kernel = test_kernel * cov_kernel * treatment_kernel
        mat = np.linalg.solve(self.kernel_mat, test_kernel)
        return np.dot(mat.T, self.outcome)

    def evaluate(self, test_data: CATETestDataSet):
        pred = self.predict(treatment=test_data.treatment, covariate=test_data.covariate)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_mean_embedding(data_config: Dict[str, Any], model_param: Dict[str, Any],
                            random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = BackDoorMeanEmbedding(**model_param)
    model.fit(train_data, data_config["name"])
    if test_data.structural is not None:
        return model.evaluate(test_data)
    else:
        return model.predict(test_data.treatment, test_data.covariate)[:, 0]
