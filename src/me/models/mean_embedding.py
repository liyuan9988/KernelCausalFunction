from typing import Optional, Dict, Any, Tuple
import numpy as np

from src.me.data import generate_train_data, generate_test_data
from src.me.data.data_class import METrainDataSet, METestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, LinearKernel
from src.utils import cal_loocv_emb, cal_loocv


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel]:
    if data_name == "simple_colangelo":
        return GaussianKernel(), LinearKernel(), GaussianKernel()
    return GaussianKernel(), GaussianKernel(), GaussianKernel()


class BackDoorMeanEmbedding:
    treatment_kernel_func: AbsKernel
    train_treatment: np.ndarray
    kernel_mat: np.ndarray
    kernel_mat2: np.ndarray
    mediate_kernel: np.ndarray
    backdoor_kernel: np.ndarray
    outcome: np.ndarray

    def __init__(self, lam1, lam2, **kwargs):
        self.lam1 = lam1
        self.lam2 = lam2

    def fit(self, train_data: METrainDataSet, data_name: str):
        self.train_treatment = np.array(train_data.treatment, copy=True)
        self.outcome = np.array(train_data.outcome, copy=True)

        funcs = get_kernel_func(data_name)
        backdoor_kernel_func = funcs[0]
        self.treatment_kernel_func = funcs[1]
        mediate_kernel_func = funcs[2]

        backdoor_kernel_func.fit(train_data.backdoor, )
        self.treatment_kernel_func.fit(train_data.treatment, )
        mediate_kernel_func.fit(train_data.mediate, )

        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
        self.backdoor_kernel = backdoor_kernel_func.cal_kernel_mat(train_data.backdoor, train_data.backdoor)
        self.mediate_kernel = mediate_kernel_func.cal_kernel_mat(train_data.mediate, train_data.mediate)
        n_data = treatment_kernel.shape[0]

        sub_kernel_mat = self.backdoor_kernel * treatment_kernel
        all_kernel_mat = self.mediate_kernel * sub_kernel_mat

        if isinstance(self.lam1, list):
            score = [cal_loocv(all_kernel_mat, self.outcome, reg) for reg in self.lam1]
            self.lam1 = self.lam1[np.argmin(score)]
            print(self.lam1)

        if isinstance(self.lam2, list):
            score = [cal_loocv_emb(sub_kernel_mat, self.mediate_kernel, reg) for reg in self.lam2]
            self.lam2 = self.lam2[np.argmin(score)]

        self.kernel_mat = all_kernel_mat + n_data * self.lam1 * np.eye(n_data)
        self.kernel_mat2 = sub_kernel_mat + n_data * self.lam2 * np.eye(n_data)

    @staticmethod
    def predict_core(treatment_kernel: np.ndarray, med_kernel_mat2_inv: np.ndarray, backdoor_kernel: np.ndarray):
        n_data = backdoor_kernel.shape[0]
        mat = backdoor_kernel.dot(backdoor_kernel.T) / n_data
        mat = mat * med_kernel_mat2_inv
        return np.dot(mat, treatment_kernel)

    def predict(self, treatment: np.ndarray, new_treatment: np.ndarray) -> np.ndarray:
        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)  # (N, M)
        new_treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, new_treatment)  # (N, M)
        med_kernel_mat2_inv = self.mediate_kernel.dot(np.linalg.inv(self.kernel_mat2))
        res = self.predict_core(treatment_kernel, med_kernel_mat2_inv, self.backdoor_kernel)
        test_kernel = res * new_treatment_kernel
        mat = np.linalg.solve(self.kernel_mat, test_kernel)
        return np.dot(mat.T, self.outcome)

    def evaluate(self, test_data: METestDataSet):
        pred = self.predict(treatment=test_data.treatment, new_treatment=test_data.new_treatment)
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
        return model.predict(test_data.treatment, test_data.new_treatment)[:, 0]
