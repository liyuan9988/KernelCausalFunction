from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from src.sate.data import generate_train_data, generate_test_data
from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, LinearKernel
from src.utils import cal_loocv, cal_loocv_emb


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel]:
    if data_name.endswith("_simple"):
        return GaussianKernel(), GaussianKernel(), LinearKernel()
    return GaussianKernel(), GaussianKernel(), GaussianKernel()


class SequentialATEMeanEmbedding:
    treatment_kernel_func: AbsKernel
    treatment_1st: np.ndarray
    treatment_2nd: np.ndarray
    kernel_all_mat: np.ndarray
    kernel_sub_mat: np.ndarray
    backdoor_mat_1st: np.ndarray
    backdoor_mat_2nd: np.ndarray
    outcome: np.ndarray

    def __init__(self, lam1, lam2, **kwargs):
        self.lam1 = lam1
        self.lam2 = lam2

    def fit(self, train_data: SATETrainDataSet, data_name: str):
        self.treatment_1st = np.array(train_data.treatment_1st, copy=True)
        self.treatment_2nd = np.array(train_data.treatment_2nd, copy=True)
        self.outcome = np.array(train_data.outcome, copy=True)

        backdoor1st_kernel_func, backdoor2nd_kernel_func, self.treatment_kernel_func = get_kernel_func(data_name)
        backdoor1st_kernel_func.fit(train_data.backdoor_1st)
        backdoor2nd_kernel_func.fit(train_data.backdoor_2nd)

        self.treatment_kernel_func.fit(np.concatenate([self.treatment_1st, self.treatment_2nd], axis=0), )

        treatment_kernel_1st = self.treatment_kernel_func.cal_kernel_mat(self.treatment_1st, self.treatment_1st)
        treatment_kernel_2nd = self.treatment_kernel_func.cal_kernel_mat(self.treatment_2nd, self.treatment_2nd)
        backdoor_kernel_1st = backdoor1st_kernel_func.cal_kernel_mat(train_data.backdoor_1st, train_data.backdoor_1st)
        backdoor_kernel_2nd = backdoor2nd_kernel_func.cal_kernel_mat(train_data.backdoor_2nd, train_data.backdoor_2nd)
        sub_kernel = treatment_kernel_1st * backdoor_kernel_1st
        all_kernel = sub_kernel * treatment_kernel_2nd * backdoor_kernel_2nd
        n_data = treatment_kernel_1st.shape[0]

        if isinstance(self.lam1, list):
            lam_score = [cal_loocv_emb(sub_kernel, treatment_kernel_2nd, lam) for lam in self.lam1]
            self.lam1 = self.lam1[np.argmin(lam_score)]

        if isinstance(self.lam2, list):
            lam_score = [cal_loocv(all_kernel, self.outcome, lam) for lam in self.lam2]
            self.lam2 = self.lam2[np.argmin(lam_score)]

        self.kernel_sub_mat = sub_kernel + n_data * self.lam1 * np.eye(n_data)
        self.kernel_all_mat = all_kernel + n_data * self.lam2 * np.eye(n_data)
        self.backdoor_mat_1st = backdoor_kernel_1st
        self.backdoor_mat_2nd = backdoor_kernel_2nd

    def predict(self, treatment_1st: np.ndarray, treatment_2nd: np.ndarray) -> np.ndarray:
        test_kernel_1st = self.treatment_kernel_func.cal_kernel_mat(self.treatment_1st, treatment_1st)
        test_kernel_2nd = self.treatment_kernel_func.cal_kernel_mat(self.treatment_2nd, treatment_2nd)
        mat = self.backdoor_mat_2nd.dot(np.linalg.inv(self.kernel_sub_mat))
        mat = mat * self.backdoor_mat_1st.dot(self.backdoor_mat_1st) / (self.backdoor_mat_1st.shape[0])
        mat = test_kernel_1st * test_kernel_2nd * mat.dot(test_kernel_1st)
        mat = np.linalg.solve(self.kernel_all_mat, mat)
        return np.dot(mat.T, self.outcome)

    def predict_incremental(self, treatment_1st: np.ndarray, treatment_2nd: np.ndarray) -> np.ndarray:
        # Predict incremental effect on 2nd treatment
        test_kernel_1st = self.treatment_kernel_func.cal_kernel_mat(self.treatment_1st, treatment_1st)
        test_kernel_2nd = self.treatment_kernel_func.cal_kernel_mat_grad(self.treatment_2nd, treatment_2nd)
        test_kernel_2nd = test_kernel_2nd[:, :, 0]
        mat = self.backdoor_mat_2nd.dot(np.linalg.inv(self.kernel_sub_mat))
        mat = mat * self.backdoor_mat_1st.dot(self.backdoor_mat_1st) / (self.backdoor_mat_1st.shape[0])
        mat = test_kernel_1st * test_kernel_2nd * mat.dot(test_kernel_1st)
        mat = np.linalg.solve(self.kernel_all_mat, mat)
        return np.dot(mat.T, self.outcome)


    def evaluate(self, test_data: SATETestDataSet):
        pred = self.predict(treatment_1st=test_data.treatment_1st,
                            treatment_2nd=test_data.treatment_2nd)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_mean_embedding(data_config: Dict[str, Any], model_param: Dict[str, Any],
                            random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = SequentialATEMeanEmbedding(**model_param)
    model.fit(train_data, data_config["name"])
    output = model_param.get("output", "ate")
    if output == "incremental":
        return model.predict_incremental(test_data.treatment_1st, test_data.treatment_2nd)[:, 0]
    else:
        if test_data.structural is not None:
            return model.evaluate(test_data)
        else:
            return model.predict(test_data.treatment_1st,
                                 test_data.treatment_2nd)[:, 0]
