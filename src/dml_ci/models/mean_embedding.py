from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.stats import norm

from src.dml_ci.data import generate_train_data, generate_test_data
from src.ate.data import ATETrainDataSet, ATETestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, LinearKernel
from src.utils import cal_loocv

from sklearn.model_selection import KFold


def get_kernel_func(data_config: Dict[str, Any]) -> Tuple[AbsKernel, AbsKernel]:
    name = data_config["name"]
    split_type = data_config.get("split_type", -1)
    if name == "job_corp" and split_type == 3:
        return GaussianKernel(), GaussianKernel()
    else:
        return GaussianKernel(), LinearKernel()


class ATEBackDoorMeanEmbeddingCI:
    treatment_kernel_func: AbsKernel
    backdoor_kernel_func: AbsKernel
    train_treatment: np.ndarray
    kernel_mat: np.ndarray
    mean_backdoor_kernel: np.ndarray
    outcome: np.ndarray

    def __init__(self, lam, quantile=0.95, nfolds=5, **kwargs):
        self.lam = lam
        self.quantile = quantile
        self.n_folds = nfolds

    def fit_gamma_and_m(self, out_of_fold: ATETrainDataSet, with_in_fold: ATETrainDataSet, test_treatment: np.ndarray):

        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment, out_of_fold.treatment)
        backdoor_kernel = self.backdoor_kernel_func.cal_kernel_mat(out_of_fold.backdoor, out_of_fold.backdoor)

        n_data = treatment_kernel.shape[0]
        kernel_mat = treatment_kernel * backdoor_kernel + n_data * self.lam * np.eye(n_data)
        test_kernel_backdoor = self.backdoor_kernel_func.cal_kernel_mat(out_of_fold.backdoor, with_in_fold.backdoor)
        test_kernel_treatment_m = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment, test_treatment)
        test_kernel_tensor = test_kernel_treatment_m[:, np.newaxis, :] * test_kernel_backdoor[:, :, np.newaxis]
        weights = np.linalg.solve(kernel_mat, out_of_fold.outcome)
        pred_m = np.einsum("i,ijk->jk", weights[:, 0], test_kernel_tensor)

        test_kernel_treatment_gamma = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment,
                                                                                with_in_fold.treatment)
        pred_gamma = weights.T @ (test_kernel_backdoor * test_kernel_treatment_gamma)

        return pred_gamma.T, pred_m  # (with_in_fold * 1, with_in_fold * n_test)

    def fit_alpha(self, out_of_fold: ATETrainDataSet, with_in_fold: ATETrainDataSet, test_treatment: np.ndarray):
        backdoor_kernel = self.backdoor_kernel_func.cal_kernel_mat(out_of_fold.backdoor, out_of_fold.backdoor)
        K1 = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment, out_of_fold.treatment)
        K1 *= backdoor_kernel
        n = out_of_fold.treatment.shape[0]
        res = []
        for i in range(test_treatment.shape[0]):
            test_row = test_treatment[[i]]
            test_kernel_row = self.treatment_kernel_func.cal_kernel_mat(test_row, out_of_fold.treatment)
            K2 = test_kernel_row.T * backdoor_kernel
            K3 = test_kernel_row * backdoor_kernel
            assert np.allclose(K2.T, K3)
            K4 = self.treatment_kernel_func.cal_kernel_mat(test_row, test_row) * backdoor_kernel
            Omega = np.block([[K1 @ K1, K1 @ K2],
                              [K3 @ K1, K3 @ K2]])
            K = np.block([[K1, K2],
                          [K3, K4]])

            v = np.sum(np.block([[K2],
                                 [K4]]), axis=1, keepdims=True)
            w = np.linalg.solve(Omega + n * self.lam * K + 0.001 * n * np.eye(2 * n), v)

            tmp = self.backdoor_kernel_func.cal_kernel_mat(out_of_fold.backdoor, with_in_fold.backdoor)
            upper_u = tmp * self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment, with_in_fold.treatment)
            lower_u = tmp * self.treatment_kernel_func.cal_kernel_mat(test_row, with_in_fold.treatment)
            u = np.block([[upper_u],
                          [lower_u]])
            res.append((w.T @ u).T)
        return np.concatenate(res, axis=1)  # with_in_fold * n_test

    def cal_lambda(self, train_data: ATETrainDataSet):
        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
        backdoor_kernel = self.backdoor_kernel_func.cal_kernel_mat(train_data.backdoor, train_data.backdoor)

        n_data = treatment_kernel.shape[0]
        lam_score = [cal_loocv(treatment_kernel * backdoor_kernel, train_data.outcome, lam) for lam in self.lam]
        self.lam = self.lam[np.argmin(lam_score)]

    def fit_and_pred(self, train_data: ATETrainDataSet, test_data: ATETestDataSet, data_config: Dict[str, Any]):
        self.backdoor_kernel_func, self.treatment_kernel_func = get_kernel_func(data_config)
        self.backdoor_kernel_func.fit(train_data.backdoor)
        self.treatment_kernel_func.fit(train_data.treatment)
        if isinstance(self.lam, list):
            self.cal_lambda(train_data)

        kf = KFold(n_splits=self.n_folds)
        n_data = train_data.treatment.shape[0]
        n_test = test_data.treatment.shape[0]
        sum1 = np.zeros((n_test))
        sum2 = np.zeros((n_test))

        for train_index, test_index in kf.split(train_data.backdoor):
            out_of_fold = ATETrainDataSet(backdoor=train_data.backdoor[train_index],
                                          treatment=train_data.treatment[train_index],
                                          outcome=train_data.outcome[train_index])
            with_in_fold = ATETrainDataSet(backdoor=train_data.backdoor[test_index],
                                           treatment=train_data.treatment[test_index],
                                           outcome=train_data.outcome[test_index])

            pred_gamma, pred_m = self.fit_gamma_and_m(out_of_fold, with_in_fold, test_data.treatment)
            pred_alpha = self.fit_alpha(out_of_fold, with_in_fold, test_data.treatment)
            tmp = pred_m + pred_alpha * (with_in_fold.outcome - pred_gamma)
            sum1 += np.sum(tmp, axis=0)
            sum2 += np.sum(tmp ** 2, axis=0)

        c = norm.ppf(1 - (self.quantile / 2.0))
        mean1 = sum1 / n_data
        mean2 = sum2 / n_data
        return mean1, c * np.sqrt(mean2 - mean1 ** 2) / np.sqrt(n_data)


def evaluate_mean_embedding(data_config: Dict[str, Any], model_param: Dict[str, Any],
                            random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = ATEBackDoorMeanEmbeddingCI(**model_param)
    mean, ci = model.fit_and_pred(train_data, test_data, data_config)
    return np.concatenate([mean, ci])
