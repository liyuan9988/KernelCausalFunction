from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.stats import norm

from src.sate_ci.data import generate_train_data, generate_test_data
from src.sate.data import SATETrainDataSet, SATETestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, LinearKernel
from src.utils import cal_loocv_emb, cal_loocv

from sklearn.model_selection import KFold


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel]:
    return GaussianKernel(), GaussianKernel(), LinearKernel()


class SATEBackDoorMeanEmbeddingCI:
    treatment_kernel_func: AbsKernel
    backdoor_1st_kernel_func: AbsKernel
    backdoor_2nd_kernel_func: AbsKernel

    def __init__(self, lam1, lam2, lam3, lam4, quantile=0.05, nfolds=5,
                 clip_propensity=False, **kwargs):
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.quantile = quantile
        self.n_folds = nfolds
        self.clip_propensity = clip_propensity

    def fit_omega(self, out_of_fold: SATETrainDataSet, with_in_fold: SATETrainDataSet,
                  test_treatment_1st: np.ndarray, test_treatment_2nd: np.ndarray):

        treatment_kernel_1st = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment_1st,
                                                                         out_of_fold.treatment_1st)
        treatment_kernel_2nd = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment_2nd,
                                                                         out_of_fold.treatment_2nd)

        backdoor_kernel_1st = self.backdoor_1st_kernel_func.cal_kernel_mat(out_of_fold.backdoor_1st,
                                                                           out_of_fold.backdoor_1st)
        backdoor_kernel_2nd = self.backdoor_2nd_kernel_func.cal_kernel_mat(out_of_fold.backdoor_2nd,
                                                                           out_of_fold.backdoor_2nd)

        n_data = treatment_kernel_1st.shape[0]
        sub_kernel = treatment_kernel_1st * backdoor_kernel_1st
        all_kernel = sub_kernel * treatment_kernel_2nd * backdoor_kernel_2nd

        kernel_mat = all_kernel + n_data * self.lam1 * np.eye(n_data)
        kernel_mat2 = sub_kernel + n_data * self.lam2 * np.eye(n_data)

        test_treatment_kernel_1st = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment_1st,
                                                                              test_treatment_1st)  # (N, M)
        test_treatment_kernel_2nd = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment_2nd,
                                                                              test_treatment_2nd)  # (N, M)

        with_in_fold_backdoor_kernel_1st = self.backdoor_1st_kernel_func.cal_kernel_mat(out_of_fold.backdoor_1st,
                                                                                        with_in_fold.backdoor_1st)
        with_in_fold_backdoor_kernel_2nd = self.backdoor_2nd_kernel_func.cal_kernel_mat(out_of_fold.backdoor_2nd,
                                                                                        with_in_fold.backdoor_2nd)

        mat = test_treatment_kernel_1st[:, :, np.newaxis] * with_in_fold_backdoor_kernel_1st[:, np.newaxis,
                                                            :]  # N * M * with_in_fold
        weight_1st = np.linalg.solve(kernel_mat2, backdoor_kernel_2nd).T
        mat = np.einsum("ij,jkl->ikl", weight_1st, mat)  # N * M * with_in_fold
        mat = mat * test_treatment_kernel_1st[:, :, np.newaxis] * test_treatment_kernel_2nd[:, :, np.newaxis]
        mat = mat * with_in_fold_backdoor_kernel_1st[:, np.newaxis, :]  # N * M * with_in_fold
        weight_2nd = np.linalg.solve(kernel_mat, out_of_fold.outcome)  # N,1
        pred = np.sum(weight_2nd[:, :, np.newaxis] * mat, axis=0)  # M * with_in_fold
        misc = dict(weight_2nd=weight_2nd,
                    with_in_fold_backdoor_kernel_1st=with_in_fold_backdoor_kernel_1st,
                    with_in_fold_backdoor_kernel_2nd=with_in_fold_backdoor_kernel_2nd,
                    test_treatment_kernel_1st=test_treatment_kernel_1st,
                    test_treatment_kernel_2nd=test_treatment_kernel_2nd,
                    backdoor_kernel_1st=backdoor_kernel_1st,
                    backdoor_kernel_2nd=backdoor_kernel_2nd,
                    treatment_kernel_1st=treatment_kernel_1st)
        return pred, misc

    def fit_gamma(self, misc: Dict[str, np.ndarray]):

        weight_2nd = misc["weight_2nd"]
        with_in_fold_backdoor_kernel_1st = misc["with_in_fold_backdoor_kernel_1st"]
        with_in_fold_backdoor_kernel_2nd = misc["with_in_fold_backdoor_kernel_2nd"]
        test_treatment_kernel_1st = misc["test_treatment_kernel_1st"]
        test_treatment_kernel_2nd = misc["test_treatment_kernel_2nd"]

        mat = with_in_fold_backdoor_kernel_1st * with_in_fold_backdoor_kernel_2nd
        mat_for_new_treatment = test_treatment_kernel_1st[:, :, np.newaxis] * mat[:, np.newaxis,
                                                                              :]  # N * M * with_in_fold
        mat_for_new_treatment = mat_for_new_treatment * test_treatment_kernel_2nd[:, :, np.newaxis]
        new_treatment_gamma = np.sum(weight_2nd[:, :, np.newaxis] * mat_for_new_treatment, axis=0)  # M * with_in_fold

        return new_treatment_gamma

    def fit_pi_and_rho(self, out_of_fold: SATETrainDataSet, with_in_fold: SATETrainDataSet,
                       test_treatment_1st: np.ndarray, test_treatment_2nd: np.ndarray, misc: Dict[str, np.ndarray]):

        backdoor_kernel_1st = misc["backdoor_kernel_1st"]
        backdoor_kernel_2nd = misc["backdoor_kernel_2nd"]
        with_in_fold_backdoor_kernel_1st = misc["with_in_fold_backdoor_kernel_1st"]
        with_in_fold_backdoor_kernel_2nd = misc["with_in_fold_backdoor_kernel_2nd"]
        treatment_kernel_1st = misc["treatment_kernel_1st"]
        test_treatment_kernel_1st = misc["test_treatment_kernel_1st"]

        n_data = backdoor_kernel_1st.shape[0]
        n_test = test_treatment_1st.shape[0]

        # fit for pi
        mat = np.linalg.solve(backdoor_kernel_1st + n_data * self.lam3 * np.eye(n_data),
                              with_in_fold_backdoor_kernel_1st)
        pred = out_of_fold.treatment_1st.T.dot(mat)  # dim_treatment * with_in_fold
        pi_1st = test_treatment_1st.dot(pred)  # M * with_in_fold

        # fit for rho
        weight = np.linalg.solve(
            backdoor_kernel_1st * backdoor_kernel_2nd * treatment_kernel_1st + n_data * self.lam4 * np.eye(n_data),
            out_of_fold.treatment_2nd) # N * dim_treatment

        mat = with_in_fold_backdoor_kernel_1st * with_in_fold_backdoor_kernel_2nd
        mat_for_new_treatment = test_treatment_kernel_1st[:, :, np.newaxis] * mat[:, np.newaxis,
                                                                              :]  # N * M * with_in_fold
        pred = np.einsum("ij,ikl->jkl", weight, mat_for_new_treatment) # dim_treatment * M * with_in_fold
        rho_2nd = np.sum(pred * test_treatment_2nd.T[:,:,np.newaxis], axis=0)

        # compute indication function
        indi_1st = with_in_fold.treatment_1st.dot(test_treatment_1st.T).T
        indi_2nd = with_in_fold.treatment_2nd.dot(test_treatment_2nd.T).T

        if self.clip_propensity:
            pi_1st = np.maximum(0.95, np.minimum(0.05, pi_1st))
            rho_2nd = np.maximum(0.95, np.minimum(0.05, rho_2nd))
        alpha1 = indi_1st * indi_2nd / np.maximum(1.0e-2, pi_1st * rho_2nd)
        alpha2 = indi_1st / np.maximum(1.0e-2, pi_1st)

        return alpha1, alpha2

    def cal_lambda(self, train_data: SATETrainDataSet):
        treatment_1st_kernel = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment_1st,
                                                                         train_data.treatment_1st)
        treatment_2nd_kernel = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment_2nd,
                                                                         train_data.treatment_2nd)
        backdoor_1st_kernel = self.backdoor_1st_kernel_func.cal_kernel_mat(train_data.backdoor_1st,
                                                                           train_data.backdoor_1st)
        backdoor_2nd_kernel = self.backdoor_2nd_kernel_func.cal_kernel_mat(train_data.backdoor_2nd,
                                                                           train_data.backdoor_2nd)

        n_data = treatment_1st_kernel.shape[0]

        sub_kernel_mat = treatment_1st_kernel * backdoor_1st_kernel
        all_kernel_mat = treatment_2nd_kernel * backdoor_2nd_kernel * sub_kernel_mat

        if isinstance(self.lam1, list):
            score = [cal_loocv(all_kernel_mat, train_data.outcome, reg) for reg in self.lam1]
            self.lam1 = self.lam1[np.argmin(score)]

        if isinstance(self.lam2, list):
            score = [cal_loocv_emb(sub_kernel_mat, backdoor_2nd_kernel, reg) for reg in self.lam2]
            self.lam2 = self.lam2[np.argmin(score)]

        if isinstance(self.lam3, list):
            score = [cal_loocv_emb(backdoor_1st_kernel, treatment_1st_kernel, reg) for reg in self.lam3]
            self.lam3 = self.lam3[np.argmin(score)]

        if isinstance(self.lam4, list):
            score = [cal_loocv_emb(backdoor_2nd_kernel * sub_kernel_mat, treatment_2nd_kernel, reg) for reg in
                     self.lam4]
            self.lam4 = self.lam4[np.argmin(score)]

    def init_kernels(self, train_data, data_config):
        kernel_funcs = get_kernel_func(data_config["name"])
        self.backdoor_1st_kernel_func = kernel_funcs[0]
        self.backdoor_2nd_kernel_func = kernel_funcs[1]
        self.treatment_kernel_func = kernel_funcs[2]

        self.backdoor_1st_kernel_func.fit(train_data.backdoor_1st)
        self.backdoor_2nd_kernel_func.fit(train_data.backdoor_2nd)
        self.treatment_kernel_func.fit(train_data.treatment_1st)

    def fit_and_pred(self, train_data: SATETrainDataSet, test_data: SATETrainDataSet, data_config: Dict[str, Any]):
        self.init_kernels(train_data, data_config)
        if isinstance(self.lam1, list) or isinstance(self.lam2, list) or isinstance(self.lam3, list) or isinstance(
                self.lam4, list):
            self.cal_lambda(train_data)

        kf = KFold(n_splits=self.n_folds)
        n_data = train_data.treatment_1st.shape[0]
        n_test = test_data.treatment_1st.shape[0]
        sum1 = np.zeros((n_test))
        sum2 = np.zeros((n_test))
        test_treatment_1st = test_data.treatment_1st
        test_treatment_2nd = test_data.treatment_2nd

        # Ensure treatment is discrete variable with one-hot encoding
        assert np.sum(train_data.treatment_1st) == train_data.treatment_1st.shape[0]
        assert np.sum(train_data.treatment_2nd) == train_data.treatment_2nd.shape[0]

        for train_index, test_index in kf.split(train_data.backdoor_1st):
            out_of_fold = SATETrainDataSet(backdoor_1st=train_data.backdoor_1st[train_index],
                                           backdoor_2nd=train_data.backdoor_2nd[train_index],
                                           treatment_1st=train_data.treatment_1st[train_index],
                                           treatment_2nd=train_data.treatment_2nd[train_index],
                                           outcome=train_data.outcome[train_index])

            with_in_fold = SATETrainDataSet(backdoor_1st=train_data.backdoor_1st[test_index],
                                            backdoor_2nd=train_data.backdoor_2nd[test_index],
                                            treatment_1st=train_data.treatment_1st[test_index],
                                            treatment_2nd=train_data.treatment_2nd[test_index],
                                            outcome=train_data.outcome[test_index])

            omega, misc = self.fit_omega(out_of_fold, with_in_fold, test_treatment_1st, test_treatment_2nd)
            gamma = self.fit_gamma(misc)

            alpha1, alpha2 = self.fit_pi_and_rho(out_of_fold, with_in_fold,
                                                 test_treatment_1st, test_treatment_2nd, misc)
            tmp = omega + alpha1 * (with_in_fold.outcome.T - gamma) + alpha2 * (gamma - omega)
            sum1 += np.sum(tmp, axis=1)
            sum2 += np.sum(tmp ** 2, axis=1)

        c = norm.ppf(1 - (self.quantile / 2.0))
        mean1 = sum1 / n_data
        mean2 = sum2 / n_data
        return mean1, c * np.sqrt(mean2 - mean1 ** 2) / np.sqrt(n_data)


def evaluate_mean_embedding(data_config: Dict[str, Any], model_param: Dict[str, Any],
                            random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = SATEBackDoorMeanEmbeddingCI(**model_param)
    mean, ci = model.fit_and_pred(train_data, test_data, data_config)
    return np.concatenate([mean, ci])
