from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.stats import norm

from src.me_ci.data import generate_train_data, generate_test_data
from src.me.data import METrainDataSet, METestDataSet
from src.utils.kernel_func import AbsKernel, GaussianKernel, LinearKernel
from src.utils import cal_loocv_emb, cal_loocv

from sklearn.model_selection import KFold


def get_kernel_func(data_config: Dict[str, Any]) -> Tuple[AbsKernel, AbsKernel, AbsKernel]:
    name = data_config["name"]
    return GaussianKernel(), LinearKernel(), GaussianKernel()


class MeBackDoorMeanEmbeddingCI:
    treatment_kernel_func: AbsKernel
    backdoor_kernel_func: AbsKernel
    mediate_kernel_func: AbsKernel

    def __init__(self, lam1, lam2, lam3, lam4, quantile=0.95, nfolds=5, **kwargs):
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.quantile = quantile
        self.n_folds = nfolds

    def fit_omega(self, out_of_fold: METrainDataSet, with_in_fold: METrainDataSet,
                  test_treatment: np.ndarray, new_test_treatment: np.ndarray):

        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment, out_of_fold.treatment)
        backdoor_kernel = self.backdoor_kernel_func.cal_kernel_mat(out_of_fold.backdoor, out_of_fold.backdoor)
        mediate_kernel = self.mediate_kernel_func.cal_kernel_mat(out_of_fold.mediate, out_of_fold.mediate)
        n_data = treatment_kernel.shape[0]

        sub_kernel_mat = backdoor_kernel * treatment_kernel
        all_kernel_mat = mediate_kernel * sub_kernel_mat

        kernel_mat = all_kernel_mat + n_data * self.lam1 * np.eye(n_data)
        kernel_mat2 = sub_kernel_mat + n_data * self.lam2 * np.eye(n_data)

        test_treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment,
                                                                          test_treatment)  # (N, M)
        new_test_treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(out_of_fold.treatment,
                                                                              new_test_treatment)  # (N, M)
        with_in_fold_backdoor_kernel = self.backdoor_kernel_func.cal_kernel_mat(out_of_fold.backdoor,
                                                                                with_in_fold.backdoor)
        with_in_fold_mediate_kernel = self.mediate_kernel_func.cal_kernel_mat(out_of_fold.mediate,
                                                                              with_in_fold.mediate)

        mat = test_treatment_kernel[:, :, np.newaxis] * with_in_fold_backdoor_kernel[:, np.newaxis,
                                                        :]  # N * M * with_in_fold
        weight_1st = np.linalg.solve(kernel_mat2, mediate_kernel).T
        mat = np.einsum("ij,jkl->ikl", weight_1st, mat)  # N * M * with_in_fold
        mat = mat * new_test_treatment_kernel[:, :, np.newaxis] * with_in_fold_backdoor_kernel[:, np.newaxis,
                                                                  :]  # N * M * with_in_fold
        weight_2nd = np.linalg.solve(kernel_mat, out_of_fold.outcome)  # N,1
        pred = np.sum(weight_2nd[:, :, np.newaxis] * mat, axis=0)  # M * with_in_fold
        misc = dict(weight_2nd=weight_2nd,
                    with_in_fold_backdoor_kernel=with_in_fold_backdoor_kernel,
                    test_treatment_kernel=test_treatment_kernel,
                    new_test_treatment_kernel=new_test_treatment_kernel,
                    backdoor_kernel=backdoor_kernel,
                    mediate_kernel=mediate_kernel,
                    with_in_fold_mediate_kernel=with_in_fold_mediate_kernel
                    )
        return pred, misc

    def fit_gamma(self, misc: Dict[str, np.ndarray]):

        weight_2nd = misc["weight_2nd"]
        with_in_fold_backdoor_kernel = misc["with_in_fold_backdoor_kernel"]
        new_test_treatment_kernel = misc["new_test_treatment_kernel"]
        with_in_fold_mediate_kernel = misc["with_in_fold_mediate_kernel"]

        mat = with_in_fold_backdoor_kernel * with_in_fold_mediate_kernel
        mat_for_new_treatment = new_test_treatment_kernel[:, :, np.newaxis] * mat[:, np.newaxis,
                                                                              :]  # N * M * with_in_fold
        new_treatment_gamma = np.sum(weight_2nd[:, :, np.newaxis] * mat_for_new_treatment, axis=0)  # M * with_in_fold

        return new_treatment_gamma

    def fit_pi_and_rho(self, out_of_fold: METrainDataSet, with_in_fold: METrainDataSet,
                       test_treatment: np.ndarray, new_test_treatment: np.ndarray, misc: Dict[str, np.ndarray]):

        backdoor_kernel = misc["backdoor_kernel"]
        mediate_kernel = misc["mediate_kernel"]
        with_in_fold_backdoor_kernel = misc["with_in_fold_backdoor_kernel"]
        with_in_fold_mediate_kernel = misc["with_in_fold_mediate_kernel"]

        n_data = backdoor_kernel.shape[0]

        # fit for pi
        mat = np.linalg.solve(backdoor_kernel + n_data * self.lam3 * np.eye(n_data),
                              with_in_fold_backdoor_kernel)
        pred = out_of_fold.treatment.T.dot(mat)  # dim_treatment * with_in_fold
        old_pi = test_treatment.dot(pred)  # M * with_in_fold

        # fit for rho
        mat = np.linalg.solve(backdoor_kernel * mediate_kernel + n_data * self.lam4 * np.eye(n_data),
                              with_in_fold_backdoor_kernel * with_in_fold_mediate_kernel)
        pred = out_of_fold.treatment.T.dot(mat)  # dim_treatment * with_in_fold
        old_rho = test_treatment.dot(pred)  # M * with_in_fold
        new_rho = new_test_treatment.dot(pred)  # M * with_in_fold

        # compute indication function
        old_indi = with_in_fold.treatment.dot(test_treatment.T).T
        new_indi = with_in_fold.treatment.dot(new_test_treatment.T).T

        alpha1 = new_indi * old_rho / np.maximum(1.0e-2, new_rho * old_pi)
        alpha2 = old_indi / np.maximum(1.0e-2, old_pi)
        return alpha1, alpha2

    def cal_lambda(self, train_data: METrainDataSet):
        treatment_kernel = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
        backdoor_kernel = self.backdoor_kernel_func.cal_kernel_mat(train_data.backdoor, train_data.backdoor)
        mediate_kernel = self.mediate_kernel_func.cal_kernel_mat(train_data.mediate, train_data.mediate)
        n_data = treatment_kernel.shape[0]

        sub_kernel_mat = backdoor_kernel * treatment_kernel
        all_kernel_mat = mediate_kernel * sub_kernel_mat

        if isinstance(self.lam1, list):
            score = [cal_loocv(all_kernel_mat, train_data.outcome, reg) for reg in self.lam1]
            self.lam1 = self.lam1[np.argmin(score)]

        if isinstance(self.lam2, list):
            score = [cal_loocv_emb(sub_kernel_mat, mediate_kernel, reg) for reg in self.lam2]
            self.lam2 = self.lam2[np.argmin(score)]

        if isinstance(self.lam3, list):
            score = [cal_loocv_emb(backdoor_kernel, treatment_kernel, reg) for reg in self.lam3]
            self.lam3 = self.lam3[np.argmin(score)]

        if isinstance(self.lam4, list):
            score = [cal_loocv_emb(backdoor_kernel * mediate_kernel, treatment_kernel, reg) for reg in self.lam4]
            self.lam4 = self.lam4[np.argmin(score)]

    def fit_and_pred(self, train_data: METrainDataSet, test_data: METestDataSet, data_config: Dict[str, Any]):
        self.backdoor_kernel_func, self.treatment_kernel_func, self.mediate_kernel_func = get_kernel_func(data_config)
        self.backdoor_kernel_func.fit(train_data.backdoor)
        self.treatment_kernel_func.fit(train_data.treatment)
        self.mediate_kernel_func.fit(train_data.mediate)
        if isinstance(self.lam1, list) or isinstance(self.lam2, list) or isinstance(self.lam3, list) or isinstance(
                self.lam4, list):
            self.cal_lambda(train_data)

        kf = KFold(n_splits=self.n_folds)
        n_data = train_data.treatment.shape[0]
        n_test = test_data.treatment.shape[0]
        sum1 = np.zeros((n_test))
        sum2 = np.zeros((n_test))
        test_treatment = test_data.treatment
        new_test_treatment = test_data.new_treatment

        # Ensure treatment is discrete variable with one-hot encoding
        assert np.sum(train_data.treatment) == train_data.treatment.shape[0]

        for train_index, test_index in kf.split(train_data.backdoor):
            out_of_fold = METrainDataSet(backdoor=train_data.backdoor[train_index],
                                         treatment=train_data.treatment[train_index],
                                         mediate=train_data.mediate[train_index],
                                         outcome=train_data.outcome[train_index])
            with_in_fold = METrainDataSet(backdoor=train_data.backdoor[test_index],
                                          treatment=train_data.treatment[test_index],
                                          mediate=train_data.mediate[test_index],
                                          outcome=train_data.outcome[test_index])


            omega, misc = self.fit_omega(out_of_fold, with_in_fold, test_treatment, new_test_treatment)
            gamma = self.fit_gamma(misc)

            alpha1, alpha2 = self.fit_pi_and_rho(out_of_fold, with_in_fold, test_treatment, new_test_treatment, misc)
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
    model = MeBackDoorMeanEmbeddingCI(**model_param)
    mean, ci = model.fit_and_pred(train_data, test_data, data_config)
    return np.concatenate([mean, ci])
