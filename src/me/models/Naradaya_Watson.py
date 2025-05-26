from typing import Optional, Dict, Any
import numpy as np

from src.me.data import generate_train_data, generate_test_data
from src.me.data.data_class import METrainDataSet, METestDataSet
from src.utils.kernel_func import GaussianKernel, EpanechnikovKernel, FourthOrderEpanechnikovKernel


class NaradayaWatson:
    treatment_kernel_func_1st: GaussianKernel
    treatment_kernel_func_2nd: GaussianKernel
    train_treatment: np.ndarray
    outcome: np.ndarray
    norm_backdoor_kernel: np.ndarray
    norm_med_backdoor_kernel: np.ndarray

    def __init__(self, **kwargs):
        pass

    def fit(self, train_data: METrainDataSet, data_name: str):
        self.train_treatment = np.copy(train_data.treatment)
        self.outcome = np.copy(train_data.outcome)

        backdoor_kernel_func = FourthOrderEpanechnikovKernel()
        self.treatment_kernel_func_1st = EpanechnikovKernel()
        self.treatment_kernel_func_2nd = EpanechnikovKernel()
        mediate_kernel_func = FourthOrderEpanechnikovKernel()

        backdoor_kernel_func.fit(train_data.backdoor, )
        self.treatment_kernel_func_1st.fit(train_data.treatment, )
        self.treatment_kernel_func_2nd.fit(train_data.treatment, )
        mediate_kernel_func.fit(train_data.mediate, )

        backdoor_kernel = backdoor_kernel_func.cal_kernel_mat(train_data.backdoor, train_data.backdoor)
        self.norm_backdoor_kernel = backdoor_kernel / np.sum(backdoor_kernel, axis=1, keepdims=True)
        med_backdoor_kernel = backdoor_kernel * mediate_kernel_func.cal_kernel_mat(train_data.mediate,
                                                                                   train_data.mediate)
        self.norm_med_backdoor_kernel = med_backdoor_kernel / np.sum(med_backdoor_kernel, axis=1, keepdims=True)

    def predict(self, treatment: np.ndarray, new_treatment: np.ndarray) -> np.ndarray:
        treatment_kernel_1st = self.treatment_kernel_func_1st.cal_kernel_mat(self.train_treatment, treatment)
        new_treatment_kernel_1st = self.treatment_kernel_func_1st.cal_kernel_mat(self.train_treatment,
                                                                                 new_treatment)
        treatment_kernel_2nd = self.treatment_kernel_func_2nd.cal_kernel_mat(self.train_treatment, treatment)
        new_treatment_kernel_2nd = self.treatment_kernel_func_2nd.cal_kernel_mat(self.train_treatment,
                                                                                 new_treatment)
        new_treatment_M_X = self.norm_med_backdoor_kernel.dot(new_treatment_kernel_1st)
        treatment_M_X = self.norm_med_backdoor_kernel.dot(treatment_kernel_1st)
        treatment_X = self.norm_backdoor_kernel.dot(treatment_kernel_1st)
        weight = new_treatment_kernel_2nd / new_treatment_M_X * treatment_M_X / treatment_X
        weight = weight / np.sum(weight, axis=0)
        return np.dot(weight.T, self.outcome)

    def evaluate(self, test_data: METestDataSet):
        pred = self.predict(treatment=test_data.treatment, new_treatment=test_data.new_treatment)
        return np.mean((pred - test_data.structural) ** 2)


def evaluate_naradaya_watson(data_config: Dict[str, Any], model_param: Dict[str, Any],
                             random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    model = NaradayaWatson(**model_param)
    model.fit(train_data, data_config["name"])
    if test_data.structural is not None:
        return model.evaluate(test_data)
    else:
        return model.predict(test_data.treatment, test_data.new_treatment)[:, 0]
