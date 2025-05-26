from typing import Dict, Any, Optional
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm

from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet


def link_func(data):
    return 0.8 * (np.exp(data)) / (1 + np.exp(data)) + 0.1


def generate_train_synthetic_data(data_config: Dict[str, Any], rand_seed: int) -> SATETrainDataSet:
    rng = default_rng(rand_seed)
    data_size = data_config["data_size"]

    backdoor_dim = 1
    backdoor_cov = np.eye(backdoor_dim)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), 1)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), -1)
    backdoor_1st = rng.multivariate_normal(np.zeros(backdoor_dim), backdoor_cov,
                                           size=data_size)  # shape of (data_size, backdoor_dim)

    theta = np.array([1.0 / ((i + 1) ** 2) for i in range(backdoor_dim)])[:, np.newaxis]
    treatment_1st_core = link_func(backdoor_1st.dot(theta))
    treatment_1st = rng.binomial(1, treatment_1st_core)
    treatment_1st_feature = np.eye(2)[treatment_1st[:, 0]]  # (data_size, 2)
    backdoor_2nd = 0.5 * (1 - treatment_1st) * backdoor_1st
    backdoor_2nd += 0.5 * rng.multivariate_normal(np.zeros(backdoor_dim), backdoor_cov,
                                                  size=data_size)
    treatment_2nd_core = link_func(0.5 * backdoor_1st.dot(theta) + backdoor_2nd.dot(theta) - 0.2 * treatment_1st)
    treatment_2nd = rng.binomial(1, treatment_2nd_core)
    treatment_2nd_feature = np.eye(2)[treatment_2nd[:, 0]]

    outcome = 1.2 * treatment_1st + 1.2 * backdoor_1st.dot(theta)
    outcome += treatment_1st ** 2 + treatment_1st * backdoor_1st[:, [0]]
    outcome /= 2
    outcome += 1.2 * treatment_2nd + 1.2 * backdoor_2nd.dot(theta)
    outcome += treatment_2nd ** 2 + treatment_2nd * backdoor_2nd[:, [0]]
    outcome += 0.5 * treatment_1st * treatment_2nd
    outcome += rng.standard_normal(size=(data_size, 1))

    return SATETrainDataSet(backdoor_1st=backdoor_1st,
                            backdoor_2nd=backdoor_2nd,
                            treatment_1st=treatment_1st_feature,
                            treatment_2nd=treatment_2nd_feature,
                            outcome=outcome)


def generate_test_synthetic_data(data_config: Dict[str, Any]) -> Optional[SATETestDataSet]:
    return SATETestDataSet(treatment_1st=np.array([[1, 0],
                                                   [0, 1],
                                                   [1, 0],
                                                   [0, 1]]),
                           treatment_2nd=np.array([[1, 0],
                                                   [1, 0],
                                                   [0, 1],
                                                   [0, 1]]),
                           structural=None)
