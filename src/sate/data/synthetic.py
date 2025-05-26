from itertools import product
import numpy as np
from numpy.random import default_rng
import logging
from scipy.stats import norm
from typing import Tuple, TypeVar

from ..data.data_class import SATETrainDataSet, SATETestDataSet

np.random.seed(42)
logger = logging.getLogger()

def link_func(data):
    return 0.8 * (np.exp(data)) / (1 + np.exp(data)) + 0.1


def generate_test_synthetic() -> SATETestDataSet:
    """
    Returns
    -------
    test_data : SATETestDataSet
    """
    treatment = np.linspace(0.0, 1.0, 11)
    data = np.array(list(product(treatment, treatment)))
    treatment_1st = data[:, [0]]
    treatment_2nd = data[:, [1]]
    structural = 0.6 * treatment_1st + 0.5 * treatment_1st * treatment_1st
    structural += 1.2 * treatment_2nd + treatment_2nd * treatment_2nd

    return SATETestDataSet(treatment_1st=treatment_1st,
                           treatment_2nd=treatment_2nd,
                           structural=structural)


def generate_train_synthetic(data_size: int, backdoor_dim: int =100,
                             rand_seed: int = 42) -> SATETrainDataSet:
    """
    Generate the squential version of data in Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments
    [Colangelo and Lee, 2020]

    Parameters
    ----------
    data_size : int
        size of data
    rand_seed : int
        random seed


    Returns
    -------
    train_data : SATETrainDataSet
    """

    rng = default_rng(seed=rand_seed)
    backdoor_cov = np.eye(backdoor_dim)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), 1)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), -1)
    backdoor_1st = rng.multivariate_normal(np.zeros(backdoor_dim), backdoor_cov,
                                           size=data_size)  # shape of (data_size, backdoor_dim)

    theta = np.array([1.0 / ((i + 1) ** 2) for i in range(backdoor_dim)])[:, np.newaxis]
    treatment_1st = link_func(backdoor_1st.dot(theta) * 3) + 0.75 * rng.standard_normal(size=(data_size, 1))
    backdoor_2nd = 0.5 * (1 - treatment_1st) * backdoor_1st
    backdoor_2nd += 0.5 * rng.multivariate_normal(np.zeros(backdoor_dim), backdoor_cov,
                                                  size=data_size)
    treatment_2nd = link_func(1.5 * backdoor_1st.dot(theta) + 3 * backdoor_2nd.dot(theta) - treatment_1st)
    treatment_2nd += 0.75 * rng.standard_normal(size=(data_size, 1))

    outcome = 1.2 * treatment_1st + 1.2 * backdoor_1st.dot(theta)
    outcome += treatment_1st ** 2 + treatment_1st * backdoor_1st[:, [0]]
    outcome /= 2
    outcome += 1.2 * treatment_2nd + 1.2 * backdoor_2nd.dot(theta)
    outcome += treatment_2nd ** 2 + treatment_2nd * backdoor_2nd[:, [0]]
    outcome += rng.standard_normal(size=(data_size, 1))

    return SATETrainDataSet(backdoor_1st=backdoor_1st,
                            backdoor_2nd=backdoor_2nd,
                            treatment_1st=treatment_1st,
                            treatment_2nd=treatment_2nd,
                            outcome=outcome)
