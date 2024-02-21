from itertools import product
import numpy as np
from numpy.random import default_rng
import logging
from scipy.stats import norm
from typing import Tuple, TypeVar

from src.ate.data.data_class import ATETrainDataSet, ATETestDataSet

np.random.seed(42)
logger = logging.getLogger()


def generate_test_colangelo() -> ATETestDataSet:
    """
    Returns
    -------
    test_data : ATETestDataSet
        Uniformly sampled from price. time and emotion is averaged to get structural.
    """

    return ATETestDataSet(treatment=np.eye(2),
                          structural=None)


def generate_train_colangelo(data_size: int,
                             rand_seed: int = 42) -> ATETrainDataSet:
    """
    Generate the data in Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments
    [Colangelo and Lee, 2020]

    Parameters
    ----------
    data_size : int
        size of data
    rand_seed : int
        random seed


    Returns
    -------
    train_data : ATETrainDataSet
    """

    rng = default_rng(seed=rand_seed)
    backdoor_dim = 100
    backdoor_cov = np.eye(backdoor_dim)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), 1)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), -1)
    backdoor = rng.multivariate_normal(np.zeros(backdoor_dim), backdoor_cov,
                                       size=data_size)  # shape of (data_size, backdoor_dim)

    theta = np.array([1.0 / ((i + 1) ** 2) for i in range(backdoor_dim)])
    treatment_org = norm.cdf(backdoor.dot(theta) * 3) + 0.75 * rng.standard_normal(size=data_size)
    treatment = np.zeros((data_size, 2))
    treatment[treatment_org > 0, 0] = 1.0
    treatment[:, 1] = 1.0 - treatment[:, 0]
    outcome = 1.2 * treatment[:, 0] + 1.2 * backdoor.dot(theta) + treatment[:, 0] ** 2 + treatment[:, 0] * backdoor[:, 0]
    outcome += rng.standard_normal(size=data_size)

    return ATETrainDataSet(backdoor=backdoor,
                           treatment=treatment,
                           outcome=outcome[:, np.newaxis])
