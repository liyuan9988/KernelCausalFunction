import numpy as np
from numpy.random import default_rng

from src.cate.data.data_class import CATETrainDataSet, CATETestDataSet

def generate_test_synthetic() -> CATETestDataSet:
    covariate = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    treatment = np.array([1, 1, 1, 1, 1])  # only test D=1
    structural = covariate * ((1 + 2 * covariate) ** 2) * ((covariate - 1) ** 2)
    return CATETestDataSet(covariate=covariate[:, np.newaxis],
                           treatment=treatment[:, np.newaxis],
                           structural=structural[:, np.newaxis])


def generate_train_synthetic(data_size: int,
                             rand_seed: int = 42) -> CATETrainDataSet:
    rng = default_rng(seed=rand_seed)
    covariate = rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    x1 = 1 + 2 * covariate + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    x2 = 1 + 2 * covariate + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    x3 = (covariate - 1) ** 2 + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    backdoor = np.c_[x1, x2, x3]
    prob = 1.0 / (1.0 + np.exp(-0.5 * (covariate + x1 + x2 + x3)))
    treatment = (rng.random(data_size) < prob).astype(float)
    outcome = covariate * x1 * x2 * x3 + rng.normal(0.0, 0.25, size=(data_size, ))
    outcome *= treatment
    return CATETrainDataSet(treatment=treatment[:, np.newaxis],
                            backdoor=backdoor,
                            covariate=covariate[:, np.newaxis],
                            outcome=outcome[:, np.newaxis])
