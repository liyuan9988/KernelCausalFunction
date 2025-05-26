from typing import Dict, Any, Optional
import numpy as np
from numpy.random import default_rng

from src.me.data.data_class import METrainDataSet, METestDataSet


def generate_train_simple_colangelo(data_size: int, rand_seed: int) -> METrainDataSet:
    rng = default_rng(rand_seed)
    u = rng.uniform(-2, 2, data_size)
    v = rng.uniform(-2, 2, data_size)
    w = rng.uniform(-2, 2, data_size)
    X = rng.uniform(-1.5, 1.5, data_size)
    D = np.zeros(data_size, dtype=int)
    D[0.3 * X + w > 0] = 1
    M = 0.3 * D + 0.3 * X + v
    Y = 0.3 * D + 0.3 * M + 0.5 * D * M + 0.3 * X + 0.25 * (D ** 3) + u
    print(np.eye(2)[D])

    return METrainDataSet(treatment=np.eye(2)[D],
                          backdoor=X[:, np.newaxis],
                          mediate=M[:, np.newaxis],
                          outcome=Y[:, np.newaxis])


def generate_test_simple_colangelo() -> Optional[METestDataSet]:
    return METestDataSet(treatment=np.array([[1, 0],
                                             [0, 1],
                                             [1, 0],
                                             [0, 1]]),
                         new_treatment=np.array([[1, 0],
                                                 [1, 0],
                                                 [0, 1],
                                                 [0, 1]]),
                         structural=None)
