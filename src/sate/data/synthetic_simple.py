from typing import Dict, Any, Optional
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm

from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet


def generate_synthetic_simple_train_data(data_config: Dict[str, Any], rand_seed: int) -> SATETrainDataSet:
    from src.sate_ci.data import generate_train_synthetic_data
    return generate_train_synthetic_data(data_config, rand_seed)


def generate_synthetic_simple_test_data(data_config: Dict[str, Any]) -> Optional[SATETestDataSet]:
    return SATETestDataSet(treatment_1st=np.array([[1, 0],
                                                   [0, 1],
                                                   [1, 0],
                                                   [0, 1]]),
                           treatment_2nd=np.array([[1, 0],
                                                   [1, 0],
                                                   [0, 1],
                                                   [0, 1]]),
                           structural=None)


def generate_jobcorp_simple_train_data() -> SATETrainDataSet:
    from src.sate_ci.data.job_corp import generate_train_jobcorp
    return generate_train_jobcorp()


def generate_jobcorp_simple_test_data() -> Optional[SATETestDataSet]:
    from src.sate_ci.data.job_corp import generate_test_jobcorp
    return generate_test_jobcorp()