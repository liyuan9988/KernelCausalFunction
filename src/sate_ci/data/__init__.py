from typing import Dict, Any, Optional
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm

from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet

from .job_corp import generate_train_jobcorp, generate_test_jobcorp
from .synthetic import generate_train_synthetic_data, generate_test_synthetic_data


def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> SATETrainDataSet:
    name = data_config["name"]
    if name == "synthetic":
        return generate_train_synthetic_data(data_config, rand_seed)
    elif name == "jobcorp":
        return generate_train_jobcorp()
    else:
        raise ValueError(f"{name} is not a valid data name")


def generate_test_data(data_config: Dict[str, Any]) -> Optional[SATETestDataSet]:
    name = data_config["name"]
    if name == "synthetic":
        return generate_test_synthetic_data()
    elif name == "jobcorp":
        return generate_test_jobcorp()
    else:
        raise ValueError(f"{name} is not a valid data name")
