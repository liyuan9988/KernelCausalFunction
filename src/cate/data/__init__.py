from typing import Dict, Any, Optional

from .job_corp import generate_train_jobcorp, generate_test_jobcorp
from .synthetic import generate_test_synthetic, generate_train_synthetic
from .data_class import CATETrainDataSet, CATETestDataSet


def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> CATETrainDataSet:
    data_name = data_config["name"]
    if data_name == "job_corp":
        return generate_train_jobcorp()
    elif data_name == "synthetic":
        return generate_train_synthetic(data_config["data_size"], rand_seed)
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_config: Dict[str, Any]) -> Optional[CATETestDataSet]:
    data_name = data_config["name"]
    if data_name == "job_corp":
        return generate_test_jobcorp()
    elif data_name == "synthetic":
        return generate_test_synthetic()
    else:
        raise ValueError(f"data name {data_name} is not valid")
