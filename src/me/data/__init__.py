from typing import Dict, Any, Optional

from .job_corp import generate_train_jobcorp, generate_test_jobcorp
from .synthetic import generate_train_synthetic, generate_test_synthetic
from .simple_colangelo import generate_train_simple_colangelo, generate_test_simple_colangelo
from .data_class import METrainDataSet, METestDataSet


def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> METrainDataSet:
    data_name = data_config["name"]
    if data_name == "job_corp":
        return generate_train_jobcorp()
    elif data_name == "synthetic":
        return generate_train_synthetic(data_config["data_size"], rand_seed)
    elif data_name == "simple_colangelo":
        return generate_train_simple_colangelo(data_config["data_size"], rand_seed)
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_config: Dict[str, Any]) -> Optional[METestDataSet]:
    data_name = data_config["name"]
    if data_name == "job_corp":
        return generate_test_jobcorp()
    elif data_name == "synthetic":
        return generate_test_synthetic()
    elif data_name == "simple_colangelo":
        return generate_test_simple_colangelo()
    else:
        raise ValueError(f"data name {data_name} is not valid")
