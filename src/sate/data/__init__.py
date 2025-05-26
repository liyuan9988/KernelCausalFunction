from typing import Dict, Any, Optional

from .synthetic import generate_test_synthetic, generate_train_synthetic
from .synthetic_linear import generate_test_synthetic_linear, generate_train_synthetic_linear
from .synthetic_simple import generate_synthetic_simple_test_data, generate_synthetic_simple_train_data, \
    generate_jobcorp_simple_train_data, generate_jobcorp_simple_test_data
from .job_corp import generate_train_jobcorp, generate_test_jobcorp
from .data_class import SATETrainDataSet, SATETestDataSet


def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> SATETrainDataSet:
    data_name = data_config["name"]
    if data_name == "synthetic":
        return generate_train_synthetic(data_config["data_size"],
                                        data_config["backdoor_dim"],
                                        rand_seed)
    elif data_name == "synthetic_simple":
        return generate_synthetic_simple_train_data(data_config, rand_seed)
    elif data_name == "jobcorp_simple":
        return generate_jobcorp_simple_train_data()
    elif data_name == "synthetic_linear":
        return generate_train_synthetic_linear(data_config["data_size"], rand_seed)
    elif data_name == "job_corp":
        return generate_train_jobcorp()
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_config: Dict[str, Any]) -> Optional[SATETestDataSet]:
    data_name = data_config["name"]
    if data_name == "synthetic":
        return generate_test_synthetic()
    elif data_name == "synthetic_simple":
        return generate_synthetic_simple_test_data(data_config)
    elif data_name == "synthetic_linear":
        return generate_test_synthetic_linear()
    elif data_name == "job_corp":
        return generate_test_jobcorp()
    elif data_name == "jobcorp_simple":
        return generate_jobcorp_simple_test_data()
    else:
        raise ValueError(f"data name {data_name} is not valid")
