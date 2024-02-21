from typing import Dict, Any, Optional

from .colangelo import generate_train_colangelo, generate_test_colangelo
from .job_corp import generate_train_jobcorp, generate_test_jobcorp
from .data_class import ATETrainDataSet, ATETestDataSet



def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> ATETrainDataSet:
    data_name = data_config["name"]
    if data_name == "colangelo":
        return generate_train_colangelo(data_config["data_size"], rand_seed)
    elif data_name == "job_corp":
        return generate_train_jobcorp()
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_config: Dict[str, Any]) -> Optional[ATETestDataSet]:
    data_name = data_config["name"]
    if data_name == "colangelo":
        return generate_test_colangelo()
    elif data_name == "job_corp":
        return generate_test_jobcorp()
    else:
        raise ValueError(f"data name {data_name} is not valid")
