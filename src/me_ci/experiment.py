from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import ray
import logging

from src.utils import grid_search_dict
from src.me_ci.models.mean_embedding import evaluate_mean_embedding

logger = logging.getLogger()


def get_run_func(mdl_name: str):
    if mdl_name == "mean_embedding":
        return evaluate_mean_embedding
    else:
        raise ValueError(f"name {mdl_name} is not known")


def me_ci_experiments(configs: Dict[str, Any],
                      dump_dir: Path,
                      num_cpus: int, num_gpu: Optional[int]):

    data_config = configs["data"]
    model_config = configs["models"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1:
        ray.init(local_mode=True, num_gpus=num_gpu)
        verbose: int = 2
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpu)
        verbose: int = 0

    run_func = get_run_func(model_config["name"])
    remote_run = ray.remote(run_func)
    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = one_dump_dir.joinpath(mdl_dump_name)
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir
            tasks = [remote_run.remote(env_param, mdl_param, idx, verbose) for idx in range(n_repeat)]
            res = ray.get(tasks)
            np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(res))
        logger.critical(f"{dump_name} ended")

    ray.shutdown()
