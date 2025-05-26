from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd

from src.me.data.data_class import METrainDataSet, METestDataSet


def generate_train_synthetic(data_size: int, rand_seed: int = 42) -> METrainDataSet:
    rng = np.random.default_rng(rand_seed)
    backdoor = rng.uniform(-1.5, 1.5, size=data_size)
    u = rng.uniform(-2, 2, size=data_size)
    v = rng.uniform(-2, 2, size=data_size)
    w = rng.uniform(-2, 2, size=data_size)
    treatment = 0.3 * backdoor + w
    mediate = 0.3 * treatment + 0.3 * backdoor + v
    outcome = 0.3 * treatment + 0.3 * mediate + 0.5 * treatment * mediate + 0.3 * backdoor + 0.25 * treatment ** 3 + u

    return METrainDataSet(treatment=treatment[:, np.newaxis],
                          backdoor=backdoor[:, np.newaxis],
                          mediate=mediate[:, np.newaxis],
                          outcome=outcome[:, np.newaxis])


def generate_test_synthetic():
    xv, yv = np.meshgrid(np.linspace(-1.5, 1.5, 9), np.linspace(-1.5, 1.5, 9))
    treatment = xv.ravel()
    new_treatment = yv.ravel()
    structural = 0.3 * new_treatment + 0.09 * treatment + 0.15 * treatment * new_treatment + 0.25 * new_treatment ** 3
    return METestDataSet(treatment=treatment[:, np.newaxis],
                         new_treatment=new_treatment[:, np.newaxis],
                         structural=structural[:, np.newaxis])