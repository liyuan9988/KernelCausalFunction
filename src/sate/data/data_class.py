from typing import NamedTuple, Optional
import numpy as np


class SATETrainDataSet(NamedTuple):
    treatment_1st: np.ndarray
    backdoor_1st: np.ndarray
    treatment_2nd: np.ndarray
    backdoor_2nd: np.ndarray
    outcome: np.ndarray


class SATETestDataSet(NamedTuple):
    treatment_1st: np.ndarray
    treatment_2nd: np.ndarray
    structural: Optional[np.ndarray]

