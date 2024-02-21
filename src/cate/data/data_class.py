from typing import NamedTuple, Optional
import numpy as np


class CATETrainDataSet(NamedTuple):
    treatment: np.ndarray
    backdoor: np.ndarray
    covariate: np.ndarray
    outcome: np.ndarray


class CATETestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: np.ndarray
    structural: Optional[np.ndarray]

