from typing import NamedTuple, Optional
import numpy as np


class ATETrainDataSet(NamedTuple):
    treatment: np.ndarray
    backdoor: np.ndarray
    outcome: np.ndarray


class ATETestDataSet(NamedTuple):
    treatment: np.ndarray
    structural: Optional[np.ndarray]

