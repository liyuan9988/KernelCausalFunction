from typing import NamedTuple, Optional
import numpy as np


class METrainDataSet(NamedTuple):
    treatment: np.ndarray
    backdoor: np.ndarray
    mediate: np.ndarray
    outcome: np.ndarray


class METestDataSet(NamedTuple):
    treatment: np.ndarray
    new_treatment: np.ndarray
    structural: Optional[np.ndarray]

