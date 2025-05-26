from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from src.sate.data import generate_train_data, generate_test_data
from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet
from src.ate.data.data_class import ATETrainDataSet, ATETestDataSet
from src.ate.models.mean_embedding import BackDoorMeanEmbedding


def evaluate_mean_embedding_ignore_sequence_ate(data_config: Dict[str, Any], model_param: Dict[str, Any],
                                                random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    ate_train_data = ATETrainDataSet(backdoor=np.c_[train_data.backdoor_1st, train_data.backdoor_2nd],
                                     treatment=np.c_[train_data.treatment_1st, train_data.treatment_2nd],
                                     outcome=train_data.outcome)
    ate_test_data = ATETestDataSet(treatment=np.c_[test_data.treatment_1st, test_data.treatment_2nd],
                                   structural=test_data.structural)

    model = BackDoorMeanEmbedding(**model_param)
    model.fit(ate_train_data, data_config["name"])
    if ate_test_data.structural is not None:
        return model.evaluate(ate_test_data)
    else:
        return model.predict(ate_test_data.treatment)[:, 0]

