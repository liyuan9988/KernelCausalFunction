from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from src.sate.data import generate_train_data, generate_test_data
from src.sate.data.data_class import SATETrainDataSet, SATETestDataSet
from src.cate.data.data_class import CATETrainDataSet, CATETestDataSet
from src.cate.models.mean_embedding import BackDoorMeanEmbedding


def evaluate_mean_embedding_ignore_sequence_cate(data_config: Dict[str, Any], model_param: Dict[str, Any],
                                                 random_seed: int = 42, verbose: int = 0):
    train_data = generate_train_data(data_config, random_seed)
    test_data = generate_test_data(data_config)
    cate_train_data = CATETrainDataSet(backdoor=np.c_[train_data.backdoor_1st, train_data.backdoor_2nd],
                                       covariate=train_data.treatment_1st,
                                       treatment=train_data.treatment_2nd,
                                       outcome=train_data.outcome)
    cate_test_data = CATETestDataSet(treatment=test_data.treatment_2nd,
                                     covariate=test_data.treatment_1st,
                                     structural=test_data.structural)

    model = BackDoorMeanEmbedding(**model_param)
    model.fit(cate_train_data, "sate_to_cate")
    if cate_test_data.structural is not None:
        return model.evaluate(cate_test_data)
    else:
        return model.predict(cate_test_data.treatment)[:, 0]
