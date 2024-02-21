from typing import Dict, Any, Iterator, Tuple
from itertools import product
import numpy as np


def cal_loocv_with_cluster(K, y, lam, cluster):
    nData = K.shape[0]
    I = np.eye(nData)
    H = I - (K*cluster).dot(np.linalg.inv((K + lam * nData * I)*cluster))
    tildeH_inv = np.diag(1.0 / np.diag(H))
    return np.linalg.norm(tildeH_inv.dot(H.dot(y)))

def cal_loocv(K, y, lam):
    nData = K.shape[0]
    I = np.eye(nData)
    H = I - K.dot(np.linalg.inv(K + lam * nData * I))
    tildeH_inv = np.diag(1.0 / np.diag(H))
    return np.linalg.norm(tildeH_inv.dot(H.dot(y)))


def cal_loocv_emb(K, kernel_y, lam):
    nData = K.shape[0]
    I = np.eye(nData)
    Q = np.linalg.inv(K + lam * nData * I)
    H = I - K.dot(Q)
    tildeH_inv = np.diag(1.0 / np.diag(H))

    return np.trace(tildeH_inv @ H @ kernel_y @ H @ tildeH_inv)


def grid_search_dict(org_params: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Iterate list in dict to do grid search.

    Examples
    --------
    >>> test_dict = dict(a=[1,2], b = [1,2,3], c = 4)
    >>> list(grid_search_dict(test_dict))
    [('a:1-b:1', {'c': 4, 'a': 1, 'b': 1}),
    ('a:1-b:2', {'c': 4, 'a': 1, 'b': 2}),
    ('a:1-b:3', {'c': 4, 'a': 1, 'b': 3}),
    ('a:2-b:1', {'c': 4, 'a': 2, 'b': 1}),
    ('a:2-b:2', {'c': 4, 'a': 2, 'b': 2}),
    ('a:2-b:3', {'c': 4, 'a': 2, 'b': 3})]
    >>> test_dict = dict(a=1, b = 2, c = 3)
    >>> list(grid_search_dict(test_dict))
    [('one', {'a': 1, 'b': 2, 'c': 3})]

    Parameters
    ----------
    org_params : Dict
        Dictionary to be grid searched

    Yields
    ------
    name : str
        Name that describes the parameter of the grid
    param: Dict[str, Any]
        Dictionary that contains the parameter at grid

    """
    search_keys = []
    non_search_keys = []
    for key in org_params.keys():
        if isinstance(org_params[key], list):
            search_keys.append(key)
        else:
            non_search_keys.append(key)
    if len(search_keys) == 0:
        yield "one", org_params
    else:
        param_generator = product(*[org_params[key] for key in search_keys])
        n_conbi = len(list(param_generator))
        param_generator = product(*[org_params[key] for key in search_keys])
        for one_param_set in param_generator:
            one_dict = {k: org_params[k] for k in non_search_keys}
            tmp = dict(list(zip(search_keys, one_param_set)))
            one_dict.update(tmp)
            one_name = "-".join([k + ":" + str(tmp[k]) for k in search_keys])
            if n_conbi > 1:
                yield one_name, one_dict
            else:
                yield "one", one_dict
