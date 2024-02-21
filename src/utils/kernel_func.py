import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import loggamma
import sympy
from typing import Callable


def log_factorial(p:int):
    return loggamma(p+1)

class AbsKernel:
    def fit(self, data: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        # Compute gradient of kernel matrix with respect to data2.
        raise NotImplementedError

    def cal_kernel_mat_jacob(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        # Compute jacobi matrix of kernel matrix. dx1 dx2 k(x1, x2)
        # assert data1 and data2 are single dimensions
        raise NotImplementedError


class LinearKernel(AbsKernel):

    def __init__(self):
        super(LinearKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        return data1 @ data2.T


class BinaryKernel(AbsKernel):

    def __init__(self):
        super(BinaryKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs):
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        res = data1.dot(data2.T)
        res += (1 - data1).dot(1 - data2.T)
        return res


class FourthOrderGaussianKernel(AbsKernel):
    bandwidth: np.float64

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.bandwidth = np.sqrt(np.median(dists))

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        diff_data = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        u = diff_data / self.bandwidth
        kernel_tensor = np.exp(- u ** 2 / 2.0) * (3 - u ** 2) / 2.0 / np.sqrt(6.28)
        return np.product(kernel_tensor, axis=2)


class SixthOrderGaussianKernel(AbsKernel):
    bandwidth: np.float64

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.bandwidth = np.sqrt(np.median(dists))

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        diff_data = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        u = diff_data / self.bandwidth
        kernel_tensor = np.exp(- u ** 2 / 2.0) * (15 - 10 * u ** 2 + u ** 4) / 8.0 / np.sqrt(6.28)
        return np.product(kernel_tensor, axis=2)


class FourthOrderEpanechnikovKernel(AbsKernel):
    bandwidth: np.float64

    def fit(self, data: np.ndarray, **kwargs) -> None:
        n_data = data.shape[0]
        assert data.shape[1] == 1
        self.bandwidth = 3.03 * np.std(data) / (n_data ** 0.12)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean') / (self.bandwidth ** 2)
        mat = (1.0 - dists) * (3 / 4) / self.bandwidth
        mat = np.maximum(mat, 0.0)
        mat = mat * (1.0 - 7 * dists / 3) * 15 / 8
        return mat


class EpanechnikovKernel(AbsKernel):
    bandwidth: np.float64

    def fit(self, data: np.ndarray, **kwargs) -> None:
        n_data = data.shape[0]
        assert data.shape[1] == 1
        self.bandwidth = 2.34 * np.std(data) / (n_data ** 0.25)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean') / (self.bandwidth ** 2)
        mat = (1.0 - dists) * (3 / 4) / self.bandwidth
        mat = np.maximum(mat, 0.0)
        return mat


class GaussianKernel(AbsKernel):
    sigma: np.float64

    def __init__(self):
        super(GaussianKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.sigma = np.median(dists)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        return np.exp(-dists)

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        res = np.exp(-dists)[:, :, np.newaxis]
        res = res * 2 / self.sigma * (data1[:, np.newaxis, :] - data2[np.newaxis, :, :])
        return res

    def cal_kernel_mat_jacob(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        res = np.exp(-dists) * (2 / self.sigma - 4 / self.sigma * dists)
        return res


class MaternKernel(AbsKernel):
    sigma: np.float64
    p: int
    f_matern: Callable

    def __init__(self, p: int):
        self.p = p
        super(MaternKernel, self).__init__()

    def construct_formulation(self):
        dist = sympy.symbols("d")
        matern = 0.0
        for i in range(self.p+1):
            const = log_factorial(self.p+i) - log_factorial(self.p-i) - log_factorial(i)
            const = const + log_factorial(self.p) - log_factorial(2*self.p)
            const = np.exp(const)
            matern = matern + const * (2*np.sqrt(2*self.p+1)*dist / self.sigma)**(self.p-i)
        matern = matern * sympy.exp(-np.sqrt(2*self.p+1)*dist / self.sigma)
        self.f_matern = sympy.utilities.lambdify(dist, matern, ["numpy", "scipy"])
        d_matern = matern.diff(dist)
        self.f_d_matern = sympy.utilities.lambdify(dist, d_matern, ["numpy", "scipy"])
        dd_matern = d_matern.diff(dist)
        self.f_dd_matern = sympy.utilities.lambdify(dist, dd_matern, ["numpy", "scipy"])

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'euclidean')
        self.sigma = np.median(dists)
        self.construct_formulation()

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'euclidean')
        return self.f_matern(dists)

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'euclidean')
        dists = dists[:, :, np.newaxis]
        d_matern = self.f_d_matern(dists)
        res = (data1[:, np.newaxis, :] - data2[np.newaxis, :, :]) / dists
        res[np.isnan(res)] = 1.0
        return -res * d_matern

    def cal_kernel_mat_jacob(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'euclidean')
        dd_matern = self.f_dd_matern(dists)
        res = (data1 - data2.T) ** 2 / (dists ** 2)
        res[np.isnan(res)] = 1.0
        return -res * dd_matern
