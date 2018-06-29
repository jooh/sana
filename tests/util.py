"""utilities for sana tests - data, assertions, etc."""
import numpy as np


def assert_shape(arrlike, shape):
    np.testing.assert_array_equal(np.array(arrlike).shape, np.array(shape))


def responses(ncon, nfeat=100):
    return np.random.rand(ncon, nfeat)


def signeddistancematrix():
    """5x5 matrix with distance from diagonal in increasing positive values below, and
    increasing negative values above."""
    return np.arange(-2, 3)[:, None] - np.arange(-2, 3)[None, :]


def signeddistancevector():
    """ground truth distance vector form for signeddistancematrix."""
    return -1 * np.array([1, 2, 3, 4, 1, 2, 3, 1, 2, 1])[:, None]
