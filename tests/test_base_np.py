"""test base, using np"""
import numpy as np
from sana import base

TESTN = [2, 3, 5, 10, 45, 990]
TESTNPAIR = [1, 3, 10, 45, 990, 489555]


def test_npairs2n():
    assert np.all(base.npairs2n(np.array(TESTNPAIR)).astype(int) == np.array(TESTN))


def test_n2npairs():
    assert np.all(base.n2npairs(np.array(TESTN)).astype(int) == np.array(TESTNPAIR))
