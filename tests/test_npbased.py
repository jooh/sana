"""test sana.backend.npbased."""
import numpy as np
from sana.backend import npbased
import util

RTOL = 1e-14


def test_allpairwisecontrasts_shape():
    util.assert_shape(npbased.allpairwisecontrasts(10), np.array([45, 10]))


def test_allpairwisecontrasts_weight():
    np.testing.assert_allclose(npbased.allpairwisecontrasts(10).sum(), 0.)


def test_euclideansq():
    responses = util.responses(3)
    rdv = npbased.euclideansq(responses)
    util.assert_shape(rdv, [3, 1])
    np.testing.assert_allclose(
        rdv[0], np.sum((responses[0, :] - responses[1, :]) ** 2), rtol=RTOL
    )


def test_square2vec():
    np.testing.assert_array_equal(
        util.signeddistancevector(), npbased.square2vec(util.signeddistancematrix())
    )


def test_vec2square():
    np.testing.assert_array_equal(
        np.abs(util.signeddistancematrix()),
        npbased.vec2square(np.abs(util.signeddistancevector())),
    )
