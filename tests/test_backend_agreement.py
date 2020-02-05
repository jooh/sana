"""test agreement between corresponding functionality across backends"""
import numpy as np
from sana.backend import npbased, tfbased
import test_tfbased
import util

# there are surprisingly substantial float precision discrepances between numpy and tf
# computations
ATOL = 1e-4
RTOL = 1e-14


def test_square2vec():
    pattern = np.random.rand(10, 10).astype("float32")
    np.testing.assert_array_equal(
        npbased.square2vec(pattern), tfbased.square2vec(pattern)
    )


def test_euclideansq():
    # cast to common precision (tf doesn't do double)
    responses = util.responses(10).astype("float32")
    np.testing.assert_allclose(
        npbased.euclideansq(responses),
        tfbased.euclideansq(responses),
        rtol=RTOL,
        atol=ATOL,
    )


def test_zscore():
    responses = util.responses(10).astype("float32")
    np.testing.assert_allclose(
        npbased.zscore(responses),
        tfbased.zscore(responses),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        npbased.zscore(responses, axis=1),
        tfbased.zscore(responses, axis=1),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        npbased.zscore(responses, axis=0),
        tfbased.zscore(responses, axis=0),
        rtol=RTOL,
        atol=ATOL,
    )
