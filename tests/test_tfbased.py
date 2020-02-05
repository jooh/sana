"""test sana.backend.tfbased."""
import numpy as np
import tensorflow as tf
from sana.backend import tfbased
import util


def test_euclideansq():
    responses = util.responses(3)
    rdv = tfbased.euclideansq(responses)
    util.assert_shape(rdv, [3, 1])
    np.testing.assert_allclose(rdv[0], np.sum((responses[0, :] - responses[1, :]) ** 2))
