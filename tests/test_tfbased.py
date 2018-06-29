"""test sana.backend.npbased."""
import numpy as np
import tensorflow as tf
from sana.backend import tfbased
import util


def runit(x):
    return tf.Session().run(x)


def test_euclideansq():
    responses = util.responses(3)
    rdv = runit(tfbased.euclideansq(responses))
    util.assert_shape(rdv, [3, 1])
    np.testing.assert_allclose(rdv[0], np.sum((responses[0, :] - responses[1, :]) ** 2))
