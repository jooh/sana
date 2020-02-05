"""representational similarity analysis in tensorflow."""
import tensorflow as tf
from sana.base import npairs2n, n2npairs


def sqsigned(x):
    """return signed square transform. Quirky concept to avoid imaginary numbers when
    working with multiple regression RSA."""
    return tf.sign(x) * (x ** 2)


def sqrtsigned(x):
    """return signed square-root transform (ie, square root on abs value, then returned
    to original sign). Quirky concept to avoid imaginary numbers when working with
    multiple regression RSA."""
    return tf.sign(x) * tf.sqrt(tf.abs(x))


def sqsignit(fun):
    """class decorator to handle squaring the input (assumed only one that needs
    squaring), square rooting the output."""

    def wrapper(self, x, *arg, **kwarg):
        return sqrtsigned(fun(self, sqsigned(x), *arg, **kwarg))

    return wrapper


class OLS(object):
    """minimal OLS RSA regression model implementation. Inputs (y, b) get squared before
    use. Outputs as returned as original units."""

    def __init__(self, X, useconstant=True):
        with tf.name_scope("rsa-OLS-init"):
            self.constant = None
            self.X = X
            if useconstant:
                constant = tf.ones([tf.shape(self.X)[0], 1], self.X.dtype)
                self.X = tf.concat([self.X, constant], axis=1)
            self.X = sqsigned(X)
            return

    @sqsignit
    def fit(self, y, **kwarg):
        with tf.name_scope("rsa-OLS-fit"):
            return tf.matrix_solve_ls(self.X, y, **kwarg)

    @sqsignit
    def predict(self, b):
        with tf.name_scope("rsa-OLS-predict"):
            return tf.matmul(self.X, b)


def debugit(*arg, feed_dict=None):
    """silly little convenience function to set up a session, run the global
    variables initializer, run the graph to get outputs in arg, and
    set_trace."""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(arg, feed_dict=feed_dict)
        print(res)
        import pdb
        pdb.set_trace()
    return


def measurement(resp, p):
    """model measurement effects in the filters by translating the response at
    each location and stimulus (first 3 axes of resp) toward the filterwise mean
    (4th axis) according to proportion p. p=1 means that all filters reduce
    to their respective means; p=0 does nothing; p<0 is possible but probably
    not something you want."""
    resp = tf.convert_to_tensor(resp)
    # average the filter dim
    meanresp = tf.reduce_mean(resp, axis=3, keepdims=False)
    # make resp the origin of meanresp and scale by p
    transresp = (meanresp[:, :, :, None] - resp) * p
    return resp + transresp


def meanresponse(resp):
    """return the mean model response to each exemplar (first dim), collapsing
    all others."""
    return tf.reduce_mean(tf.convert_to_tensor(resp), axis=[1, 2, 3])


def flatten(resp):
    """flatten the ND response to 2D (exemplar x feature)."""
    with tf.name_scope("rsa-flatten"):
        resp = tf.convert_to_tensor(resp)
        rshape = tf.shape(resp)
        return tf.reshape(resp, [rshape[0], tf.reduce_prod(rshape[1:])])


def euclideansq(resp):
    """return the squared euclidean distance matrix for the first axis in
    resp."""
    with tf.name_scope("rsa-euclidean"):
        # reshape to 2D (and convert to tensor if necessary)
        respflat = flatten(resp)
        r = tf.reduce_sum(respflat * respflat, axis=1)
        rdm = r[:, None] - 2 * tf.matmul(respflat, tf.transpose(respflat)) + r[None, :]
        # take advantage of known properties of distance matrices to correct
        # rounding error
        rdm = (rdm + tf.transpose(rdm)) / 2.
        # (NB, rdm diagonal won't be exactly zero but those values are not returned so it
        # doesn't matter)
        return square2vec(rdm)


def square2vec(mat):
    """convert square distance matrix in rdm to vector
    form (n by 1)."""
    with tf.name_scope("rsa-square2vec"):
        mat = tf.convert_to_tensor(mat)
        return tf.expand_dims(tf.boolean_mask(mat, triuind(mat)), axis=1)


def zscore(x, axis=0):
    with tf.name_scope("rsa-zscore"):
        x = tf.convert_to_tensor(x)
        m, v = tf.nn.moments(x, axis, keep_dims=True)
        return tf.nn.batch_normalization(x, m, v, None, None, 1e-12)


def trilind(x):
    """return boolean indices into the lower triangular part of rdm, excluding
    the diagonal. x contains a square distance matrix along the two innermost
    dims."""
    return _matrix_band_part_inverted(x, 0, -1)


def triuind(x):
    """return boolean indices into the upper triangular part of rdm, excluding
    the diagonal. x contains a square distance matrix along the two innermost
    dims."""
    return _matrix_band_part_inverted(x, -1, 0)


def _matrix_band_part_inverted(x, *arg):
    with tf.name_scope("rsa-matrix_band_inv"):
        x = tf.convert_to_tensor(x)
        xs = tf.shape(x)
        # matrix_band_part always -includes- the diagonal, so need to flip the logic
        # to always -exclude- it.
        return tf.equal(
            tf.linalg.matrix_band_part(tf.ones(xs, tf.bool), *arg), tf.constant(False)
        )


def vec2square(rdv):
    # TODO - tf doesn't support boolean assign so this is tricky.
    # something like this, but then rdm needs to be set as Variable, which is
    # problematic.
    # test = tf.scatter_nd_update(tf.Variable(tf.zeros([10,10],tf.float32)),tf.transpose(maskind),tfg['rdv'])
    # and the problem is that we need to initialize the variable (the rdm) to
    # update it.
    raise NotImplemented("no support for in-graph vec2square transforms yet")
    xs = tf.shape(rdv)
    ncon = npairs2n(xs[-1])
    rdm = tf.zeros([xs[0], ncon, ncon], rdv.dtype)
    rdm[trilind(rdm)] = rdv
    rdm[triuind(rdm)] = rdv
    return rdm
