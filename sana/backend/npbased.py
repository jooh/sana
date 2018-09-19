"""representational similarity analysis in numpy."""
import numpy as np
from sana.base import n2npairs, npairs2n


def square2vec(rdm):
    """map 2D distance matrix to [n,1] vector of unique distances. Returns distances in
    same order as scipy.spatial.distance.squareform."""
    return rdm[np.triu_indices_from(rdm, k=1)][:, None]


def allpairwisecontrasts(n):
    """return a npair by n matrix of contrast vectors. The result differences should be
    compatible with the default scipy distance matrix code (pdist, squareform)."""
    outsize = [n, int(n2npairs(n))]
    # as in tf.square2vec, use triu instead of tril since numpy is row major and matlab
    # column major
    mask = np.triu(np.ones((n, n), dtype="bool"), 1)
    rows, cols = mask.nonzero()
    ind = np.arange(outsize[1])
    posind = np.ravel_multi_index((rows, ind), outsize)
    negind = np.ravel_multi_index((cols, ind), outsize)
    result = np.zeros(outsize)
    result[np.unravel_index(posind, outsize)] = 1.
    result[np.unravel_index(negind, outsize)] = -1.
    return result.T


def sqsigned(x):
    """return signed square transform. Quirky concept to avoid imaginary numbers when
    working with multiple regression RSA."""
    return np.sign(x) * (x ** 2)


def sqrtsigned(x):
    """return signed square-root transform (ie, square root on abs value, then returned
    to original sign). Quirky concept to avoid imaginary numbers when working with
    multiple regression RSA."""
    return np.sign(x) * np.sqrt(np.abs(x))


def zscore(rdv, axis=0):
    return (rdv - np.mean(rdv, axis=axis, keepdims=True)) / np.std(
        rdv, axis=axis, keepdims=True
    )


def pearsonz_1vN(rdv, rdvn):
    """Fisher Z-transformed pearson correlation between rdv (f by 1 array) and
    rdvn (f by n array)."""
    return np.arctanh(np.linalg.lstsq(zscore(rdv), zscore(rdvn), rcond=None)[0])


# TODO - the two other convenient correlation algorithms - from the covariance
# matrix (to get a full n by n matrix) and from unit length cosines (to get
# pairwise correlations between arbitrary n by f and m by f arrays)

def flatten(resp):
    """flatten dimensions 1: of the input ND array resp."""
    return np.reshape(resp, [resp.shape[0], np.prod(resp.shape[1:])])


def euclideansq(resp):
    """squared euclidean distance matrix"""
    # resp is exemplar by features
    # reshape to put examplars in rows, features in columns
    resp = flatten(resp)
    # sum of squares over feature dim
    r = np.sum(resp * resp, axis=1)
    rdm = r[:, None] - 2 * np.matmul(resp, resp.T) + r[None, :]
    # take advantage of known properties of distance matrices to correct
    # rounding error - symmetry
    rdm = (rdm + rdm.T) / 2.
    # (NB, rdm diagonal won't be exactly zero but those values are not returned so it
    # doesn't matter)
    return square2vec(rdm)


def vec2square(rdv):
    xs = rdv.shape
    ncon = npairs2n(xs[0])
    assert np.allclose(
        ncon, np.round(ncon)
    ), "rdv is not convertible to square distance matrix"
    ncon = int(ncon)
    uind = np.triu_indices(ncon, k=1)
    rdm = np.zeros([ncon, ncon], rdv.dtype)
    rdm[uind] = rdv.flatten()
    rdm += rdm.T
    return rdm


def rdmsplitter(rdm):
    """leave-one-out splitter for input rdm (n by ncon by ncon array). Returns trainrdm
    (the sum RDM for all-but-one) and testrdm (the left-out RDM) on each call."""
    rdm = np.asarray(rdm)
    # singleton dimensions are impossible (condition dims are at least 2, and if you
    # have less than 2 entries in split dim, you have a problem)
    assert np.all(np.array(rdm.shape) > 1)
    assert np.ndim(rdm) == 3
    # assume stacked in rows
    nrdm = rdm.shape[0]
    # always leave one out for now
    allind = np.arange(nrdm)
    for ind in allind:
        testrdm = rdm[ind, :, :]
        # pearson is invariant to whether you sum or average. And with sum you
        # can easily add another rdv later (e.g. in noiseceiling)
        trainrdm = np.sum(rdm[np.setdiff1d(allind, ind), :, :], axis=0)
        yield trainrdm, testrdm


def noiseceiling(metric, trainrdv, testrdv):
    return (metric(trainrdv, testrdv), metric(trainrdv + testrdv, testrdv))
