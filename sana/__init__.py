"""similarity analysis in python, using numpy and tensorflow. Main API - for
low-level functionality, see backend modules."""
__version__ = "alpha"
__all__ = ["base", "plot"]
import itertools
import numpy as np
# only auto import module(s) with no dependencies beyond numpy
from sana import base
from sana.backend import npbased

def kfold(chunks, k=1):
    """return kfold train / test boolean indices in chunks, splitting on unique elements
    thereof (we ignore any nan so this can be used to code observations that should be
    excluded from the splits)."""
    chunks = chunks.flatten()
    uchunk = np.unique(chunks[~np.isnan(chunks)])
    for testchunk in itertools.combinations(uchunk, k):
        trainchunk = np.setdiff1d(uchunk,testchunk)
        yield npbased.chunk2bool(chunks, trainchunk), npbased.chunk2bool(chunks, testchunk)

def crossvalidate(splitter, axis, *arg):
    """split each input in arg into train and test splits by indexing along the axis
    using the indices returned by splitter. Both returns are lists."""
    for trainind, testind in splitter:
        # from list of arg where each entry is [train, test] to [trainarg, testarg]
        yield zip(*[(np.compress(trainind, thisdata, axis=axis),
            np.compress(testind, thisdata, axis=axis)) for thisdata in arg])


def noiseceiling(rdm, axis=1, metric=npbased.pearsonz_1vN):
    # assumes independent observations along axis
    chunks = np.arange(rdm.shape[axis])
    cv = crossvalidate(kfold(chunks, k=1), axis, rdm)
    for train, test in cv:
        # calculate means
        # (nb, indexing here since crossvalidate always returns lists)
        trainmean = np.mean(train[0], axis=axis, keepdims=True)
        testmean = np.mean(test[0], axis=axis, keepdims=True)
        # in principle one could deduce this from the above but this is more didactic
        allmean = np.mean(np.concatenate(train+test, axis=axis), axis=axis, keepdims=True)
        low = metric(trainmean, testmean)
        high = metric(allmean, testmean)
        yield low, high

def crossnobis(splitter, contrast, testcontrast=None, weightfun=npbased.discriminant,
        confun=npbased.discriminantcontrast, **kwarg):
    if testcontrast is None:
        testcontrast = contrast
    for (traindesign, traindata), (testdesign, testdata) in splitter:
        weights = weightfun(traindesign, traindata, contrast, **kwarg)
        yield confun(testdesign, testdata, testcontrast, weights)
