"""plotting functionality using matplotlib."""
import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

plt.interactive(True)


def plotdim(npanel, maxcol=12, mode="square"):
    """work out a rectangular plot grid that achieves a total of npanel. maxcol
    limits how many columns we allow. mode can be 'square' (keep approximately
    equal nrow and ncol), 'cols' (as few rows as possible), 'rows' (as few
    columns as possible)."""
    # here's what we know: the most rows we'll consider is the square case
    maxrow = int(np.ceil(np.sqrt(npanel)))
    # and the max number of columns is the min of npanel and some sensible max
    maxcol = min(maxcol, npanel)
    # so all the possible pairings would be
    row, col = np.meshgrid(np.arange(maxrow) + 1, np.arange(maxcol) + 1)
    nactual = row * col
    hits = nactual == npanel
    if not np.any(hits):
        # allow for empty panels
        npanel += 1
        return plotdim(npanel, maxcol=maxcol)
    if mode == "square":
        # try to preserve square aspect ratio
        asym = col - row
    elif mode == "cols":
        # as few rows as possible
        asym = row
    elif mode == "rows":
        asym = col
    else:
        raise ValueError(f"invalid mode: {mode}")
    # we only want cases where we have more columns than rows
    asym[asym < 0] = np.iinfo(asym.dtype).max
    # now we want the smallest asym that is also a hit
    bestasym = asym[hits].min()
    final = np.all([hits, asym == bestasym], axis=0)
    # so now it's something like
    bestrows = row[final]
    bestcols = col[final]
    return bestrows[0], bestcols[0]


def plotrdm(rdv, label, **kwarg):
    nrow, ncol = plotdim(rdv.shape[1])
    plt.figure("plotrdm")
    plt.clf()
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, num="plotrdm")
    ax = list(ax.flatten())
    ax.reverse()
    for ind in range(rdv.shape[1]):
        rdm = squareform(rdv[:, ind])
        thisax = ax.pop()
        plt.sca(thisax)
        plt.imshow(rdm, interpolation="nearest", **kwarg)
        plt.title(label[ind])
        thisax.axis("off")
    return fig


def plotimages(*arg, colorbar=True, **kwarg):
    """plot all the tensorflow-formatted (leading 1st dim, filters/images
    stacked along 4th) input args. if colorbar=True we plot a colorbar for each
    panel. All additional keyword arguments are passed to plotdim."""
    paneldim = np.vstack([x.shape for x in arg])
    sharedim = np.all(paneldim == paneldim[0, :], axis=0)
    sharemap = {True: "all", False: "none"}
    n = np.sum(paneldim[:, -1])
    nrow, ncol = plotdim(n, **kwarg)
    fig = plt.figure("plotit")
    fig.clf()
    fig, ax = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        sharex=sharemap[sharedim[2]],
        sharey=sharemap[sharedim[3]],
        num="plotit",
    )
    ax = list(ax.flatten())
    ax.reverse()
    for argind, thisarg in enumerate(arg):
        for pind in range(paneldim[argind, -1]):
            thisax = ax.pop()
            plt.sca(thisax)
            plt.imshow(thisarg[0, :, :, pind], cmap="gray", interpolation="nearest")
            thisax.axis("off")
            if colorbar:
                plt.colorbar()
    # remove whatever is left
    [thisax.remove for thisax in ax]
    return fig
