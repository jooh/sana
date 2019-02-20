"""plotting functionality using matplotlib."""
import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker

def imagelabels(images, ax=None, axis="y", nrow=1, zoom=0.1, offset=20, linewidth=None):
    """draw images at each tick along the axis ('x', 'y', or ['x','y'] of axis handle
    ax, potentially arranging the images into multiple rows. images is a dict where each
    key is a potential xtick and the value is an image matrix.

    Parameters
    ----------
    images : dict, required, default: 1
        dict where each key is a tick and each value is an image matrix.
    ax : matplotlib.Axes handle, optional, default: None
        axis to draw on
    axis : str, optional, default: 'y'
        axis direction to add labels to (lists are supported e.g. ['x','y'])
    nrow : int, optional, default: 1
        number of rows or columns to stack images along.
    zoom : float, optional, default: .1
        matplotlib.offsetbox.OffsetImage argument. As proportion of actual image size in
        figure, so a bit of tweaking is usually required to achieve something that looks
        reasonable.
    offset : float, optional, default: 20
        offset for each additional row of images (barring the first, which gets
        offset*.25). As zoom, this will require tuning.
    linewidth: float, optional, default: None
        width of connecting line. If 0, no line is drawn. If None (the default), the
        line width is set by the current major tick line width *.5 (why .5? Don't know,
        but it seems to work).
        
    If printing with 'bbox_inches'='tight' (or equivalently using matplotlib inline in a
    notebook), the annotations get cropped. This is a known Matplotlib issue (see
    https://github.com/matplotlib/matplotlib/issues/12699). Workaround this by passing
    bbox_extra_artists=[a.offsetbox for a in ab] when saving."""
    if ax is None:
        ax = plt.gca()
    axis = list(axis)
    ab = []
    for thisax in axis:
        axhand = getattr(ax, thisax + 'axis')
        if linewidth is None:
            linewidth = .5 * axhand.get_majorticklines()[0].properties()['linewidth']
        if linewidth > 0:
            # we replace the existing ticks
            axhand.set_tick_params(length=0)
        targetticks = list(set(axhand.get_ticklocs()).intersection(images.keys()))
        targetticks.sort()
        # start on the back row to reduce overdrawing of lines
        currentoffset = offset * (nrow+0.2)
        for thisrow in range(nrow):
            thistarget = targetticks[thisrow::nrow]
            currentoffset -= offset
            for key in thistarget:
                posarg = dict(xy = (0, key),
                        xybox = (-currentoffset, 0),
                        xycoords = ("axes fraction", "data"),
                        box_alignment = (1, .5))
                # map to correct order for axis
                if thisax.lower() == 'x':
                    posarg = {k: tuple(list(v)[::-1]) for k,v in posarg.items()}
                imbox = matplotlib.offsetbox.OffsetImage(images[key], zoom=zoom, 
                        cmap='gray')
                imbox.image.axes = ax
                ab.append(matplotlib.offsetbox.AnnotationBbox(imbox, boxcoords="offset points",
                        bboxprops=dict(edgecolor='w', facecolor='w'),
                        arrowprops=dict(linewidth=linewidth,
                            arrowstyle='-'),
                        pad=0.,
                        annotation_clip=False,
                        **posarg))
                artist = ax.add_artist(ab[-1])
    return ab

class SpecialTickLocator(matplotlib.ticker.LinearLocator):
    """LinearLocator sub-class to support manually inserting extra ticks as indicated
    by special_values. Otherwise identical functionality to LinearLocator.

    Example usage:

    ax_cb.yaxis.set_major_locator(SpecialTickLocator(numticks=2, special_values=0.))"""
    def __init__(self, special_values=[], **kwarg):
        super().__init__(**kwarg)
        self.special_values = np.asarray(special_values).flatten()

    def tick_values(self, vmin, vmax):
        # post filter
        ticklocs = super().tick_values(vmin, vmax)
        if self.special_values.size > 0:
            ticklocs = np.concatenate((ticklocs, self.special_values))
        return ticklocs


class ImageTickLocator(SpecialTickLocator):
    """SpecialTickLocator sub-class to support image plots, where the center of mass
    is offset (typically by 0.5) from the plot limit, and where only integer ticks make 
    sense. Otherwise identical functionality to SpecialTickLocator.

    Example usage:

    ax_plt.xaxis.set_major_locator(ImageTickLocator(numticks=5))
    """

    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def rescore(self, x, tickfun):
        x = np.asarray(tickfun(x))
        # fix ugly rounding error
        x[x == -0.] = 0.
        if x.size == 1:
            x = float(x)
        return x

    def tick_values(self, vmin, vmax):
        # we need to prefilter the method to place the ticks away from lims
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        vmin = self.rescore(vmin, np.ceil)
        vmax = self.rescore(vmax, np.floor)
        ticklocs = super().tick_values(vmin, vmax)
        # now any non-lim values can just be rounded
        ticklocs[1:-1] = self.rescore(ticklocs[1:-1], np.round)
        return ticklocs


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
