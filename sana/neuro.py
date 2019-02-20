"""utility functions for neuroimaging-specific applications."""
import logging
import numpy as np
import sana
from scipy.ndimage import affine_transform
#import nipy.modalities.fmri.hemodynamic_models as hrf
import nistats.hemodynamic_models as hrf
import nibabel as nib

logging.basicConfig(format="%(asctime)s %(filename)s %(funcName)s %(message)s",
                    datefmt="%Y/%m/%d %H:%M",
                    level="INFO")
LOGGER = logging.getLogger(__name__)

def checkregistration(im1, im2, **kwarg):
    """check that the nibabel image instances have similar affines and identical image
    dimensions. Useful check before running array-based operations over images. Any
    kwarg are passed on to numpy.allclose (rtol and atol might be especially useful)."""
    registered = np.allclose(im1.affine, im2.affine, **kwarg)
    shaped = np.all(np.asarray(im1.shape[:3]) == np.asarray(im2.shape[:3]))
    return registered and shaped

def combineroi(roilist):
    refroi =  roilist[0]
    refmat = refroi.get_fdata()
    for thisroi in roilist[1:]:
        assert checkregistration(refroi, thisroi), "images are not registered"
        # get ROI data but don't cache
        thisroimat = np.asarray(thisroi.dataobj)
        boolhit = (thisroimat!=0) & (refmat==0)
        refmat[boolhit] = thisroimat[boolhit]
    return nib.Nifti1Image(refmat, thisroi.affine)

def resliceroi(roi, epi, *, matchn=True, **kwarg):
    """reslice the nibabel instance roi to the space of the nibabel instance epi, and
    return a boolean matrix. if matchn, we ensure that the resliced roimat has the same
    number of non-zero voxels are the original roimat. All kwarg are passed to
    scipy.ndimage.affine_transform (order is probably the main one of interest - 1 for
    nearest neighbour)."""
    roi2epi = np.linalg.inv(roi.affine) @ epi.affine
    roimat = roi.get_data()
    roimat_epi = affine_transform(
        (roimat != 0).astype("float"),
        roi2epi,
        output_shape=epi.shape[:3],
        mode="constant",
        cval=0.0,
        **kwarg
    )
    thresh = 1.0 - np.finfo(roimat_epi.flatten()[0]).eps
    if matchn:
        thresh = 1 - np.percentile(
            1 - roimat_epi, 100 * ((roimat != 0).astype("float").sum() / roimat.size)
        )
        LOGGER.info(f"thresholding epi at {thresh} to match n")
    return roimat_epi >= thresh

def loadroidata(roimat, epi):
    """return a samples by features matrix where each feature is a non-zero index in
    roimat and every row is a timepoint from the nibabel instance epi.

    NB big assumption that roimat is resliced to match epi - see resliceroi."""
    roidata = []
    roiind = np.where(roimat != 0)
    # I had hoped that something like this would work, but alas...
    # epidata = nepi.dataobj[roiind[0], roiind[1], roiind[2], :]
    # unlike thisepi.get_data(), this avoids cacheing
    epidata = np.asarray(epi.dataobj)
    # samples by features
    return epidata[roiind[0],roiind[1],roiind[2],:].T

def convolveevents(evtable, epi, *, hrf_model='spm', target_col='stim_id',
        regressors=None, **kwarg):
    """return a convolved design matrix for the design in pd.DataFrame-like evtable,
    using the meta data in nibabel.Nifti1Image-like epi to deduce tr and nvol."""
    if not "amplitude" in evtable:
        LOGGER.info("no amplitude field in evtable, creating")
        evtable = evtable.assign(amplitude=1)
    if regressors is None:
        LOGGER.info("no regressors input, using unique entries in target_col")
        regressors = evtable[target_col].unique()
        regressors = np.sort(regressors[np.isnan(regressors) == False])
    tr = epi.header.get_zooms()[-1]
    nvol = epi.shape[-1]
    frametimes = np.arange(0,nvol*tr,tr)
    convolved = []
    for thisreg in regressors:
        regtable = evtable.loc[evtable[target_col] == thisreg]
        vals = regtable[['onset', 'duration', 'amplitude']].values.T
        convolved.append(hrf.compute_regressor(vals, hrf_model, frametimes, **kwarg)[0])
    return np.concatenate(convolved, axis=1)

def vol2covdeg(nvol, tr):
    """Kay heuristic for selecting polynomial degree by run duration."""
    return int((nvol * tr / 60 / 2).round())

def bidsiterator(layout, subject, sessions, **kwarg):
    """iterate over a pybids layout instance, yielding a file and its associated
    events. Any keyword arguments are passed to layout.get (use to filter the type of
    preprocessed data you want, e.g., extensions='bold_space-T1w_preproc.nii.gz')."""
    for sess in sessions:
        sessruns = layout.get_runs(session=sess, subject=subject)
        for run in sessruns:
            runfile = layout.get(return_type='file', subject=subject, session=sess,
                    run=run, **kwarg)
            assert len(runfile) == 1
            runevents = layout.get(return_type='file', type='events', subject=subject,
                    session=sess, run=run)
            assert len(runevents) == 1
            yield runevents[0], runfile[0]

def preparefmrirun(event, func, roi, *, polydeg="adaptive", **kwarg):
    """
    Any **kwarg are passed to convolveevents.

    NB if you want to combine bidsiterator and preparefmrirun you need to take care
    of instancing the correct classes from the file paths (nibabel.Nifti1Image and
    pandas.DataFrame probably)."""
    # data
    data = loadroidata(roi, func)
    nvol, nvox = data.shape
    tr = func.header.get_zooms()[-1]
    # events
    convolved = convolveevents(event, func, **kwarg)
    # filter
    if polydeg == "adaptive":
        polydeg = vol2covdeg(nvol, tr)
        LOGGER.info(f"set adaptive polynomial degree: {polydeg}")
    trends = sana.npbased.polynomialmatrix(nvol, polydeg)
    return sana.npbased.projectout(convolved, trends), sana.npbased.projectout(data, trends)

def enumeratechunk(iterator):
    """for each item in iterator, prepend to the returns a chunkvec given by the index
    of that entry (from enumerate) replicated to correspond to item[0].shape[0]. This is
    useful to wrap an iterator based on preparefmrirun."""
    for chunk, item in enumerate(iterator):
        chunkvec = np.tile(chunk, item[0].shape[0])
        yield chunkvec, item
