import pandas as pd
import numpy as np
import os
import math
from osgeo import gdal
import logging
logger = logging.getLogger('pyimpute')


def load_training_vector(response_shapes, explanatory_rasters, response_field, metric='mean'):
    """
    Parameters
    ----------
    response_shapes : Source of vector features for raster_stats; 
                      can be OGR file path or iterable of geojson-like features
    response_field : Field name containing the known response category (must be numeric)
    explanatory_rasters : List of Paths to GDAL rasters containing explanatory variables
    metric : Statistic to aggregate explanatory data across line and polygon vector features
             Defaults to 'mean' (optional)

    Returns
    -------
    train_xs : Array of explanatory variables
    train_ys : 1xN array of known responses
    """
    from rasterstats import raster_stats
    explanatory_dfs = []
    fldnames = []

    for i, raster in enumerate(explanatory_rasters):
        logger.debug("Rasters stats on %s" % raster)
        stats = raster_stats(response_shapes, raster, stats="mean", copy_properties=True)
        df = pd.DataFrame(stats, columns=["__fid__", response_field, metric])
        fldname = "%s_%d" % (metric, i)
        fldnames.append(fldname)

        df.rename(columns={metric: fldname,}, inplace=True)
        orig_count = df.shape[0]
        df = df[pd.notnull(df[fldname])]
        new_count = df.shape[0]
        if (orig_count - new_count) > 0:
            logger.warn('Dropping %d rows due to nans' % (orig_count - new_count))
        explanatory_dfs.append(df)

    current = explanatory_dfs[0]
    for i, frame in enumerate(explanatory_dfs[1:], 2):
        current = current.merge(frame, on=['__fid__', response_field])

    train_y = np.array(current[response_field])
    train_xs = np.array(current[fldnames])

    return train_xs, train_y


def load_training_rasters(response_raster, explanatory_rasters, selected=None):
    """
    Parameters
    ----------
    response_raster : Path to GDAL raster containing responses
    explanatory_rasters : List of Paths to GDAL rasters containing explanatory variables

    Returns
    -------
    train_xs : Array of explanatory variables
    train_ys : 1xN array of known responses
    """

    ds = gdal.Open(response_raster)
    if ds is None:
        raise Exception("%s not found" % response_raster)
    response_data = ds.ReadAsArray().flatten()

    if selected is None:
        train_y = response_data
    else:
        train_y = response_data[selected]

    selected_data = []
    for rast in explanatory_rasters:
        ds = gdal.Open(rast)
        if ds is None:
            raise Exception("%s not found" % rast)
        explanatory_data = ds.ReadAsArray().flatten()
        assert explanatory_data.size == response_data.size
        if selected is None:
            selected_data.append(explanatory_data)
        else:
            selected_data.append(explanatory_data[selected])

    train_xs = np.asarray(selected_data).T
    return train_xs, train_y


def load_targets(explanatory_rasters):
    """
    Parameters
    ----------
    explanatory_rasters : List of Paths to GDAL rasters containing explanatory variables

    Returns
    -------
    expl : Array of explanatory variables
    raster_info : dict of raster info
    """

    explanatory_raster_arrays = []
    gt = None
    shape = None
    srs = None

    for raster in explanatory_rasters:
        logger.debug(raster)
        r = gdal.Open(raster)
        if r is None:
            raise Exception("%s not found" % raster)
        ar = r.ReadAsArray()
        
        # Save or check the geotransform
        if not gt:
            gt = r.GetGeoTransform()
        else:
            assert gt == r.GetGeoTransform()
        
        # Save or check the shape
        if not shape:
            shape = ar.shape
        else:
            assert shape == ar.shape
            
        # Save or check the geotransform
        if not srs:
            srs = r.GetProjection()
        else:
            assert srs == r.GetProjection()

        # Flatten in one dimension
        arf = ar.flatten()
        explanatory_raster_arrays.append(arf)

        # TODO scale

    expl = np.array(explanatory_raster_arrays).T
    raster_info = {
        'gt': gt,
        'shape': shape,
        'srs': srs
    }
    return expl, raster_info


def impute(target_xs, clf, raster_info, outdir="output",
           linechunk=1000, class_prob=True, certainty=True):
    """
    Parameters
    ----------
    target_xs: Array of explanatory variables for which to predict responses
    clf: instance of a scikit-learn Classifier
    raster_info: dictionary of raster attributes with key 'gt', 'shape' and 'srs'

    Options
    -------
    outdir : output directory
    linechunk : number of lines to process per pass; reduce only if memory is constrained
    class_prob : Boolean. Should we create a probability raster for each class?
    certainty : Boolean. Should we produce a raster of overall classification certainty? 
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    driver = gdal.GetDriverByName('HFA')
    gt = raster_info['gt']
    shape = raster_info['shape']
    srs = raster_info['srs']

    ## Create a new raster for responses
    outds_response = driver.Create(os.path.join(outdir, "responses.img"),
                                   shape[1], shape[0], 1, gdal.GDT_UInt16)
    outds_response.SetGeoTransform(gt)
    outds_response.SetProjection(srs)
    outband_response = outds_response.GetRasterBand(1)

    ## Create a new raster for certainty
    ## We interpret certainty to be the max probability across classes
    if certainty:
        outds_certainty = driver.Create(os.path.join(outdir, "certainty.img"), 
                                        shape[1], shape[0], 1, gdal.GDT_Float32)
        outds_certainty.SetGeoTransform(gt)
        outds_certainty.SetProjection(srs)
        outband_certainty = outds_certainty.GetRasterBand(1)

    ## Create a new rasters for probability of each class
    if class_prob:
        classes = list(clf.classes_)
        # classes.index(70)  # find the idx of class 70
        outdss_classprob = []
        outbands_classprob = []
        for i, c in enumerate(classes):
            ods = driver.Create(os.path.join(outdir, "probability_%s.img" % c), 
                                shape[1], shape[0], 1, gdal.GDT_Float32)
            ods.SetGeoTransform(gt)
            ods.SetProjection(srs)
            outdss_classprob.append(ods)
            outbands_classprob.append(ods.GetRasterBand(1))

        class_gdal = zip(outdss_classprob, outbands_classprob)

    if not linechunk:
        linechunk = shape[1]

    chunks = int(math.ceil(shape[0] / float(linechunk)))
    for chunk in range(chunks):
        logger.debug("Writing chunk %d of %d" % (chunk+1, chunks))
        row = chunk * linechunk
        if row + linechunk > shape[0]:
            linechunk = shape[0] - row

        start = shape[1] * row
        end = start + shape[1] * linechunk 
        line = target_xs[start:end,:]

        responses = clf.predict(line)
        responses2D = responses.reshape((linechunk, shape[1]))
        outband_response.WriteArray(responses2D, xoff=0, yoff=row)

        if certainty or class_prob:
            proba = clf.predict_proba(line)

        if certainty:
            certaintymax = proba.max(axis=1)
            certainty2D = certaintymax.reshape((linechunk, shape[1]))
            outband_certainty.WriteArray(certainty2D, xoff=0, yoff=row)
          
        # write out probabilities for each class as a separate raster
        if class_prob:
            for cls_index, ds_band in enumerate(class_gdal):
                proba_class = proba[:, cls_index]
                classcert2D = proba_class.reshape((linechunk, shape[1]))
                ds, band = ds_band
                band.WriteArray(classcert2D, xoff=0, yoff=row)
                ds.FlushCache()

        if certainty:
            outds_certainty.FlushCache()
        outds_response.FlushCache()

    outds_certainty = None
    outds_response = None



def stratified_sample_raster(strata_data, 
                              target_sample_size=30,
                              min_sample_proportion=0.1):
    """
    Parameters
    ----------
    strata_data: Path to raster dataset containing strata to sample from (e.g. zones)

    Returns
    -------
    selected: array of selected indices
    """
    ds = gdal.Open(strata_data)
    if ds is None:
        raise Exception("%s not found" % strata_data)
    strata2D = ds.ReadAsArray()
    strata = strata2D.flatten()
    index_array = np.arange(strata.size)

    # construct a dictionary of lists,
    # keys are stratum ids
    # values are list of indices
    sample = dict([(int(s),[]) for s in np.unique(strata)])
    satisfied = []

    # counts for proportion-based constraints
    bins = np.bincount(strata)
    ii = np.nonzero(bins)[0]
    stratum_count = dict(zip(ii,bins[ii]))

    # shuffle the indices and loop until the sample satisfied our constraints
    np.random.shuffle(index_array)
    for idx in index_array:
        stratum = strata[index_array[idx]]
        if stratum in satisfied:
            continue
        sample[stratum].append(idx)
        nsamples = len(sample[stratum])
        # constraints -> hit the target sample size OR proportion of total 
        # (whichever is highest)
        target = stratum_count[stratum] * min_sample_proportion
        if target < target_sample_size:
            target = target_sample_size
        if nsamples >= target:
            satisfied.append(stratum)
        if len(satisfied) == len(sample.keys()):
            break
    
    # convert sampled indicies into a list of indicies
    selected = []
    for k, v in sample.items():
        # check for stratum with < target sample size
        if len(v) < target_sample_size:
            # if we have too few samples, drop them
            #warnings.warn("Stratum %s has only %d samples, dropped" % (k, len(v)))
            pass
        else:
            selected.extend(v)

    return np.array(selected)
 
