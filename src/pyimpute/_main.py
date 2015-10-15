from __future__ import print_function
import rasterio
import numpy as np
import os
import math
import logging
from sklearn import metrics
from sklearn import cross_validation
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
    train_y : 1xN array of known responses
    """
    from rasterstats import zonal_stats
    all_means = []
    all_zones = None

    for i, raster in enumerate(explanatory_rasters):
        logger.debug("Rasters stats on %s" % raster)

        stats = zonal_stats(response_shapes, raster, stats=metric, prefix="pyimpute_", geojson_out=True)

        zones = [x['properties']['ZONE'] for x in stats]
        if all_zones:
            assert zones == all_zones
        else:
            all_zones = zones

        means = [x['properties']['pyimpute_' + metric] for x in stats]
        all_means.append(means)

    train_y = np.array(all_zones)
    train_xs = np.array(all_means).T

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

    with rasterio.open(response_raster) as src:
        response_data = src.read().flatten()

    if selected is None:
        train_y = response_data
    else:
        train_y = response_data[selected]

    selected_data = []
    for rast in explanatory_rasters:
        with rasterio.open(rast) as src:
            explanatory_data = src.read().flatten()
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
    aff = None
    shape = None
    crs = None

    for raster in explanatory_rasters:
        logger.debug(raster)
        with rasterio.open(raster) as src:
            ar = src.read(1)  # TODO band num? 

            # Save or check the geotransform
            if not aff:
                aff = src.affine
            else:
                assert aff == src.affine

            # Save or check the shape
            if not shape:
                shape = ar.shape
            else:
                assert shape == ar.shape

            # Save or check the geotransform
            if not crs:
                crs = src.crs
            else:
                assert crs == src.crs

        # Flatten in one dimension
        arf = ar.flatten()
        explanatory_raster_arrays.append(arf)

    expl = np.array(explanatory_raster_arrays).T

    raster_info = {
        'affine': aff,
        'shape': shape,
        'crs': crs
    }
    return expl, raster_info


def impute(target_xs, clf, raster_info, outdir="output", linechunk=1000, class_prob=True, certainty=True):
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

    shape = raster_info['shape']

    profile = {
        'affine': raster_info['affine'],
        'blockxsize': shape[1],
        'height': shape[0],
        'blockysize': 1,
        'count': 1,
        'crs': raster_info['crs'],
        'driver': u'GTiff',
        'dtype': 'int16',
        'nodata': -32768,
        'tiled': False,
        'transform': raster_info['affine'].to_gdal(),
        'width': shape[1]}

    try:
        response_path = os.path.join(outdir, "responses.tif")
        response_ds = rasterio.open(response_path, 'w', **profile)

        profile['dtype'] = 'float32'
        if certainty:
            certainty_path = os.path.join(outdir, "certainty.tif")
            certainty_ds = rasterio.open(certainty_path, 'w', **profile)

        class_dss = []
        if class_prob:
            classes = list(clf.classes_)
            class_paths = []
            for i, c in enumerate(classes):
                ods = os.path.join(outdir, "probability_%s.tif" % c)
                class_paths.append(ods)
            for p in class_paths:
                class_dss.append(rasterio.open(p, 'w', **profile))

        # Chunky logic
        if not linechunk:
            linechunk = shape[0]
        chunks = int(math.ceil(shape[0] / float(linechunk)))

        for chunk in range(chunks):
            logger.debug("Writing chunk %d of %d" % (chunk+1, chunks))
            row = chunk * linechunk
            if row + linechunk > shape[0]:
                linechunk = shape[0] - row
            # in 1D space
            start = shape[1] * row
            end = start + shape[1] * linechunk
            line = target_xs[start:end, :]

            window = ((row, row + linechunk), (0, shape[1]))

            # Predict
            responses = clf.predict(line)
            responses2D = responses.reshape((linechunk, shape[1])).astype('int16')
            response_ds.write_band(1, responses2D, window=window)

            if certainty or class_prob:
                proba = clf.predict_proba(line)

            # Certainty
            if certainty:
                certaintymax = proba.max(axis=1)
                certainty2D = certaintymax.reshape((linechunk, shape[1])).astype('float32')
                certainty_ds.write_band(1, certainty2D, window=window)

            # write out probabilities for each class as a separate raster
            for i, class_ds in enumerate(class_dss):
                proba_class = proba[:, i]
                classcert2D = proba_class.reshape((linechunk, shape[1])).astype('float32')
                class_ds.write_band(1, classcert2D, window=window)

    finally:
        response_ds.close()
        if certainty:
            certainty_ds.close()
        for class_ds in class_dss:
            class_ds.close()


def stratified_sample_raster(strata_data, target_sample_size=30, min_sample_proportion=0.1):
    """
    Parameters
    ----------
    strata_data: Path to raster dataset containing strata to sample from (e.g. zones)

    Returns
    -------
    selected: array of selected indices
    """
    with rasterio.open(strata_data) as src:
        strata = src.read().flatten()
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


def evaluate_clf(clf, X, y, k=4, test_size=0.5, scoring="f1", feature_names=None):
    """
    Evalate the classifier on the FULL training dataset
    This takes care of fitting on train/test splits
    """
    X_train, X_test, y_train, y_true = cross_validation.train_test_split(
        X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy Score: %f" % metrics.accuracy_score(y_true, y_pred))
    print()

    print("Classification report")
    print(metrics.classification_report(y_true, y_pred))
    print()

    print("Confussion matrix")
    print(metrics.confusion_matrix(y_true, y_pred))
    print()

    print("Feature importances")
    if not feature_names:
        feature_names = ["%d" % i for i in xrange(X.shape[1])]
    for f, imp in zip(feature_names, clf.feature_importances_):
        print("%20s: %s" % (f, round(imp * 100, 1)))
    print()

    if k:
        print("Cross validation")
        kf = cross_validation.KFold(len(y), n_folds=k)
        scores = cross_validation.cross_val_score(clf, X, y, cv=kf, scoring=scoring)
        print(scores)
        print("%d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (k, scores.mean() * 100, scores.std() * 200))

