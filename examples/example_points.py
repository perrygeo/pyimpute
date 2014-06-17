from __future__ import print_function
import sys
import os
from pyimpute import load_training_vector, load_targets, impute
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
import json
import numpy as np

import logging
logger = logging.getLogger('pyimpute')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(sh)


TRAINING_DIR = "./_usfs_data"


def main():

    # Define the known data points or "training" data
    explanatory_fields = "d100 dd0 dd5 fday ffp gsdd5 gsp map mat_tenths mmax_tenths mmindd0 mmin_tenths mtcm_tenths mtwm_tenths sday".split()
    explanatory_rasters = [os.path.join(TRAINING_DIR, "current_" + r + ".img") for r in explanatory_fields]
    response_shapes = os.path.join(TRAINING_DIR, "DF.shp")

    # Load the training rasters using the sampled subset
    try:
        cached = json.load(open("_cached_training.json"))
        train_xs = np.array(cached['train_xs'])
        train_y = np.array(cached['train_y'])
    except IOError:
        train_xs, train_y = load_training_vector(response_shapes, 
            explanatory_rasters, response_field='GRIDCODE')
        cache = {'train_xs': train_xs.tolist(), 'train_y': train_y.tolist()}
        with open("_cached_training.json", 'w') as fh:
            fh.write(json.dumps(cache))

    print(train_xs.shape, train_y.shape)

    # Train the classifier
    clf = ExtraTreesClassifier(n_estimators=120, n_jobs=3)
    clf.fit(train_xs, train_y)
    print(clf)

    # Cross validate
    k = 5
    scores = cross_validation.cross_val_score(clf, train_xs, train_y, cv=k)
    print("%d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (k, scores.mean() * 100, scores.std() * 200))

    # Run the model on the current data; i.e. predict current conditions
    print("Imputing response rasters FOR CURRENT DATA")
    target_xs, raster_info = load_targets(explanatory_rasters)

    impute(target_xs, clf, raster_info, outdir="_usfs_output_current",
           linechunk=400, class_prob=True, certainty=True)

    years = ['2060']
    for year in years:
        print("Loading target explanatory raster data, swapping out for %s climate data" % year)

        # Swap out for future climate rasters
        new_explanatory_rasters = [os.path.join(TRAINING_DIR, "Ensemble_rcp60_y%s_%s.img" % (year, r)) 
                                    for r in explanatory_fields]

        target_xs, raster_info = load_targets(new_explanatory_rasters)

        print("Imputing response rasters")
        impute(target_xs, clf, raster_info, outdir="_usfs_output_%s" % year,
               linechunk=400, class_prob=True, certainty=True)

if __name__ == '__main__':
    main()
