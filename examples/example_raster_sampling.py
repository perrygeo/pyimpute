from __future__ import print_function
import sys
import os
from pyimpute import load_training_rasters, load_targets, impute
from pyimpute import stratified_sample_raster
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from collections import OrderedDict

import logging
logger = logging.getLogger('pyimpute')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(sh)


TRAINING_DIR = "./_aez_data/training"


def main():
    # Define the known data points or "training" data
    explanatory_fields = "tmin12c tmax8c p_ph_c pmean_wntrc pmean_sumrc irr_lands gt_demc grwsnc d2u2c".split()
    explanatory_rasters = [os.path.join(TRAINING_DIR, r, "hdr.adf") for r in explanatory_fields]
    response_raster = os.path.join(TRAINING_DIR, 'iso_zns3-27/hdr.adf')

    # Take a random stratified sample
    selected = stratified_sample_raster(response_raster,
        target_sample_size=20, min_sample_proportion=0.01)
  
    # Load the training rasters using the sampled subset
    train_xs, train_y = load_training_rasters(response_raster, 
        explanatory_rasters, selected)
    print(train_xs.shape, train_y.shape)

    # Train the classifier
    clf = ExtraTreesClassifier(n_estimators=10, n_jobs=1)
    clf.fit(train_xs, train_y)
    print(clf)

    # Cross validate
    k = 5
    scores = cross_validation.cross_val_score(clf, train_xs, train_y, cv=k)
    print("%d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (k, scores.mean() * 100, scores.std() * 200))

    # ... Other model assessment

    # Run the model on the current data; i.e. predict itself
    print("Imputing response rasters FOR CURRENT DATA")
    target_xs, raster_info = load_targets(explanatory_rasters)

    impute(target_xs, clf, raster_info, outdir="_aez_output_current",
           linechunk=400, class_prob=True, certainty=True)

    sys.exit()

    years = ['2070s']
    for year in years:
        print("Loading target explanatory raster data, swapping out for %s climate data" % year)

        fdir = os.path.join(TRAINING_DIR, "../RCP85/%s/" % year)

        # swap out datasets that are predicted to change over time (i.e the climate data only)
        climate_rasters = "grwsnc pmean_sumrc pmean_wntrc tmax8c tmin12c".split()
        new_explanatory_rasters = OrderedDict(zip(explanatory_fields, explanatory_rasters))
        for cr in climate_rasters:
            new_explanatory_rasters[cr] = fdir + cr + "/hdr.adf"

        target_xs, raster_info = load_targets(new_explanatory_rasters.values())

        print("Imputing response rasters")
        impute(target_xs, clf, raster_info, outdir="_aez_output_%s" % year,
               linechunk=40, class_prob=True, certainty=True)

if __name__ == '__main__':
    main()
