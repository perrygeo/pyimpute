from __future__ import print_function
import sys
import os
from pyimpute import load_training_vector, evaluate_clf
from sklearn.ensemble import ExtraTreesClassifier
import json
import numpy as np

import logging
logger = logging.getLogger('pyimpute')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(sh)


TRAINING_DIR = "./_usfs_data"


def main():
    """
    Main function.

    Args:
    """

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

    evaluate_clf(clf, train_xs, train_y, feature_names=explanatory_fields)

if __name__ == '__main__':
    main()
