from impute import load_training, load_training_from_rasters, load_targets, impute
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.filterwarnings('ignore')

###############################################################################
# Load the known data points or "training" data
# x ~ the explanatory variables which are full coverage/inexpensive
# y ~ the response variables which are sparse/expensive/impossible to collect
print "Loading training data by random stratified sampling"

explanatory_rasters = {
    'GT_DEM': 'data/gt_dem.img',
    'PMEAN_ALL': 'data/pmean_all.img',
    'TMAX8': 'data/tmax8.img',
    'TMEAN_ALL': 'data/tmean_all.img',
    'DTMEAN8_12': 'data/dtmean8_12.img',
    'P_N': 'data/p_n.img',
    'INT_CNL_EUC': 'data/int_cnl_euc.img'
}
explanatory_fields = explanatory_rasters.keys()
response_raster = 'data/responses.tif'

from sampling import raster_stratified_sample
selected = raster_stratified_sample(response_raster)
# TODO: May want to cache this so you can work with a consistent training set

train_xs, train_y = load_training_from_rasters(
    selected, response_raster, explanatory_rasters)

###############################################################################
# Set up classifier
print "Training classifier"
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(train_xs, train_y)  # fit the classifier to the training data

###############################################################################
# Assess predictive accuracy
from sklearn import cross_validation
scores = cross_validation.cross_val_score(rf, train_xs, train_y, cv=2)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean() * 100,
                                         scores.std() * 200))

###############################################################################
# Load the target/explanatory raster data 
# will be used to predict resposes
print "Loading explanatory raster data"
target_xs, gt, shape = load_targets( explanatory_rasters, explanatory_fields)

###############################################################################
# Impute response rasters
# default to standard naming convention for outputs
# data gets dumped to an output directory
print "Imputing response rasters"
impute(target_xs, rf, gt, shape, outdir="out1", linechunk=None)
