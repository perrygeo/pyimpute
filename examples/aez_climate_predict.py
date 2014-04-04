import sys
import json
import numpy as np
sys.path.append('/usr/local/src/geopredict')

from impute import load_training, load_training_from_rasters, load_targets, impute
from sklearn.ensemble import RandomForestClassifier

###############################################################################
# Load the known data points or "training" data
# x ~ the explanatory variables which are full coverage/inexpensive
# y ~ the response variables which are sparse/expensive/impossible to collect

tdir = "/storage/moore_aez/Predict/training/"

rs = """tmin12c
tmax8c
p_ph_c
pmean_wntrc
pmean_sumrc
irr_lands
gt_demc
grwsnc
d2u2c""".split()

# leave out """hy200"""

explanatory_rasters = dict([(r, tdir + r + "/hdr.adf") for r in rs])

# explanatory_rasters = {
#     'gt_demc': '....gt_dem/hdr.adf',
#     ...
# }
explanatory_fields = explanatory_rasters.keys()
response_raster = tdir + 'iso_zns3-27/hdr.adf'

# cache this so we can work with a consistent training set
sfile = "cache/selected.json"
try:
    print "Loading training data"
    selected = np.array(json.load(open(sfile)))
    print "\tcached"
except IOError:
    print "\trandom stratified sampling"
    from sampling import raster_stratified_sample
    selected = raster_stratified_sample(response_raster, 
                                    target_sample_size=20,
                                    min_sample_proportion=0.1)
    with open(sfile, 'w') as fh:
        fh.write(json.dumps(list(selected)))
               
print len(selected), "samples"

###############################################################################
# Set up classifier

from sklearn.externals import joblib
pfile = "cache/cache_classifier.pkl"
try:
    print "Loading classifier;"
    rf = joblib.load(pfile)
    print "\tUsing cached @ %s" % pfile
except:
    print "\ttraining classifier..."
    train_xs, train_y = load_training_from_rasters(
        selected, response_raster, explanatory_rasters, explanatory_fields)

    rf = RandomForestClassifier(n_estimators=80, n_jobs=4)
    rf.fit(train_xs, train_y)  # fit the classifier to the training data
    joblib.dump(rf, pfile)

###############################################################################
# Assess predictive accuracy
from sklearn import cross_validation
cvfile = "cache/cross_validation.txt"
try:
    acc = open(cvfile).read()
except IOError:
    cv = 3
    scores = cross_validation.cross_val_score(rf, train_xs, train_y, cv=cv)
    acc = "%d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv, scores.mean() * 100, scores.std() * 200)
    with open(cvfile, 'w') as fh:
        fh.write(acc)
print acc

###############################################################################
# Load the target/explanatory raster data 
# will be used to predict resposes
rcps = ["RCP45", "RCP85"] 
years = ["2030s", "2050s", "2070s", "2080s"]

print "Imputing response rasters FOR CURRENT DATA"
target_xs, raster_info = load_targets(explanatory_rasters, explanatory_fields)
impute(target_xs, rf, raster_info, outdir="out_aezs_CURRENT",
       linechunk=502, class_prob=True, certainty=True)

for rcp in rcps:
    for year in years:
        print "Loading target explanatory raster data, swapping out for %s %s climate data" % (rcp, year)

        fdir = "/storage/moore_aez/Predict/%s/%s/" % (rcp, year)

        climate_rasters = """grwsnc
pmean_sumrc
pmean_wntrc
tmax8c
tmin12c""".split()
        for cr in climate_rasters:
            explanatory_rasters[cr] = fdir + cr + "/hdr.adf"

        target_xs, raster_info = load_targets(explanatory_rasters, explanatory_fields)

        ###############################################################################
        # Impute response rasters
        # default to standard naming convention for outputs
        # data gets dumped to an output directory
        print "Imputing response rasters"
        impute(target_xs, rf, raster_info, outdir="out_aezs_%s_%s" % (rcp, year),
               linechunk=402, class_prob=True, certainty=True)
