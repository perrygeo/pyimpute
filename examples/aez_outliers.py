import sys
import json
import numpy as np
sys.path.append('/usr/local/src/geopredict')

from impute import load_training, load_training_from_rasters, load_targets, impute
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

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
                                    min_sample_proportion=0.05)
    with open(sfile, 'w') as fh:
        fh.write(json.dumps(list(selected)))
               
print len(selected), "samples"

###############################################################################
# Set up classifier

from sklearn.externals import joblib
pfile = "cache/cache_classifier.pkl"
try:
    print "Loading classifier;"
    clf = joblib.load(pfile)
    print "\tUsing cached @ %s" % pfile
except:
    print "\ttraining classifier..."
    train_xs, train_y = load_training_from_rasters(
        selected, response_raster, explanatory_rasters, explanatory_fields)

    clf = svm.OneClassSVM(kernel="rbf")
    clf.fit(train_xs)  # only Xs; just looking for outliers in explanatory data
    joblib.dump(clf, pfile)

###############################################################################
# Load the target/explanatory raster data 
rcps = ["RCP45", "RCP85"] 
#rcps = ["RCP45"] 
years = ["2030s", "2050s", "2070s", "2080s"]
#years = ["2080s"]

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
        import numpy as np
        target_xs = target_xs[np.random.uniform(0, 1, len(target_xs)) <= .1]
        res = clf.predict(target_xs)
        print rcp, year, len(res[res != -1]), len(res[res == -1])
