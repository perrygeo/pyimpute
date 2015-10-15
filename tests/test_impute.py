# test zonal stats
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
explanatory_rasters = [os.path.join(DATA, x) for x in
                       "dtmean8_12.img gt_dem.img int_cnl_euc.img pmean_all.img "
                       "p_n.img tmax8.img tmean_all.img".split()]
response_raster = os.path.join(DATA, 'responses.tif')
TMPOUT = "/tmp/pyimpute_test"


def test_impute():
    from pyimpute import load_training_rasters, load_targets, impute

    # Load training data
    train_xs, train_y = load_training_rasters(response_raster, explanatory_rasters)

    # Train a classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, n_jobs=1)
    clf.fit(train_xs, train_y)

    # Load targets
    target_xs, raster_info = load_targets(explanatory_rasters)

    # Go...
    impute(target_xs, clf, raster_info, outdir=TMPOUT,
           linechunk=400, class_prob=True, certainty=True)

    assert os.path.exists(os.path.join(TMPOUT, "responses.tif"))
    assert os.path.exists(os.path.join(TMPOUT, "certainty.tif"))
    assert os.path.exists(os.path.join(TMPOUT, "probability_90.tif"))


def test_load_training_rasters():
    from pyimpute import load_training_rasters
    train_xs, train_y = load_training_rasters(response_raster, explanatory_rasters)
    assert train_xs.shape == (38304, 7)
    assert len(explanatory_rasters) == train_xs.shape[1]
    assert train_y.shape == (38304,)

def test_load_training_vector():
    from pyimpute import load_training_vector
    response_shapes = os.path.join(DATA, "points.geojson")

    train_xs, train_y = load_training_vector(response_shapes,
                                             explanatory_rasters,
                                             response_field='ZONE')

    assert train_xs.shape == (250, 7)
    assert train_y.shape == (250, )


def test_load_targets():
    from pyimpute import load_targets
    target_xs, raster_info = load_targets(explanatory_rasters)
    assert sorted(raster_info.keys()) == ['affine', 'crs', 'shape']
    assert target_xs.shape == (38304, 7)


def test_stratified_sample_raster():
    from pyimpute import load_training_rasters, stratified_sample_raster

    # Take a random stratified sample
    selected = stratified_sample_raster(response_raster,
                                        target_sample_size=20,
                                        min_sample_proportion=0.1)
    assert selected.shape == (3849,)

    train_xs, train_y = load_training_rasters(response_raster, explanatory_rasters, selected)
    assert train_xs.shape == (3849, 7)
    assert train_y.shape == (3849,)
