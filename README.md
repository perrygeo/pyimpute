## Spatial Imputation for Python

*This is just a placeholder for a project that is hopefully coming soon.*
*For now, it's mostly ideas and very little code*

### Overview

TODO use imputation as a more general term instead of GNN

[Gradient Nearest Neighbor](http://www.forestencyclopedia.net/p/p3453) mapping is
a technique for estimating detailed characteristics of the landscape based on 
sparse point observations. The observations are imputed to the landscape based on
their correspondence with several underlying explanatory variables for which we
have full coverage raster data. [This poster](http://www.fsl.orst.edu/clams/download/posters/gnn_scaling.pdf)
does a good job of visualizing the process.

### The idea

Build a python library to streamline the creation of GNN maps in order to

* frequently update with new information (e.g. new Landsat imagery as it becomes available)
* explore new variables more easily
* bring the technique to other disciplines and geographies (has been applied primarily to pacific northwest forestry)

There are a number of existing tools that could be leveraged to support this workflow:

### Loading spatial data
* User inputs explanatory rasters, observation vector dataset, response columns
* Use python-rasters-stats to grab the data from explanatory rasters
* Use pandas/numpy to construct arrays: explanatory variables vs. obs and response variable vs. observations

#### Calibrate a classification model

* All done using scikit-learn classifiers
* optional: scale and optionally reduce dimensionality of data
* optional: calibrate using grid search cv to find optimal parameters
* optional: create your own ensemble [2]
* Evaluate:
    * crossvalidation (average score over k-folds)
    * train_test split
    * metrics  [3]
    * confusion matrix
    * compare to dummy estimators
    * identify most informative features [1]
  
#### Fit model and generate spatial prediction
* Fit and predict using scikit learn
* GDAL to write predicted classes array to new raster
* write raster of prediction probability for each pixel
* write rasters (one for each class) with probability of that class over space



### Some resources

[1] http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers?rq=1

[2] http://stackoverflow.com/questions/21506128/best-way-to-combine-probabilistic-classifiers-in-scikit-learn/21544196#21544196

[3] http://scikit-learn.org/stable/modules/model_evaluation.html#prediction-error-metrics

[4] http://scikit-learn.org/stable/auto_examples/plot_classification_probability.html

### Example

Here we take a table of known data points (*training data*) and use it to train a classifier
and predict a raster surface of the response variable.

```python
from impute import load_training, load_targets, impute
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.filterwarnings('ignore')

#------------------ Load the known data points or "training" data
# x ~ the explanatory variables which are full coverage/inexpensive
# y ~ the response variables which are sparse/expensive/impossible to collect
print "Loading training data"
train_xs, train_y, explanatory_fields = load_training(
    # A csv containing point observations with known responses
    'data/zone_sample.csv',
    # the column name holding the response; should be integer
    "ISO_AG11_15",  
    # the column names holding explanatory variables
    ['GT_DEM', 'PMEAN_ALL', 'TMAX8', 'TMEAN_ALL', 'DTMEAN8_12', 'P_N', 'INT_CNL_EUC'])

#------------------ Set up classifier
print "Training classifier"
rf = RandomForestClassifier(n_estimators=10, n_jobs=3)
rf.fit(train_xs, train_y)  # fit the classifier to the training data

#------------------ Assess predictive accuracy
from sklearn import cross_validation
scores = cross_validation.cross_val_score(rf, train_xs, train_y, cv=2)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 200))

#------------------ Load the target/explanatory raster data 
#------------------ will be used to predict resposes
print "Loading explanatory raster data"
target_xs, gt, shape = load_targets({  
    # one for each explanatory field
    'GT_DEM': 'data/gt_dem.img',
    'PMEAN_ALL': 'data/pmean_all.img',
    'TMAX8': 'data/tmax8.img',
    'TMEAN_ALL': 'data/tmean_all.img',
    'DTMEAN8_12': 'data/dtmean8_12.img',
    'P_N': 'data/p_n.img',
    'INT_CNL_EUC': 'data/int_cnl_euc.img'}, explanatory_fields)

#------------------ Impute response rasters
#------------------ default to standard naming convention for outputs
#------------------ data gets dumped to an output directory
print "Imputing responses; check ./out1/*.tif"
impute(target_xs, rf, gt, shape, outdir="out1")
```

The example output would be something like...
```
Loading training data
Training classifier
	Accuracy: 84.74 (+/- 2.48)
Loading explanatory raster data
Imputing responses; check ./out1/*.tif
```

And the resulting predicted zones might look like...
![alt tag](https://raw.github.com/perrygeo/python-impute/master/img/example_responses.png)

While the certainty estimates might look like...
![alt tag](https://raw.github.com/perrygeo/python-impute/master/img/example_certainty.png)

