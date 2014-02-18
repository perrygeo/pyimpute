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

### Possible API

Not really an API as much as it is a series of utility functions that play nice with scikit-learn

```python
"""
training grids = IVs to fit
training data = raster or vector containing classes
target grids = IVs to use for prediction
output grids = prediction rasters
"""
from impute import load_training, load_targets, cvreport, impute, plot

X, y, class_names, feature_names = load_training('aez_pts.shp', class="aez", fields=['precip'],
                                                 rasters={'dem', 'elev10m.tif'})

rf = RandomForestClassifier()
rf.fit(X, y)

target_X, spatial_info = load_targets({
    'precip': 'precip_2060.tif',
    'dem': 'dem.tif',
)


# default to standard naming convention for outputs
rf = impute(target_X, spatial_info, rf, outdir="test2", class_names, feature_names)

