## Python module for geospatial prediction using scikit-learn and GDAL

*Impute*: Estimating missing data by inference.

The naming and core concept is based on the [yaImpute](http://cran.r-project.org/web/packages/yaImpute/index.html) R Package.


### Overview

*Imputation* and *Geospatial prediction* are broad terms for techniques aimed at estimating spatially-explicity characteristics of the landscape based on sparse observations. 

The observations, known as the **training data**, contain:

* **explanatory** variables: relatively inexpensive to measure or predict; explain the spatial patterns of response variables 
* **response** variables: relatively expensive or impossible to obtain

The **target data** contains *only* explanatory variables represented by full coverage raster data. There are no response variables available for the target data.

The ultimate goal is to predict spatially-explicit responses based on the target data. It is very similar to the concept of "supervised classification" in remote sensing but can also be used to predict continuious variables (i.e. regression)

### Installation

Install some prerequisites (unless you want to have pip compile them for you)...
```
python setup.py python-numpy python-gdal python-pandas 
```

and go...
```
pip install pyimpute
```

Then check out the [examples](https://github.com/perrygeo/python-impute/blob/master/examples/).

### The goal of pyimpute

`pyimpute` helps optimize and streamline the process of predicting spatial data through supervised classification and regression. 

`pyimpute` doesn't do much other than provide some powerful, high-level utility functions 
that leverage the raster data tools of GDAL and the machine learning algorithms of scikit-learn

Having a clean workflow for spatial predictions will allow us to:
* frequently update predictions with new information (e.g. new Landsat imagery as it becomes available)
* explore new variables more easily
* bring the technique to other disciplines and geographies

### The process

1. Loading spatial data
	* Easiest method: import table containing training observations with both explanatory and response variables.  
	* Alternate method: perform random stratified sampling on a response data
	and generate training data from rasters
	* Another alternate method: Use python-rasters-stats to grab the data from explanatory rasters using point data

2. Fit a classification or regression model
	* Leverage scikit-learn classifiers or regressors
	* Fit to training data
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
  
3. Generate spatial prediction from target data
	* scikit classifiers to make predictions and generate certainty estimates
	* GDAL to write predicted classes array to new raster
	* write raster of prediction probability for each pixel
	* write rasters (one for each class) with probability of that class over space



### Some resources

[1] http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers?rq=1

[2] http://stackoverflow.com/questions/21506128/best-way-to-combine-probabilistic-classifiers-in-scikit-learn/21544196#21544196

[3] http://scikit-learn.org/stable/modules/model_evaluation.html#prediction-error-metrics

[4] http://scikit-learn.org/stable/auto_examples/plot_classification_probability.html

### Example

Please check out [the examples](https://github.com/perrygeo/python-impute/blob/master/examples/)

This example walks through the main steps of loading training data, setting up and evaluating a classifier, and using it to predict a raster of the response variable.

The resulting prediction raster

![alt tag](https://raw.github.com/perrygeo/python-impute/master/img/example_responses.png)

The certainty estimates

![alt tag](https://raw.github.com/perrygeo/python-impute/master/img/example_certainty.png)


#### Note about performance and memory limitations
Depending on the classifier you use, memory and/or time might become limited.

The `impute` method takes an optional argument, (`linechunk`) which calibrates the performance. 
Specifically, it determines how many lines/rows of the raster file are processed at once. 

**tl;dr;** You want to set `linechunk` as high as possible without exceeding your memory capacity.

In this example, I use the RandomForest classifier. Other classifiers may exhibit different behavior
but, in general, there is a tradeoff between speed and memory;

as you increase `linechunk` memory increases *linearly*

![alt tag](https://raw.github.com/perrygeo/python-impute/master/img/memory.png)

while performance increases *exponentially*. 

![alt tag](https://raw.github.com/perrygeo/python-impute/master/img/time.png)



#### Note about geostatistics
While kriging and other geostatistical techniques are technically "geospatial prediction", they rely on spatial dependence between observations. The problems for which this module was built are landscape scale 
and rarely suited to such approaches. There is great potential to meld the two approaches (i.e. consider spatial autocorrelation between training data as well as explanatory variables) but this is currently outside the scope of this module.

