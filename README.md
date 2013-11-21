## Gradient Nearest Neighbor Mapping for Python

*This is just a placeholder for a project that is hopefully coming soon.*
*For now, it's mostly ideas and very little code*

### Overview

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

* User inputs explanatory rasters, observation vector dataset, response columns
* Use python-rasters-stats to grab the data from explanatory rasters
* Use pandas to construct data frames: explanatory variables vs. obs and response variable vs. observations
* Use RPy2 to bridge the gap to the vegan R library to run canonical correspondence analysis (*stepwise? test statistical significance? pick number of axes?*)
* Use GDAL to apply cca coefficients to each pixel 
* SciPy K Nearest Neighbor routine to select the nearest observation(s) in gradient space (after adding X & Y)
* GDAL to write to new raster
* R? to perform some sort of cross validation and statistical testing

### Possible API

Example: Impute canopy cover and basal area of douglas fir and red alder across 
the landscape using existing forest plots and explanatory terrain variables. 

    from gnn import gnn_cca, apply_nn, cross_validate
    
    cca_results = gnn_cca(
        ['dem','tpi', 'slope', 'aspect'], 
        'forest_plots.shp', 
        ['DougFirBA', 'RedAlderBA', 'CanopyCover'])

    nn_results = apply_nn(cca_results, 'output.tif', 'log.json')
    cv_results = cross_validate(cca_results)

Or maybe more object oriented like scikit-learn? I dunno, you get the idea though.
