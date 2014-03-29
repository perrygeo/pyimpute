import pandas as pd
import numpy as np
import os
import math
from osgeo import gdal


def load_training(infile, response_field, explanatory_fields, rasters=None):
    """
    train_x, train_y, class_names, feature_names = load_training(
    'pts.shp',  # A shapefile or csv containing point observations with known responses
        response_field="zone",  # the column name holding the response; should be integer
        explanatory_fields=['precip'],  # the column names holding explanatory variables
        rasters={'dem': 'elev10m.tif'}  # optional. Rasters to derive explanatory variables with zonal stats.
    """
    data = pd.read_csv(infile)
    train_y = np.asarray(data[response_field])
    train_xs = np.float32(np.asarray(data[explanatory_fields]))

    # todo scale
    return train_xs, train_y, explanatory_fields


def load_targets(targets, explanatory_fields):
    explanatory_raster_arrays = []
    gt = None
    shape = None
    for fld in explanatory_fields:
        raster = targets[fld]
        r = gdal.Open(raster)
        if r is None:
            raise Exception("%s not found" % raster)
        ar = r.ReadAsArray()
        
        # Save or check the geotransform
        if not gt:
            gt = r.GetGeoTransform()
        else:
            assert gt == r.GetGeoTransform()
        
        # Save or check the shape
        if not shape:
            shape = ar.shape
        else:
            assert shape == ar.shape
            
        # Flatten in one dimension
        arf = ar.flatten()
        explanatory_raster_arrays.append(arf)

        # TODO scale

    expl = np.array(explanatory_raster_arrays).T
    return expl, gt, shape


def impute(target_xs, rf, gt, shape, outdir="output", linechunk=25):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    driver = gdal.GetDriverByName('GTiff')

    ## Create a new raster for responses
    outds_response = driver.Create(os.path.join(outdir, "responses.tif"),
                                   shape[1], shape[0], 1, gdal.GDT_UInt16)
    outds_response.SetGeoTransform(gt)
    # outds.SetProjection(rzones.GetProjection())
    outband_response = outds_response.GetRasterBand(1)

    ## Create a new raster for certainty
    outds_certainty = driver.Create(os.path.join(outdir, "certainty.tif"), 
                                    shape[1], shape[0], 1, gdal.GDT_Float32)
    outds_certainty.SetGeoTransform(gt)
    # outds.SetProjection(rzones.GetProjection())
    outband_certainty = outds_certainty.GetRasterBand(1)


    # Do it one line at a time to avoid memory contraints on large rasters
    if linechunk:
        chunks = int(math.ceil(shape[0] / float(linechunk)))
        for chunk in range(chunks):
            print "Writing chunk %d of %d" % (chunk+1, chunks)
            row = chunk * linechunk
            if row + linechunk > shape[0]:
                linechunk = shape[0] - row

            start = shape[1] * row
            end = start + shape[1] * linechunk 
            line = target_xs[start:end,:]

            responses = rf.predict(line)
            responses2D = responses.reshape((linechunk, shape[1]))
            outband_response.WriteArray(responses2D, xoff=0, yoff=row)

            proba = rf.predict_proba(line)
            certainty = proba.max(axis=1)
            certainty2D = certainty.reshape((linechunk, shape[1]))
            outband_certainty.WriteArray(certainty2D, xoff=0, yoff=row)

            outds_certainty.FlushCache()
            outds_response.FlushCache()
    else:
        predicted_responses = rf.predict(target_xs)
        responses2D = predicted_responses.reshape(shape)
        outband_response.WriteArray(responses2D)

        proba = rf.predict_proba(target_xs)
        certainty = proba.max(axis=1)
        cert2D = certainty.reshape(shape)
        outband_certainty.WriteArray(cert2D)

    outds_certainty = None
    outds_response = None
