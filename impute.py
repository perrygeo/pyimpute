import pandas as pd
import numpy as np
import os
import math
from osgeo import gdal


def load_training(infile, response_field, explanatory_fields):
    data = pd.read_csv(infile)
    train_y = np.asarray(data[response_field])
    train_xs = np.float32(np.asarray(data[explanatory_fields]))

    # todo scale
    return train_xs, train_y

def load_training_from_rasters(selected, response_raster, 
                               explanatory_rasters, explanatory_fields):
    ds = gdal.Open(response_raster)
    if ds is None:
        raise Exception("%s not found" % strata_data)
    response_data = ds.ReadAsArray().flatten()
    train_y = response_data[selected]

    selected_data = []
    for var in explanatory_fields:
        rast = explanatory_rasters[var]
        ds = gdal.Open(rast)
        if ds is None:
            raise Exception("%s not found" % strata_data)
        explanatory_data = ds.ReadAsArray().flatten()
        assert explanatory_data.size == response_data.size
        selected_data.append(explanatory_data[selected])

    train_xs = np.asarray(selected_data).T
    return train_xs, train_y

def load_targets(targets, explanatory_fields):
    explanatory_raster_arrays = []
    gt = None
    shape = None
    srs = None
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
            
        # Save or check the geotransform
        if not srs:
            srs = r.GetProjection()
        else:
            assert srs == r.GetProjection()

        # Flatten in one dimension
        arf = ar.flatten()
        explanatory_raster_arrays.append(arf)

        # TODO scale

    expl = np.array(explanatory_raster_arrays).T
    raster_info = {
        'gt': gt,
        'shape': shape,
        'srs': srs
    }
    return expl, raster_info


def impute(target_xs, rf, raster_info, outdir="output",
           linechunk=1000, class_prob=True, certainty=True):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    driver = gdal.GetDriverByName('GTiff')
    gt = raster_info['gt']
    shape = raster_info['shape']
    srs = raster_info['srs']

    ## Create a new raster for responses
    outds_response = driver.Create(os.path.join(outdir, "responses.tif"),
                                   shape[1], shape[0], 1, gdal.GDT_UInt16)
    outds_response.SetGeoTransform(gt)
    outds_response.SetProjection(srs)
    outband_response = outds_response.GetRasterBand(1)

    ## Create a new raster for certainty
    ## We interpret certainty to be the max probability across classes
    if certainty:
        outds_certainty = driver.Create(os.path.join(outdir, "certainty.tif"), 
                                        shape[1], shape[0], 1, gdal.GDT_Float32)
        outds_certainty.SetGeoTransform(gt)
        outds_certainty.SetProjection(srs)
        outband_certainty = outds_certainty.GetRasterBand(1)

    ## Create a new rasters for probability of each class
    if class_prob:
        classes = list(rf.classes_)
        # classes.index(70)  # find the idx of class 70
        outdss_classprob = []
        outbands_classprob = []
        for i, c in enumerate(classes):
            ods = driver.Create(os.path.join(outdir, "probability_%s.tif" % c), 
                                shape[1], shape[0], 1, gdal.GDT_Float32)
            ods.SetGeoTransform(gt)
            ods.SetProjection(srs)
            outdss_classprob.append(ods)
            outbands_classprob.append(ods.GetRasterBand(1))

        class_gdal = zip(outdss_classprob, outbands_classprob)

    if not linechunk:
        linechunk = shape[1]

    chunks = int(math.ceil(shape[0] / float(linechunk)))
    for chunk in range(chunks):
        #print "Writing chunk %d of %d" % (chunk+1, chunks)
        row = chunk * linechunk
        if row + linechunk > shape[0]:
            linechunk = shape[0] - row

        start = shape[1] * row
        end = start + shape[1] * linechunk 
        line = target_xs[start:end,:]

        responses = rf.predict(line)
        responses2D = responses.reshape((linechunk, shape[1]))
        outband_response.WriteArray(responses2D, xoff=0, yoff=row)

        if certainty or class_prob:
            proba = rf.predict_proba(line)

        if certainty:
            certaintymax = proba.max(axis=1)
            certainty2D = certaintymax.reshape((linechunk, shape[1]))
            outband_certainty.WriteArray(certainty2D, xoff=0, yoff=row)
          
        # write out probabilities for each class as a separate raster
        if class_prob:
            for cls_index, ds_band in enumerate(class_gdal):
                proba_class = proba[:, cls_index]
                classcert2D = proba_class.reshape((linechunk, shape[1]))
                ds, band = ds_band
                band.WriteArray(classcert2D, xoff=0, yoff=row)
                ds.FlushCache()

        if certainty:
            outds_certainty.FlushCache()
        outds_response.FlushCache()

    outds_certainty = None
    outds_response = None
