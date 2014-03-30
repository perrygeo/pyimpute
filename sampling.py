from osgeo import gdal
import numpy as np
import warnings


def raster_stratified_sample(strata_data, 
                              target_sample_size=30,
                              min_sample_proportion=0.1):
    """
    in: raster dataset containing strata to sample from (e.g. zones)
    out: array of selected indices
    """
    ds = gdal.Open(strata_data)
    if ds is None:
        raise Exception("%s not found" % strata_data)
    strata2D = ds.ReadAsArray()
    strata = strata2D.flatten()
    index_array = np.arange(strata.size)

    # construct a dictionary of lists,
    # keys are stratum ids
    # values are list of indices
    sample = dict([(int(s),[]) for s in np.unique(strata)])
    satisfied = []

    # counts for proportion-based constraints
    bins = np.bincount(strata)
    ii = np.nonzero(bins)[0]
    stratum_count = dict(zip(ii,bins[ii]))

    # shuffle the indices and loop until the sample satisfied our constraints
    np.random.shuffle(index_array)
    for idx in index_array:
        stratum = strata[index_array[idx]]
        if stratum in satisfied:
            continue
        sample[stratum].append(idx)
        nsamples = len(sample[stratum])
        # constraints -> hit the target sample size OR proportion of total 
        # (whichever is highest)
        target = stratum_count[stratum] * min_sample_proportion
        if target < target_sample_size:
            target = target_sample_size
        if nsamples >= target:
            satisfied.append(stratum)
        if len(satisfied) == len(sample.keys()):
            break
    
    # convert sampled indicies into a list of indicies
    selected = []
    for k, v in sample.items():
        # check for stratum with < target sample size
        # print k, len(v)
        if len(v) < target_sample_size:
            # if we have too few samples, drop them
            #warnings.warn("Stratum %s has only %d samples, dropped" % (k, len(v)))
            pass
        else:
            selected.extend(v)

    return np.array(selected)




if __name__ == "__main__":
    response_raster = "data/responses.tif"
    selected = raster_stratified_sample(response_raster)
    print selected


