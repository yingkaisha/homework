"""
   bin modis data into regular latitude and longitude bins   
"""

import numpy as np

def reproj_L1B(raw_data, raw_x, raw_y, xlim, ylim, res):
    
    '''
    =========================================================================================
    Reproject MODIS L1B file to a regular grid
    -----------------------------------------------------------------------------------------
    d_array, x_array, y_array, bin_count = reproj_L1B(raw_data, raw_x, raw_y, xlim, ylim, res)
    -----------------------------------------------------------------------------------------
    Input:
            raw_data: L1B data, N*M 2-D array.
            raw_x: longitude info. N*M 2-D array.
            raw_y: latitude info. N*M 2-D array.
            xlim: range of longitude, a list.
            ylim: range of latitude, a list.
            res: resolution, single value.
    Output:
            d_array: L1B reprojected data.
            x_array: reprojected longitude.
            y_array: reprojected latitude.
            bin_count: how many raw data point included in a reprojected grid.
    Note:
            function do not performs well if "res" is larger than the resolution of input data.
            size of "raw_data", "raw_x", "raw_y" must agree.
    =========================================================================================
    '''
    import numpy as np
    
    x_bins=np.arange(xlim[0], xlim[1], res)
    y_bins=np.arange(ylim[0], ylim[1], res)
    x_indices=np.searchsorted(x_bins, raw_x.flat, 'right')
    y_indices=np.searchsorted(y_bins, raw_y.flat, 'right')
        
    y_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    x_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    d_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    bin_count=np.zeros([len(y_bins), len(x_bins)], dtype=np.int)
    
    for n in range(len(y_indices)): #indices
        bin_row=y_indices[n]-1 # '-1' is because we call 'right' in np.searchsorted.
        bin_col=x_indices[n]-1
        bin_count[bin_row, bin_col] += 1
        x_array[bin_row, bin_col] += raw_x.flat[n]
        y_array[bin_row, bin_col] += raw_y.flat[n]
        d_array[bin_row, bin_col] += raw_data.flat[n]
                   
    for i in range(x_array.shape[0]):
        for j in range(x_array.shape[1]):
            if bin_count[i, j] > 0:
                x_array[i, j]=x_array[i, j]/bin_count[i, j]
                y_array[i, j]=y_array[i, j]/bin_count[i, j]
                d_array[i, j]=d_array[i, j]/bin_count[i, j] 
            else:
                d_array[i, j]=np.nan
                x_array[i, j]=np.nan
                y_array[i,j]=np.nan
                
    return d_array, x_array, y_array, bin_count
