'''
Functions for ATSC-500 Final Proj - YS
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
# verification stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def mean_bias(A, B):
    return np.mean(np.abs(A-B))

def cal_verif_stats(A, B):
    # verification stats
    R, _ = pearsonr(A, B)
    RMSE = np.sqrt(mean_squared_error(A, B))
    MAE  = mean_absolute_error(A, B)
    return R, MAE, RMSE

def height_lon_grid(z, lon, elev):
    '''
    Create 2-D grid from height and longitude 1-D arrays
    '''
    zgrid, xgrid = np.meshgrid(z, lon)
    for i in range(zgrid.shape[1]):
        zgrid[:, i] += elev
    return zgrid, xgrid

def get_wrf_dir(wrfdir, CALIPSO_dt):
    '''
    String operations to get WRF filename from a given datetime variable
    '''
    if CALIPSO_dt.minute >= 30:
        CALIPSO_dt = CALIPSO_dt + timedelta(hours=1)
    return wrfdir + (CALIPSO_dt - timedelta(days=1)).strftime("%y%m%d00/")+\
            CALIPSO_dt.strftime("wrfout_d02_%Y-%m-%d_%H:00:00")
    
def grid_transfer(raw_x, raw_y, raw_data, nav_lon, nav_lat, method='cubic'):
    LatLonPair=(raw_x.flatten(), raw_y.flatten())
    data_interp = griddata(LatLonPair, raw_data.flatten(), (nav_lon, nav_lat), method=method)
    return data_interp

def CALIPSO_to_datetime(CALIPSO_time):
    '''
    Convert CALIPSO time fraction variable (e.g. Profile_UTC_Time) to
    Python datrtime object
    '''
    date, frac = divmod(CALIPSO_time, 1)
    date = str(date); seconds = 86400*frac
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return datetime(int('20'+date[0:2]), int(date[2:4]), int(date[4:6]), int(h), int(m), int(s))

def CBL_Retrieval(std_temp, z, elev, ratio_thres=10, val_thres=0.5):
    '''
    Retreving CBL height through maximum variance method 
    '''
    CBL_H = np.zeros(len(std_temp))*np.nan
    for i in range(len(std_temp)):
        temp = std_temp[i, :]
        L = len(temp); top = -1*int(0.01*L)
        top_ind = np.argpartition(temp, top)[top:]
        top_flag = np.in1d(range(len(temp)), top_ind)
        ratio = np.nanmean(temp[top_flag])/np.nanmean(temp[~top_flag])
        if ratio > ratio_thres and np.nanmax(temp) > val_thres:
            CBL_H[i] = z[np.nanargmax(temp)]+elev[i]
        else:
            CBL_H[i] = np.nan
    return CBL_H

def CALIPSO_plot(z, lon_temp, elev_temp, Beta_log10, ind, lev, title):
    '''
    Plot the log10(TAB532) of a given CALIPSO footprint
    '''
    # Colors
    land_c = [0.3, 0.3, 0]
    missing_c = 'gray'
    zgrid, xgrid = height_lon_grid(z, lon_temp, elev_temp)
    # figure setup
    fig = plt.figure(figsize=(9.5, 4)); 
    ax = fig.gca(); ax.grid(False)
    ax.set_ylim([-2, 15])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    [j.set_linewidth(2.5) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitude [degree]', fontsize=14)
    ax.set_ylabel('Height [km]', fontsize=14)
    # If CALIPSO footprint has a breaking point
    ax.patch.set_color(missing_c)
    if np.isnan(ind):
        # If CALIPSO footprint is continuous
        CS = ax.contourf(xgrid, zgrid, Beta_log10, lev, cmap=plt.cm.gist_ncar_r, extend='both')
        #ax.plot(lon_temp, zT_temp, 'k--', lw=3.0, label='GMAO Tropopause height')
        ax.fill_between(lon_temp, elev_temp, -2, where=elev_temp>=-2, lw=0.0, facecolor=land_c, interpolate=True)
    else:
        CS = ax.contourf(xgrid[:ind, :], zgrid[:ind, :], Beta_log10[:ind, :], lev, cmap=plt.cm.gist_ncar_r, extend='both')
        ax.contourf(xgrid[ind+1:, :], zgrid[ind+1:, :], Beta_log10[ind+1:, :], lev, cmap=plt.cm.gist_ncar_r, extend='both')
        #ax.plot(lon_temp[:ind], zT_temp[:ind], 'k--', lw=3.0, label='GMAO Tropopause height')
        #ax.plot(lon_temp[ind+1:], zT_temp[ind+1:], 'k--', lw=3.0)
        ax.fill_between(lon_temp[:ind], elev_temp[:ind], -2, where=elev_temp[:ind]>=-2, lw=0.0, facecolor=land_c, interpolate=True)
        ax.fill_between(lon_temp[ind+1:], elev_temp[ind+1:], -2, where=elev_temp[ind+1]>=-2, lw=0.0, facecolor=land_c, interpolate=True)
    cax = fig.add_axes([0.94, 0.1, 0.0225, 0.7])
    CBar = fig.colorbar(CS, cax=cax, orientation='vertical')
    CBar.ax.tick_params(axis='y', length=0, direction='in', labelsize=14)
    CBar.ax.xaxis.set_label_position('top')
    CBar.set_label(r"[$\log_{10}\ \beta'_{532}$]", y=1.125, labelpad=-40, rotation=360, fontsize=14)
    return fig, ax

