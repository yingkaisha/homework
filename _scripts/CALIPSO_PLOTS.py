'''
This is the script that batches all the downloaded 
CALIPSO L1B 532-nm Total Attenuated Backscatter files and plots the transect 
'''
import numpy as np
import netCDF4 as nc
from pyhdf import SD
from glob import glob
from os.path import basename
import matplotlib.pyplot as plt
from matplotlib.path import Path
# The plotting function
def CALIPSO_plot(z, lon_temp, elev_temp, zT_temp, Beta_log10, key, ind, lev, title):
    # Colors
    land_c = [0.3, 0.3, 0]
    missing_c = 'gray'
    # create 2-D grids for plots
    zgrid, xgrid = np.meshgrid(z, lon_temp)
    for i in range(zgrid.shape[1]):
        zgrid[:, i] += elev_temp
    # figure setup
    fig = plt.figure(figsize=(9.5, 4)); 
    ax = fig.gca(); ax.grid(False)
    ax.set_ylim([-2, 22.5])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    [j.set_linewidth(2.5) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom="off", top="off", \
           labelbottom="on", left="off", right="off", labelleft="on")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitude [degree]', fontsize=14)
    ax.set_ylabel('Height [km]', fontsize=14)
    # If CALIPSO footprint has a breaking point
    ax.patch.set_color(missing_c)
    if np.isnan(ind):
        # If CALIPSO footprint is continuous
        CS = ax.contourf(xgrid, zgrid, Beta_log10, lev, cmap=plt.cm.gist_ncar_r, extend='both')
        #ax.plot(lon_temp, elev_temp, '-', lw=3.0, color=[0.3, 0.3, 0], label='Elevation')
        ax.plot(lon_temp, zT_temp, 'k--', lw=3.0, label='GMAO Tropopause height')
        ax.fill_between(lon_temp, elev_temp, -2, where=elev_temp>=-2, lw=0, \
                        facecolor=land_c, interpolate=True)
    else:
        CS = ax.contourf(xgrid[:ind, :], zgrid[:ind, :], Beta_log10[:ind, :], lev, \
                         cmap=plt.cm.gist_ncar_r, extend='both')
        ax.contourf(xgrid[ind+1:, :], zgrid[ind+1:, :], Beta_log10[ind+1:, :], lev, \
                         cmap=plt.cm.gist_ncar_r, extend='both')
        #ax.plot(lon_temp[:ind], elev_temp[:ind], '-', lw=3.0, color=[0.3, 0.3, 0], label='Elevation')
        ax.plot(lon_temp[ind+1:], elev_temp[ind+1:], '-', lw=3.0, color=[0.3, 0.3, 0])
        ax.plot(lon_temp[:ind], zT_temp[:ind], 'k--', lw=3.0, label='GMAO Tropopause height')
        ax.plot(lon_temp[ind+1:], zT_temp[ind+1:], 'k--', lw=3.0)
        ax.fill_between(lon_temp[:ind], elev_temp[:ind], -2, where=elev_temp[:ind]>=-2, lw=0, \
                    facecolor=land_c, interpolate=True)
        ax.fill_between(lon_temp[ind+1:], elev_temp[ind+1:], -2, where=elev_temp[ind+1]>=-2, lw=0, \
                    facecolor=land_c, interpolate=True)
        
    #LG = ax.legend(bbox_to_anchor=(1, 0.9), prop={'size':14}); LG.draw_frame(False)
    cax = fig.add_axes([0.94, 0.1, 0.0225, 0.7])
    CBar = fig.colorbar(CS, cax=cax, orientation='vertical')
    CBar.ax.tick_params(axis='y', length=0, direction='in', labelsize=14)
    CBar.ax.xaxis.set_label_position('top')
    CBar.set_label(r"[$\log_{10}\ \beta'_{532}$]", y=1.125, labelpad=-40, rotation=360, fontsize=14)
    fig.savefig(key+'.png', dpi=250, orientation='portrait', papertype='a4', format='png',
               bbox_inches='tight', pad_inches=0)
# Get CALIPSO filenames
CALIPSO_path = '../_data/ATSC-500/BACKUP_CALIPSO/Daytime/CAL_LID_L1-Standard-V4-10*'
CALIPSO_daytime = glob(CALIPSO_path)
# Create vertical bins [km]
z1 = np.linspace(40, 30.1, 33-1+1)
z2 = np.linspace(30.1, 20.2, 88-34+1)
z3 = np.linspace(20.2, 8.3, 288-89+1)
z4 = np.linspace(8.3, -0.5, 578-289+1)
z5 = np.linspace(-0.5, -2, 583-579+1)
z = np.hstack([z1, z2, z3, z4, z5])
# Import BC boundary
TEMP_obj = np.load('BC_Boundary.npy', encoding='latin1')
BC_lon = TEMP_obj[()]['BC_lon']
BC_lat = TEMP_obj[()]['BC_lat']
BC_polygon = Path(list(zip(BC_lon, BC_lat)))
# "Read file --> Plot --> close" routine
lev = np.arange(-4.5, 1.0, 0.25)
for num, name in enumerate(CALIPSO_daytime):
    print("Processing "+name)
    key = basename(name)[26:-4]
    # read HDF4
    hdf4_obj = SD.SD(name, SD.SDC.READ)
    y = hdf4_obj.select('Latitude')[:, 0]
    x = hdf4_obj.select('Longitude')[:, 0]
    # indices inside the domain
    inds = BC_polygon.contains_points(list(zip(x, y)))
    lon_temp = x[inds]; lat_temp = y[inds]
    # detect breaking points
    if np.abs(np.diff(lon_temp)).max() > 10*(np.abs(lon_temp[1] - lon_temp[0])):
        ind = np.argmin(np.diff(lon_temp))
    else:
        ind = np.nan
    # import variables
    zT_temp = hdf4_obj.select('Tropopause_Height')[:, 0][inds]
    elev_temp = hdf4_obj.select('Surface_Elevation')[:, 0][inds]
    Beta_temp = hdf4_obj.select('Total_Attenuated_Backscatter_532')[:][inds, :]
    hdf4_obj.end()
    # post-processing Beta
    Beta_temp[Beta_temp<0] = np.nan
    Beta_log10 = np.log10(Beta_temp)
    # Plotting
    title = ('UTC Time: '+key+' | File index: {}').format(num)
    CALIPSO_plot(z, lon_temp, elev_temp, zT_temp, Beta_log10, key, ind, lev, title)

