# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from shapely.geometry.polygon import LinearRing
import cartopy.feature as cfeature
import matplotlib.path as mpath
from cmcrameri import cm
import re
import xarray as xr

# Matplotlib colors
npcols = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
]


MinMax_colors= np.array([
[10,50,120],
[15,75,165],
[30,110,200],
[60,160,240],
[80,180,250],
[130,210,255],
[160,240,255],
[230,255,255],
[255,255,255],
[255,255,255],
[255,250,220],
[255,232,120],
[255,192,60],
[255,160,0],
[255,96,0],
[255,50,0],
[225,20,0],
[200,10,0]])/255
MinMax = ListedColormap(MinMax_colors)
MinMax.set_under([10/300,50/300,120/300])
MinMax.set_over([192/300,0,0])


# Define map projections
data_crs = ccrs.PlateCarree()
proj_Global = ccrs.EqualEarth(central_longitude=200)
proj_North = ccrs.NorthPolarStereo()
proj_South = ccrs.NorthPolarStereo()
proj_Global_PC = ccrs.PlateCarree(central_longitude=210)

from utils.DIRECTORIES import SHAPE_DIRECTORY

def setup_figure(projection,nCols=1,nRows=1,size=(15,15),mask=True):

    circ = False
    if projection.lower() == 'north':
        map_proj = proj_North
        circ = True
    elif projection.lower() == 'south':
        map_proj = proj_South
        circ = True
    elif projection.lower() == 'global':
        map_proj = proj_Global
    elif projection.lower() == 'global_pc':
        map_proj = proj_Global_PC
    else:
        raise NotImplementedError("Projection not implemented.")

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)


    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor=(0.8,0.8,0.8),
        edgecolor='k',
        linewidth=.25,
    )
    

    fig = plt.figure(figsize=size)
    if nCols==1 and nRows==1:
        ax = fig.add_subplot(1, 1, 1, projection=map_proj)

        ax.coastlines('50m', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )

        if circ:
            ax.set_boundary(circle, transform=ax.transAxes)
        if mask:
            ax.add_feature(land_feature)
    elif nCols > 1 and nRows > 1:
        ax = np.empty((nCols,nRows),dtype=object)
        iF = 1
        for jj in range(nRows):
            for ii in range(nCols):
                ax[ii,jj] = fig.add_subplot(nRows, nCols, iF, projection=map_proj)

                ax[ii,jj].coastlines('50m', linewidth=0.8)
                ax[ii,jj].tick_params(axis='both', which='major', labelsize=10)
                ax[ii,jj].gridlines(
                    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
                )
                if circ:
                    ax[ii,jj].set_boundary(circle, transform=ax.transAxes)
                if mask:
                    ax[ii,jj].add_feature(land_feature)
                iF += 1
    elif nCols > 1:
        ax = np.empty((nCols),dtype=object)
        for ii in range(nCols):
            ax[ii] = fig.add_subplot(1, nCols, ii+1, projection=map_proj)

            ax[ii].coastlines('50m', linewidth=0.8)
            ax[ii].tick_params(axis='both', which='major', labelsize=10)
            ax[ii].gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            if circ:
                ax[ii].set_boundary(circle, transform=ax.transAxes)
            if mask:
                ax[ii].add_feature(land_feature)
    elif nRows > 1:
        ax = np.empty((nRows),dtype=object)
        for ii in range(nRows):
            ax[ii] = fig.add_subplot(nRows, 1, ii+1, projection=map_proj)

            ax[ii].coastlines('50m', linewidth=0.8)
            ax[ii].tick_params(axis='both', which='major', labelsize=10)
            ax[ii].gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            if circ:
                ax[ii].set_boundary(circle, transform=ax.transAxes)
            if mask:
                ax[ii].add_feature(land_feature)

            
    return fig, ax


def add_square(ax,region,crs,**kwargs):
    # Region = lon1, lat1; lon2, lat2

    if np.isnan(region[0]) and np.isnan(region[2]):
        ax.gridlines(draw_labels=False, xlocs=[], ylocs=[region[1],region[3]], color='red',**kwargs)
    else:
        lons = [region[0],region[2],region[2],region[0]]
        lats = [region[1],region[1],region[3],region[3]]
        ring = LinearRing(list(zip(lons,lats)))

        ax.add_geometries([ring],crs,**kwargs)

    return ax


def add_loc_square(ax,settings,**kwargs):
    if "mask_region" in settings.keys():
        maskout=settings["mask_region"]
    else:
        return ax

    if maskout == "indonesia":
        min_lon, max_lon = 75., 130.
        min_lat, max_lat = -15., 10.
    elif maskout == "westpacific":
        min_lon, max_lon = 170., 230.
        min_lat, max_lat = -30., -5.
    elif maskout == "eastpacific":
        min_lon, max_lon = 210., 280.
        min_lat, max_lat = -10., 10.
    elif maskout == "caribbean":
        min_lon, max_lon = 250., 320.
        min_lat, max_lat = 5., 25.
    elif maskout == "brazil":
        min_lon, max_lon = 310., 350.
        min_lat, max_lat = -25., -5.
    elif maskout == "namibia":
        min_lon, max_lon = 0., 20.
        min_lat, max_lat = -45., -10.
    else:
        raise NotImplementedError("no such mask type.")
    
    if min_lon > 180 and max_lon > 180:
        min_lon = min_lon - 360
        max_lon = max_lon - 360
    
    add_square(ax,[min_lon,min_lat,max_lon,max_lat],data_crs,**kwargs)

    return ax


def add_mask(ax,region,lon=None,lat=None):
    
    if region == "land":
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='50m',
            facecolor='gray',
            edgecolor='k',
            linewidth=.25,
        )
        ax.add_feature(land_feature)

    elif region[-3:] == ".nc":

        mask = xr.load_dataarray(SHAPE_DIRECTORY + region).to_numpy()

        mask[mask>0.5] = np.nan
        mask_cyc, lons_cyc = add_cyclic_point(mask, coord=lon)
        ax.pcolormesh(lons_cyc,lat,mask_cyc,cmap="gray")

    return ax


def round_to_n(x,n):
    if x == 0:
        return x
    else:
        return np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))
    
def num_lab(x,n):
    return str(round_to_n(x,n))


def get_area(lat,lon,mask=None):
    _, Y = np.meshgrid(lon,lat)

    dy = np.deg2rad(np.diff(lat)[0])
    dx = np.deg2rad(np.diff(lon)[0]) * np.cos(np.deg2rad(Y))

    area = dy * dx

    if mask is not None:
        area = area * mask
        area[np.isnan(area)] = 0.

    return area/np.sum(area)