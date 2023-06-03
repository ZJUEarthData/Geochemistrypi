# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from mpl_toolkits.basemap import Basemap

import cartopy.crs as ccrs
import cartopy
from ..global_variable import MAP_IMAGE_PATH
from ..utils.base import save_fig

logging.captureWarnings(True)


def map_projected(col: pd.Series, longitude: pd.DataFrame, latitude: pd.DataFrame) -> None:
    """Project an element data into world map.

    Parameters
    ----------
    col : pd.Series
        One selected column from the data sheet.

    longitude : pd.DataFrame
        Longitude data of data items.

    latitude : pd.DataFrame
        Latitude data of data items.
    """

    M = col
    # Create a new figure with the desired size and DPI
    plt.figure(figsize=(24, 16), dpi=300)
    # Set the font style
    plt.rcParams["font.sans-serif"] = "Arial"

    # Create a Robinson projection centered at the equator
    projection = ccrs.Robinson(central_longitude=0,globe=None, false_easting=0, false_northing=0)
    # Create a new axis with the Robinson projection
    ax = plt.axes(projection=projection)

    # Add coastlines and borders
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)

    # Set the map boundaries and fill color
    ax.set_global()
    ax.set_facecolor('white')

    # Define parallels and meridians
    parallels = np.arange(-90.0, 90.0, 45.0)
    meridians = np.arange(-180.0, 180.0, 60.0)

    # Draw parallels and meridians
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                 linestyle='--', xlocs=meridians, ylocs=parallels)

    # Set the color values
    if type(M) != type(longitude):
        M = [0.5 for i in range(len(M))]

    # Create a scatter plot with color and size parameters
    sc = ax.scatter(longitude, latitude, c=pd.DataFrame(M),  edgecolor="grey", transform=ccrs.PlateCarree(), linewidths=0.5,
                    vmax=3, vmin=0, s=25, alpha=0.6, cmap="BuPu")

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.01, pad=0.1)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("Counts", fontsize=30)
    
    # save figure
    save_fig(f"Map Projection - {col.name}", MAP_IMAGE_PATH)


# def map_projected(col: pd.Series, longitude: pd.DataFrame, latitude: pd.DataFrame) -> None:
#     """Project an element data into world map.
#
#     Parameters
#     ----------
#     col : pd.Series
#         One selected column from the data sheet.
#
#     longitude : pd.DataFrame
#         Longitude data of data items.
#
#     latitude : pd.DataFrame
#         Latitude data of data items.
#     """
#     M = col
#     plt.figure(figsize=(24, 16), dpi=300)
#     plt.rcParams["font.sans-serif"] = "Arial"
#     m = Basemap(projection="robin", lat_0=0, lon_0=0)
#     m.drawcoastlines()
#     m.drawcountries()
#     m.drawmapboundary(fill_color="white")

#     parallels = np.arange(-90.0, 90.0, 45.0)
#     m.drawparallels(parallels, labels=[True, True, True, False], fontsize=30)
#     meridians = np.arange(-180.0, 180.0, 60.0)
#     m.drawmeridians(meridians, labels=[True, False, True, True], fontsize=30)
#     lon, lat = m(longitude, latitude)
#     if type(M) != type(longitude):
#         M = [0.5 for i in range(len(M))]
#     m.scatter(lon, lat, c=M, edgecolor="grey", marker="D", linewidths=0.5, vmax=3, vmin=0, s=25, alpha=0.3, cmap="BuPu")
#     cb = m.colorbar(pad=1)
#     cb.ax.tick_params(labelsize=30)
#     cb.set_label("Counts", fontsize=50)
#
#     save_fig(f"Map Projection - {col.name}", MAP_IMAGE_PATH)

# def map_projected(col: pd.Series, longitude: pd.DataFrame, latitude: pd.DataFrame) -> None:
#     """Project an element data into world map.

#     Parameters
#     ----------
#     col : pd.Series
#         One selected column from the data sheet.

#     longitude : pd.DataFrame
#         Longitude data of data items.

#     latitude : pd.DataFrame
#         Latitude data of data items.
#     """
#     # Create point geometries
#     geometry = geopandas.points_from_xy(longitude, latitude)

#     geo_df = geopandas.GeoDataFrame(pd.concat([col, longitude, latitude], axis=1), geometry=geometry)
#     world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

#     # Make figure
#     fig, ax = plt.subplots(figsize=(24, 18))
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     world.plot(ax=ax, alpha=0.4, color='grey', edgecolor='black')
#     geo_df.plot(col.name, ax=ax, s=20, cmap='gist_heat_r', cax=cax, legend=True)
#     plt.title('colorbar')
#     save_fig(f"Map Projection - {col.name}", MAP_IMAGE_PATH)
