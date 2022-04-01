# -*- coding: utf-8 -*-
# import sys
import matplotlib.pyplot as plt
from utils.base import save_fig
from global_variable import MAP_IMAGE_PATH
import logging
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable

# sys.path.append("..")
logging.captureWarnings(True)


def map_projected(col, df):
    """

    :param col: pd.Series, one selected column from the data sheet
    :param df: pd.DataFrame, the data sheet
    """
    # Create point geometries
    geometry = geopandas.points_from_xy(df['LONGITUDE'], df['LATITUDE'])
    geo_df = geopandas.GeoDataFrame(df[[col.name, 'LONGITUDE', 'LATITUDE']], geometry=geometry)
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    # Make figure
    fig, ax = plt.subplots(figsize=(24, 18))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    world.plot(ax=ax, alpha=0.4, color='grey', edgecolor='black')
    geo_df.plot(col.name, ax=ax, s=20, cmap='gist_heat_r', cax=cax, legend=True)
    plt.title('colorbar')
    save_fig(f"map projection - {col.name}", MAP_IMAGE_PATH)
