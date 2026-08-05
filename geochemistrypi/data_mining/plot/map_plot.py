# -*- coding: utf-8 -*-
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from ..constants import MLFLOW_ARTIFACT_IMAGE_MAP_PATH, OPTION, SECTION
from ..data.data_readiness import limit_num_input, num2option, num_input, show_data_columns
from ..utils.base import clear_output, get_os, save_data, save_fig

logging.captureWarnings(True)

WORLD_MAP_CONFIG_SCHEMA_VERSION = 1
_MAX_WORLD_MAP_CONFIG_CHARACTERS = 16_384
_UNSAFE_ARTIFACT_NAME = re.compile(r'[<>:"/\\|?*]')


class WorldMapConfigurationError(ValueError):
    """Raised when a semantic world-map request is unsafe or invalid."""


class WorldMapRendererError(RuntimeError):
    """Raised when the platform renderer is unavailable or cannot render."""


@dataclass(frozen=True)
class WorldMapConfiguration:
    """Versioned non-interactive world-map choices owned by the CLI."""

    enabled: bool
    longitude_column: Optional[str] = None
    latitude_column: Optional[str] = None
    value_columns: Tuple[str, ...] = ()

    @classmethod
    def from_json(cls, raw: str) -> "WorldMapConfiguration":
        if len(raw) > _MAX_WORLD_MAP_CONFIG_CHARACTERS:
            raise WorldMapConfigurationError("World-map configuration exceeds the 16384-character safety limit.")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WorldMapConfigurationError("World-map configuration must be valid JSON.") from exc
        if not isinstance(value, dict):
            raise WorldMapConfigurationError("World-map configuration must be a JSON object.")
        expected = {
            "schema_version",
            "enabled",
            "longitude_column",
            "latitude_column",
            "value_columns",
        }
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if unknown or missing:
            raise WorldMapConfigurationError(f"World-map configuration fields are invalid; unknown={unknown}, missing={missing}.")
        if value["schema_version"] != WORLD_MAP_CONFIG_SCHEMA_VERSION:
            raise WorldMapConfigurationError("Unsupported world-map configuration schema; expected version 1.")
        if not isinstance(value["enabled"], bool):
            raise WorldMapConfigurationError("World-map enabled must be a boolean.")
        longitude = value["longitude_column"]
        latitude = value["latitude_column"]
        raw_columns = value["value_columns"]
        if not isinstance(raw_columns, list) or len(raw_columns) > 20:
            raise WorldMapConfigurationError("World-map value_columns must be a JSON array with at most 20 entries.")
        if any(not isinstance(column, str) for column in raw_columns):
            raise WorldMapConfigurationError("World-map value_columns must contain only strings.")
        columns = tuple(column.strip() for column in raw_columns)
        if any(not column or "\n" in column or "\r" in column for column in columns):
            raise WorldMapConfigurationError("World-map value columns must be non-blank single-line names.")
        if len(columns) != len(set(columns)):
            raise WorldMapConfigurationError("World-map value columns must not contain duplicates.")
        if not value["enabled"]:
            if longitude is not None or latitude is not None or columns:
                raise WorldMapConfigurationError("Disabled world-map configuration must not include coordinate or value columns.")
            return cls(False)
        if not isinstance(longitude, str) or not longitude.strip():
            raise WorldMapConfigurationError("Enabled world-map configuration requires longitude_column.")
        if not isinstance(latitude, str) or not latitude.strip():
            raise WorldMapConfigurationError("Enabled world-map configuration requires latitude_column.")
        longitude = longitude.strip()
        latitude = latitude.strip()
        if longitude == latitude:
            raise WorldMapConfigurationError("World-map longitude and latitude columns must be different.")
        conflicts = sorted({longitude, latitude}.intersection(columns))
        if conflicts:
            raise WorldMapConfigurationError(f"World-map coordinate columns cannot also be projected values: {conflicts}.")
        unsafe = sorted(column for column in columns if _UNSAFE_ARTIFACT_NAME.search(column))
        if unsafe:
            raise WorldMapConfigurationError(f"World-map value columns contain characters unsafe for artifact names: {unsafe}.")
        return cls(True, longitude, latitude, columns)


def _finite_numeric_column(data: pd.DataFrame, column: str, role: str) -> pd.Series:
    if column not in data.columns:
        raise WorldMapConfigurationError(f"World-map {role} column is absent from the dataset: {column!r}.")
    try:
        numeric = pd.to_numeric(data[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise WorldMapConfigurationError(f"World-map {role} column must contain only numeric values: {column!r}.") from exc
    finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    if not bool(finite.all()):
        rows = [int(index) + 2 for index in np.flatnonzero(~finite)[:10]]
        raise WorldMapConfigurationError(f"World-map {role} column contains missing or non-finite values at data rows {rows}: {column!r}.")
    return numeric


def _configured_map_series(data: pd.DataFrame, configuration: WorldMapConfiguration) -> Tuple[pd.Series, pd.Series, Tuple[pd.Series, ...]]:
    longitude = _finite_numeric_column(data, configuration.longitude_column, "longitude")
    latitude = _finite_numeric_column(data, configuration.latitude_column, "latitude")
    if bool(((longitude < -180) | (longitude > 180)).any()):
        raise WorldMapConfigurationError("World-map longitude values must be between -180 and 180 degrees.")
    if bool(((latitude < -90) | (latitude > 90)).any()):
        raise WorldMapConfigurationError("World-map latitude values must be between -90 and 90 degrees.")
    if configuration.value_columns:
        values = tuple(_finite_numeric_column(data, column, "projected value") for column in configuration.value_columns)
    else:
        values = (pd.Series(0.5, index=data.index, name="Locations", dtype=float),)
    return longitude, latitude, values


def map_projected_by_cartopy(col: pd.Series, name_column: str, longitude: pd.DataFrame, latitude: pd.DataFrame) -> None:
    """Project an element data into world map using cartopy.

    Parameters
    ----------
    col : pd.Series
        One selected column from the data sheet.

    longitude : pd.DataFrame
        Longitude data of data items.

    latitude : pd.DataFrame
        Latitude data of data items.
    """
    try:
        import cartopy
        import cartopy.crs as ccrs
    except ImportError as exc:
        raise WorldMapRendererError(
            "Cartopy is required for world-map rendering on macOS but is not installed in the GeochemistryPi CLI environment. "
            "Install the supported macOS package before retrying; GeochemistryPi never installs map packages during an analysis."
        ) from exc
    M = col
    # Create a new figure with the desired size and DPI
    plt.figure(figsize=(24, 16), dpi=300)
    # Set the font style
    plt.rcParams["font.sans-serif"] = "Arial"

    # Create a Robinson projection centered at the equator
    projection = ccrs.Robinson(central_longitude=0, globe=None, false_easting=0, false_northing=0)
    # Create a new axis with the Robinson projection
    ax = plt.axes(projection=projection)

    # Add coastlines and borders
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)

    # Set the map boundaries and fill color
    ax.set_global()
    ax.set_facecolor("white")

    # Define parallels and meridians
    parallels = np.arange(-90.0, 90.0, 45.0)
    meridians = np.arange(-180.0, 180.0, 60.0)

    # Draw parallels and meridians
    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--", xlocs=meridians, ylocs=parallels)

    # Set the color values
    if type(M) != type(longitude):
        M = [0.5 for i in range(len(M))]

    # Create a scatter plot with color and size parameters
    sc = ax.scatter(longitude, latitude, c=pd.DataFrame(M), edgecolor="grey", transform=ccrs.PlateCarree(), linewidths=0.5, vmax=3, vmin=0, s=25, alpha=0.6, cmap="BuPu")

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.01, pad=0.1)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(str(col.name), fontsize=30)

    # save figure and data
    data = pd.concat([col, longitude, latitude], axis=1)
    save_fig(f"Map Projection - {col.name}", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MAP_PATH"), MLFLOW_ARTIFACT_IMAGE_MAP_PATH)
    save_data(data, name_column, f"Map Projection - {col.name}", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MAP_PATH"), MLFLOW_ARTIFACT_IMAGE_MAP_PATH)


def map_projected_by_basemap(col: pd.Series, name_column: str, longitude: pd.DataFrame, latitude: pd.DataFrame) -> None:
    """Project an element data into world map using basemap.

    Parameters
    ----------
    col : pd.Series
        One selected column from the data sheet.

    longitude : pd.DataFrame
        Longitude data of data items.

    latitude : pd.DataFrame
        Latitude data of data items.
    """
    try:
        from mpl_toolkits.basemap import Basemap
    except ImportError as exc:
        raise WorldMapRendererError(
            "Basemap is required for world-map rendering on Windows and Linux but is not installed in the GeochemistryPi CLI environment. "
            "Install the declared GeochemistryPi dependency before retrying; GeochemistryPi never installs map packages during an analysis."
        ) from exc
    M = col
    plt.figure(figsize=(24, 16), dpi=300)
    plt.rcParams["font.sans-serif"] = "Arial"
    m = Basemap(projection="robin", lat_0=0, lon_0=0)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color="white")

    parallels = np.arange(-90.0, 90.0, 45.0)
    m.drawparallels(parallels, labels=[True, True, True, False], fontsize=30)
    meridians = np.arange(-180.0, 180.0, 60.0)
    m.drawmeridians(meridians, labels=[True, False, True, True], fontsize=30)
    lon, lat = m(longitude, latitude)
    if type(M) != type(longitude):
        M = [0.5 for i in range(len(M))]
    m.scatter(lon, lat, c=M, edgecolor="grey", marker="D", linewidths=0.5, vmax=3, vmin=0, s=25, alpha=0.3, cmap="BuPu")
    cb = m.colorbar(pad=1)
    cb.ax.tick_params(labelsize=30)
    cb.set_label(str(col.name), fontsize=50)

    data = pd.concat([col, longitude, latitude], axis=1)
    save_fig(f"Map Projection - {col.name}", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MAP_PATH"), MLFLOW_ARTIFACT_IMAGE_MAP_PATH)
    save_data(data, name_column, f"Map Projection - {col.name}", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MAP_PATH"), MLFLOW_ARTIFACT_IMAGE_MAP_PATH)


def process_world_map(
    data: pd.DataFrame,
    name_column: str,
    configuration: Optional[WorldMapConfiguration] = None,
) -> None:
    """The process of projecting the data on the world map."""
    if configuration is not None:
        if not configuration.enabled:
            print("World map projection is explicitly disabled for this analysis.")
            clear_output()
            return
        longitude, latitude, values = _configured_map_series(data, configuration)
        my_os = get_os()
        if my_os in {"Windows", "Linux"}:
            renderer = map_projected_by_basemap
        elif my_os == "macOS":
            renderer = map_projected_by_cartopy
        else:
            raise WorldMapRendererError(f"World-map rendering is unsupported on operating system {my_os!r}.")
        for value in values:
            try:
                renderer(value, name_column, longitude, latitude)
            except WorldMapRendererError:
                raise
            except Exception as exc:
                raise WorldMapRendererError(f"World-map renderer failed while projecting {value.name!r}: {exc}") from exc
        print(f"Configured world map projection completed for {len(values)} map artifact(s).")
        clear_output()
        return

    map_flag = 0
    is_map_projection = 0
    detection_index = 0
    lon = ["LONGITUDE", "Longitude (°E)", "longitude", "Longitude", "经度 (°N)", "经度", "lng"]
    lat = ["LATITUDE", "Latitude (°N)", "latitude", "Latitude", "纬度 (°E)", "纬度", "lat"]
    j = [j for j in lat if j in data.columns]
    i = [i for i in lon if i in data.columns]
    if bool(len(j) > 0):
        detection_index += 1
    if bool(len(i) > 0):
        detection_index += 2
    if detection_index == 2:
        print("The provided data set is lack of 'LATITUDE' data.")
    elif detection_index == 1:
        print("The provided data set is lack of 'LONGITUDE' data.")
    elif detection_index == 0:
        print("The provided data set is lack of 'LONGITUDE' and 'LATITUDE' data.")
    if detection_index != 3:
        print("Hence, world map projection functionality will be skipped!")
        clear_output()
    # If the data set contains both longitude and latitude data, then the user can choose to project the data on the world map.
    while detection_index == 3:
        if map_flag != 1:
            # Check if the user wants to project the data on the world map.
            print("World Map Projection for A Specific Element Option:")
            num2option(OPTION)
            is_map_projection = limit_num_input(OPTION, SECTION[3], num_input)
            clear_output()
        if is_map_projection == 1:
            # If the user chooses to project the data on the world map, then the user can select the element to be projected.
            print("[bold green]-*-*- Distribution in World Map -*-*-[/bold green]")
            print("Select one of the elements below to be projected in the World Map: ")
            show_data_columns(data.columns)
            elm_num = limit_num_input(data.columns, SECTION[3], num_input)
            clear_output()
            latitude = data.loc[:, j]
            longitude = data.loc[:, i]
            print("Longitude and latitude data are selected from the provided data set.")
            # If OS is Windows or Linux, then use basemap to project the data on the world map.
            # If OS is macOS, then use cartopy to project the data on the world map.
            my_os = get_os()
            if my_os == "Windows" or my_os == "Linux":
                map_projected_by_basemap(data.iloc[:, elm_num - 1], name_column, longitude, latitude)
            elif my_os == "macOS":
                map_projected_by_cartopy(data.iloc[:, elm_num - 1], name_column, longitude, latitude)
            clear_output()
            print("Do you want to continue to project a new element in the World Map?")
            num2option(OPTION)
            map_flag = limit_num_input(OPTION, SECTION[3], num_input)
            if map_flag == 1:
                clear_output()
                continue
            else:
                print("Exit Map Projection Mode.")
                clear_output()
                break
        elif is_map_projection == 2:
            break


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
