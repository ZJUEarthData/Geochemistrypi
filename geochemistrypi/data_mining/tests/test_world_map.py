import json

import pandas as pd
import pytest

from geochemistrypi.data_mining.plot import map_plot


def _configuration(**overrides):
    value = {
        "schema_version": 1,
        "enabled": True,
        "longitude_column": "custom_lon",
        "latitude_column": "custom_lat",
        "value_columns": ["SIO2", "TIO2"],
    }
    value.update(overrides)
    return map_plot.WorldMapConfiguration.from_json(json.dumps(value))


def test_world_map_configuration_is_strict_and_disabled_is_explicit() -> None:
    disabled = map_plot.WorldMapConfiguration.from_json(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": False,
                "longitude_column": None,
                "latitude_column": None,
                "value_columns": [],
            }
        )
    )

    assert disabled.enabled is False
    with pytest.raises(map_plot.WorldMapConfigurationError, match="unknown"):
        map_plot.WorldMapConfiguration.from_json(
            json.dumps(
                {
                    "schema_version": 1,
                    "enabled": False,
                    "longitude_column": None,
                    "latitude_column": None,
                    "value_columns": [],
                    "surprise": True,
                }
            )
        )
    with pytest.raises(map_plot.WorldMapConfigurationError, match="must not include"):
        _configuration(enabled=False)


def test_configured_world_map_uses_explicit_columns_and_multiple_values(
    monkeypatch,
) -> None:
    data = pd.DataFrame(
        {
            "SampleID": ["A", "B"],
            "custom_lon": [120.0, 121.0],
            "custom_lat": [30.0, 31.0],
            "SIO2": [50.0, 51.0],
            "TIO2": [1.0, 1.1],
        }
    )
    rendered = []
    monkeypatch.setattr(map_plot, "get_os", lambda: "Windows")
    monkeypatch.setattr(map_plot, "clear_output", lambda: None)
    monkeypatch.setattr(
        map_plot,
        "map_projected_by_basemap",
        lambda value, name, longitude, latitude: rendered.append(
            (value.name, tuple(longitude), tuple(latitude))
        ),
    )

    map_plot.process_world_map(
        data,
        data["SampleID"],
        _configuration(),
    )

    assert rendered == [
        ("SIO2", (120.0, 121.0), (30.0, 31.0)),
        ("TIO2", (120.0, 121.0), (30.0, 31.0)),
    ]


def test_configured_world_map_supports_location_only_projection(monkeypatch) -> None:
    data = pd.DataFrame(
        {
            "SampleID": ["A"],
            "custom_lon": [120.0],
            "custom_lat": [30.0],
        }
    )
    rendered = []
    monkeypatch.setattr(map_plot, "get_os", lambda: "Linux")
    monkeypatch.setattr(map_plot, "clear_output", lambda: None)
    monkeypatch.setattr(
        map_plot,
        "map_projected_by_basemap",
        lambda value, *args: rendered.append((value.name, tuple(value))),
    )

    map_plot.process_world_map(
        data,
        data["SampleID"],
        _configuration(value_columns=[]),
    )

    assert rendered == [("Locations", (0.5,))]


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("custom_lon", [181.0], "between -180 and 180"),
        ("custom_lat", [91.0], "between -90 and 90"),
        ("SIO2", [float("nan")], "missing or non-finite"),
    ],
)
def test_configured_world_map_rejects_invalid_values_before_rendering(
    column, values, message
) -> None:
    data = pd.DataFrame(
        {
            "SampleID": ["A"],
            "custom_lon": [120.0],
            "custom_lat": [30.0],
            "SIO2": [50.0],
            "TIO2": [1.0],
        }
    )
    data[column] = values

    with pytest.raises(map_plot.WorldMapConfigurationError, match=message):
        map_plot.process_world_map(data, data["SampleID"], _configuration())
