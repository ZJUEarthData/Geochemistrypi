"""Strict client requests and wrapper responses."""

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictBool, StrictFloat, StrictInt, StrictStr, computed_field, field_validator, model_serializer, model_validator

from ..contracts.anomaly_detection import MODEL_ORDER as ANOMALY_DETECTION_MODEL_ORDER
from ..contracts.classification import MODEL_ORDER as CLASSIFICATION_MODEL_ORDER
from ..contracts.clustering import MODEL_ORDER as CLUSTERING_MODEL_ORDER
from ..contracts.decomposition import MODEL_ORDER as DECOMPOSITION_MODEL_ORDER
from ..contracts.regression import MODEL_ORDER as REGRESSION_MODEL_ORDER
from ..contracts.regression import MODELS_WITHOUT_AUTOML

_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_UNSAFE_PATH_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
ColumnName = Annotated[str, Field(min_length=1, max_length=128)]
ArtifactRequirementId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]+$", max_length=120)]
SemanticLabel = Union[StrictBool, StrictInt, StrictFloat, StrictStr]
TabularCell = Union[
    StrictBool,
    StrictInt,
    StrictFloat,
    Annotated[StrictStr, Field(max_length=256)],
    None,
]
AnalysisTaskName = Literal[
    "classification",
    "regression",
    "clustering",
    "decomposition",
    "anomaly_detection",
    "time_series",
]
START_READY_METHODS_BY_TASK: dict[str, tuple[str, ...]] = {
    "classification": CLASSIFICATION_MODEL_ORDER,
    "regression": REGRESSION_MODEL_ORDER,
    "clustering": CLUSTERING_MODEL_ORDER,
    "decomposition": (*DECOMPOSITION_MODEL_ORDER, "embedding_label_overlay"),
    "anomaly_detection": ANOMALY_DETECTION_MODEL_ORDER,
    "time_series": (
        "subaerial_proportion",
        "continuous",
        "element_mean",
        "reference_anomaly_series",
    ),
}


def validate_cli_execution_interval(
    cli_started_at: str | None,
    cli_finished_at: str | None,
    cli_execution_duration_seconds: float | None,
) -> None:
    """Validate one complete route-native CLI child-process interval."""

    values = (cli_started_at, cli_finished_at, cli_execution_duration_seconds)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("CLI execution timing must provide start, finish, and duration together")
    assert cli_started_at is not None
    assert cli_finished_at is not None
    try:
        started = datetime.fromisoformat(cli_started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(cli_finished_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("CLI execution timestamps must be ISO 8601") from exc
    if started.tzinfo is None or started.utcoffset() is None or finished.tzinfo is None or finished.utcoffset() is None:
        raise ValueError("CLI execution timestamps must include a timezone")
    if finished < started:
        raise ValueError("CLI execution finish cannot precede its start")


_DATASET_PATH_RESOLUTION_DESCRIPTION = (
    "Accepts an absolute local path or a relative path inside the MCP server's "
    "startup working directory. Relative paths are resolved from that fixed directory; "
    "parent-directory and symbolic-link escapes are rejected."
)


def _dataset_path_field(description: str) -> Any:
    return Field(
        None,
        description=f"{description} {_DATASET_PATH_RESOLUTION_DESCRIPTION}",
        json_schema_extra={
            "x-path-resolution-base": "mcp_startup_working_directory",
            "x-relative-path-must-remain-within-base": True,
        },
    )


_MAX_COMPACT_RESULT_JSON_BYTES = 64 * 1024
_MAX_COMPACT_REPORTED_METRICS_BYTES = 8 * 1024
_MAX_COMPACT_ARTIFACT_REFERENCES = 32
_MAX_COMPACT_REQUIREMENT_IDS_PER_ARTIFACT = 4
_MAX_COMPACT_MISSING_REQUIREMENT_IDS = 16
_MAX_COMPACT_CHILD_RESULTS = 16
_MAX_COMPACT_LIMITATIONS = 8
_MAX_COMPACT_CHILD_TEXT_JSON_BYTES = 256
_MAX_COMPACT_LIMITATION_TEXT_JSON_BYTES = 512
_MAX_COMPACT_ARTIFACT_PATH_JSON_BYTES = 1024
_MAX_COMPACT_ARTIFACT_RESERVE_BYTES = 8 * 1024
_MAX_REQUIRED_TABULAR_OBSERVATIONS = 32
_MAX_REQUIRED_TABULAR_COLUMNS = 64
_MAX_REQUIRED_TABULAR_ROWS = 512
_MAX_REQUIRED_TABULAR_CELLS = 512
_MAX_REQUIRED_TABULAR_REQUIREMENT_IDS = 4
_MAX_REQUIRED_TABULAR_JSON_BYTES = 16 * 1024


def _non_null_property(name: str) -> dict[str, Any]:
    """Match an explicitly supplied, non-null JSON property."""
    return {
        "required": [name],
        "properties": {name: {"not": {"type": "null"}}},
    }


def _non_empty_array_property(name: str) -> dict[str, Any]:
    """Match an explicitly supplied JSON array containing at least one item."""
    return {
        "required": [name],
        "properties": {name: {"type": "array", "minItems": 1}},
    }


def _exactly_one_non_null(left: str, right: str) -> dict[str, Any]:
    """Require exactly one of two nullable input alternatives."""
    return {
        "oneOf": [
            {
                "allOf": [
                    _non_null_property(left),
                    {"not": _non_null_property(right)},
                ]
            },
            {
                "allOf": [
                    _non_null_property(right),
                    {"not": _non_null_property(left)},
                ]
            },
        ]
    }


def _at_most_one_non_null(left: str, right: str) -> dict[str, Any]:
    """Reject only the case where both nullable alternatives are populated."""
    return {
        "not": {
            "allOf": [
                _non_null_property(left),
                _non_null_property(right),
            ]
        }
    }


def _append_schema_conditions(schema: dict[str, Any], *conditions: dict[str, Any]) -> None:
    """Add Draft 2020-12 conditions without replacing generated field guidance."""
    schema.setdefault("allOf", []).extend(conditions)


def _forbid_explicit_properties(*names: str) -> dict[str, Any]:
    """Forbid mode-inapplicable fields without duplicating their public schemas."""
    return {"not": {"anyOf": [{"required": [name]} for name in names]}}


def _source_row_identity_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": {
                "required": ["strategy"],
                "properties": {"strategy": {"const": "column_values"}},
            },
            "then": _non_empty_array_property("columns"),
            "else": {"not": _non_empty_array_property("columns")},
        },
        {
            "not": {
                "oneOf": [
                    _non_null_property("source_mapping_path"),
                    _non_null_property("source_mapping_sha256"),
                ]
            }
        },
    )


def _dataset_filter_json_schema(schema: dict[str, Any]) -> None:
    comparison_operators = [
        "equal",
        "not_equal",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
    ]
    no_value = {"not": _non_null_property("value")}
    no_values = {"not": _non_empty_array_property("values")}
    no_minimum = {"not": _non_null_property("minimum")}
    no_maximum = {"not": _non_null_property("maximum")}
    _append_schema_conditions(
        schema,
        {
            "oneOf": [
                {
                    "properties": {"operator": {"const": "not_null"}},
                    "allOf": [no_value, no_values, no_minimum, no_maximum],
                },
                {
                    "properties": {"operator": {"enum": comparison_operators}},
                    "allOf": [
                        _non_null_property("value"),
                        no_values,
                        no_minimum,
                        no_maximum,
                    ],
                },
                {
                    "properties": {"operator": {"const": "between"}},
                    "allOf": [
                        _non_null_property("minimum"),
                        _non_null_property("maximum"),
                        no_value,
                        no_values,
                    ],
                },
                {
                    "properties": {"operator": {"const": "in"}},
                    "allOf": [
                        _non_empty_array_property("values"),
                        no_value,
                        no_minimum,
                        no_maximum,
                    ],
                },
            ]
        },
    )


def _dataset_preparation_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": _non_empty_array_property("worksheets"),
            "then": {
                "required": ["union_mode"],
                "properties": {
                    "worksheets": {"minItems": 2},
                    "union_mode": {"const": "rows"},
                },
                "allOf": [
                    _non_null_property("source_sheet_column"),
                    _non_null_property("source_row_column"),
                    _non_empty_array_property("selected_columns"),
                    {"not": _non_null_property("worksheet")},
                ],
            },
            "else": {
                "allOf": [
                    {"not": _non_null_property("union_mode")},
                    {"not": _non_null_property("source_sheet_column")},
                ]
            },
        },
        {
            "if": _non_empty_array_property("header_row_indices"),
            "then": {"not": {"required": ["header_row_index"]}},
        },
        {
            "not": {
                "allOf": [
                    _non_empty_array_property("selected_columns"),
                    _non_empty_array_property("excluded_columns"),
                ]
            }
        },
    )


def _evaluation_json_schema(schema: dict[str, Any]) -> None:
    source_choice = _exactly_one_non_null("evaluation_dataset_path", "evaluation_dataset")
    no_source = {
        "allOf": [
            {"not": _non_null_property("evaluation_dataset_path")},
            {"not": _non_null_property("evaluation_dataset")},
        ]
    }
    _append_schema_conditions(
        schema,
        _at_most_one_non_null("evaluation_dataset_path", "evaluation_dataset"),
        {
            "if": {
                "required": ["mode"],
                "properties": {"mode": {"const": "external_labeled"}},
            },
            "then": source_choice,
            "else": no_source,
        },
        {
            "if": {
                "required": ["mode"],
                "properties": {"mode": {"const": "cross_validation"}},
            },
            "then": _non_null_property("folds"),
        },
        {
            "if": {
                "properties": {
                    "mode": {
                        "enum": [
                            "cli_default",
                            "quality_report",
                            "reference_comparison",
                        ]
                    }
                }
            },
            "then": {"not": _non_null_property("folds")},
        },
        {
            "if": _non_null_property("external_identifier_column"),
            "then": {
                "required": ["mode"],
                "properties": {"mode": {"const": "external_labeled"}},
            },
        },
        {
            "if": {
                "required": ["split_strategy"],
                "properties": {"split_strategy": {"enum": ["random_holdout", "stratified_holdout"]}},
            },
            "then": {
                "required": ["mode"],
                "properties": {"mode": {"const": "holdout"}},
            },
        },
    )


def _all_models_json_condition(*, unsupervised: bool = False) -> dict[str, Any]:
    nested_properties: dict[str, Any] = {"mode": {"const": "all"}}
    if unsupervised:
        nested_properties["tuning"] = {"const": "manual"}
    return {
        "if": {
            "required": ["model_selection"],
            "properties": {
                "model_selection": {
                    "required": ["mode"],
                    "properties": {"mode": {"const": "all"}},
                }
            },
        },
        "then": {
            "properties": {"model_selection": {"properties": nested_properties}},
            "not": {
                "anyOf": [
                    {"required": ["model"]},
                    {"required": ["tuning"]},
                ]
            },
        },
    }


def _classification_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        _exactly_one_non_null("training_dataset_path", "training_dataset"),
        _at_most_one_non_null("application_dataset_path", "application_dataset"),
        _all_models_json_condition(),
        {
            "if": {
                "required": ["metric_average"],
                "properties": {"metric_average": {"const": "binary"}},
            },
            "then": _non_null_property("positive_label"),
        },
        {
            "if": {
                "required": ["metric_average"],
                "properties": {"metric_average": {"enum": ["micro", "macro", "weighted"]}},
            },
            "then": {"not": _non_null_property("positive_label")},
        },
        {
            "if": {
                "required": ["tuning"],
                "properties": {"tuning": {"const": "automl"}},
            },
            "then": {"properties": {"model": {"maxProperties": 1}}},
        },
    )


def _regression_json_schema(schema: dict[str, Any]) -> None:
    target_choice = {
        "oneOf": [
            {
                "allOf": [
                    _non_null_property("target_column"),
                    {"not": _non_empty_array_property("target_columns")},
                ]
            },
            {
                "allOf": [
                    _non_empty_array_property("target_columns"),
                    {"not": _non_null_property("target_column")},
                ]
            },
        ]
    }
    _append_schema_conditions(
        schema,
        _exactly_one_non_null("training_dataset_path", "training_dataset"),
        _at_most_one_non_null("application_dataset_path", "application_dataset"),
        target_choice,
        _all_models_json_condition(),
        {
            "if": {
                "required": ["tuning"],
                "properties": {"tuning": {"const": "automl"}},
            },
            "then": {
                "required": ["model"],
                "properties": {
                    "model": {
                        "maxProperties": 1,
                        "properties": {"type": {"not": {"enum": list(MODELS_WITHOUT_AUTOML)}}},
                    }
                },
            },
        },
    )


def _unsupervised_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        _exactly_one_non_null("training_dataset_path", "training_dataset"),
        _all_models_json_condition(unsupervised=True),
    )


def _decomposition_json_schema(schema: dict[str, Any]) -> None:
    overlay = {
        "required": ["mode"],
        "properties": {"mode": {"const": "embedding_label_overlay"}},
    }
    _append_schema_conditions(
        schema,
        _exactly_one_non_null("training_dataset_path", "training_dataset"),
        _at_most_one_non_null("application_dataset_path", "application_dataset"),
        _all_models_json_condition(unsupervised=True),
        {
            "if": overlay,
            "then": {
                "required": ["scaling"],
                "properties": {
                    "feature_columns": {"minItems": 2, "maxItems": 2},
                    "metadata_columns": {"maxItems": 0},
                    "engineered_features": {"maxItems": 0},
                    "scaling": {"const": "none"},
                    "missing_values": {"properties": {"method": {"const": "error"}}},
                    "model_selection": {"properties": {"mode": {"const": "single"}}},
                    "world_map": {"properties": {"enabled": {"const": False}}},
                },
                "allOf": [
                    _forbid_explicit_properties("model", "model_selection"),
                    _exactly_one_non_null("application_dataset_path", "application_dataset"),
                    _non_null_property("label_identifier_column"),
                    _non_null_property("label_column"),
                    _non_empty_array_property("positive_label_values"),
                ],
            },
            "else": {
                "allOf": [
                    _forbid_explicit_properties(
                        "coordinate_sheet",
                        "label_sheet",
                        "label_identifier_column",
                        "label_column",
                        "positive_label_values",
                    ),
                    {"not": _non_null_property("application_dataset_path")},
                    {"not": _non_null_property("application_dataset")},
                ]
            },
        },
    )


_TIME_SERIES_FIELDS_BY_MODE = {
    "subaerial_proportion": frozenset(
        {
            "bin_width",
            "iterations",
            "seed",
            "age_column",
            "maximum_age_column",
            "probability_column",
            "latitude_column",
            "longitude_column",
            "age_unit",
            "fit_curve",
        }
    ),
    "continuous": frozenset(
        {
            "bin_width",
            "iterations",
            "seed",
            "age_column",
            "minimum_age_column",
            "maximum_age_column",
            "value_column",
            "latitude_column",
            "longitude_column",
            "filter_column",
            "filter_minimum",
            "filter_maximum",
            "minimum_samples_per_bin",
            "relative_value_two_sigma",
            "age_unit",
            "fit_curve",
            "compact_y_axis",
        }
    ),
    "element_mean": frozenset(
        {
            "bin_width",
            "age_column",
            "element_columns",
            "filter_column",
            "filter_minimum",
            "filter_maximum",
            "aggregation",
            "uncertainty",
            "minimum_samples_per_bin",
        }
    ),
    "reference_anomaly_series": frozenset(
        {
            "time_column",
            "signal_columns",
            "reference_label_column",
            "reference_positive_values",
            "reference_label_provenance",
            "comparison_label_column",
            "comparison_positive_values",
            "comparison_label_provenance",
            "event_dataset_path",
            "event_sheet",
            "event_time_column",
            "event_identifier_column",
            "event_filter_column",
            "event_filter_values",
            "association_window_days",
            "association_direction",
        }
    ),
}
_TIME_SERIES_MODE_SPECIFIC_FIELDS = frozenset().union(*_TIME_SERIES_FIELDS_BY_MODE.values())


def _time_series_mode_ownership_json_conditions() -> tuple[dict[str, Any], ...]:
    """Require each explicitly supplied mode-owned field to select an owning mode."""
    grouped_fields: dict[tuple[str, ...], list[str]] = {}
    for field in sorted(_TIME_SERIES_MODE_SPECIFIC_FIELDS):
        owners = tuple(mode for mode, allowed_fields in _TIME_SERIES_FIELDS_BY_MODE.items() if field in allowed_fields)
        grouped_fields.setdefault(owners, []).append(field)

    conditions = []
    for owners, fields in grouped_fields.items():
        mode_constraint: dict[str, Any]
        if len(owners) == 1:
            mode_constraint = {"const": owners[0]}
        else:
            mode_constraint = {"enum": list(owners)}
        consequence: dict[str, Any] = {
            "properties": {"mode": mode_constraint},
        }
        if "subaerial_proportion" not in owners:
            consequence["required"] = ["mode"]
        conditions.append(
            {
                "if": {"anyOf": [{"required": [field]} for field in fields]},
                "then": consequence,
            }
        )
    return tuple(conditions)


def _time_series_json_schema(schema: dict[str, Any]) -> None:
    filter_semantics = {
        "if": {
            "anyOf": [
                _non_null_property("filter_minimum"),
                _non_null_property("filter_maximum"),
            ]
        },
        "then": _non_null_property("filter_column"),
    }
    _append_schema_conditions(
        schema,
        _exactly_one_non_null("training_dataset_path", "training_dataset"),
        *_time_series_mode_ownership_json_conditions(),
        {
            "if": {"properties": {"mode": {"const": "subaerial_proportion"}}},
            "then": _non_null_property("bin_width"),
        },
        {
            "if": {
                "required": ["mode"],
                "properties": {"mode": {"const": "continuous"}},
            },
            "then": {
                "allOf": [
                    _non_null_property("bin_width"),
                    _non_null_property("minimum_age_column"),
                    _non_null_property("value_column"),
                    filter_semantics,
                ]
            },
        },
        {
            "if": {
                "required": ["mode"],
                "properties": {"mode": {"const": "element_mean"}},
            },
            "then": {
                "allOf": [
                    _non_null_property("bin_width"),
                    _non_empty_array_property("element_columns"),
                    filter_semantics,
                ]
            },
        },
        {
            "if": {
                "required": ["mode"],
                "properties": {"mode": {"const": "reference_anomaly_series"}},
            },
            "then": {
                "properties": {
                    "bin_width": {"type": "null"},
                    "missing_values": {"properties": {"method": {"const": "error"}}},
                },
                "allOf": [
                    _non_null_property("time_column"),
                    _non_empty_array_property("signal_columns"),
                    _non_null_property("reference_label_column"),
                    _non_empty_array_property("reference_positive_values"),
                    {
                        "not": {
                            "oneOf": [
                                _non_null_property("comparison_label_column"),
                                _non_empty_array_property("comparison_positive_values"),
                            ]
                        }
                    },
                    {
                        "if": {
                            "anyOf": [
                                _non_null_property("event_time_column"),
                                _non_null_property("event_identifier_column"),
                                _non_null_property("event_filter_column"),
                                _non_null_property("association_window_days"),
                                _non_empty_array_property("event_filter_values"),
                            ]
                        },
                        "then": _non_null_property("event_dataset_path"),
                    },
                    {
                        "if": _non_null_property("event_dataset_path"),
                        "then": _non_null_property("event_time_column"),
                    },
                    {
                        "not": {
                            "oneOf": [
                                _non_null_property("event_filter_column"),
                                _non_empty_array_property("event_filter_values"),
                            ]
                        }
                    },
                ],
            },
        },
    )


def _dataset_inspection_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        _exactly_one_non_null("dataset_path", "dataset"),
    )


def _impute_missing_values_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": {
                "required": ["strategy"],
                "properties": {"strategy": {"const": "constant"}},
            },
            "then": _non_null_property("fill_value"),
            "else": {"not": _non_null_property("fill_value")},
        },
    )


def _logistic_regression_json_schema(schema: dict[str, Any]) -> None:
    no_l1_ratio = {"not": _non_null_property("l1_ratio")}
    _append_schema_conditions(
        schema,
        {
            "if": {"properties": {"penalty": {"const": "l2"}}},
            "then": {
                "properties": {"solver": {"enum": ["newton-cg", "lbfgs", "sag", "saga"]}},
                "allOf": [no_l1_ratio],
            },
        },
        {
            "if": {
                "required": ["penalty"],
                "properties": {"penalty": {"const": "l1"}},
            },
            "then": {
                "required": ["solver"],
                "properties": {"solver": {"enum": ["liblinear", "saga"]}},
                "allOf": [no_l1_ratio],
            },
        },
        {
            "if": {
                "required": ["penalty"],
                "properties": {"penalty": {"const": "elasticnet"}},
            },
            "then": {
                "required": ["solver"],
                "properties": {"solver": {"const": "saga"}},
                "allOf": [_non_null_property("l1_ratio")],
            },
        },
    )


def _support_vector_machine_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": {
                "required": ["kernel"],
                "properties": {"kernel": {"const": "poly"}},
            },
            "else": {"properties": {"degree": {"const": 3}}},
        },
        {
            "if": {
                "required": ["kernel"],
                "properties": {"kernel": {"const": "linear"}},
            },
            "then": {"properties": {"gamma": {"const": 0.1}}},
        },
    )


def _k_nearest_neighbors_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": {
                "required": ["algorithm"],
                "properties": {"algorithm": {"enum": ["ball_tree", "kd_tree"]}},
            },
            "else": {"properties": {"leaf_size": {"const": 30}}},
        },
        {
            "if": {
                "required": ["metric"],
                "properties": {"metric": {"enum": ["euclidean", "manhattan"]}},
            },
            "then": {"properties": {"power": {"const": 2}}},
        },
    )


def _stochastic_gradient_descent_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": {
                "required": ["penalty"],
                "properties": {"penalty": {"const": "elasticnet"}},
            },
            "else": {"properties": {"l1_ratio": {"const": 0.15}}},
        },
    )


def _forest_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": {"properties": {"bootstrap": {"const": True}}},
            "then": {"properties": {"maximum_samples": {"not": {"type": "null"}}}},
            "else": {
                "required": ["maximum_samples", "out_of_bag_score"],
                "properties": {
                    "maximum_samples": {"type": "null"},
                    "out_of_bag_score": {"const": False},
                },
            },
        },
    )


def _affine_target_transformation_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {"properties": {"scale": {"not": {"const": 0}}}},
    )


def _dbscan_json_schema(schema: dict[str, Any]) -> None:
    _append_schema_conditions(
        schema,
        {
            "if": {
                "required": ["metric"],
                "properties": {"metric": {"const": "minkowski"}},
            },
            "then": _non_null_property("power"),
            "else": {"not": _non_null_property("power")},
        },
    )


class StrictModel(BaseModel):
    """Immutable model that rejects undeclared client fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_shared_all_models_contract(self) -> "StrictModel":
        selection = getattr(self, "model_selection", None)
        if selection is None or getattr(selection, "mode", None) != "all":
            return self
        conflicting = sorted(field for field in ("model", "tuning") if field in self.model_fields_set)
        if conflicting:
            raise ValueError("model_selection.mode='all' replaces explicit legacy fields: " f"{conflicting}")
        unsupervised_task = getattr(self, "task", None) in {
            "clustering",
            "decomposition",
            "anomaly_detection",
        }
        if unsupervised_task and selection.tuning != "manual":
            raise ValueError("AutoML is not public for unsupervised all-models workflows")
        return self


class SourceRowIdentityContract(StrictModel):
    """Stable source-row identity used to audit a prepared dataset view."""

    model_config = ConfigDict(json_schema_extra=_source_row_identity_json_schema)

    strategy: Literal["source_row", "column_values"] = "source_row"
    columns: tuple[ColumnName, ...] = Field(default=(), max_length=16)
    expected_ordered_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    source_mapping_path: Path | None = None
    source_mapping_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("row-identity columns must contain unique, non-blank names")
        return normalized

    @model_validator(mode="after")
    def validate_strategy(self) -> "SourceRowIdentityContract":
        if self.strategy == "column_values" and not self.columns:
            raise ValueError("column_values row identity requires at least one column")
        if self.strategy == "source_row" and self.columns:
            raise ValueError("source_row identity does not accept columns")
        if (self.source_mapping_path is None) != (self.source_mapping_sha256 is None):
            raise ValueError("source row mapping path and SHA256 must be provided together")
        return self


class DatasetFilterRule(StrictModel):
    """One deterministic row predicate applied before column projection."""

    model_config = ConfigDict(json_schema_extra=_dataset_filter_json_schema)

    column: ColumnName
    operator: Literal[
        "not_null",
        "equal",
        "not_equal",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "between",
        "in",
    ]
    value: str | int | float | bool | None = None
    values: tuple[str | int | float | bool, ...] = Field(default=(), max_length=256)
    minimum: int | float | None = None
    maximum: int | float | None = None
    inclusive: bool = True

    @field_validator("column")
    @classmethod
    def validate_column(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("filter column must be a non-blank single-line name")
        return normalized

    @model_validator(mode="after")
    def validate_operands(self) -> "DatasetFilterRule":
        comparison_operators = {
            "equal",
            "not_equal",
            "greater_than",
            "greater_than_or_equal",
            "less_than",
            "less_than_or_equal",
        }
        if self.operator == "not_null":
            if self.value is not None or self.values or self.minimum is not None or self.maximum is not None:
                raise ValueError("not_null does not accept filter operands")
        elif self.operator in comparison_operators:
            if self.value is None:
                raise ValueError(f"{self.operator} requires value")
            if self.values or self.minimum is not None or self.maximum is not None:
                raise ValueError(f"{self.operator} accepts only value")
        elif self.operator == "between":
            if self.minimum is None or self.maximum is None:
                raise ValueError("between requires minimum and maximum")
            if self.minimum > self.maximum:
                raise ValueError("between minimum must not exceed maximum")
            if self.value is not None or self.values:
                raise ValueError("between accepts only minimum, maximum, and inclusive")
        else:
            if not self.values:
                raise ValueError("in requires at least one value")
            if len(self.values) != len(set(self.values)):
                raise ValueError("in values must not contain duplicates")
            if self.value is not None or self.minimum is not None or self.maximum is not None:
                raise ValueError("in accepts only values")
        numeric_values = (
            *(item for item in (self.value, self.minimum, self.maximum) if isinstance(item, (int, float)) and not isinstance(item, bool)),
            *(item for item in self.values if isinstance(item, (int, float)) and not isinstance(item, bool)),
        )
        if any(not math.isfinite(float(item)) for item in numeric_values):
            raise ValueError("numeric filter operands must be finite")
        return self


class DatasetPreparationContract(StrictModel):
    """Paper-agnostic table selection and source-row lineage controls."""

    model_config = ConfigDict(json_schema_extra=_dataset_preparation_json_schema)

    worksheet: str | None = Field(None, min_length=1, max_length=255)
    worksheets: tuple[str, ...] = Field(default=(), min_length=0, max_length=16)
    union_mode: Literal["rows"] | None = None
    source_sheet_column: ColumnName | None = None
    source_row_column: ColumnName | None = None
    header_row_index: int = Field(0, ge=0, le=1_000_000)
    header_row_indices: tuple[Annotated[int, Field(ge=0, le=1_000_000)], ...] = Field(default=(), max_length=16)
    header_join_separator: str = Field(" | ", min_length=1, max_length=16)
    empty_header_policy: Literal["forward_fill", "skip", "error"] = "forward_fill"
    header_whitespace_policy: Literal["reject", "strip"] = "reject"
    header_bom_policy: Literal["preserve", "strip"] = "preserve"
    duplicate_header_policy: Literal["reject", "suffix"] = "reject"
    selected_columns: tuple[ColumnName, ...] = Field(default=(), max_length=256)
    excluded_columns: tuple[ColumnName, ...] = Field(default=(), max_length=256)
    filters: tuple[DatasetFilterRule, ...] = Field(default=(), max_length=64)
    row_identity: SourceRowIdentityContract = Field(default_factory=SourceRowIdentityContract)
    operations: tuple[
        Literal["missing_value_handling", "filtering", "transformation", "feature_selection"],
        ...,
    ] = Field(default=(), max_length=4)

    @field_validator("worksheet", "source_sheet_column", "source_row_column")
    @classmethod
    def validate_worksheet(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("worksheet must be a non-blank single-line name")
        return normalized

    @field_validator("worksheets")
    @classmethod
    def validate_worksheets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized):
            raise ValueError("worksheets must contain non-blank single-line names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("worksheets must not contain duplicates")
        return normalized

    @field_validator("header_row_indices")
    @classmethod
    def validate_header_row_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(index < 0 or index > 1_000_000 for index in value):
            raise ValueError("header_row_indices must contain zero-based bounded indices")
        if tuple(sorted(value)) != value or len(value) != len(set(value)):
            raise ValueError("header_row_indices must be unique and strictly increasing")
        return value

    @field_validator("header_join_separator")
    @classmethod
    def validate_header_join_separator(cls, value: str) -> str:
        if not value or "\n" in value or "\r" in value:
            raise ValueError("header_join_separator must be a non-blank single-line value")
        return value

    @field_validator("selected_columns", "excluded_columns")
    @classmethod
    def validate_selected_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("dataset column selections must contain unique, non-blank names")
        return normalized

    @model_validator(mode="after")
    def validate_preparation_steps(self) -> "DatasetPreparationContract":
        if self.worksheet is not None and self.worksheets:
            raise ValueError("worksheet and worksheets are mutually exclusive")
        if self.worksheets and len(self.worksheets) < 2:
            raise ValueError("a row union requires at least two worksheets")
        if bool(self.worksheets) != (self.union_mode == "rows"):
            raise ValueError("worksheets and union_mode='rows' must be declared together")
        if self.worksheets and (self.source_sheet_column is None or self.source_row_column is None):
            raise ValueError("a worksheet row union requires source_sheet_column and source_row_column")
        if not self.worksheets and self.source_sheet_column is not None:
            raise ValueError("source_sheet_column is used only by a worksheet row union")
        if self.header_row_indices and "header_row_index" in self.model_fields_set:
            raise ValueError("provide either header_row_index or header_row_indices, not both")
        if self.selected_columns and self.excluded_columns:
            raise ValueError("selected_columns and excluded_columns are mutually exclusive")
        if len(self.operations) != len(set(self.operations)):
            raise ValueError("dataset preparation operations must not contain duplicates")
        filter_identities = tuple((rule.column, rule.operator) for rule in self.filters)
        if len(filter_identities) != len(set(filter_identities)):
            raise ValueError("dataset preparation filters must not contain duplicates")
        missing_identity_columns = sorted(set(self.row_identity.columns) - set(self.selected_columns))
        if self.selected_columns and missing_identity_columns:
            raise ValueError(f"selected_columns must retain row-identity columns: {missing_identity_columns}")
        excluded_identity_columns = sorted(set(self.row_identity.columns) & set(self.excluded_columns))
        if excluded_identity_columns:
            raise ValueError(f"excluded_columns must retain row-identity columns: {excluded_identity_columns}")
        generated = {value for value in (self.source_sheet_column, self.source_row_column) if value is not None}
        if generated and not generated <= set(self.selected_columns):
            raise ValueError("selected_columns must retain every generated source sheet or row column")
        return self


class ExplicitDatasetReference(StrictModel):
    """One explicit local path, preserved for users who already know it."""

    source: Literal["path"] = "path"
    path: Path = Field(
        description=_DATASET_PATH_RESOLUTION_DESCRIPTION,
        json_schema_extra={
            "x-path-resolution-base": "mcp_startup_working_directory",
            "x-relative-path-must-remain-within-base": True,
        },
    )
    expected_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    preparation: DatasetPreparationContract = Field(default_factory=DatasetPreparationContract)


class BuiltInDatasetReference(StrictModel):
    """One stable dataset ID shipped with the installed CLI."""

    source: Literal["builtin"] = "builtin"
    dataset_id: str = Field(
        pattern=r"^builtin:[a-z0-9_]+$",
        max_length=80,
        description="Exact stable ID returned by list_datasets; retain the required 'builtin:' prefix.",
    )
    expected_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    preparation: DatasetPreparationContract = Field(default_factory=DatasetPreparationContract)


class DesktopDatasetReference(StrictModel):
    """One immediate child of the CLI's Desktop/geopi_input directory."""

    source: Literal["desktop"] = "desktop"
    file_name: str = Field(min_length=1, max_length=255)
    expected_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    preparation: DatasetPreparationContract = Field(default_factory=DatasetPreparationContract)

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized in {".", ".."} or Path(normalized).name != normalized or "/" in normalized or "\\" in normalized or _UNSAFE_PATH_CHARACTERS.search(normalized):
            raise ValueError("must be a plain file name inside Desktop/geopi_input")
        if Path(normalized).suffix.lower() not in {".csv", ".xlsx"}:
            raise ValueError("must end in .csv or .xlsx")
        return normalized


DatasetReference = Annotated[
    Union[ExplicitDatasetReference, BuiltInDatasetReference, DesktopDatasetReference],
    Field(discriminator="source"),
]


class DisabledWorldMap(StrictModel):
    """Explicitly skip map rendering even when coordinate columns exist."""

    enabled: Literal[False] = False


class EnabledWorldMap(StrictModel):
    """Semantic coordinate roles and zero or more projected value columns."""

    enabled: Literal[True] = True
    longitude_column: ColumnName
    latitude_column: ColumnName
    value_columns: tuple[ColumnName, ...] = Field(default=(), max_length=20)

    @field_validator("longitude_column", "latitude_column")
    @classmethod
    def validate_coordinate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("must be a non-blank single-line column name")
        return normalized

    @field_validator("value_columns")
    @classmethod
    def validate_value_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column or "\n" in column or "\r" in column or _UNSAFE_PATH_CHARACTERS.search(column) for column in normalized):
            raise ValueError("must contain unique, non-blank artifact-safe column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @model_validator(mode="after")
    def validate_roles(self) -> "EnabledWorldMap":
        if self.longitude_column == self.latitude_column:
            raise ValueError("longitude_column and latitude_column must be different")
        conflicts = sorted({self.longitude_column, self.latitude_column}.intersection(self.value_columns))
        if conflicts:
            raise ValueError(f"coordinate columns must not also be projected values: {conflicts}")
        return self


WorldMapConfiguration = Annotated[Union[DisabledWorldMap, EnabledWorldMap], Field(discriminator="enabled")]


class SingleModelSelection(StrictModel):
    """Run the single model described by the request's model field."""

    mode: Literal["single"] = "single"


class AllModelsSelection(StrictModel):
    """Run every public model in the selected task with isolated child results."""

    mode: Literal["all"] = "all"
    tuning: Literal["manual", "automl"] = "manual"


ModelSelection = Annotated[Union[SingleModelSelection, AllModelsSelection], Field(discriminator="mode")]


def _validate_dataset_choice(
    path: Path | None,
    reference: DatasetReference | None,
    field_name: str,
    required: bool = True,
) -> None:
    supplied = int(path is not None) + int(reference is not None)
    if required and supplied != 1:
        raise ValueError(f"provide exactly one of {field_name}_path or {field_name}")
    if not required and supplied > 1:
        raise ValueError(f"provide at most one of {field_name}_path or {field_name}")


ScalingMethod = Literal["none", "min_max", "standardization", "mean_normalization"]
ClassificationModelName = Literal[
    "logistic_regression",
    "support_vector_machine",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "xgboost",
    "multi_layer_perceptron",
    "gradient_boosting",
    "k_nearest_neighbors",
    "stochastic_gradient_descent",
    "adaboost",
]
RegressionModelName = Literal[
    "linear_regression",
    "polynomial_regression",
    "k_nearest_neighbors",
    "support_vector_machine",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "multi_layer_perceptron",
    "lasso_regression",
    "elastic_net",
    "stochastic_gradient_descent",
    "bayesian_ridge",
    "ridge_regression",
]
ClusteringModelName = Literal[
    "kmeans",
    "dbscan",
    "agglomerative",
    "affinity_propagation",
    "mean_shift",
]
DecompositionModelName = Literal["pca", "tsne", "mds"]
AnomalyDetectionModelName = Literal["isolation_forest", "local_outlier_factor"]
ModelParameterValue = str | int | float | bool | None | tuple[int, ...]


class EncodeOriginalLabels(StrictModel):
    strategy: Literal["encode_original"] = "encode_original"


class MapLabels(StrictModel):
    strategy: Literal["map"] = "map"
    mapping: dict[str, str] = Field(min_length=2, max_length=256)

    @field_validator("mapping")
    @classmethod
    def validate_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = {str(source).strip(): str(target).strip() for source, target in value.items()}
        if any(not source or not target for source, target in normalized.items()):
            raise ValueError("mapping keys and values must not be blank")
        if any(any(character in item for character in ":;\r\n") for pair in normalized.items() for item in pair):
            raise ValueError("mapping keys and values must not contain ':', ';', or line breaks")
        if len(set(normalized.values())) < 2:
            raise ValueError("mapping must produce at least two final classes")
        return normalized


class IntervalLabels(StrictModel):
    strategy: Literal["interval"] = "interval"
    cut_points: tuple[float, ...] = Field(min_length=1, max_length=19)
    labels: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_intervals(self) -> "IntervalLabels":
        if any(left >= right for left, right in zip(self.cut_points, self.cut_points[1:])):
            raise ValueError("cut_points must be strictly increasing")
        if self.labels is not None:
            if len(self.labels) != len(self.cut_points) + 1:
                raise ValueError("labels must contain exactly one more item than cut_points")
            if any(not label.strip() or ";" in label or "\n" in label or "\r" in label for label in self.labels):
                raise ValueError("labels must be non-blank single-line values without semicolons")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("labels must be unique")
        return self


class QuantileLabels(StrictModel):
    strategy: Literal["quantile"] = "quantile"
    number_of_classes: int = Field(2, ge=2, le=20)
    labels: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> "QuantileLabels":
        if self.labels is not None:
            if len(self.labels) != self.number_of_classes:
                raise ValueError("labels must match number_of_classes")
            if any(not label.strip() or ";" in label or "\n" in label or "\r" in label for label in self.labels):
                raise ValueError("labels must be non-blank single-line values without semicolons")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("labels must be unique")
        return self


LabelCustomization = Annotated[
    Union[EncodeOriginalLabels, MapLabels, IntervalLabels, QuantileLabels],
    Field(discriminator="strategy"),
]


class RejectMissingValues(StrictModel):
    method: Literal["error"] = "error"


class KeepMissingValues(StrictModel):
    method: Literal["keep"] = "keep"


class DropMissingRows(StrictModel):
    method: Literal["drop_rows"] = "drop_rows"
    columns: tuple[ColumnName, ...] = Field(default=(), max_length=256)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("columns must contain unique, non-blank names")
        return normalized


class ImputeMissingValues(StrictModel):
    model_config = ConfigDict(json_schema_extra=_impute_missing_values_json_schema)

    method: Literal["impute"] = "impute"
    strategy: Literal["mean", "median", "most_frequent", "constant"] = Field(
        "mean",
        description="Imputation strategy; constant requires an explicit fill_value.",
    )
    fill_value: float | None = Field(
        None,
        description="Numeric replacement used only when strategy='constant'.",
    )

    @model_validator(mode="after")
    def validate_fill_value(self) -> "ImputeMissingValues":
        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError("fill_value is required when strategy is constant")
        if self.strategy != "constant" and self.fill_value is not None:
            raise ValueError("fill_value is only valid when strategy is constant")
        return self


MissingValueHandling = Annotated[
    Union[RejectMissingValues, KeepMissingValues, DropMissingRows, ImputeMissingValues],
    Field(discriminator="method"),
]
TimeSeriesMissingValueHandling = Annotated[
    Union[RejectMissingValues, DropMissingRows],
    Field(discriminator="method"),
]
UnsupervisedMissingValueHandling = Annotated[
    Union[RejectMissingValues, DropMissingRows, ImputeMissingValues],
    Field(discriminator="method"),
]


class NoFeatureSelection(StrictModel):
    method: Literal["none"] = "none"


class SelectFeatures(StrictModel):
    method: Literal["generic_univariate", "select_k_best"]
    retain_count: int = Field(ge=1, le=255)


FeatureSelection = Annotated[Union[NoFeatureSelection, SelectFeatures], Field(discriminator="method")]


class EngineeredFeature(StrictModel):
    name: ColumnName
    formula: str = Field(min_length=1, max_length=500)

    @field_validator("name", "formula")
    @classmethod
    def validate_single_line(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("must be a non-blank single-line value")
        return normalized


class LogisticRegressionSettings(StrictModel):
    """Supported logistic-regression choices in the existing CLI."""

    model_config = ConfigDict(json_schema_extra=_logistic_regression_json_schema)

    type: Literal["logistic_regression"] = "logistic_regression"
    penalty: Literal["l1", "l2", "elasticnet"] = Field(
        "l2",
        description="Regularization penalty; it determines the allowed solver and whether l1_ratio is used.",
    )
    regularization_strength: float = Field(1.0, gt=0)
    solver: Literal["liblinear", "newton-cg", "lbfgs", "sag", "saga"] = Field(
        "lbfgs",
        description="CLI solver compatible with the selected penalty.",
    )
    l1_ratio: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Elastic-net mixing ratio; required only for penalty='elasticnet'.",
    )
    maximum_iterations: int = Field(200, ge=1, le=100_000)
    class_weight: Literal["none", "balanced"] = "none"

    @model_validator(mode="after")
    def validate_solver(self) -> "LogisticRegressionSettings":
        allowed = {
            "l1": {"liblinear", "saga"},
            "l2": {"newton-cg", "lbfgs", "sag", "saga"},
            "elasticnet": {"saga"},
        }[self.penalty]
        if self.solver not in allowed:
            raise ValueError(f"solver {self.solver!r} is not offered by the CLI for penalty {self.penalty!r}")
        if self.penalty == "elasticnet" and self.l1_ratio is None:
            raise ValueError("l1_ratio is required for elasticnet")
        if self.penalty != "elasticnet" and self.l1_ratio is not None:
            raise ValueError("l1_ratio is only valid for elasticnet")
        return self


class SupportVectorMachineSettings(StrictModel):
    model_config = ConfigDict(json_schema_extra=_support_vector_machine_json_schema)

    type: Literal["support_vector_machine"] = "support_vector_machine"
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    degree: int = Field(3, ge=1, le=100)
    gamma: float = Field(0.1, gt=0)
    regularization_strength: float = Field(1.0, gt=0)
    shrinking: bool = True

    @model_validator(mode="after")
    def validate_kernel_parameters(self) -> "SupportVectorMachineSettings":
        if self.kernel != "poly" and self.degree != 3:
            raise ValueError("degree is consumed only for kernel='poly'; use degree=3 or choose the poly kernel")
        if self.kernel == "linear" and self.gamma != 0.1:
            raise ValueError("gamma is not consumed for kernel='linear'; use gamma=0.1 or choose poly, rbf, or sigmoid")
        return self


class DecisionTreeSettings(StrictModel):
    type: Literal["decision_tree"] = "decision_tree"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    maximum_depth: int = Field(4, ge=1, le=100_000)
    minimum_samples_split: int = Field(2, ge=2)
    minimum_samples_leaf: int = Field(1, ge=1)
    maximum_features: int = Field(1, ge=1)


class ForestSettings(StrictModel):
    model_config = ConfigDict(json_schema_extra=_forest_json_schema)

    number_of_estimators: int = Field(100, ge=1, le=100_000)
    maximum_depth: int | None = Field(4, ge=1, le=100_000)
    minimum_samples_split: int = Field(2, ge=2)
    minimum_samples_leaf: int = Field(1, ge=1)
    maximum_features: int = Field(1, ge=1)
    bootstrap: bool = Field(
        True,
        description="Whether each tree uses a bootstrap sample.",
    )
    maximum_samples: float | None = Field(
        0.8,
        gt=0,
        le=1,
        description="Bootstrap sample fraction; required only when bootstrap is true.",
    )
    out_of_bag_score: bool = Field(
        True,
        description="Compute out-of-bag evidence; requires bootstrap=true.",
    )

    @model_validator(mode="after")
    def validate_bootstrap_options(self) -> "ForestSettings":
        if self.bootstrap and self.maximum_samples is None:
            raise ValueError("maximum_samples is required when bootstrap is true")
        if not self.bootstrap and self.maximum_samples is not None:
            raise ValueError("maximum_samples is only valid when bootstrap is true")
        if not self.bootstrap and self.out_of_bag_score:
            raise ValueError("out_of_bag_score requires bootstrap=true")
        return self


class RandomForestSettings(ForestSettings):
    type: Literal["random_forest"] = "random_forest"


class ExtraTreesSettings(ForestSettings):
    type: Literal["extra_trees"] = "extra_trees"


class XGBoostSettings(StrictModel):
    type: Literal["xgboost"] = "xgboost"
    number_of_estimators: int = Field(100, ge=1, le=100_000)
    learning_rate: float = Field(0.1, gt=0)
    maximum_depth: int = Field(4, ge=1, le=100_000)
    subsample: float = Field(1.0, gt=0, le=1)
    column_subsample: float = Field(1.0, gt=0, le=1)
    l1_regularization: float = Field(0.0, ge=0)
    l2_regularization: float = Field(1.0, ge=0)
    gamma: float = Field(0.0, ge=0)
    tree_method: Literal["auto", "exact", "approx", "hist", "gpu_hist"] = "auto"
    objective: Literal[
        "auto",
        "binary:logistic",
        "multi:softprob",
        "multi:softmax",
    ] = "auto"
    importance_type: Literal[
        "gain",
        "weight",
        "cover",
        "total_gain",
        "total_cover",
    ] = "gain"


class MultiLayerPerceptronSettings(StrictModel):
    type: Literal["multi_layer_perceptron"] = "multi_layer_perceptron"
    hidden_layer_sizes: tuple[int, ...] = Field((50, 25, 5), min_length=1, max_length=20)
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    solver: Literal["lbfgs", "sgd", "adam"] = "adam"
    alpha: float = Field(0.0001, ge=0)
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant"
    maximum_iterations: int = Field(200, ge=1, le=100_000)

    @field_validator("hidden_layer_sizes")
    @classmethod
    def validate_hidden_layers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(size < 1 for size in value):
            raise ValueError("every hidden layer size must be positive")
        return value


class GradientBoostingSettings(StrictModel):
    type: Literal["gradient_boosting"] = "gradient_boosting"
    number_of_estimators: int = Field(100, ge=1, le=100_000)
    learning_rate: float = Field(0.1, gt=0)
    maximum_depth: int = Field(4, ge=1, le=100_000)
    minimum_samples_split: int = Field(2, ge=2)
    minimum_samples_leaf: int = Field(1, ge=1)
    maximum_features: int = Field(1, ge=1)
    subsample: float = Field(1.0, gt=0, le=1)
    loss: Literal["log_loss", "exponential"] = "log_loss"


class KNearestNeighborsSettings(StrictModel):
    model_config = ConfigDict(json_schema_extra=_k_nearest_neighbors_json_schema)

    type: Literal["k_nearest_neighbors"] = "k_nearest_neighbors"
    number_of_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = Field(30, ge=1)
    metric: Literal["euclidean", "manhattan", "minkowski"] = "minkowski"
    power: int = Field(2, ge=1)

    @model_validator(mode="after")
    def validate_conditional_parameters(self) -> "KNearestNeighborsSettings":
        if self.algorithm not in {"ball_tree", "kd_tree"} and self.leaf_size != 30:
            raise ValueError("leaf_size is consumed only for algorithm='ball_tree' or 'kd_tree'; " "use leaf_size=30 or choose a tree algorithm")
        if self.metric != "minkowski" and self.power != 2:
            raise ValueError("power is consumed only for metric='minkowski'; use power=2 or choose minkowski")
        return self


class StochasticGradientDescentSettings(StrictModel):
    model_config = ConfigDict(json_schema_extra=_stochastic_gradient_descent_json_schema)

    type: Literal["stochastic_gradient_descent"] = "stochastic_gradient_descent"
    loss: Literal["log_loss", "modified_huber"] = "log_loss"
    penalty: Literal["l2", "l1", "elasticnet", "none"] = "l2"
    l1_ratio: float = Field(0.15, ge=0, le=1)
    alpha: float = Field(0.0001, gt=0)
    fit_intercept: bool = True
    maximum_iterations: int = Field(1000, ge=1, le=100_000)
    tolerance: float = Field(0.001, gt=0)
    shuffle: bool = True
    learning_rate: Literal["constant", "optimal", "invscaling", "adaptive"] = "optimal"
    initial_learning_rate: float = Field(0.0, ge=0)
    power: float = Field(0.5, gt=0)
    early_stopping: bool = False
    validation_fraction: float = Field(0.1, gt=0, lt=1)
    iterations_without_improvement: int = Field(5, ge=1)

    @model_validator(mode="after")
    def validate_l1_ratio(self) -> "StochasticGradientDescentSettings":
        if self.penalty != "elasticnet" and self.l1_ratio != 0.15:
            raise ValueError("l1_ratio is consumed only for penalty='elasticnet'; " "use l1_ratio=0.15 or choose elasticnet")
        return self


class AdaBoostSettings(StrictModel):
    type: Literal["adaboost"] = "adaboost"
    number_of_estimators: int = Field(100, ge=1, le=100_000)
    learning_rate: float = Field(0.1, gt=0)
    maximum_depth: int = Field(3, ge=1, le=100_000)


ClassificationModelSettings = Annotated[
    Union[
        LogisticRegressionSettings,
        SupportVectorMachineSettings,
        DecisionTreeSettings,
        RandomForestSettings,
        ExtraTreesSettings,
        XGBoostSettings,
        MultiLayerPerceptronSettings,
        GradientBoostingSettings,
        KNearestNeighborsSettings,
        StochasticGradientDescentSettings,
        AdaBoostSettings,
    ],
    Field(discriminator="type"),
]


class LinearRegressionSettings(StrictModel):
    type: Literal["linear_regression"] = "linear_regression"
    fit_intercept: bool = True


class PolynomialRegressionSettings(StrictModel):
    type: Literal["polynomial_regression"] = "polynomial_regression"
    degree: int = Field(2, ge=1, le=20)
    interaction_only: bool = False
    include_bias: bool = True


class RegressionDecisionTreeSettings(StrictModel):
    type: Literal["decision_tree"] = "decision_tree"
    criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"] = "squared_error"
    maximum_depth: int = Field(4, ge=1, le=100_000)
    minimum_samples_split: int = Field(2, ge=2)
    minimum_samples_leaf: int = Field(1, ge=1)
    maximum_features: int = Field(1, ge=1)


class RegressionGradientBoostingSettings(StrictModel):
    type: Literal["gradient_boosting"] = "gradient_boosting"
    number_of_estimators: int = Field(100, ge=1, le=100_000)
    learning_rate: float = Field(0.1, gt=0)
    maximum_depth: int = Field(4, ge=1, le=100_000)
    minimum_samples_split: int = Field(2, ge=2)
    minimum_samples_leaf: int = Field(1, ge=1)
    maximum_features: int = Field(1, ge=1)
    subsample: float = Field(1.0, gt=0, le=1)
    loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = "squared_error"


class RegressionXGBoostSettings(StrictModel):
    type: Literal["xgboost"] = "xgboost"
    number_of_estimators: int = Field(100, ge=1, le=100_000)
    learning_rate: float = Field(0.01, gt=0)
    maximum_depth: int = Field(4, ge=1, le=100_000)
    subsample: float = Field(1.0, gt=0, le=1)
    column_subsample: float = Field(1.0, gt=0, le=1)
    gamma: float = Field(0.0, ge=0)
    tree_method: Literal["auto", "exact", "approx", "hist", "gpu_hist"] = "auto"
    l1_regularization: float = Field(0.0, ge=0)
    l2_regularization: float = Field(1.0, ge=0)
    base_score: float | None = None
    booster: Literal["gbtree", "gblinear", "dart"] = "gbtree"
    column_subsample_by_level: float = Field(1.0, gt=0, le=1)
    column_subsample_by_node: float = Field(1.0, gt=0, le=1)
    importance_type: Literal["gain", "weight", "cover", "total_gain", "total_cover"] = "gain"
    maximum_delta_step: float = Field(0.0, ge=0)
    minimum_child_weight: float = Field(1.0, ge=0)
    number_of_jobs: int = 1
    verbosity: int = Field(1, ge=0, le=3)

    @field_validator("number_of_jobs")
    @classmethod
    def validate_number_of_jobs(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("must be -1 or a positive integer")
        return value


class LassoRegressionSettings(StrictModel):
    type: Literal["lasso_regression"] = "lasso_regression"
    alpha: float = Field(0.01, ge=0)
    fit_intercept: bool = True
    maximum_iterations: int = Field(1000, ge=1, le=100_000)
    tolerance: float = Field(0.0001, gt=0)
    selection: Literal["cyclic", "random"] = "cyclic"


class ElasticNetSettings(StrictModel):
    type: Literal["elastic_net"] = "elastic_net"
    alpha: float = Field(1.0, ge=0)
    l1_ratio: float = Field(0.5, ge=0, le=1)
    fit_intercept: bool = True
    maximum_iterations: int = Field(1000, ge=1, le=100_000)
    tolerance: float = Field(0.0001, gt=0)
    selection: Literal["cyclic", "random"] = "cyclic"


class StochasticGradientDescentRegressionSettings(StrictModel):
    model_config = ConfigDict(json_schema_extra=_stochastic_gradient_descent_json_schema)

    type: Literal["stochastic_gradient_descent"] = "stochastic_gradient_descent"
    loss: Literal["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"] = "squared_error"
    penalty: Literal["l2", "l1", "elasticnet", "none"] = "l2"
    l1_ratio: float = Field(0.15, ge=0, le=1)
    alpha: float = Field(0.0001, gt=0)
    fit_intercept: bool = True
    maximum_iterations: int = Field(1000, ge=1, le=100_000)
    tolerance: float = Field(0.001, gt=0)
    shuffle: bool = True
    learning_rate: Literal["constant", "optimal", "invscaling", "adaptive"] = "invscaling"
    initial_learning_rate: float = Field(0.01, gt=0)
    power: float = Field(0.25, gt=0)

    @model_validator(mode="after")
    def validate_l1_ratio(self) -> "StochasticGradientDescentRegressionSettings":
        if self.penalty != "elasticnet" and self.l1_ratio != 0.15:
            raise ValueError("l1_ratio is consumed only for penalty='elasticnet'; " "use l1_ratio=0.15 or choose elasticnet")
        return self


class BayesianRidgeSettings(StrictModel):
    type: Literal["bayesian_ridge"] = "bayesian_ridge"
    tolerance: float = Field(0.0001, gt=0)
    alpha_1: float = Field(0.000001, gt=0)
    alpha_2: float = Field(0.000001, gt=0)
    lambda_1: float = Field(0.000001, gt=0)
    lambda_2: float = Field(0.000001, gt=0)
    alpha_initial: float = Field(1.0, gt=0)
    lambda_initial: float = Field(1.0, gt=0)
    compute_score: bool = False
    fit_intercept: bool = True
    copy_x: bool = True
    verbose: bool = False


class RidgeRegressionSettings(StrictModel):
    type: Literal["ridge_regression"] = "ridge_regression"
    alpha: float = Field(0.01, ge=0)
    fit_intercept: bool = True
    maximum_iterations: int = Field(1000, ge=1, le=100_000)
    tolerance: float = Field(0.0001, gt=0)


class AffineTargetTransformation(StrictModel):
    """Paper-agnostic unit or reference transformation for one numeric target."""

    model_config = ConfigDict(json_schema_extra=_affine_target_transformation_json_schema)

    scale: float = Field(
        1.0,
        allow_inf_nan=False,
        description="Finite non-zero multiplicative target transformation.",
    )
    offset: float = Field(0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_non_degenerate_scale(self) -> "AffineTargetTransformation":
        if self.scale == 0:
            raise ValueError("target transformation scale must be non-zero")
        return self


RegressionModelSettings = Annotated[
    Union[
        LinearRegressionSettings,
        PolynomialRegressionSettings,
        KNearestNeighborsSettings,
        SupportVectorMachineSettings,
        RegressionDecisionTreeSettings,
        RandomForestSettings,
        ExtraTreesSettings,
        RegressionGradientBoostingSettings,
        RegressionXGBoostSettings,
        MultiLayerPerceptronSettings,
        LassoRegressionSettings,
        ElasticNetSettings,
        StochasticGradientDescentRegressionSettings,
        BayesianRidgeSettings,
        RidgeRegressionSettings,
    ],
    Field(discriminator="type"),
]


class KMeansClusteringSettings(StrictModel):
    type: Literal["kmeans"] = "kmeans"
    number_of_clusters: int = Field(3, ge=2)
    initialization: Literal["k-means++", "random"] = "k-means++"
    maximum_iterations: int = Field(300, ge=1, le=100_000)
    tolerance: float = Field(0.0005, gt=0)
    algorithm: Literal["auto", "full", "elkan"] = "elkan"


class DBSCANClusteringSettings(StrictModel):
    model_config = ConfigDict(json_schema_extra=_dbscan_json_schema)

    type: Literal["dbscan"] = "dbscan"
    epsilon: float = Field(0.5, gt=0)
    minimum_samples: int = Field(5, ge=1)
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    metric: Literal[
        "euclidean",
        "l2",
        "minkowski",
        "p",
        "manhattan",
        "cityblock",
        "l1",
        "chebyshev",
        "infinity",
        "seuclidean",
        "mahalanobis",
        "hamming",
        "canberra",
        "braycurtis",
        "jaccard",
        "dice",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "haversine",
        "cosine",
        "correlation",
    ] = "euclidean"
    leaf_size: int = Field(30, ge=1)
    power: int | None = Field(
        None,
        ge=1,
        description="Minkowski distance power; required only when metric='minkowski'.",
    )

    @model_validator(mode="after")
    def validate_metric(self) -> "DBSCANClusteringSettings":
        kd_tree_metrics = {
            "euclidean",
            "l2",
            "minkowski",
            "p",
            "manhattan",
            "cityblock",
            "l1",
            "chebyshev",
            "infinity",
        }
        ball_tree_metrics = kd_tree_metrics | {
            "seuclidean",
            "mahalanobis",
            "hamming",
            "canberra",
            "braycurtis",
            "jaccard",
            "dice",
            "rogerstanimoto",
            "russellrao",
            "sokalmichener",
            "sokalsneath",
            "haversine",
        }
        general_metrics = {
            "euclidean",
            "manhattan",
            "chebyshev",
            "minkowski",
            "cosine",
            "correlation",
        }
        allowed = kd_tree_metrics if self.algorithm == "kd_tree" else ball_tree_metrics if self.algorithm == "ball_tree" else general_metrics
        if self.metric not in allowed:
            raise ValueError(f"metric {self.metric!r} is not offered by the CLI for algorithm {self.algorithm!r}")
        if self.metric == "minkowski" and self.power is None:
            raise ValueError("power is required for the Minkowski metric")
        if self.metric != "minkowski" and self.power is not None:
            raise ValueError("power is only valid for the Minkowski metric")
        return self


class AgglomerativeClusteringSettings(StrictModel):
    type: Literal["agglomerative"] = "agglomerative"
    number_of_clusters: int = Field(3, ge=2)
    linkage: Literal["ward", "complete", "average", "single"] = "ward"


class AffinityPropagationClusteringSettings(StrictModel):
    type: Literal["affinity_propagation"] = "affinity_propagation"
    damping: float = Field(0.5, ge=0.5, lt=1)
    maximum_iterations: int = Field(200, ge=1, le=100_000)
    convergence_iterations: int = Field(15, ge=1, le=100_000)
    affinity: Literal["euclidean", "precomputed"] = "euclidean"


class MeanShiftClusteringSettings(StrictModel):
    type: Literal["mean_shift"] = "mean_shift"
    bandwidth: int | None = Field(None, ge=1)
    cluster_all: bool = True
    bin_seeding: bool = False
    minimum_bin_frequency: int = Field(1, ge=1)
    number_of_jobs: int = Field(1, ge=1)
    maximum_iterations: int = Field(300, ge=1, le=100_000)


ClusteringModelSettings = Annotated[
    Union[
        KMeansClusteringSettings,
        DBSCANClusteringSettings,
        AgglomerativeClusteringSettings,
        AffinityPropagationClusteringSettings,
        MeanShiftClusteringSettings,
    ],
    Field(discriminator="type"),
]


class PCADecompositionSettings(StrictModel):
    type: Literal["pca"] = "pca"
    number_of_components: int = Field(2, ge=1)
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto"


class TSNEDecompositionSettings(StrictModel):
    type: Literal["tsne"] = "tsne"
    number_of_components: int = Field(2, ge=1)
    perplexity: int = Field(30, ge=1)
    learning_rate: float = Field(200.0, gt=0)
    number_of_iterations: int = Field(1000, ge=250, le=100_000)
    early_exaggeration: float = Field(12.0, ge=1)


class MDSDecompositionSettings(StrictModel):
    type: Literal["mds"] = "mds"
    number_of_components: int = Field(2, ge=1)
    metric: bool = True
    number_of_initializations: int = Field(4, ge=1, le=100_000)
    maximum_iterations: int = Field(300, ge=1, le=100_000)


DecompositionModelSettings = Annotated[
    Union[
        PCADecompositionSettings,
        TSNEDecompositionSettings,
        MDSDecompositionSettings,
    ],
    Field(discriminator="type"),
]


class IsolationForestAnomalyDetectionSettings(StrictModel):
    type: Literal["isolation_forest"] = "isolation_forest"
    number_of_estimators: int = Field(100, ge=1, le=100_000)
    contamination: float = Field(0.3, gt=0, le=0.5)
    maximum_features: int = Field(1, ge=1)
    bootstrap: bool = Field(
        False,
        description="Whether isolation trees use bootstrap samples.",
    )
    maximum_samples: Literal["auto"] | Annotated[int, Field(ge=1)] = Field(
        "auto",
        description=("Samples drawn per isolation tree. 'auto' resolves to min(256, n_samples); " "bootstrap independently controls sampling with or without replacement."),
    )


class LocalOutlierFactorAnomalyDetectionSettings(StrictModel):
    type: Literal["local_outlier_factor"] = "local_outlier_factor"
    number_of_neighbors: int = Field(20, ge=1)
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = Field(30, ge=1)
    metric: Literal["euclidean", "manhattan", "minkowski"] = "minkowski"
    power: float = Field(2.0, gt=0)
    contamination: float = Field(0.3, gt=0, le=0.5)
    number_of_jobs: int = 1
    detection_mode: Literal["training_outlier", "novelty_detection"] = "novelty_detection"

    @field_validator("number_of_jobs")
    @classmethod
    def validate_number_of_jobs(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("must be -1 or a positive integer")
        return value


AnomalyDetectionModelSettings = Annotated[
    Union[
        IsolationForestAnomalyDetectionSettings,
        LocalOutlierFactorAnomalyDetectionSettings,
    ],
    Field(discriminator="type"),
]


class ArtifactRequirement(StrictModel):
    """One caller-declared scientific output that must be present after execution."""

    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$", max_length=120)
    scientific_type: str = Field(min_length=1, max_length=120)
    output_role: str | None = Field(None, min_length=1, max_length=120)
    required: bool = True
    category: Literal["artifacts", "metrics", "parameters", "summary"] | None = None
    media_types: tuple[str, ...] = Field(default=(), max_length=16)
    expected_relative_path: str | None = Field(None, min_length=1, max_length=512)
    path_pattern: str | None = Field(None, min_length=1, max_length=512)
    minimum_count: int = Field(1, ge=0, le=10_000)
    maximum_count: int | None = Field(None, ge=1, le=10_000)
    required_json_keys: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("expected_relative_path", "path_pattern")
    @classmethod
    def validate_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/").strip()
        candidate = Path(normalized)
        if not normalized or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("must be a safe relative output path")
        return candidate.as_posix()

    @field_validator("media_types", "required_json_keys")
    @classmethod
    def validate_unique_artifact_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized):
            raise ValueError("must contain non-blank single-line values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_counts(self) -> "ArtifactRequirement":
        if self.required and self.minimum_count < 1:
            raise ValueError("required artifact requirements need minimum_count >= 1")
        if self.maximum_count is not None and self.maximum_count < self.minimum_count:
            raise ValueError("maximum_count must not be smaller than minimum_count")
        return self


class TimeSeriesArtifactRequirement(StrictModel):
    """Compact evidence requirement for Time Series outputs."""

    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$", max_length=120)
    scientific_type: str = Field(min_length=1, max_length=120)
    path_pattern: str | None = Field(None, min_length=1, max_length=512)
    count: int = Field(1, ge=1, le=10_000)
    required_json_keys: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("path_pattern")
    @classmethod
    def validate_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/").strip()
        candidate = Path(normalized)
        if not normalized or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("must be a safe relative output path")
        return candidate.as_posix()

    @field_validator("required_json_keys")
    @classmethod
    def validate_json_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized):
            raise ValueError("must contain non-blank single-line values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicates")
        return normalized


class EvaluationContract(StrictModel):
    """Scientific evaluation intent, distinct from application/inference data."""

    model_config = ConfigDict(json_schema_extra=_evaluation_json_schema)

    mode: Literal[
        "cli_default",
        "holdout",
        "external_labeled",
        "cross_validation",
        "quality_report",
        "reference_comparison",
    ] = "cli_default"
    metrics: tuple[str, ...] = Field(default=(), max_length=64)
    metric_artifact_bindings: dict[str, str] = Field(default_factory=dict, max_length=64)
    split_strategy: Literal["cli_default", "random_holdout", "stratified_holdout"] = "cli_default"
    evaluation_dataset_path: Path | None = _dataset_path_field("External labeled evaluation dataset path.")
    evaluation_dataset: DatasetReference | None = None
    external_identifier_column: str | None = Field(None, min_length=1, max_length=128)
    folds: int | None = Field(None, ge=2, le=100)
    required_artifact_ids: tuple[str, ...] = Field(default=(), max_length=64)
    class_order: tuple[str, ...] = Field(default=(), max_length=256)
    confusion_matrix_normalization: Literal["none", "true", "predicted", "all"] | None = None

    @field_validator("metrics", "required_artifact_ids", "class_order")
    @classmethod
    def validate_unique_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized):
            raise ValueError("must contain non-blank single-line names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_evaluation_design(self) -> "EvaluationContract":
        _validate_dataset_choice(
            self.evaluation_dataset_path,
            self.evaluation_dataset,
            "evaluation_dataset",
            required=False,
        )
        has_dataset = self.evaluation_dataset_path is not None or self.evaluation_dataset is not None
        if self.mode == "external_labeled" and not has_dataset:
            raise ValueError("external_labeled evaluation requires an evaluation dataset")
        if self.mode != "external_labeled" and has_dataset:
            raise ValueError("an evaluation dataset is used only by external_labeled evaluation")
        if self.mode != "external_labeled" and self.external_identifier_column is not None:
            raise ValueError("external_identifier_column is used only by external_labeled evaluation")
        if self.mode == "cross_validation" and self.folds is None:
            raise ValueError("cross_validation evaluation requires folds")
        if self.mode not in {"cross_validation", "holdout", "external_labeled"} and self.folds is not None:
            raise ValueError("folds is used only when a supervised adapter produces cross-validation evidence")
        if self.mode != "holdout" and self.split_strategy != "cli_default":
            raise ValueError("an explicit split_strategy is used only by holdout evaluation")
        unknown_metrics = sorted(set(self.metric_artifact_bindings) - set(self.metrics))
        if unknown_metrics:
            raise ValueError(f"metric artifact bindings reference undeclared metrics: {unknown_metrics}")
        invalid_ids = sorted(artifact_id for artifact_id in self.metric_artifact_bindings.values() if not re.fullmatch(r"[a-z][a-z0-9_.-]+", artifact_id) or len(artifact_id) > 120)
        if invalid_ids:
            raise ValueError(f"metric artifact bindings contain invalid requirement ids: {invalid_ids}")
        return self


class EnvironmentContract(StrictModel):
    """Exact desired CLI runtime identity, separate from observed provenance."""

    expected_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    python: str | None = Field(None, min_length=1, max_length=80)
    geochemistrypi: str | None = Field(None, min_length=1, max_length=80)
    mcp: str | None = Field(None, min_length=1, max_length=80)
    platform: str | None = Field(None, min_length=1, max_length=255)
    runtime: str | None = Field(None, min_length=1, max_length=80)
    dependency_versions: dict[str, str] = Field(default_factory=dict, max_length=2_000)

    @field_validator("python", "geochemistrypi", "mcp", "platform", "runtime")
    @classmethod
    def validate_environment_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("environment values must be non-blank single-line exact values")
        return normalized

    @field_validator("dependency_versions")
    @classmethod
    def validate_dependency_versions(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for package, version in value.items():
            package_name = package.strip().lower().replace("_", "-")
            exact_version = version.strip()
            if not package_name or not exact_version or "\n" in package_name or "\n" in exact_version:
                raise ValueError("dependency versions must use non-blank single-line names and exact values")
            normalized[package_name] = exact_version
        return normalized


class EnvironmentProfileContract(StrictModel):
    """Named exact runtime requirements selected before a CLI process can start."""

    profile_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    expected_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    python: str | None = Field(None, min_length=1, max_length=80)
    geochemistrypi: str | None = Field(None, min_length=1, max_length=80)
    mcp: str | None = Field(None, min_length=1, max_length=80)
    package_versions: dict[str, str] = Field(default_factory=dict, max_length=2_000)
    runtime_constraints: dict[str, str] = Field(default_factory=dict, max_length=8)

    @field_validator("python", "geochemistrypi", "mcp")
    @classmethod
    def validate_exact_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("environment profile versions must be non-blank exact values")
        return normalized

    @field_validator("package_versions")
    @classmethod
    def validate_package_versions(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for package, version in value.items():
            package_name = package.strip().lower().replace("_", "-")
            exact_version = version.strip()
            if not package_name or not exact_version or any(token in exact_version for token in "<>=!~,*\n\r"):
                raise ValueError("environment profile packages require normalized names and exact versions")
            normalized[package_name] = exact_version
        return normalized

    @field_validator("runtime_constraints")
    @classmethod
    def validate_runtime_constraints(cls, value: dict[str, str]) -> dict[str, str]:
        supported = {"kind", "python_implementation", "platform", "cli_executable_sha256"}
        unknown = sorted(set(value) - supported)
        if unknown:
            raise ValueError(f"unsupported runtime constraint keys: {unknown}")
        normalized: dict[str, str] = {}
        for name, expected in value.items():
            exact = expected.strip()
            if not exact or "\n" in exact or "\r" in exact:
                raise ValueError("runtime constraints require non-blank exact values")
            if name == "cli_executable_sha256" and not re.fullmatch(r"[0-9a-f]{64}", exact):
                raise ValueError("cli_executable_sha256 must be a lowercase SHA256 value")
            normalized[name] = exact
        return normalized


class ReproducibilityContract(StrictModel):
    """Desired seeds and dependency constraints; observed values belong in the manifest."""

    split_seed: int | None = Field(None, ge=0, le=2**32 - 1)
    model_seed: int | None = Field(None, ge=0, le=2**32 - 1)
    tuning_seed: int | None = Field(None, ge=0, le=2**32 - 1)
    dependency_profile: str | None = Field(None, min_length=1, max_length=120)
    dependency_constraints: dict[str, str] = Field(default_factory=dict, max_length=32)
    environment: EnvironmentContract = Field(default_factory=EnvironmentContract)
    model_parameter_assertions: dict[str, ModelParameterValue] = Field(default_factory=dict, max_length=128)
    deterministic_policy: Literal[
        "adapter_default",
        "fixed_seed_required",
        "fixed_seed_and_dependency_required",
        "nondeterministic_allowed",
    ] = "adapter_default"

    @field_validator("dependency_constraints")
    @classmethod
    def validate_dependency_constraints(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for package, constraint in value.items():
            package_name = package.strip()
            version_constraint = constraint.strip()
            if not package_name or not version_constraint or "\n" in package_name or "\n" in version_constraint:
                raise ValueError("dependency constraints must use non-blank single-line names and values")
            normalized[package_name] = version_constraint
        return normalized

    @field_validator("model_parameter_assertions")
    @classmethod
    def validate_model_parameter_assertions(cls, value: dict[str, ModelParameterValue]) -> dict[str, ModelParameterValue]:
        normalized: dict[str, ModelParameterValue] = {}
        for parameter, expected in value.items():
            name = parameter.strip()
            if not name or "\n" in name or "\r" in name:
                raise ValueError("model parameter assertions require non-blank single-line names")
            normalized[name] = expected
        return normalized


class TimeSeriesEvaluationContract(StrictModel):
    """Evaluation intent that is meaningful for the binned Time Series adapter."""

    mode: Literal["cli_default", "reference_comparison"] = "cli_default"
    required_artifact_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("required_artifact_ids")
    @classmethod
    def validate_unique_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized):
            raise ValueError("must contain non-blank single-line names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicates")
        return normalized


class TimeSeriesEnvironmentContract(StrictModel):
    """Compact exact environment identity for Time Series requests."""

    expected_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    dependency_versions: dict[str, str] = Field(default_factory=dict, max_length=2_000)


class TimeSeriesReproducibilityContract(StrictModel):
    """Time Series reproducibility controls; its effective seed is the request seed."""

    environment: TimeSeriesEnvironmentContract = Field(default_factory=TimeSeriesEnvironmentContract)


class ScientificRequest(StrictModel):
    """Additive scientific requirements normalized before an existing CLI adapter is selected."""

    scientific_contract_version: Literal[2] = 2
    evaluation: EvaluationContract = Field(default_factory=EvaluationContract)
    reproducibility: ReproducibilityContract = Field(default_factory=ReproducibilityContract)
    environment_profile: EnvironmentProfileContract | None = None
    artifact_requirements: tuple[ArtifactRequirement, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def validate_requirement_references(self) -> "ScientificRequest":
        legacy_environment = self.reproducibility.environment.model_dump(mode="json")
        legacy_environment_specified = any(value not in (None, {}, (), []) for value in legacy_environment.values())
        if self.environment_profile is not None and legacy_environment_specified:
            raise ValueError("environment_profile replaces reproducibility.environment; provide only one environment contract")
        identifiers = tuple(item.requirement_id for item in self.artifact_requirements)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("artifact requirement ids must be unique")
        unknown = sorted(set(self.evaluation.required_artifact_ids) - set(identifiers))
        if unknown:
            raise ValueError(f"evaluation references unknown artifact requirement ids: {unknown}")
        metric_artifact_ids = set(getattr(self.evaluation, "metric_artifact_bindings", {}).values())
        unknown_metric_artifacts = sorted(metric_artifact_ids - set(identifiers))
        if unknown_metric_artifacts:
            raise ValueError(f"metric requirements reference unknown artifact requirement ids: {unknown_metric_artifacts}")
        return self


class ClassificationRequest(ScientificRequest):
    """Scientific inputs for the validated classification reference workflow."""

    model_config = ConfigDict(json_schema_extra=_classification_json_schema)

    task: Literal["classification"] = "classification"
    training_dataset_path: Path | None = _dataset_path_field("Top-level legacy path alternative to training_dataset; provide exactly one.")
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    target_column: ColumnName
    application_dataset_path: Path | None = _dataset_path_field("Optional top-level path alternative to application_dataset; provide at most one.")
    application_dataset: DatasetReference | None = None
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: MissingValueHandling = Field(default_factory=RejectMissingValues)
    engineered_features: tuple[EngineeredFeature, ...] = Field(default=(), max_length=20)
    label_customization: LabelCustomization = Field(default_factory=EncodeOriginalLabels)
    metric_average: Literal["auto", "binary", "micro", "macro", "weighted"] = Field(
        "auto",
        description=(
            "Metric averaging contract. 'auto' preserves backward-compatible native " "binary-versus-multiclass resolution from observed training labels; " "'binary' requires positive_label."
        ),
    )
    positive_label: SemanticLabel | None = Field(
        None,
        description=("Typed semantic positive class used by binary metrics and probability " "outputs. Type is part of the identity: numeric 1 and string '1' are " "different labels."),
    )
    scaling: ScalingMethod = "standardization"
    feature_selection: FeatureSelection = Field(default_factory=NoFeatureSelection)
    sample_balancing: Literal["none"] = "none"
    test_ratio: float = Field(0.2, gt=0, lt=1)
    tuning: Literal["manual", "automl"] = "manual"
    model: ClassificationModelSettings = Field(default_factory=LogisticRegressionSettings)

    @model_validator(mode="after")
    def validate_metric_semantics(self) -> "ClassificationRequest":
        """Require an explicit semantic positive class only for explicit binary metrics."""
        if self.metric_average == "binary" and self.positive_label is None:
            raise ValueError("metric_average='binary' requires an explicit positive_label")
        if self.metric_average in {"micro", "macro", "weighted"} and self.positive_label is not None:
            raise ValueError("positive_label is valid only when metric_average is 'auto' or 'binary'")
        return self

    @field_validator("experiment_name", "run_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        """Keep CLI directory names safe and portable."""
        normalized = value.strip()
        if normalized in {"", ".", ".."} or _UNSAFE_PATH_CHARACTERS.search(normalized):
            raise ValueError("contains characters that are unsafe in an output directory name")
        if normalized.endswith((" ", ".")):
            raise ValueError("must not end with a space or period")
        if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("uses a Windows-reserved output directory name")
        return normalized

    @field_validator("identifier_column", "target_column")
    @classmethod
    def validate_required_column_name(cls, value: str) -> str:
        """Reject blank semantic column names."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require uniquely named, non-blank features."""
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized):
            raise ValueError("must not contain blank column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @model_validator(mode="after")
    def validate_column_roles(self) -> "ClassificationRequest":
        """Prevent a source column from receiving conflicting roles."""
        _validate_dataset_choice(self.training_dataset_path, self.training_dataset, "training_dataset")
        _validate_dataset_choice(
            self.application_dataset_path,
            self.application_dataset,
            "application_dataset",
            required=False,
        )
        if self.identifier_column == self.target_column:
            raise ValueError("identifier_column and target_column must be different")
        if self.identifier_column in self.feature_columns:
            raise ValueError("identifier_column must not also be a feature")
        if self.target_column in self.feature_columns:
            raise ValueError("target_column must not also be a feature")
        engineered_names = [feature.name for feature in self.engineered_features]
        if len(engineered_names) != len(set(engineered_names)):
            raise ValueError("engineered feature names must be unique")
        source_names = {
            self.identifier_column,
            self.target_column,
            *self.feature_columns,
        }
        conflicts = sorted(set(engineered_names) & source_names)
        if conflicts:
            raise ValueError(f"engineered feature names conflict with source columns: {conflicts}")
        if self.tuning == "automl":
            manual_fields = sorted(set(self.model.model_fields_set) - {"type"})
            if manual_fields:
                raise ValueError(f"manual model settings are not used when tuning='automl': {manual_fields}; send only the model type")
        return self


class RegressionRequest(ScientificRequest):
    """Scientific inputs for one validated numeric-target regression run."""

    model_config = ConfigDict(json_schema_extra=_regression_json_schema)

    task: Literal["regression"] = "regression"
    training_dataset_path: Path | None = _dataset_path_field("Top-level legacy path alternative to training_dataset; provide exactly one.")
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    target_column: ColumnName | None = Field(
        default=None,
        description="Backward-compatible single-target field. Do not combine it with target_columns.",
    )
    target_columns: tuple[ColumnName, ...] = Field(
        default=(),
        max_length=256,
        description="One or more numeric regression targets. Do not combine this field with target_column.",
    )
    target_transformations: dict[ColumnName, AffineTargetTransformation] = Field(
        default_factory=dict,
        max_length=256,
    )
    application_dataset_path: Path | None = _dataset_path_field("Optional top-level path alternative to application_dataset; provide at most one.")
    application_dataset: DatasetReference | None = None
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: MissingValueHandling = Field(default_factory=RejectMissingValues)
    engineered_features: tuple[EngineeredFeature, ...] = Field(default=(), max_length=20)
    scaling: ScalingMethod = "standardization"
    feature_selection: FeatureSelection = Field(default_factory=NoFeatureSelection)
    test_ratio: float = Field(0.2, gt=0, lt=1)
    tuning: Literal["manual", "automl"] = "manual"
    model: RegressionModelSettings = Field(default_factory=LinearRegressionSettings)

    @field_validator("experiment_name", "run_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        normalized = value.strip()
        if normalized in {"", ".", ".."} or _UNSAFE_PATH_CHARACTERS.search(normalized):
            raise ValueError("contains characters that are unsafe in an output directory name")
        if normalized.endswith((" ", ".")):
            raise ValueError("must not end with a space or period")
        if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("uses a Windows-reserved output directory name")
        return normalized

    @field_validator("identifier_column")
    @classmethod
    def validate_required_column_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("target_column")
    @classmethod
    def validate_legacy_target_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized):
            raise ValueError("must not contain blank column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @field_validator("target_columns")
    @classmethod
    def validate_target_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized):
            raise ValueError("must not contain blank column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @property
    def resolved_target_columns(self) -> tuple[str, ...]:
        """Return the uniform one-or-more target contract for old and new clients."""
        if self.target_columns:
            return self.target_columns
        return (self.target_column,) if self.target_column is not None else ()

    @model_validator(mode="after")
    def validate_regression_contract(self) -> "RegressionRequest":
        _validate_dataset_choice(self.training_dataset_path, self.training_dataset, "training_dataset")
        _validate_dataset_choice(
            self.application_dataset_path,
            self.application_dataset,
            "application_dataset",
            required=False,
        )
        if (self.target_column is None) == (not self.target_columns):
            raise ValueError("provide exactly one of target_column or target_columns")
        targets = self.resolved_target_columns
        unknown_target_transformations = sorted(set(self.target_transformations) - set(targets))
        if unknown_target_transformations:
            raise ValueError("target transformations reference non-target columns: " f"{unknown_target_transformations}")
        if self.identifier_column in targets:
            raise ValueError("identifier_column and regression targets must be different")
        if self.identifier_column in self.feature_columns:
            raise ValueError("identifier_column must not also be a feature")
        target_features = sorted(set(targets) & set(self.feature_columns))
        if target_features:
            raise ValueError(f"regression targets must not also be features: {target_features}")
        engineered_names = [feature.name for feature in self.engineered_features]
        if len(engineered_names) != len(set(engineered_names)):
            raise ValueError("engineered feature names must be unique")
        source_names = {
            self.identifier_column,
            *targets,
            *self.feature_columns,
        }
        conflicts = sorted(set(engineered_names) & source_names)
        if conflicts:
            raise ValueError(f"engineered feature names conflict with source columns: {conflicts}")
        if len(targets) > 1 and self.feature_selection.method != "none":
            raise ValueError("multiple-target regression requires feature_selection.method='none' because the public CLI selectors are univariate")
        if self.tuning == "automl" and self.model.type in MODELS_WITHOUT_AUTOML:
            raise ValueError(f"the public CLI does not offer AutoML for {self.model.type}")
        if self.tuning == "automl":
            manual_fields = sorted(set(self.model.model_fields_set) - {"type"})
            if manual_fields:
                raise ValueError(f"manual model settings are not used when tuning='automl': {manual_fields}; send only the model type")
        return self


class ClusteringRequest(ScientificRequest):
    """Scientific inputs for one validated unsupervised clustering run."""

    model_config = ConfigDict(json_schema_extra=_unsupervised_json_schema)

    task: Literal["clustering"] = "clustering"
    training_dataset_path: Path | None = _dataset_path_field("Top-level legacy path alternative to training_dataset; provide exactly one.")
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: UnsupervisedMissingValueHandling = Field(default_factory=RejectMissingValues)
    engineered_features: tuple[EngineeredFeature, ...] = Field(default=(), max_length=20)
    scaling: ScalingMethod = "standardization"
    model: ClusteringModelSettings = Field(default_factory=KMeansClusteringSettings)

    @field_validator("experiment_name", "run_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        normalized = value.strip()
        if normalized in {"", ".", ".."} or _UNSAFE_PATH_CHARACTERS.search(normalized):
            raise ValueError("contains characters that are unsafe in an output directory name")
        if normalized.endswith((" ", ".")):
            raise ValueError("must not end with a space or period")
        if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("uses a Windows-reserved output directory name")
        return normalized

    @field_validator("identifier_column")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized):
            raise ValueError("must not contain blank column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @model_validator(mode="after")
    def validate_clustering_contract(self) -> "ClusteringRequest":
        _validate_dataset_choice(self.training_dataset_path, self.training_dataset, "training_dataset")
        if self.identifier_column in self.feature_columns:
            raise ValueError("identifier_column must not also be a feature")
        engineered_names = [feature.name for feature in self.engineered_features]
        if len(engineered_names) != len(set(engineered_names)):
            raise ValueError("engineered feature names must be unique")
        source_names = {self.identifier_column, *self.feature_columns}
        conflicts = sorted(set(engineered_names) & source_names)
        if conflicts:
            raise ValueError(f"engineered feature names conflict with source columns: {conflicts}")
        if self.missing_values.method == "keep":
            raise ValueError("the public CLI exposes no clustering models when missing values remain unprocessed")
        return self


class DecompositionRequest(ScientificRequest):
    """Scientific inputs for one validated unsupervised decomposition run."""

    model_config = ConfigDict(json_schema_extra=_decomposition_json_schema)

    task: Literal["decomposition"] = "decomposition"
    mode: Literal["model", "embedding_label_overlay"] = "model"
    training_dataset_path: Path | None = _dataset_path_field("Top-level legacy path alternative to training_dataset; provide exactly one.")
    training_dataset: DatasetReference | None = None
    application_dataset_path: Path | None = _dataset_path_field("Optional top-level path alternative to application_dataset.")
    application_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    metadata_columns: tuple[ColumnName, ...] = Field(default=(), max_length=256)
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: UnsupervisedMissingValueHandling = Field(default_factory=RejectMissingValues)
    engineered_features: tuple[EngineeredFeature, ...] = Field(default=(), max_length=20)
    scaling: ScalingMethod = "standardization"
    model: DecompositionModelSettings = Field(default_factory=PCADecompositionSettings)
    coordinate_sheet: str = Field("0", min_length=1, max_length=128)
    label_sheet: str = Field("0", min_length=1, max_length=128)
    label_identifier_column: ColumnName | None = None
    label_column: ColumnName | None = None
    positive_label_values: tuple[str, ...] = Field(default=(), max_length=64)

    @property
    def mode_inapplicable_fields(self) -> frozenset[str]:
        """Fields omitted when the strict request is persisted and revalidated."""
        if self.mode == "embedding_label_overlay":
            return frozenset({"model", "model_selection"})
        return frozenset(
            {
                "coordinate_sheet",
                "label_sheet",
                "label_identifier_column",
                "label_column",
                "positive_label_values",
            }
        )

    @model_serializer(mode="wrap")
    def serialize_selected_mode(self, handler: Any) -> dict[str, Any]:
        """Keep ordinary model_dump output valid under the mode discriminator."""
        value = handler(self)
        for field in self.mode_inapplicable_fields:
            value.pop(field, None)
        return value

    @field_validator("experiment_name", "run_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        normalized = value.strip()
        if normalized in {"", ".", ".."} or _UNSAFE_PATH_CHARACTERS.search(normalized):
            raise ValueError("contains characters that are unsafe in an output directory name")
        if normalized.endswith((" ", ".")):
            raise ValueError("must not end with a space or period")
        if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("uses a Windows-reserved output directory name")
        return normalized

    @field_validator("identifier_column", "label_identifier_column", "label_column")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("feature_columns", "metadata_columns")
    @classmethod
    def validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized):
            raise ValueError("must not contain blank column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @model_validator(mode="after")
    def validate_decomposition_contract(self) -> "DecompositionRequest":
        _validate_dataset_choice(self.training_dataset_path, self.training_dataset, "training_dataset")
        if self.mode == "embedding_label_overlay":
            ignored_model_fields = sorted({"model", "model_selection"} & self.model_fields_set)
            if ignored_model_fields:
                raise ValueError("embedding_label_overlay does not fit a model and cannot accept " f"{ignored_model_fields}; remove them or change mode='model'")
            _validate_dataset_choice(
                self.application_dataset_path,
                self.application_dataset,
                "application_dataset",
            )
            if len(self.feature_columns) != 2:
                raise ValueError("embedding_label_overlay requires exactly two coordinate feature columns")
            if self.label_identifier_column is None or self.label_column is None:
                raise ValueError("embedding_label_overlay requires label_identifier_column and label_column")
            if not self.positive_label_values:
                raise ValueError("embedding_label_overlay requires positive_label_values")
            if self.label_column in self.feature_columns:
                raise ValueError("label_column must be distinct from coordinate feature columns")
            if self.label_column == self.identifier_column:
                raise ValueError("label_column must be distinct from the coordinate identifier")
            if self.label_identifier_column in self.feature_columns:
                raise ValueError("label_identifier_column must be distinct from coordinate feature columns")
            if self.metadata_columns:
                raise ValueError("embedding_label_overlay does not use metadata_columns")
            if self.engineered_features:
                raise ValueError("embedding_label_overlay does not perform feature engineering")
            if self.scaling != "none":
                raise ValueError("embedding_label_overlay requires scaling='none' because coordinates are already calculated")
            if self.missing_values.method != "error":
                raise ValueError("embedding_label_overlay requires complete explicitly supplied rows")
            if self.model_selection.mode != "single":
                raise ValueError("embedding_label_overlay is one artifact-composition producer")
            if self.world_map.enabled:
                raise ValueError("embedding_label_overlay does not use world-map rendering")
            return self
        ignored_overlay_fields = sorted(
            {
                "coordinate_sheet",
                "label_sheet",
                "label_identifier_column",
                "label_column",
                "positive_label_values",
            }
            & self.model_fields_set
        )
        if ignored_overlay_fields:
            raise ValueError("decomposition model mode cannot accept overlay-only fields " f"{ignored_overlay_fields}; remove them or change mode='embedding_label_overlay'")
        if self.application_dataset_path is not None or self.application_dataset is not None:
            raise ValueError("decomposition model mode does not accept an application dataset")
        if self.identifier_column in self.feature_columns:
            raise ValueError("identifier_column must not also be a feature")
        overlap = sorted(set(self.metadata_columns) & set(self.feature_columns))
        if overlap:
            raise ValueError(f"metadata_columns must not overlap feature_columns: {overlap}")
        if self.identifier_column in self.metadata_columns:
            raise ValueError("identifier_column must not also be metadata")
        engineered_names = [feature.name for feature in self.engineered_features]
        if len(engineered_names) != len(set(engineered_names)):
            raise ValueError("engineered feature names must be unique")
        source_names = {self.identifier_column, *self.feature_columns, *self.metadata_columns}
        conflicts = sorted(set(engineered_names) & source_names)
        if conflicts:
            raise ValueError(f"engineered feature names conflict with source columns: {conflicts}")
        if self.missing_values.method == "keep":
            raise ValueError("the public CLI exposes no decomposition models when missing values remain unprocessed")
        return self


class AnomalyDetectionRequest(ScientificRequest):
    """Scientific inputs for one validated unsupervised anomaly-detection run."""

    model_config = ConfigDict(json_schema_extra=_unsupervised_json_schema)

    task: Literal["anomaly_detection"] = "anomaly_detection"
    training_dataset_path: Path | None = _dataset_path_field("Top-level legacy path alternative to training_dataset; provide exactly one.")
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: UnsupervisedMissingValueHandling = Field(default_factory=RejectMissingValues)
    engineered_features: tuple[EngineeredFeature, ...] = Field(default=(), max_length=20)
    scaling: ScalingMethod = "standardization"
    model: AnomalyDetectionModelSettings = Field(default_factory=IsolationForestAnomalyDetectionSettings)

    @field_validator("experiment_name", "run_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        normalized = value.strip()
        if normalized in {"", ".", ".."} or _UNSAFE_PATH_CHARACTERS.search(normalized):
            raise ValueError("contains characters that are unsafe in an output directory name")
        if normalized.endswith((" ", ".")):
            raise ValueError("must not end with a space or period")
        if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("uses a Windows-reserved output directory name")
        return normalized

    @field_validator("identifier_column")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized):
            raise ValueError("must not contain blank column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @model_validator(mode="after")
    def validate_anomaly_detection_contract(self) -> "AnomalyDetectionRequest":
        _validate_dataset_choice(self.training_dataset_path, self.training_dataset, "training_dataset")
        if self.identifier_column in self.feature_columns:
            raise ValueError("identifier_column must not also be a feature")
        engineered_names = [feature.name for feature in self.engineered_features]
        if len(engineered_names) != len(set(engineered_names)):
            raise ValueError("engineered feature names must be unique")
        source_names = {self.identifier_column, *self.feature_columns}
        conflicts = sorted(set(engineered_names) & source_names)
        if conflicts:
            raise ValueError(f"engineered feature names conflict with source columns: {conflicts}")
        if self.missing_values.method == "keep":
            raise ValueError("the public CLI exposes no anomaly-detection models when missing values remain unprocessed")
        return self


class TimeSeriesRequest(ScientificRequest):
    """Generic binned Time Series request with mode-specific scientific roles."""

    model_config = ConfigDict(json_schema_extra=_time_series_json_schema)

    evaluation: TimeSeriesEvaluationContract = Field(default_factory=TimeSeriesEvaluationContract)
    reproducibility: TimeSeriesReproducibilityContract = Field(default_factory=TimeSeriesReproducibilityContract)
    artifact_requirements: tuple[TimeSeriesArtifactRequirement, ...] = Field(default=(), max_length=128)

    task: Literal["time_series"] = Field(
        "time_series",
        description="Select the generic Time Series workflow family.",
    )
    mode: Literal[
        "subaerial_proportion",
        "continuous",
        "element_mean",
        "reference_anomaly_series",
    ] = "subaerial_proportion"
    training_dataset_path: Path | None = Field(
        None,
        description=("Top-level local-path alternative to training_dataset; provide exactly one " f"of the two. {_DATASET_PATH_RESOLUTION_DESCRIPTION}"),
        json_schema_extra={
            "x-path-resolution-base": "mcp_startup_working_directory",
            "x-relative-path-must-remain-within-base": True,
        },
    )
    training_dataset: DatasetReference | None = Field(
        None,
        description="Required top-level input reference alternative; use this field, not dataset, and provide exactly one training input form.",
    )
    experiment_name: str = Field("Time Series", min_length=1, max_length=40)
    run_name: str = Field("Subaerial Proportion", min_length=1, max_length=40)
    bin_width: float | None = Field(
        None,
        gt=0,
        description="Required for binned modes and unused by reference_anomaly_series.",
    )
    iterations: int = Field(
        100,
        ge=1,
        le=10_000,
        description="Top-level bootstrap iterations; use iterations, not bootstrap_iterations.",
    )
    seed: int = Field(
        2025,
        ge=0,
        le=2**32 - 1,
        description="Top-level deterministic random seed; use seed, not random_seed.",
    )
    age_column: ColumnName = Field("R_AGE", description="Top-level central-age column role.")
    minimum_age_column: ColumnName | None = Field(
        None,
        description="Minimum-age column required by continuous mode.",
    )
    maximum_age_column: ColumnName = Field("R_MAX_AGE", description="Top-level comparison/maximum-age column role.")
    probability_column: ColumnName = Field("SBAP", description="Top-level subaerial-proportion probability column role.")
    value_column: ColumnName | None = Field(
        None,
        description="Numeric response column required by continuous mode.",
    )
    latitude_column: ColumnName = Field("LATITUDE", description="Top-level latitude column role.")
    longitude_column: ColumnName = Field("LONGITUDE", description="Top-level longitude column role.")
    element_columns: tuple[ColumnName, ...] = Field(
        default=(),
        max_length=128,
        description="One or more numeric value columns for element_mean mode.",
    )
    filter_column: ColumnName | None = Field(
        default=None,
        description="Optional numeric filter role for element_mean and continuous modes.",
    )
    filter_minimum: float | None = None
    filter_maximum: float | None = None
    aggregation: Literal["mean"] = "mean"
    uncertainty: Literal["standard_error"] = "standard_error"
    minimum_samples_per_bin: int = Field(1, ge=1)
    relative_value_two_sigma: float = Field(
        0.0,
        ge=0,
        description="Relative analytical two-sigma uncertainty for continuous values.",
    )
    age_unit: Literal["Ma", "Ga"] = "Ma"
    fit_curve: bool = True
    compact_y_axis: bool = False
    sheet: str = Field(
        "0",
        min_length=1,
        max_length=128,
        description="Observation Excel sheet index or name; ignored for CSV.",
    )
    time_column: ColumnName | None = Field(
        None,
        description="Observation date/time role for reference_anomaly_series mode.",
    )
    signal_columns: tuple[ColumnName, ...] = Field(
        default=(),
        max_length=128,
        description="Numeric signals for reference_anomaly_series mode.",
    )
    reference_label_column: ColumnName | None = None
    reference_positive_values: tuple[str, ...] = Field(default=(), max_length=64)
    reference_label_provenance: str = Field(
        "external_reference",
        min_length=1,
        max_length=128,
    )
    comparison_label_column: ColumnName | None = None
    comparison_positive_values: tuple[str, ...] = Field(default=(), max_length=64)
    comparison_label_provenance: Literal["calculated", "external", "reference"] = "calculated"
    event_dataset_path: Path | None = Field(
        None,
        description=("Optional event dataset for reference_anomaly_series evaluation and display. " f"{_DATASET_PATH_RESOLUTION_DESCRIPTION}"),
        json_schema_extra={
            "x-path-resolution-base": "mcp_startup_working_directory",
            "x-relative-path-must-remain-within-base": True,
        },
    )
    event_sheet: str = Field("0", min_length=1, max_length=128)
    event_time_column: ColumnName | None = None
    event_identifier_column: ColumnName | None = None
    event_filter_column: ColumnName | None = None
    event_filter_values: tuple[str, ...] = Field(default=(), max_length=64)
    association_window_days: float | None = Field(None, ge=0)
    association_direction: Literal[
        "before_event",
        "after_event",
        "symmetric",
    ] = "before_event"
    identifier_column: ColumnName | None = Field(
        default=None,
        description="Optional sample-name column used by the interactive data-preparation workflow.",
    )
    selected_columns: tuple[ColumnName, ...] = Field(
        default=(),
        max_length=256,
        description="Columns selected before missing-value handling; an empty value selects the five analysis-role columns.",
    )
    missing_values: TimeSeriesMissingValueHandling = Field(default_factory=RejectMissingValues)
    feature_engineering: Literal["none"] = "none"

    @property
    def mode_inapplicable_fields(self) -> frozenset[str]:
        """Fields omitted when the strict request is persisted and revalidated."""
        return _TIME_SERIES_MODE_SPECIFIC_FIELDS - _TIME_SERIES_FIELDS_BY_MODE[self.mode]

    @model_serializer(mode="wrap")
    def serialize_selected_mode(self, handler: Any) -> dict[str, Any]:
        """Keep ordinary model_dump output valid under the mode discriminator."""
        value = handler(self)
        for field in self.mode_inapplicable_fields:
            value.pop(field, None)
        return value

    @field_validator("experiment_name", "run_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        normalized = value.strip()
        if normalized in {"", ".", ".."} or _UNSAFE_PATH_CHARACTERS.search(normalized):
            raise ValueError("contains characters that are unsafe in an output directory name")
        if normalized.endswith((" ", ".")):
            raise ValueError("must not end with a space or period")
        if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("uses a Windows-reserved output directory name")
        return normalized

    @field_validator(
        "age_column",
        "minimum_age_column",
        "maximum_age_column",
        "probability_column",
        "value_column",
        "latitude_column",
        "longitude_column",
        "time_column",
        "reference_label_column",
        "comparison_label_column",
        "event_time_column",
        "event_identifier_column",
        "event_filter_column",
    )
    @classmethod
    def validate_column_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("must be a non-blank single-line column name")
        return normalized

    @field_validator("sheet", "event_sheet", "reference_label_provenance")
    @classmethod
    def validate_single_line_value(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("must be a non-blank single-line value")
        return normalized

    @field_validator("identifier_column")
    @classmethod
    def validate_identifier_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("must be a non-blank single-line column name")
        return normalized

    @field_validator("filter_column")
    @classmethod
    def validate_filter_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("must be a non-blank single-line column name")
        return normalized

    @field_validator("element_columns", "signal_columns")
    @classmethod
    def validate_element_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column or "\n" in column or "\r" in column for column in normalized):
            raise ValueError("must contain non-blank single-line column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @field_validator(
        "reference_positive_values",
        "comparison_positive_values",
        "event_filter_values",
    )
    @classmethod
    def validate_semantic_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized):
            raise ValueError("must contain non-blank single-line values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate values")
        return normalized

    @field_validator("selected_columns")
    @classmethod
    def validate_selected_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column or "\n" in column or "\r" in column for column in normalized):
            raise ValueError("must contain non-blank single-line column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
        return normalized

    @property
    def resolved_selected_columns(self) -> tuple[str, ...]:
        """Return the explicit selected-data range or the active mode's role columns."""
        if self.selected_columns:
            return self.selected_columns
        if self.mode == "reference_anomaly_series":
            return tuple(
                dict.fromkeys(
                    (
                        *((self.time_column,) if self.time_column is not None else ()),
                        *self.signal_columns,
                        *((self.reference_label_column,) if self.reference_label_column is not None else ()),
                        *((self.comparison_label_column,) if self.comparison_label_column is not None else ()),
                    )
                )
            )
        if self.mode == "element_mean":
            return tuple(
                dict.fromkeys(
                    (
                        self.age_column,
                        *self.element_columns,
                        *((self.filter_column,) if self.filter_column is not None else ()),
                    )
                )
            )
        if self.mode == "continuous":
            return tuple(
                dict.fromkeys(
                    (
                        self.age_column,
                        *((self.minimum_age_column,) if self.minimum_age_column is not None else ()),
                        self.maximum_age_column,
                        *((self.value_column,) if self.value_column is not None else ()),
                        self.latitude_column,
                        self.longitude_column,
                        *((self.filter_column,) if self.filter_column is not None else ()),
                    )
                )
            )
        return (
            self.age_column,
            self.maximum_age_column,
            self.probability_column,
            self.latitude_column,
            self.longitude_column,
        )

    @field_validator("bin_width")
    @classmethod
    def validate_finite_bin_width(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @field_validator("filter_minimum", "filter_maximum", "relative_value_two_sigma")
    @classmethod
    def validate_finite_filter_bound(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @model_validator(mode="after")
    def validate_time_series_contract(self) -> "TimeSeriesRequest":
        _validate_dataset_choice(self.training_dataset_path, self.training_dataset, "training_dataset")
        irrelevant_mode_fields = sorted((_TIME_SERIES_MODE_SPECIFIC_FIELDS - _TIME_SERIES_FIELDS_BY_MODE[self.mode]) & self.model_fields_set)
        if irrelevant_mode_fields:
            raise ValueError(f"{self.mode} does not accept fields owned by another Time Series mode: " f"{irrelevant_mode_fields}")
        if self.mode == "reference_anomaly_series":
            if self.bin_width is not None:
                raise ValueError("reference_anomaly_series does not use bin_width")
            if not self.signal_columns:
                raise ValueError("reference_anomaly_series requires at least one signal column")
            if self.time_column is None:
                raise ValueError("reference_anomaly_series requires time_column")
            if self.reference_label_column is None:
                raise ValueError("reference_anomaly_series requires reference_label_column")
            if not self.reference_positive_values:
                raise ValueError("reference_anomaly_series requires reference_positive_values")
            roles = {
                self.time_column,
                *self.signal_columns,
                self.reference_label_column,
                *((self.comparison_label_column,) if self.comparison_label_column is not None else ()),
            }
            expected_role_count = 2 + len(self.signal_columns) + (1 if self.comparison_label_column is not None else 0)
            if len(roles) != expected_role_count:
                raise ValueError("reference anomaly observation columns must have distinct scientific roles")
            if bool(self.comparison_label_column) != bool(self.comparison_positive_values):
                raise ValueError("comparison_label_column and comparison_positive_values must be supplied together")
            has_event_semantics = any(
                value is not None
                for value in (
                    self.event_time_column,
                    self.event_identifier_column,
                    self.event_filter_column,
                    self.association_window_days,
                )
            ) or bool(self.event_filter_values)
            if self.event_dataset_path is None and has_event_semantics:
                raise ValueError("event semantics require event_dataset_path")
            if self.event_dataset_path is not None and self.event_time_column is None:
                raise ValueError("event_dataset_path requires event_time_column")
            if bool(self.event_filter_column) != bool(self.event_filter_values):
                raise ValueError("event_filter_column and event_filter_values must be supplied together")
            if self.selected_columns:
                missing_roles = sorted(set(self.resolved_selected_columns) - set(self.selected_columns))
                if missing_roles:
                    raise ValueError(f"selected_columns must include every reference anomaly role: {missing_roles}")
            if self.missing_values.method != "error":
                raise ValueError("reference_anomaly_series requires complete explicitly supplied rows")
            return self
        if self.bin_width is None:
            raise ValueError(f"{self.mode} mode requires bin_width")
        if self.mode == "continuous":
            if self.minimum_age_column is None or self.value_column is None:
                raise ValueError("continuous mode requires minimum_age_column and value_column")
            roles = {
                self.age_column,
                self.minimum_age_column,
                self.maximum_age_column,
                self.value_column,
                self.latitude_column,
                self.longitude_column,
            }
            if len(roles) != 6:
                raise ValueError("continuous Time Series roles must identify six different columns")
            if self.filter_column is not None and self.filter_column in roles:
                raise ValueError("filter_column must have a distinct scientific role")
            has_filter_bounds = self.filter_minimum is not None or self.filter_maximum is not None
            if has_filter_bounds and self.filter_column is None:
                raise ValueError("filter bounds require filter_column")
            if self.filter_minimum is not None and self.filter_maximum is not None and self.filter_minimum > self.filter_maximum:
                raise ValueError("filter_minimum must not exceed filter_maximum")
            missing_roles = sorted(set(self.resolved_selected_columns) - set(self.selected_columns)) if self.selected_columns else []
            if missing_roles:
                raise ValueError(f"selected_columns must include every continuous Time Series role: {missing_roles}")
            if self.missing_values.method not in {"error", "drop_rows"}:
                raise ValueError("Time Series missing_values supports only 'error' or 'drop_rows'")
            drop_columns = tuple(getattr(self.missing_values, "columns", ()))
            unknown_drop_columns = sorted(set(drop_columns) - set(self.resolved_selected_columns))
            if unknown_drop_columns:
                raise ValueError(f"missing_values.columns were not selected: {unknown_drop_columns}")
            return self
        if self.mode == "element_mean":
            if not self.element_columns:
                raise ValueError("element_mean mode requires at least one element column")
            if self.age_column in self.element_columns:
                raise ValueError("age_column must not also be an element column")
            if self.filter_column is not None and self.filter_column in {
                self.age_column,
                *self.element_columns,
            }:
                raise ValueError("filter_column must have a distinct scientific role")
            has_filter_bounds = self.filter_minimum is not None or self.filter_maximum is not None
            if has_filter_bounds and self.filter_column is None:
                raise ValueError("filter bounds require filter_column")
            if self.filter_minimum is not None and self.filter_maximum is not None and self.filter_minimum > self.filter_maximum:
                raise ValueError("filter_minimum must not exceed filter_maximum")
            required_element_roles = {
                self.age_column,
                *self.element_columns,
                *((self.filter_column,) if self.filter_column is not None else ()),
            }
            missing_roles = sorted(required_element_roles - set(self.selected_columns)) if self.selected_columns else []
            if missing_roles:
                raise ValueError(f"selected_columns must include every element_mean role: {missing_roles}")
            if self.missing_values.method not in {"error", "drop_rows"}:
                raise ValueError("Time Series missing_values supports only 'error' or 'drop_rows'")
            drop_columns = tuple(getattr(self.missing_values, "columns", ()))
            unknown_drop_columns = sorted(set(drop_columns) - set(self.resolved_selected_columns))
            if unknown_drop_columns:
                raise ValueError(f"missing_values.columns were not selected: {unknown_drop_columns}")
            return self
        columns = (
            self.age_column,
            self.maximum_age_column,
            self.probability_column,
            self.latitude_column,
            self.longitude_column,
        )
        if len(set(columns)) != len(columns):
            raise ValueError("Time Series roles must identify five different columns")
        missing_roles = sorted(set(columns) - set(self.resolved_selected_columns))
        if missing_roles:
            raise ValueError(f"selected_columns must include every Time Series role: {missing_roles}")
        if self.missing_values.method not in {"error", "drop_rows"}:
            raise ValueError("Time Series missing_values supports only 'error' or 'drop_rows'")
        drop_columns = tuple(getattr(self.missing_values, "columns", ()))
        unknown_drop_columns = sorted(set(drop_columns) - set(self.resolved_selected_columns))
        if unknown_drop_columns:
            raise ValueError(f"missing_values.columns were not selected: {unknown_drop_columns}")
        return self


AnalysisRequestValue = Annotated[
    Union[
        ClassificationRequest,
        RegressionRequest,
        ClusteringRequest,
        DecompositionRequest,
        AnomalyDetectionRequest,
        TimeSeriesRequest,
    ],
    Field(discriminator="task"),
]


class AnalysisRequest(RootModel[AnalysisRequestValue]):
    """Task-discriminated request accepted by start_analysis."""

    @model_validator(mode="before")
    @classmethod
    def preserve_classification_default(cls, value: Any) -> Any:
        if isinstance(value, dict) and "task" not in value:
            return {"task": "classification", **value}
        return value

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        # MCP tool input schemas must advertise an object at the root. Both
        # discriminated alternatives are strict JSON objects.
        schema["type"] = "object"
        return schema


class DatasetInspectionRequest(StrictModel):
    """Bounded, read-only inspection options for one explicit local dataset."""

    model_config = ConfigDict(json_schema_extra=_dataset_inspection_json_schema)

    dataset_path: Path | None = _dataset_path_field("Direct path alternative to dataset; provide exactly one input form.")
    dataset: DatasetReference | None = None
    sample_rows: int = Field(0, ge=0, le=10)
    detail: Literal["full", "names"] = "names"

    @model_validator(mode="before")
    @classmethod
    def default_names_to_no_samples(cls, value: Any) -> Any:
        """Keep names-only discovery small unless samples are explicitly requested."""
        if isinstance(value, dict) and value.get("detail") == "names" and "sample_rows" not in value:
            return {**value, "sample_rows": 0}
        return value

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetInspectionRequest":
        _validate_dataset_choice(self.dataset_path, self.dataset, "dataset")
        return self


class ListDatasetsRequest(StrictModel):
    """Select which safe CLI-owned dataset catalogs to discover."""

    source: Literal["all", "builtin", "desktop"] = "all"


class DatasetCatalogEntry(StrictModel):
    dataset_id: str
    source: Literal["builtin", "desktop"]
    role: Literal["training", "application", "unspecified"]
    task: Literal[
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    ] | None = None
    file_name: str
    path: str
    format: Literal["csv", "xlsx"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int | None = Field(None, ge=0)
    column_count: int | None = Field(None, ge=0)
    analysis_blockers: tuple[str, ...] = ()
    supported_for_analysis: bool


class ListDatasetsResponse(StrictModel):
    schema_version: int = 1
    source_filter: Literal["all", "builtin", "desktop"]
    supported_formats: tuple[Literal["csv", "xlsx"], ...]
    desktop_root: str | None = None
    datasets: tuple[DatasetCatalogEntry, ...]
    warnings: tuple[str, ...] = ()


class ListExperimentsRequest(StrictModel):
    """Bound the number of active persistent MLflow experiments returned."""

    maximum_experiments: int = Field(100, ge=1, le=100)


class GetExperimentRequest(StrictModel):
    """Select one stable MLflow experiment and bound its recent runs."""

    experiment_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    maximum_runs: int = Field(50, ge=0, le=100)


class ExperimentSummary(StrictModel):
    experiment_id: str
    name: str
    lifecycle_stage: Literal["active"]
    artifact_location: str
    tags: dict[str, str] = Field(default_factory=dict)


class ExperimentRunSummary(StrictModel):
    run_id: str
    run_name: str
    status: str
    start_time: int | None = None
    end_time: int | None = None
    artifact_uri: str
    metrics: dict[str, float | None] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)


class ListExperimentsResponse(StrictModel):
    schema_version: Literal[1] = 1
    tracking_root: str
    experiment_count: int = Field(ge=0)
    experiments: tuple[ExperimentSummary, ...]


class GetExperimentResponse(StrictModel):
    schema_version: Literal[1] = 1
    tracking_root: str
    experiment: ExperimentSummary
    run_count: int = Field(ge=0)
    runs: tuple[ExperimentRunSummary, ...]


class StartMlflowUiRequest(StrictModel):
    """Start the local-only managed MLflow UI on one explicit port."""

    port: int = Field(5000, ge=1024, le=65535)


class MlflowUiStatusResponse(StrictModel):
    schema_version: Literal[1] = 1
    state: Literal["stopped", "starting", "running", "ownership_mismatch"]
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int | None = Field(None, ge=1024, le=65535)
    url: str | None = None
    pid: int | None = Field(None, ge=1)
    started_at: str | None = None
    tracking_root: str
    message: str


class RunLookupRequest(StrictModel):
    """Identify one wrapper-owned run."""

    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")


class RunStatusRequest(RunLookupRequest):
    """Read one run immediately or wait for at most five minutes."""

    wait_seconds: float = Field(0, ge=0, le=300)


class RunResultRequest(RunLookupRequest):
    """Read one bounded terminal-result view, optionally using a conditional receipt."""

    # Terminal results return immediately, so this longest bounded default
    # avoids an otherwise redundant pending/result round trip without slowing
    # runs that finish sooner. A caller may still request a shorter wait.
    wait_seconds: float = Field(300, ge=0, le=300)
    artifact_offset: int = Field(0, ge=0)
    artifact_limit: int | None = Field(None, ge=1, le=200)
    artifact_view: Literal["canonical", "all"] = "canonical"
    detail: Literal["compact", "full"] = "compact"
    if_result_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_conditional_view(self) -> "RunResultRequest":
        if self.if_result_sha256 is None:
            return self
        conflicts = []
        if self.detail != "compact":
            conflicts.append("detail")
        if self.artifact_view != "canonical":
            conflicts.append("artifact_view")
        if self.artifact_offset != 0:
            conflicts.append("artifact_offset")
        if self.artifact_limit is not None:
            conflicts.append("artifact_limit")
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError("if_result_sha256 is a compact identity check and cannot be combined " f"with full, all-artifact, or paginated delivery ({joined})")
        return self


class StartAnalysisByValidationRequest(StrictModel):
    """Start the exact immutable request identified by validate_analysis."""

    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnalysisValidationDetailRequest(StrictModel):
    """Read the complete immutable detail for one validation reference."""

    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    detail: Literal["full"]


class DatasetColumnSummary(StrictModel):
    """Type information inferred from a bounded local sample."""

    name: str = Field(max_length=128)
    inferred_type: Literal["empty", "boolean", "integer", "number", "string", "mixed"]
    sampled_non_null: int = Field(ge=0)


class DatasetInspectionResponse(StrictModel):
    """Small dataset summary safe to return through MCP."""

    source_path: str
    resolved_path: str
    original_source_path: str | None = None
    original_source_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    dataset_preparation: dict[str, Any] | None = None
    format: Literal["csv", "xlsx"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    row_count_exact: bool
    column_count: int = Field(ge=1)
    detail: Literal["full", "names"] = "full"
    columns: tuple[DatasetColumnSummary, ...] = ()
    column_names: tuple[str, ...] = ()
    header_warnings: tuple[str, ...] = ()
    sample_rows: tuple[dict[str, Any], ...]
    sample_truncated: bool

    @computed_field(return_type=str)
    @property
    def source_sha256(self) -> str:
        """Hash of the immutable source file before any prepared-view projection."""
        return self.original_source_sha256 or self.sha256

    @computed_field(return_type=str)
    @property
    def prepared_view_sha256(self) -> str:
        """Hash of the exact tabular view inspected by this response."""
        return self.sha256


class CompactDatasetInspectionResponse(StrictModel):
    """Names-only dataset identity without repeated preparation or sample payloads."""

    detail: Literal["names"] = "names"
    format: Literal["csv", "xlsx"]
    size_bytes: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_view_is_source: bool
    preparation_contract_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    row_count_exact: bool
    column_count: int = Field(ge=1)
    column_names: tuple[str, ...]
    header_warnings: tuple[str, ...] = ()
    sample_rows: tuple[dict[str, Any], ...] = ()
    sample_truncated: bool


class CompatibilityPolicy(StrictModel):
    """Versioned runtime and release boundary advertised to every MCP client."""

    schema_version: int = 1
    release_channel: Literal["development", "stable"]
    public_release_ready: bool
    mcp_python_requires: str
    cli_python_requires: str
    mcp_sdk_requires: str
    supported_cli_versions: tuple[str, ...]
    interaction_plan_version: int
    cli_automation_contract_version: int
    artifact_index_schema_version: int
    target_operating_systems: tuple[str, ...]
    pending_release_gates: tuple[str, ...]


class ResourceLimits(StrictModel):
    """Installer-owned limits that bound local resource consumption."""

    maximum_dataset_bytes: int = Field(ge=1)
    maximum_columns: int = Field(ge=1)
    maximum_artifact_references: int = Field(ge=1)
    maximum_concurrent_runs: int = Field(ge=1)
    maximum_pending_runs: int = Field(ge=1)
    maximum_process_seconds: int = Field(ge=1)


class CapabilitySummary(StrictModel):
    """One stable CLI/MCP capability status from the packaged inventory."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$", max_length=160)
    category: str = Field(min_length=1, max_length=80)
    status: Literal["implemented", "verified", "known_gap", "not_public"]
    cli_public: bool
    mcp_supported: bool
    evidence: tuple[str, ...] = ()


class CapabilitiesRequest(StrictModel):
    """Select a capability view or resolve one exact contract."""

    detail: Literal["compact", "full", "start_ready"] = "compact"
    task: AnalysisTaskName | None = Field(
        None,
        description="Task filter; required by start_ready. Full is the audit inventory.",
    )
    method: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="Public method or mode; used only by start_ready.",
    )
    if_capabilities_sha256: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{64}$",
        description="Conditional SHA for unfiltered full; use the view SHA otherwise.",
    )
    if_capability_view_sha256: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{64}$",
        description="Conditional SHA bound to this exact view projection.",
    )
    output_contract_sha256: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{64}$",
        description="Resolve an exact tools/list output schema; exclusive with view fields.",
    )
    request_schema_sha256: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{64}$",
        description="Resolve an exact compact/start_ready request schema; exclusive with view fields.",
    )

    @model_validator(mode="after")
    def validate_conditional_identity(self) -> "CapabilitiesRequest":
        if self.output_contract_sha256 is not None or self.request_schema_sha256 is not None:
            lookup_name = "output_contract_sha256" if self.output_contract_sha256 is not None else "request_schema_sha256"
            if self.output_contract_sha256 is not None and self.request_schema_sha256 is not None:
                raise ValueError("provide only one of output_contract_sha256 or request_schema_sha256")
            conflicts = []
            if self.detail != "compact":
                conflicts.append("detail")
            if self.task is not None:
                conflicts.append("task")
            if self.method is not None:
                conflicts.append("method")
            if self.if_capabilities_sha256 is not None:
                conflicts.append("if_capabilities_sha256")
            if self.if_capability_view_sha256 is not None:
                conflicts.append("if_capability_view_sha256")
            if conflicts:
                raise ValueError(f"{lookup_name} is an exact contract lookup and cannot be combined with: " f"{', '.join(conflicts)}")
            return self
        if self.detail == "start_ready":
            if self.task is None or self.method is None:
                raise ValueError("detail='start_ready' requires both task and method")
            allowed = START_READY_METHODS_BY_TASK[self.task]
            if self.method not in allowed:
                raise ValueError(f"method must be one of the public {self.task} identities: " f"{', '.join(allowed)}")
        elif self.method is not None:
            raise ValueError("method is accepted only with detail='start_ready'")
        if self.detail == "full" and self.task is not None:
            raise ValueError("task filtering is available only for the compact capability view")
        if self.if_capabilities_sha256 is not None and self.if_capability_view_sha256 is not None:
            raise ValueError("provide only one conditional capability identity")
        if self.if_capabilities_sha256 is not None and (self.detail != "full" or self.task is not None):
            raise ValueError("if_capabilities_sha256 is safe only for the unfiltered full view; " "use if_capability_view_sha256 for compact or task-filtered views")
        return self


class ScientificAttestationCapabilities(StrictModel):
    """Public method coverage and the exact boundary of v4 sidecar evidence."""

    scientific_config_contract_version: Literal[4] = 4
    public_manual_method_count: Literal[36] = 36
    v4_attested_method_count: Literal[27] = 27
    legacy_without_v4_attestation_method_count: Literal[9] = 9
    v4_attested_methods_by_task: dict[str, tuple[str, ...]] = Field(
        description=("Public manual task/method combinations accepted by scientific-config v4 " "and required to emit a verified Scientific Execution Attestation.json on success.")
    )
    legacy_methods_without_v4_attestation_by_task: dict[str, tuple[str, ...]] = Field(
        description=("Public manual task/method combinations that remain executable through the legacy " "CLI adapter but do not emit the scientific-config v4 sidecar attestation.")
    )
    selection_modes_without_v4_scientific_config_sidecar: tuple[
        Literal["automl", "all_models"],
        ...,
    ] = ("automl", "all_models")
    time_series_contract: Literal["separate_route_native_contract"] = Field(
        "separate_route_native_contract",
        description=("Time Series uses its independent route-native request, validation, execution, " "and artifact contracts and is not part of the 36/27/9 manual-model counts."),
    )


class CapabilitiesResponse(StrictModel):
    """Installed wrapper and supported CLI workflow information."""

    response_detail: Literal["full"] = "full"
    capabilities_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    capability_view_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    server_name: Literal["GeochemistryPi MCP"] = "GeochemistryPi MCP"
    server_version: str
    supported_cli_versions: tuple[str, ...]
    supported_tasks: tuple[str, ...]
    analysis_schema_task_scope: Literal[
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    ] | None = None
    analysis_start_modes: tuple[Literal["validation_reference", "legacy_full_request"], ...] = (
        "validation_reference",
        "legacy_full_request",
    )
    supported_models: tuple[str, ...]
    supported_dataset_formats: tuple[str, ...]
    maximum_dataset_bytes: int
    default_concurrency: int
    capability_manifest_schema_version: int
    capability_manifest_id: str
    cli_automation_contract_version: int
    capabilities: tuple[CapabilitySummary, ...]
    known_gaps: tuple[str, ...]
    supported_data_sources: tuple[Literal["path", "builtin", "desktop"], ...]
    supported_clients: tuple[str, ...]
    compatibility: CompatibilityPolicy
    resource_limits: ResourceLimits
    classification_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    regression_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    clustering_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    decomposition_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    anomaly_detection_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    time_series_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    supported_models_by_task: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    scientific_attestation: ScientificAttestationCapabilities
    unsupported_interactions: tuple[str, ...] = ()
    notes: tuple[str, ...]


class OutputContractSchemaResponse(StrictModel):
    """One publicly resolvable, hash-bound serialized tool-output contract."""

    response_detail: Literal["output_contract"] = "output_contract"
    output_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_contract_utf8_bytes: int = Field(ge=2)
    output_contract_schema: dict[str, Any]

    @model_validator(mode="after")
    def validate_output_contract_identity(self) -> "OutputContractSchemaResponse":
        encoded = json.dumps(
            self.output_contract_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) != self.output_contract_utf8_bytes:
            raise ValueError("output_contract_utf8_bytes does not match the returned schema")
        if hashlib.sha256(encoded).hexdigest() != self.output_contract_sha256:
            raise ValueError("output_contract_sha256 does not match the returned schema")
        return self


class RequestSchemaLookupArguments(StrictModel):
    """Exact content-addressed request-schema lookup arguments."""

    request_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RequestSchemaResolver(StrictModel):
    """Public lossless resolver for one strict validate_analysis request schema."""

    tool: Literal["get_capabilities"] = "get_capabilities"
    arguments: RequestSchemaLookupArguments
    response_field: Literal["request_schema"] = "request_schema"


class RequestSchemaResponse(StrictModel):
    """One exact hash-bound task-level validate_analysis request schema."""

    response_detail: Literal["request_schema"] = "request_schema"
    task: AnalysisTaskName
    request_schema_utf8_bytes: int = Field(ge=1)
    request_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_schema: dict[str, Any]

    @model_validator(mode="after")
    def validate_request_schema_identity(self) -> "RequestSchemaResponse":
        encoded = json.dumps(
            self.request_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) != self.request_schema_utf8_bytes:
            raise ValueError("request_schema_utf8_bytes does not match the returned schema")
        if hashlib.sha256(encoded).hexdigest() != self.request_schema_sha256:
            raise ValueError("request_schema_sha256 does not match the returned schema")
        return self


class CompactCapabilityBoundary(StrictModel):
    """One unsupported or non-public boundary without maintainer-only evidence paths."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$", max_length=160)
    category: str = Field(min_length=1, max_length=80)
    status: Literal["implemented", "verified", "known_gap", "not_public"]
    cli_public: bool
    mcp_supported: bool


class ValidationRequestNavigation(StrictModel):
    """Stable field locations needed to assemble one task-specific validation request."""

    training_dataset_one_of: tuple[
        Literal["training_dataset"],
        Literal["training_dataset_path"],
    ] = ("training_dataset", "training_dataset_path")
    dataset_reference_discriminator_path: Literal["training_dataset.source"] = "training_dataset.source"
    dataset_reference_sources: tuple[
        Literal["path"],
        Literal["builtin"],
        Literal["desktop"],
    ] = ("path", "builtin", "desktop")
    path_resolution_policy: Literal["absolute_or_mcp_startup_working_directory"] = "absolute_or_mcp_startup_working_directory"
    application_dataset_at_most_one_of: tuple[str, ...] = ()
    regression_target_exactly_one_of: tuple[str, ...] = ()
    model_selection_discriminator_path: str | None = None
    model_settings_discriminator_path: str | None = None
    reproducibility_container_path: Literal["reproducibility"] = "reproducibility"
    split_seed_path: str | None = None
    model_seed_path: str | None = None
    tuning_seed_path: str | None = None
    workflow_seed_path: str | None = None


class TaskValidationRequestContract(StrictModel):
    """Exact task-level contract for constructing one validate_analysis call."""

    contract_schema_version: Literal[1] = 1
    task: AnalysisTaskName
    validation_tool: Literal["validate_analysis"] = "validate_analysis"
    strict_top_level_object: Literal[True] = True
    top_level_fields: tuple[str, ...]
    top_level_required_fields: tuple[str, ...]
    navigation: ValidationRequestNavigation
    minimal_legal_request_example: dict[str, Any] | None = Field(
        None,
        description=("Schema-derived structural example containing placeholder names and a placeholder path. " "It proves request shape only; replace every placeholder before dataset validation."),
    )
    request_schema_utf8_bytes: int = Field(ge=1)
    request_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_schema: dict[str, Any]


class CompactCapabilitiesResponse(StrictModel):
    """Token-bounded planning view that preserves task indexes and scientific limits."""

    response_detail: Literal["compact"] = "compact"
    capabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_filter: AnalysisTaskName | None = None
    server_name: Literal["GeochemistryPi MCP"] = "GeochemistryPi MCP"
    server_version: str
    supported_cli_versions: tuple[str, ...]
    supported_tasks: tuple[AnalysisTaskName, ...]
    analysis_schema_task_scope: AnalysisTaskName | None = None
    analysis_start_modes: tuple[Literal["validation_reference", "legacy_full_request"], ...]
    capability_manifest_schema_version: int
    capability_manifest_id: str
    cli_automation_contract_version: int
    supported_dataset_formats: tuple[str, ...]
    supported_data_sources: tuple[Literal["path", "builtin", "desktop"], ...]
    compatibility: CompatibilityPolicy
    resource_limits: ResourceLimits
    supported_models_by_task: dict[str, tuple[str, ...]]
    scientific_attestation: ScientificAttestationCapabilities
    task_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    known_gaps: tuple[str, ...]
    capability_boundaries: tuple[CompactCapabilityBoundary, ...]
    unsupported_interactions: tuple[str, ...]
    validation_request_contract: TaskValidationRequestContract | None = None
    next_action: str


class StartReadyNextAction(StrictModel):
    """One deterministic transition from discovery to strict validation."""

    next_tool: Literal["validate_analysis"] = "validate_analysis"
    arguments_source: Literal["request_template_after_replacing_placeholders_and_overlaying_user_values"] = "request_template_after_replacing_placeholders_and_overlaying_user_values"
    full_capabilities_required: Literal[False] = False
    dataset_inspection_required: Literal[False] = False


class StartReadyCapabilitiesResponse(StrictModel):
    """Small method-specific request-construction view over the unchanged strict contract."""

    response_detail: Literal["start_ready"] = "start_ready"
    capabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_filter: AnalysisTaskName
    method_filter: str = Field(min_length=1, max_length=80)
    server_name: Literal["GeochemistryPi MCP"] = "GeochemistryPi MCP"
    server_version: str
    available_methods: tuple[str, ...]
    top_level_required_fields: tuple[str, ...]
    navigation: ValidationRequestNavigation
    selected_method_constraints: dict[str, Any]
    request_template: dict[str, Any]
    placeholder_paths: tuple[str, ...]
    template_is_structural_only: Literal[True] = True
    template_merge_policy: Literal[
        "replace_every_placeholder_then_overlay_and_preserve_all_user_supplied_scientific_values"
    ] = "replace_every_placeholder_then_overlay_and_preserve_all_user_supplied_scientific_values"
    template_runtime_model_validated: Literal[True] = True
    request_schema_utf8_bytes: int = Field(ge=1)
    request_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_schema_resolver: RequestSchemaResolver
    next_action: StartReadyNextAction = Field(default_factory=StartReadyNextAction)
    guidance: str


class StartReadyCapabilitiesNotModifiedResponse(StrictModel):
    """Small receipt for an explicitly cached method-specific start-ready view."""

    response_detail: Literal["start_ready_not_modified"] = "start_ready_not_modified"
    not_modified: Literal[True] = True
    capabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_filter: AnalysisTaskName
    method_filter: str = Field(min_length=1, max_length=80)
    server_name: Literal["GeochemistryPi MCP"] = "GeochemistryPi MCP"
    server_version: str
    requery_required: Literal[False] = False
    message: str = "The explicitly cached start-ready view is unchanged. If its original structured payload " "is no longer available, repeat the same start_ready request without a conditional hash."


class CapabilitiesNotModifiedResponse(StrictModel):
    """Small receipt proving that the requested capability view is unchanged."""

    response_detail: Literal["not_modified"] = "not_modified"
    not_modified: Literal[True] = True
    capabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_filter: AnalysisTaskName | None = None
    server_name: Literal["GeochemistryPi MCP"] = "GeochemistryPi MCP"
    server_version: str
    capability_manifest_id: str
    requery_required: Literal[False] = False
    message: str = "Requested capability view is unchanged; its inventory was not replayed."


class StartAnalysisResponse(StrictModel):
    """Immediate acknowledgement for a queued local CLI run."""

    run_id: str
    state: Literal["queued", "running"]
    models: tuple[str, ...]
    estimated_model_count: int = Field(ge=1)
    status_hint: str
    request_hash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    started_from_validation: bool = False


class AnalysisValidationResponse(StrictModel):
    """Read-only execution preview produced before any analysis process starts."""

    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_expires_at: str
    valid: Literal[True] = True
    execution_ready: bool
    comparison_ready: bool = False
    claim_ready: Literal[False] = False
    schema_status: Literal["valid"] = "valid"
    scientific_status: Literal["valid", "requirements_unmet"]
    adapter_status: Literal["available", "unavailable", "requirements_unmet"]
    artifact_status: Literal["planned", "requirements_unmet"]
    environment_status: Literal["READY", "MISMATCH", "UNSPECIFIED"] = "UNSPECIFIED"
    workflow_family: Literal[
        "time_series",
        "supervised_learning",
        "dimension_reduction",
        "clustering",
        "anomaly_detection",
        "artifact_composition",
    ]
    workflow_mode: str = Field(min_length=1, max_length=80)
    method: str = Field(min_length=1, max_length=120)
    scientific_contract_id: str = Field(min_length=1, max_length=255)
    adapter_id: str | None = Field(None, max_length=160)
    adapter_version: str | None = Field(None, max_length=40)
    adapter_identity: str | None = Field(None, max_length=220)
    artifact_requirements: tuple[ArtifactRequirement | TimeSeriesArtifactRequirement, ...] = ()
    blocking_issues: tuple[str, ...] = ()
    task: Literal[
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    ]
    models: tuple[str, ...]
    estimated_model_count: int = Field(ge=1)
    tuning: Literal["manual", "automl", "not_applicable"]
    training_source: Literal["path", "builtin", "desktop"]
    training_dataset_path: str
    training_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_size_bytes: int = Field(ge=0)
    source_dataset_path: str | None = None
    source_dataset_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    dataset_preparation: dict[str, Any] = Field(default_factory=dict)
    dataset_preparation_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    environment_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    environment_profile: dict[str, Any] = Field(default_factory=dict)
    environment_profile_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    requested_seeds: dict[str, int] = Field(default_factory=dict, max_length=8)
    effective_seeds: dict[Annotated[str, Field(min_length=1, max_length=128)], int] = Field(default_factory=dict, max_length=8)
    execution_decisions: dict[str, Any] = Field(
        default_factory=dict,
        max_length=4,
        description=("Bounded, controller-readable evaluation, preprocessing, application, and binding decisions. " "The complete immutable request and plan remain protected by their hashes."),
    )
    parameter_binding: dict[str, Any] = Field(default_factory=dict)
    adapter_artifact_mappings: tuple[dict[str, Any], ...] = ()
    source_row_count: int | None = Field(None, ge=0)
    row_identity_scheme: str | None = None
    row_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    columns: tuple[str, ...]
    identifier_column: str | None = None
    feature_columns: tuple[str, ...] = ()
    selected_columns: tuple[str, ...] = ()
    target_column: str | None = None
    target_columns: tuple[str, ...] = ()
    resolved_model_parameters: dict[str, ModelParameterValue] = Field(default_factory=dict, max_length=64)
    application_source: Literal["path", "builtin", "desktop"] | None = None
    application_dataset_path: str | None = None
    application_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    application_source_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    application_preparation: dict[str, Any] | None = None
    application_source_row_count: int | None = Field(None, ge=0)
    application_row_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    event_dataset_path: str | None = None
    event_source_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    event_size_bytes: int | None = Field(None, ge=0)
    experiment_mode: Literal["new", "existing", "not_applicable"]
    experiment_name: str
    existing_experiment_id: str | None = None
    interaction_plan: str
    analysis_process_started: Literal[False] = False
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_event_dataset_identity(self) -> "AnalysisValidationResponse":
        event_identity = (
            self.event_dataset_path,
            self.event_source_sha256,
            self.event_size_bytes,
        )
        if any(value is not None for value in event_identity) and not all(value is not None for value in event_identity):
            raise ValueError("event dataset path, SHA-256, and size must be reported together")
        if self.event_dataset_path is not None and self.task != "time_series":
            raise ValueError("event dataset identity is available only for Time Series validation")
        return self


def validate_terminal_error_projection(
    error: str | None,
    error_truncated: bool,
    error_sha256: str | None,
    error_total_utf8_bytes: int | None,
) -> None:
    """Validate one bounded terminal-error prefix and its complete identity."""

    if error is None:
        if error_truncated or error_sha256 is not None or error_total_utf8_bytes is not None:
            raise ValueError("terminal error metadata requires an error")
        return
    if error_sha256 is None or error_total_utf8_bytes is None:
        raise ValueError("a terminal error requires complete-text identity metadata")
    displayed_bytes = error.encode("utf-8")
    if error_total_utf8_bytes < len(displayed_bytes):
        raise ValueError("terminal error bytes cannot exceed the complete error size")
    if error_truncated != (len(displayed_bytes) < error_total_utf8_bytes):
        raise ValueError("terminal error truncation metadata is inconsistent")
    if not error_truncated and hashlib.sha256(displayed_bytes).hexdigest() != error_sha256:
        raise ValueError("complete terminal error must match its SHA-256")


class RunStatusResponse(StrictModel):
    """Durable state of one local run."""

    run_id: str
    state: Literal["queued", "running", "succeeded", "partial_failure", "failed", "cancelled"]
    stage: Literal[
        "queued",
        "running_cli",
        "indexing_outputs",
        "completed",
        "failed",
        "cancelled",
    ] = "queued"
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    cli_pid: int | None = None
    progress_message: str
    error: str | None = Field(None, max_length=1000)
    error_truncated: bool = False
    error_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    error_total_utf8_bytes: int | None = Field(None, ge=1)
    result_type: Literal[
        "input_integrity_changed",
        "cli_execution_failed",
        "run_state_invalid",
        "internal_error",
    ] | None = None
    retryable: bool | None = None

    @model_validator(mode="after")
    def validate_terminal_error_identity(self) -> "RunStatusResponse":
        validate_terminal_error_projection(
            self.error,
            self.error_truncated,
            self.error_sha256,
            self.error_total_utf8_bytes,
        )
        if self.state == "failed":
            if self.result_type is None or self.retryable is None:
                raise ValueError("a failed run status requires a typed recovery contract")
        elif self.result_type is not None or self.retryable is not None:
            raise ValueError("failure recovery fields are available only for failed runs")
        return self


class PendingRunResultResponse(StrictModel):
    """Non-error receipt returned when a bounded result wait ends before terminal state."""

    response_detail: Literal["pending"] = "pending"
    terminal: Literal[False] = False
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    state: Literal["queued", "running"]
    stage: Literal["queued", "running_cli", "indexing_outputs"]
    created_at: str
    started_at: str | None = None
    cli_pid: int | None = Field(None, ge=1)
    progress_message: str = Field(min_length=1, max_length=1000)
    wait_seconds: float = Field(ge=0, le=300)
    result_available: Literal[False] = False
    requery_required: Literal[True] = True
    recommended_wait_seconds: int = Field(5, ge=1, le=300)
    message: Literal["The run is still active; no terminal scientific result is available yet."] = "The run is still active; no terminal scientific result is available yet."

    @model_validator(mode="after")
    def validate_pending_state(self) -> "PendingRunResultResponse":
        if self.state == "queued" and self.stage != "queued":
            raise ValueError("a queued pending result must remain at the queued stage")
        if self.state == "running" and self.stage == "queued":
            raise ValueError("a running pending result cannot remain at the queued stage")
        return self

    @classmethod
    def from_status(
        cls,
        status: RunStatusResponse,
        *,
        wait_seconds: float,
    ) -> "PendingRunResultResponse":
        if status.state not in {"queued", "running"}:
            raise ValueError("a pending result receipt requires a queued or running status")
        if status.stage not in {"queued", "running_cli", "indexing_outputs"}:
            raise ValueError("a pending result receipt requires a non-terminal stage")
        return cls(
            run_id=status.run_id,
            state=status.state,
            stage=status.stage,
            created_at=status.created_at,
            started_at=status.started_at,
            cli_pid=status.cli_pid,
            progress_message=status.progress_message,
            wait_seconds=wait_seconds,
        )


class RequiredTabularObservation(StrictModel):
    """Hash-bound structural view over one requirement-bound native output table.

    Rows are returned only when the complete native table fits every cell and
    byte budget.  Metadata-only observations remain useful for proving the row
    and column shape of large prediction, assignment, and coordinate outputs.
    """

    artifact_id: str = Field(min_length=1, max_length=80)
    relative_path: str = Field(min_length=1, max_length=2048)
    requirement_ids: tuple[ArtifactRequirementId, ...] = Field(
        default=(),
        min_length=1,
        max_length=_MAX_REQUIRED_TABULAR_REQUIREMENT_IDS,
    )
    requirement_ids_total_count: int = Field(ge=1, le=256)
    requirement_ids_truncated: bool = False
    requirement_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    format: Literal["csv", "xlsx", "json", "txt"]
    sheet: str | None = Field(None, min_length=1, max_length=255)
    row_count: int = Field(ge=0)
    row_count_semantics: Literal["nonempty_data_rows_after_first_nonempty_header"] = "nonempty_data_rows_after_first_nonempty_header"
    column_count: int = Field(ge=1)
    columns: tuple[Annotated[str, Field(max_length=256)], ...] = Field(
        max_length=_MAX_REQUIRED_TABULAR_COLUMNS,
    )
    columns_truncated: bool = False
    columns_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows_included: bool
    rows: tuple[
        Annotated[tuple[TabularCell, ...], Field(max_length=_MAX_REQUIRED_TABULAR_COLUMNS)],
        ...,
    ] = Field(default=(), max_length=_MAX_REQUIRED_TABULAR_ROWS)
    returned_cell_count: int = Field(0, ge=0, le=_MAX_REQUIRED_TABULAR_CELLS)
    rows_omission_reason: Literal[
        "large_table",
        "column_limit",
        "cell_length_limit",
        "total_cell_budget",
        "response_byte_budget",
    ] | None = None

    @model_validator(mode="after")
    def validate_tabular_receipt(self) -> "RequiredTabularObservation":
        if self.format == "xlsx" and self.sheet is None:
            raise ValueError("an XLSX observation requires a worksheet identity")
        if self.format != "xlsx" and self.sheet is not None:
            raise ValueError("only XLSX observations carry a worksheet identity")
        if len(self.requirement_ids) > self.requirement_ids_total_count:
            raise ValueError("returned requirement IDs cannot exceed their full count")
        if self.requirement_ids_truncated != (len(self.requirement_ids) < self.requirement_ids_total_count):
            raise ValueError("requirement ID truncation fields are inconsistent")
        if not self.requirement_ids_truncated and self.requirement_ids_sha256 != _canonical_json_sha256(list(self.requirement_ids)):
            raise ValueError("complete requirement IDs must match their SHA-256")
        if len(self.columns) > self.column_count:
            raise ValueError("returned columns cannot exceed the observed column count")
        if self.columns_truncated != (len(self.columns) < self.column_count):
            raise ValueError("column truncation fields are inconsistent")
        if not self.columns_truncated and self.columns_sha256 != _canonical_json_sha256(list(self.columns)):
            raise ValueError("complete columns must match their SHA-256")
        observed_cells = sum(len(row) for row in self.rows)
        if observed_cells != self.returned_cell_count:
            raise ValueError("returned_cell_count must equal the delivered row cells")
        if self.rows_included:
            if self.rows_omission_reason is not None:
                raise ValueError("complete rows cannot have an omission reason")
            if len(self.rows) != self.row_count:
                raise ValueError("complete rows must equal the observed output row count")
            if any(len(row) != self.column_count for row in self.rows):
                raise ValueError("every complete row must match the observed column count")
        elif self.rows or self.returned_cell_count or self.rows_omission_reason is None:
            raise ValueError("metadata-only observations require empty rows and an omission reason")
        return self


class RequiredTabularObservationSummary(StrictModel):
    """Strict global budget and immutable-index identity for native table views."""

    artifact_index_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    observations: tuple[RequiredTabularObservation, ...] = Field(
        default=(),
        max_length=_MAX_REQUIRED_TABULAR_OBSERVATIONS,
    )
    total_count: int = Field(0, ge=0, le=10_000)
    returned_count: int = Field(0, ge=0, le=_MAX_REQUIRED_TABULAR_OBSERVATIONS)
    truncated: bool = False
    observations_sha256: str = Field(
        default_factory=lambda: hashlib.sha256(b"[]").hexdigest(),
        pattern=r"^[0-9a-f]{64}$",
    )
    returned_cell_count: int = Field(0, ge=0, le=_MAX_REQUIRED_TABULAR_CELLS)
    returned_utf8_bytes: int = Field(2, ge=0, le=_MAX_REQUIRED_TABULAR_JSON_BYTES)
    omitted_artifact_count: int = Field(0, ge=0, le=10_000)
    omission_reason_counts: dict[str, int] = Field(default_factory=dict, max_length=8)
    omissions_sha256: str = Field(
        default_factory=lambda: hashlib.sha256(b"[]").hexdigest(),
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_observation_budget(self) -> "RequiredTabularObservationSummary":
        if self.returned_count != len(self.observations):
            raise ValueError("returned_count must equal the observation tuple length")
        if self.returned_count > self.total_count:
            raise ValueError("returned observations cannot exceed their full count")
        if self.truncated != (self.returned_count < self.total_count):
            raise ValueError("tabular observation truncation fields are inconsistent")
        cells = sum(item.returned_cell_count for item in self.observations)
        if cells != self.returned_cell_count:
            raise ValueError("global returned_cell_count must equal delivered observation cells")
        observed_bytes = _json_size_bytes([item.model_dump(mode="json") for item in self.observations])
        if observed_bytes != self.returned_utf8_bytes:
            raise ValueError("returned_utf8_bytes must equal the delivered observation JSON bytes")
        if self.total_count == 0 and self.observations_sha256 != _canonical_json_sha256([]):
            raise ValueError("an empty observation set must use the canonical empty-list SHA-256")
        if sum(self.omission_reason_counts.values()) != self.omitted_artifact_count:
            raise ValueError("omission reason counts must equal omitted_artifact_count")
        if any(not key or value < 1 for key, value in self.omission_reason_counts.items()):
            raise ValueError("omission reason counts must contain positive, named counts")
        if self.omitted_artifact_count == 0 and self.omissions_sha256 != _canonical_json_sha256([]):
            raise ValueError("an empty omission set must use the canonical empty-list SHA-256")
        return self


class ArtifactReference(StrictModel):
    """Reference to an original file produced by the CLI."""

    artifact_id: str
    category: Literal["artifacts", "metrics", "parameters", "summary"]
    relative_path: str
    local_path: str
    size_bytes: int = Field(ge=0)
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirement_id: ArtifactRequirementId | None = None
    requirement_ids: tuple[ArtifactRequirementId, ...] = Field(default=(), max_length=256)
    scientific_type: str | None = Field(None, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def validate_requirement_identities(self) -> "ArtifactReference":
        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("artifact requirement IDs must be unique")
        if self.requirement_id is not None and self.requirement_ids and self.requirement_id not in self.requirement_ids:
            raise ValueError("legacy requirement_id must be present in requirement_ids")
        return self


class CompactArtifactReference(StrictModel):
    """Token-bounded reference to one indexed CLI artifact."""

    category: Literal["artifacts", "metrics", "parameters", "summary"]
    relative_path: str = Field(min_length=1, max_length=1024)
    relative_path_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    relative_path_truncated: bool = False
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirement_ids: tuple[ArtifactRequirementId, ...] = Field(
        default=(),
        max_length=_MAX_COMPACT_REQUIREMENT_IDS_PER_ARTIFACT,
    )
    requirement_ids_total_count: int = Field(0, ge=0, le=256)
    requirement_ids_truncated: bool = False
    requirement_ids_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    scientific_type: str | None = Field(None, max_length=120)

    @model_validator(mode="after")
    def validate_requirement_id_summary(self) -> "CompactArtifactReference":
        if self.relative_path_truncated != (self.relative_path_sha256 is not None):
            raise ValueError("a compact artifact path requires a separate SHA-256 only when truncated")
        if len(self.requirement_ids) > self.requirement_ids_total_count:
            raise ValueError("compact requirement IDs cannot exceed their full count")
        if self.requirement_ids_truncated != (len(self.requirement_ids) < self.requirement_ids_total_count):
            raise ValueError("compact requirement ID truncation fields are inconsistent")
        if self.requirement_ids_truncated != (self.requirement_ids_sha256 is not None):
            raise ValueError("compact requirement IDs require a separate SHA-256 only when truncated")
        return self

    @model_serializer(mode="wrap")
    def serialize_without_derived_hashes(self, handler):
        """Omit identities fully derivable from untruncated values."""

        value = handler(self)
        if self.relative_path_sha256 is None:
            value.pop("relative_path_sha256", None)
        if self.requirement_ids_sha256 is None:
            value.pop("requirement_ids_sha256", None)
        return value


class CompactDatasetPreparationSummary(StrictModel):
    """Bounded audit identity and decisions for the immutable preparation record."""

    canonical_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    preparation_contract_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    source_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    prepared_input_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    materialized_view: bool | None = None
    input_row_count: int | None = Field(None, ge=0)
    prepared_row_count: int | None = Field(None, ge=0)
    filtered_row_count: int | None = Field(None, ge=0)
    worksheet: str | None = Field(None, min_length=1, max_length=255)
    worksheet_count: int = Field(0, ge=0, le=16)
    header_row_indices: tuple[int, ...] = Field(default=(), max_length=16)
    projection_mode: Literal["all", "selected", "excluded"] = "all"
    projected_column_count: int = Field(0, ge=0, le=256)
    projected_columns_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    filter_count: int = Field(0, ge=0, le=64)
    filters_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    declared_operations: tuple[
        Literal["missing_value_handling", "filtering", "transformation", "feature_selection"],
        ...,
    ] = Field(default=(), max_length=4)
    executed_view_operations: tuple[
        Annotated[str, Field(min_length=1, max_length=64)],
        ...,
    ] = Field(default=(), max_length=16)
    row_identity_scheme: str | None = Field(None, min_length=1, max_length=64)
    row_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ChildModelResult(StrictModel):
    """One isolated child in an all-models aggregate run."""

    model: str = Field(min_length=1, max_length=80)
    state: Literal["succeeded", "failed"]
    output_relative_path: str = Field(min_length=1, max_length=255)
    artifact_count: int = Field(ge=0)
    error: str | None = Field(None, max_length=1000)

    @model_validator(mode="after")
    def validate_error_state(self) -> "ChildModelResult":
        if self.state == "succeeded" and self.error is not None:
            raise ValueError("a succeeded child must not contain an error")
        if self.state == "failed" and not self.error:
            raise ValueError("a failed child must contain a bounded error")
        return self


class CompactChildModelResult(StrictModel):
    """Bounded aggregate-child state with hashes for any shortened text."""

    model: str = Field(min_length=1, max_length=80)
    state: Literal["succeeded", "failed"]
    output_relative_path: str = Field(min_length=1, max_length=255)
    output_relative_path_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_relative_path_truncated: bool = False
    artifact_count: int = Field(ge=0)
    error: str | None = Field(None, max_length=256)
    error_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    error_truncated: bool = False

    @model_validator(mode="after")
    def validate_error_state(self) -> "CompactChildModelResult":
        if not self.output_relative_path_truncated and self.output_relative_path_sha256 != _canonical_json_sha256(self.output_relative_path):
            raise ValueError("complete compact child output path must match its SHA-256")
        if self.state == "succeeded":
            if self.error is not None or self.error_sha256 is not None or self.error_truncated:
                raise ValueError("a succeeded compact child cannot contain an error summary")
        elif self.error is None or self.error_sha256 is None:
            raise ValueError("a failed compact child requires a bounded error and full-error SHA-256")
        elif not self.error_truncated and self.error_sha256 != _canonical_json_sha256(self.error):
            raise ValueError("complete compact child error must match its SHA-256")
        return self


class AggregateResultSummary(StrictModel):
    """Bounded parent counts so child failures are visible at a glance."""

    expected_model_count: int = Field(ge=1)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "AggregateResultSummary":
        if self.expected_model_count != self.succeeded_count + self.failed_count:
            raise ValueError("aggregate expected_model_count must equal succeeded_count + failed_count")
        return self


def validate_success_result_state(
    *,
    state: str,
    contract_status: str,
    model: str,
    aggregate_state: str | None,
    aggregate_summary: AggregateResultSummary | None,
    children: tuple[Any, ...],
    children_total_count: int,
    children_complete: bool,
) -> None:
    """Bind aggregate evidence, artifact completeness, and the top-level state."""

    aggregate = model == "all_models"
    if not aggregate:
        if aggregate_state is not None or aggregate_summary is not None or children_total_count:
            raise ValueError("a single-model result cannot contain aggregate state, summary, or children")
    else:
        if aggregate_state is None or aggregate_summary is None:
            raise ValueError("an all-models result requires aggregate_state and aggregate_summary")
        if aggregate_summary.expected_model_count != children_total_count:
            raise ValueError("aggregate expected_model_count must equal the complete child count")
        returned_succeeded = sum(child.state == "succeeded" for child in children)
        returned_failed = sum(child.state == "failed" for child in children)
        if returned_succeeded > aggregate_summary.succeeded_count or returned_failed > aggregate_summary.failed_count:
            raise ValueError("aggregate child states exceed the published summary counts")
        if children_complete and (returned_succeeded != aggregate_summary.succeeded_count or returned_failed != aggregate_summary.failed_count):
            raise ValueError("complete aggregate children must match the published summary counts")
        expected_aggregate_state = "partial_failure" if aggregate_summary.failed_count else "complete"
        if aggregate_state != expected_aggregate_state:
            raise ValueError("aggregate_state must reflect whether any aggregate child failed")

    expected_state = "partial_failure" if contract_status == "incomplete" or aggregate_state == "partial_failure" else "succeeded"
    if state != expected_state:
        raise ValueError("top-level state must reflect artifact-contract and aggregate completeness")


class PreprocessingSummary(StrictModel):
    """Strict row counts copied from an original CLI preprocessing record."""

    input_row_count: int = Field(ge=0, strict=True)
    analysis_row_count: int = Field(ge=0, strict=True)
    dropped_row_count: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_row_count_invariants(self) -> "PreprocessingSummary":
        if self.analysis_row_count > self.input_row_count:
            raise ValueError("analysis_row_count must not exceed input_row_count")
        if self.dropped_row_count != self.input_row_count - self.analysis_row_count:
            raise ValueError("dropped_row_count must equal input_row_count - analysis_row_count")
        return self


class RunResultResponse(StrictModel):
    """Bounded result composed only from wrapper metadata and original CLI outputs."""

    response_detail: Literal["full"] = "full"
    run_id: str
    result_record_path: str | None = None
    result_record_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    canonical_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_contract_id: str = Field(min_length=1, max_length=256)
    scientific_execution_contract_bound: bool
    provenance_manifest_path: str = Field(min_length=1, max_length=4096)
    provenance_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_status: Literal["complete", "incomplete"] = "complete"
    missing_artifact_requirement_ids: tuple[ArtifactRequirementId, ...] = Field(default=(), max_length=256)
    state: Literal["succeeded", "partial_failure"]
    task: Literal[
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    ]
    model: ClassificationModelName | RegressionModelName | ClusteringModelName | DecompositionModelName | AnomalyDetectionModelName | Literal[
        "subaerial_proportion_bootstrap",
        "spatiotemporal_weighted_continuous_bootstrap",
        "element_mean",
        "reference_label_event_overlay",
        "embedding_label_overlay",
        "all_models",
    ]
    tuning: Literal["manual", "automl", "not_applicable"] = "manual"
    output_directory: str
    interaction_trace: str
    cli_stdout_log: str
    cli_stderr_log: str
    cli_exit_code: int
    cli_started_at: str | None = Field(
        None,
        description="Actual CLI child start from the immutable trace; null only when no child was created.",
    )
    cli_finished_at: str | None = Field(
        None,
        description="Actual CLI child finish from the immutable trace; never managed-run time.",
    )
    cli_execution_duration_seconds: float | None = Field(
        None,
        ge=0,
        allow_inf_nan=False,
        description="Monotonic seconds for the CLI child; null only when no child was created.",
    )
    cli_version: str
    input_sha256: str
    input_hash_verified: bool
    source_input_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    dataset_preparation: dict[str, Any] = Field(default_factory=dict)
    environment_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    effective_seeds: dict[str, int] = Field(default_factory=dict, max_length=8)
    source_row_count: int | None = Field(None, ge=0)
    row_identity_scheme: str | None = None
    row_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    source_row_pairing_verified: bool | None = None
    source_row_pairing_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    preprocessing_summary: PreprocessingSummary | None = Field(
        None,
        description="Optional strict row counts copied from the original indexed CLI parameter artifact; MCP never recalculates them.",
    )
    application_input_sha256: str | None = None
    application_input_hash_verified: bool | None = None
    event_input_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    event_input_hash_verified: bool | None = None
    reported_metrics: dict[str, Any]
    required_tabular_observations: RequiredTabularObservationSummary = Field(
        default_factory=RequiredTabularObservationSummary,
        description=(
            "Requirement-bound, artifact-index-bound native table metadata. Complete rows are included only "
            "for small outputs that fit the global cell and UTF-8 budgets; large outputs retain exact output "
            "row/column metadata without substituting validation input counts."
        ),
    )
    artifact_count: int = Field(ge=0)
    canonical_artifact_count: int | None = Field(None, ge=0)
    summary_mirror_count: int = Field(0, ge=0)
    artifact_index_path: str | None = None
    artifact_index_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    artifact_view: Literal["canonical", "all"] = "all"
    artifact_view_count: int | None = Field(None, ge=0)
    artifact_offset: int = Field(0, ge=0)
    returned_artifact_count: int = Field(0, ge=0)
    next_artifact_offset: int | None = Field(None, ge=0)
    artifacts: tuple[ArtifactReference, ...]
    artifacts_truncated: bool
    aggregate_state: Literal["complete", "partial_failure"] | None = None
    aggregate_summary: AggregateResultSummary | None = None
    children: tuple[ChildModelResult, ...] = ()
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_preprocessing_summary_source(self) -> "RunResultResponse":
        validate_cli_execution_interval(
            self.cli_started_at,
            self.cli_finished_at,
            self.cli_execution_duration_seconds,
        )
        if self.cli_started_at is None:
            raise ValueError("a successful or partial scientific result requires a CLI child-process interval")
        if self.contract_status == "complete" and self.missing_artifact_requirement_ids:
            raise ValueError("a complete artifact contract cannot list missing requirements")
        if self.contract_status == "incomplete" and not self.missing_artifact_requirement_ids:
            raise ValueError("an incomplete artifact contract must list missing requirements")
        if self.missing_artifact_requirement_ids and self.state != "partial_failure":
            raise ValueError("missing required artifacts must produce an explicit partial_failure result")
        validate_success_result_state(
            state=self.state,
            contract_status=self.contract_status,
            model=self.model,
            aggregate_state=self.aggregate_state,
            aggregate_summary=self.aggregate_summary,
            children=self.children,
            children_total_count=len(self.children),
            children_complete=True,
        )
        if self.preprocessing_summary is None:
            return self
        if self.task != "time_series":
            raise ValueError("preprocessing_summary is available only for Time Series results")
        if self.source_row_count is None or self.preprocessing_summary.input_row_count != self.source_row_count:
            raise ValueError("preprocessing_summary input rows must match source_row_count")
        return self


def _json_size_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def _compact_reported_metrics(
    metrics: dict[str, Any],
    maximum_bytes: int = _MAX_COMPACT_REPORTED_METRICS_BYTES,
) -> tuple[dict[str, Any], int, int]:
    original_size = _json_size_bytes(metrics)
    if original_size <= maximum_bytes:
        return metrics, original_size, 0
    if maximum_bytes < _json_size_bytes({}):
        return {}, original_size, len(metrics)
    bounded: dict[str, Any] = {}
    omitted = 0
    for key, value in metrics.items():
        candidate = {**bounded, key: value}
        if _json_size_bytes(candidate) <= maximum_bytes:
            bounded[key] = value
        else:
            omitted += 1
    return bounded, original_size, omitted


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded_json_text(value: str, maximum_json_bytes: int) -> tuple[str, bool]:
    """Keep a readable prefix whose escaped JSON representation is bounded."""

    if _json_size_bytes(value) <= maximum_json_bytes:
        return value, False
    lower = 1
    upper = len(value)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if _json_size_bytes(value[:middle]) <= maximum_json_bytes:
            lower = middle
        else:
            upper = middle - 1
    return value[:lower], True


def _compact_requirement_ids(
    reference: ArtifactReference,
) -> tuple[tuple[str, ...], int, bool, str | None]:
    full_ids = tuple(
        dict.fromkeys(
            (
                *((reference.requirement_id,) if reference.requirement_id is not None else ()),
                *reference.requirement_ids,
            )
        )
    )
    compact_ids = full_ids[:_MAX_COMPACT_REQUIREMENT_IDS_PER_ARTIFACT]
    return (
        compact_ids,
        len(full_ids),
        len(compact_ids) < len(full_ids),
        (_canonical_json_sha256(list(full_ids)) if len(compact_ids) < len(full_ids) else None),
    )


def _compact_artifact_reference(reference: ArtifactReference) -> CompactArtifactReference:
    """Project one artifact without dropping its path, content, or requirement identities."""

    compact_ids, total_ids, ids_truncated, ids_sha256 = _compact_requirement_ids(reference)
    relative_path, relative_path_truncated = _bounded_json_text(
        reference.relative_path,
        _MAX_COMPACT_ARTIFACT_PATH_JSON_BYTES,
    )
    return CompactArtifactReference(
        category=reference.category,
        relative_path=relative_path,
        relative_path_sha256=(_canonical_json_sha256(reference.relative_path) if relative_path_truncated else None),
        relative_path_truncated=relative_path_truncated,
        size_bytes=reference.size_bytes,
        sha256=reference.sha256,
        requirement_ids=compact_ids,
        requirement_ids_total_count=total_ids,
        requirement_ids_truncated=ids_truncated,
        requirement_ids_sha256=ids_sha256,
        scientific_type=reference.scientific_type,
    )


def _compact_child_result(child: ChildModelResult) -> CompactChildModelResult:
    output_path, output_truncated = _bounded_json_text(
        child.output_relative_path,
        _MAX_COMPACT_CHILD_TEXT_JSON_BYTES,
    )
    error = None
    error_sha256 = None
    error_truncated = False
    if child.error is not None:
        error, error_truncated = _bounded_json_text(
            child.error,
            _MAX_COMPACT_CHILD_TEXT_JSON_BYTES,
        )
        error_sha256 = _canonical_json_sha256(child.error)
    return CompactChildModelResult(
        model=child.model,
        state=child.state,
        output_relative_path=output_path,
        output_relative_path_sha256=_canonical_json_sha256(child.output_relative_path),
        output_relative_path_truncated=output_truncated,
        artifact_count=child.artifact_count,
        error=error,
        error_sha256=error_sha256,
        error_truncated=error_truncated,
    )


def _compact_limitations(limitations: tuple[str, ...]) -> tuple[tuple[str, ...], int, bool, str]:
    compact = []
    any_text_truncated = False
    for limitation in limitations[:_MAX_COMPACT_LIMITATIONS]:
        bounded, truncated = _bounded_json_text(
            limitation,
            _MAX_COMPACT_LIMITATION_TEXT_JSON_BYTES,
        )
        compact.append(bounded)
        any_text_truncated = any_text_truncated or truncated
    return (
        tuple(compact),
        len(limitations),
        any_text_truncated or len(compact) < len(limitations),
        _canonical_json_sha256(list(limitations)),
    )


def _sha256_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def _nonnegative_int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _bounded_text_or_none(value: object, maximum: int) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= maximum else None


def _bounded_string_tuple(value: object, *, maximum_items: int, maximum_length: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > maximum_items:
        return ()
    result = tuple(item for item in value if isinstance(item, str) and 0 < len(item) <= maximum_length)
    return result if len(result) == len(value) else ()


def _compact_dataset_preparation(
    preparation: dict[str, Any],
    *,
    fallback_source_row_count: int | None,
) -> CompactDatasetPreparationSummary:
    contract = preparation.get("contract") if isinstance(preparation.get("contract"), dict) else {}
    table = preparation.get("table") if isinstance(preparation.get("table"), dict) else {}
    source_file = preparation.get("source_file") if isinstance(preparation.get("source_file"), dict) else {}
    prepared_input = preparation.get("prepared_input") if isinstance(preparation.get("prepared_input"), dict) else {}
    row_identity = table.get("row_identity") if isinstance(table.get("row_identity"), dict) else {}

    worksheets = _bounded_string_tuple(table.get("worksheets"), maximum_items=16, maximum_length=255)
    worksheet = _bounded_text_or_none(table.get("worksheet"), 255)
    if worksheet is None:
        worksheet = _bounded_text_or_none(contract.get("worksheet"), 255)
    if not worksheets:
        worksheets = _bounded_string_tuple(contract.get("worksheets"), maximum_items=16, maximum_length=255)

    header_indices_value = table.get("header_row_indices") or contract.get("header_row_indices")
    header_indices = (
        tuple(header_indices_value)
        if isinstance(header_indices_value, (list, tuple))
        and len(header_indices_value) <= 16
        and all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 1_000_000 for item in header_indices_value)
        else ()
    )
    if not header_indices:
        header_index = table.get("header_row_index", contract.get("header_row_index"))
        if isinstance(header_index, int) and not isinstance(header_index, bool) and 0 <= header_index <= 1_000_000:
            header_indices = (header_index,)

    selected = contract.get("selected_columns")
    excluded = contract.get("excluded_columns")
    table_selected = table.get("selected_columns")
    projection_mode: Literal["all", "selected", "excluded"] = "all"
    projected: object = ()
    if isinstance(selected, (list, tuple)) and selected:
        projected = selected
        projection_mode = "selected"
    elif isinstance(excluded, (list, tuple)) and excluded:
        projected = excluded
        projection_mode = "excluded"
    if projection_mode == "all" and isinstance(table_selected, (list, tuple)) and table_selected:
        projected = table_selected
        projection_mode = "selected"
    projected_items = _bounded_string_tuple(projected, maximum_items=256, maximum_length=128)
    projected_hash = _canonical_json_sha256(list(projected_items)) if projected_items else None

    filters = table.get("filters") if isinstance(table.get("filters"), (list, tuple)) else contract.get("filters")
    if not isinstance(filters, (list, tuple)) or len(filters) > 64:
        filters = ()
    filters_hash = _canonical_json_sha256(filters) if filters else None

    declared = _bounded_string_tuple(preparation.get("declared_operations"), maximum_items=4, maximum_length=32)
    allowed_declared = {"missing_value_handling", "filtering", "transformation", "feature_selection"}
    if any(item not in allowed_declared for item in declared):
        declared = ()
    executed = _bounded_string_tuple(preparation.get("executed_view_operations"), maximum_items=16, maximum_length=64)

    input_rows = _nonnegative_int_or_none(table.get("input_row_count"))
    prepared_rows = _nonnegative_int_or_none(table.get("source_row_count"))
    if prepared_rows is None:
        prepared_rows = fallback_source_row_count
    if input_rows is None:
        input_rows = prepared_rows
    filtered_rows = _nonnegative_int_or_none(table.get("filtered_row_count"))
    if filtered_rows is None and input_rows is not None and prepared_rows is not None and prepared_rows <= input_rows:
        filtered_rows = input_rows - prepared_rows

    source_sha256 = _sha256_or_none(source_file.get("sha256"))
    prepared_sha256 = _sha256_or_none(prepared_input.get("sha256"))
    materialized = None if source_sha256 is None or prepared_sha256 is None else source_sha256 != prepared_sha256
    contract_sha256 = _canonical_json_sha256(contract) if contract else None

    return CompactDatasetPreparationSummary(
        canonical_record_sha256=_canonical_json_sha256(preparation),
        contract_sha256=contract_sha256,
        preparation_contract_sha256=_sha256_or_none(preparation.get("contract_hash")),
        source_sha256=source_sha256,
        prepared_input_sha256=prepared_sha256,
        materialized_view=materialized,
        input_row_count=input_rows,
        prepared_row_count=prepared_rows,
        filtered_row_count=filtered_rows,
        worksheet=worksheet,
        worksheet_count=len(worksheets) if worksheets else (1 if worksheet is not None else 0),
        header_row_indices=header_indices,
        projection_mode=projection_mode,
        projected_column_count=len(projected_items),
        projected_columns_sha256=projected_hash,
        filter_count=len(filters),
        filters_sha256=filters_hash,
        declared_operations=declared,
        executed_view_operations=executed,
        row_identity_scheme=_bounded_text_or_none(row_identity.get("strategy"), 64),
        row_identity_sha256=_sha256_or_none(row_identity.get("ordered_sha256")),
    )


class CompactRunResultResponse(StrictModel):
    """Independent bounded terminal view over an immutable full result."""

    response_detail: Literal["compact"] = "compact"
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    result_record_path: str = Field(min_length=1, max_length=4096)
    result_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    canonical_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_contract_id: str = Field(min_length=1, max_length=256)
    scientific_execution_contract_bound: bool
    provenance_manifest_path: str = Field(min_length=1, max_length=4096)
    provenance_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal["succeeded", "partial_failure"]
    contract_status: Literal["complete", "incomplete"]
    missing_artifact_requirement_ids: tuple[ArtifactRequirementId, ...] = Field(
        default=(),
        max_length=_MAX_COMPACT_MISSING_REQUIREMENT_IDS,
    )
    missing_artifact_requirement_ids_total_count: int = Field(0, ge=0, le=256)
    missing_artifact_requirement_ids_truncated: bool = False
    missing_artifact_requirement_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: AnalysisTaskName
    model: Annotated[str, Field(min_length=1, max_length=80)]
    tuning: Literal["manual", "automl", "not_applicable"]
    output_directory: str = Field(min_length=1, max_length=4096)
    cli_exit_code: int
    cli_started_at: str | None = Field(
        None,
        description="Actual CLI child start, copied unchanged from the full result.",
    )
    cli_finished_at: str | None = Field(
        None,
        description="Actual CLI child finish, copied unchanged from the full result.",
    )
    cli_execution_duration_seconds: float | None = Field(
        None,
        ge=0,
        allow_inf_nan=False,
        description="CLI child monotonic seconds, copied unchanged from the full result.",
    )
    cli_version: str = Field(min_length=1, max_length=128)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_hash_verified: bool
    source_input_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    dataset_preparation: CompactDatasetPreparationSummary
    environment_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    effective_seeds: dict[str, int] = Field(default_factory=dict, max_length=8)
    source_row_count: int | None = Field(None, ge=0)
    row_identity_scheme: str | None = Field(None, min_length=1, max_length=128)
    row_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    source_row_pairing_verified: bool | None = None
    source_row_pairing_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    preprocessing_summary: PreprocessingSummary | None = None
    application_input_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    application_input_hash_verified: bool | None = None
    event_input_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    event_input_hash_verified: bool | None = None
    reported_metrics: dict[str, Any]
    reported_metrics_truncated: bool = False
    reported_metrics_original_bytes: int = Field(ge=0)
    reported_metric_groups_omitted: int = Field(0, ge=0)
    required_tabular_observations: RequiredTabularObservationSummary = Field(
        default_factory=RequiredTabularObservationSummary,
    )
    artifact_count: int = Field(ge=0)
    canonical_artifact_count: int = Field(ge=0)
    summary_mirror_count: int = Field(0, ge=0)
    artifact_index_path: str = Field(min_length=1, max_length=4096)
    artifact_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_view: Literal["canonical", "all"]
    artifact_view_count: int = Field(ge=0)
    artifact_offset: int = Field(ge=0)
    returned_artifact_count: int = Field(ge=0, le=_MAX_COMPACT_ARTIFACT_REFERENCES)
    next_artifact_offset: int | None = Field(None, ge=0)
    artifacts: tuple[CompactArtifactReference, ...] = Field(max_length=_MAX_COMPACT_ARTIFACT_REFERENCES)
    artifacts_truncated: bool
    aggregate_state: Literal["complete", "partial_failure"] | None = None
    aggregate_summary: AggregateResultSummary | None = None
    children: tuple[CompactChildModelResult, ...] = Field(
        default=(),
        max_length=_MAX_COMPACT_CHILD_RESULTS,
    )
    children_total_count: int = Field(0, ge=0, le=10_000)
    children_truncated: bool = False
    children_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations: tuple[Annotated[str, Field(min_length=1, max_length=512)], ...] = Field(
        default=(),
        max_length=_MAX_COMPACT_LIMITATIONS,
    )
    limitations_total_count: int = Field(0, ge=0, le=10_000)
    limitations_truncated: bool = False
    limitations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_compact_invariants(self) -> "CompactRunResultResponse":
        validate_cli_execution_interval(
            self.cli_started_at,
            self.cli_finished_at,
            self.cli_execution_duration_seconds,
        )
        if self.cli_started_at is None:
            raise ValueError("a successful or partial scientific result requires a CLI child-process interval")
        if self.returned_artifact_count != len(self.artifacts):
            raise ValueError("returned_artifact_count must equal the compact artifact page length")
        if self.returned_artifact_count > self.artifact_view_count:
            raise ValueError("a compact artifact page cannot exceed its selected view")
        if self.artifact_offset + self.returned_artifact_count > self.artifact_view_count:
            raise ValueError("a compact artifact page cannot extend beyond its selected view")
        if self.artifact_view == "all" and self.artifact_view_count != self.artifact_count:
            raise ValueError("the all-artifact view count must equal artifact_count")
        if self.artifact_view == "canonical" and self.artifact_view_count != self.canonical_artifact_count:
            raise ValueError("the canonical view count must equal canonical_artifact_count")
        if self.canonical_artifact_count + self.summary_mirror_count != self.artifact_count:
            raise ValueError("canonical artifacts plus proven summary mirrors must equal artifact_count")
        if self.artifacts_truncated != (self.next_artifact_offset is not None):
            raise ValueError("artifacts_truncated must agree with next_artifact_offset")
        compact_metrics_bytes = _json_size_bytes(self.reported_metrics)
        if compact_metrics_bytes > _MAX_COMPACT_REPORTED_METRICS_BYTES:
            raise ValueError("reported_metrics exceeds the compact response budget")
        if self.reported_metrics_original_bytes < compact_metrics_bytes:
            raise ValueError("reported_metrics_original_bytes cannot be smaller than the compact metrics")
        if self.reported_metrics_truncated != (self.reported_metric_groups_omitted > 0):
            raise ValueError("reported_metrics truncation fields are inconsistent")
        if len(self.missing_artifact_requirement_ids) > self.missing_artifact_requirement_ids_total_count:
            raise ValueError("compact missing requirement IDs cannot exceed their full count")
        if self.missing_artifact_requirement_ids_truncated != (len(self.missing_artifact_requirement_ids) < self.missing_artifact_requirement_ids_total_count):
            raise ValueError("compact missing requirement ID truncation fields are inconsistent")
        if not self.missing_artifact_requirement_ids_truncated and self.missing_artifact_requirement_ids_sha256 != _canonical_json_sha256(list(self.missing_artifact_requirement_ids)):
            raise ValueError("complete compact missing requirement IDs must match their SHA-256")
        if len(self.children) > self.children_total_count:
            raise ValueError("compact children cannot exceed their full count")
        if self.children_truncated != (len(self.children) < self.children_total_count):
            raise ValueError("compact child truncation fields are inconsistent")
        if len(self.limitations) > self.limitations_total_count:
            raise ValueError("compact limitations cannot exceed their full count")
        if len(self.limitations) < self.limitations_total_count and not self.limitations_truncated:
            raise ValueError("compact limitation truncation fields are inconsistent")
        if not self.limitations_truncated and self.limitations_sha256 != _canonical_json_sha256(list(self.limitations)):
            raise ValueError("complete compact limitations must match their SHA-256")
        if self.contract_status == "complete" and self.missing_artifact_requirement_ids_total_count:
            raise ValueError("a complete artifact contract cannot list missing requirements")
        if self.contract_status == "incomplete" and not self.missing_artifact_requirement_ids_total_count:
            raise ValueError("an incomplete artifact contract must list missing requirements")
        if self.missing_artifact_requirement_ids_total_count and self.state != "partial_failure":
            raise ValueError("missing required artifacts must remain an explicit partial_failure")
        validate_success_result_state(
            state=self.state,
            contract_status=self.contract_status,
            model=self.model,
            aggregate_state=self.aggregate_state,
            aggregate_summary=self.aggregate_summary,
            children=self.children,
            children_total_count=self.children_total_count,
            children_complete=not self.children_truncated,
        )
        if _json_size_bytes(self.model_dump(mode="json")) > _MAX_COMPACT_RESULT_JSON_BYTES:
            raise ValueError("compact result exceeds the 64 KiB structured JSON budget")
        return self

    @classmethod
    def from_full(cls, response: RunResultResponse) -> "CompactRunResultResponse":
        if (
            response.result_record_path is None
            or response.result_record_sha256 is None
            or response.artifact_index_path is None
            or response.artifact_index_sha256 is None
            or response.canonical_artifact_count is None
            or response.artifact_view_count is None
        ):
            raise ValueError("a compact result requires complete immutable result and artifact-index identities")
        full_missing_ids = response.missing_artifact_requirement_ids
        compact_missing_ids = list(full_missing_ids[:_MAX_COMPACT_MISSING_REQUIREMENT_IDS])
        full_children = response.children
        compact_children = [_compact_child_result(child) for child in full_children[:_MAX_COMPACT_CHILD_RESULTS]]
        compact_limitations, limitations_total_count, limitation_text_truncated, limitations_sha256 = _compact_limitations(response.limitations)
        compact_limitations = list(compact_limitations)
        children_sha256 = _canonical_json_sha256([child.model_dump(mode="json") for child in full_children])
        missing_ids_sha256 = _canonical_json_sha256(list(full_missing_ids))

        compact_artifact_candidates = [_compact_artifact_reference(artifact) for artifact in response.artifacts[:_MAX_COMPACT_ARTIFACT_REFERENCES]]

        metrics_budget = _MAX_COMPACT_REPORTED_METRICS_BYTES

        def build_payload() -> dict[str, Any]:
            metrics, original_size, omitted = _compact_reported_metrics(
                response.reported_metrics,
                metrics_budget,
            )
            return {
                "response_detail": "compact",
                "run_id": response.run_id,
                "result_record_path": response.result_record_path,
                "result_record_sha256": response.result_record_sha256,
                "request_hash": response.request_hash,
                "validation_id": response.validation_id,
                "canonical_contract_hash": response.canonical_contract_hash,
                "compiled_plan_hash": response.compiled_plan_hash,
                "scientific_contract_id": response.scientific_contract_id,
                "scientific_execution_contract_bound": response.scientific_execution_contract_bound,
                "provenance_manifest_path": response.provenance_manifest_path,
                "provenance_manifest_sha256": response.provenance_manifest_sha256,
                "state": response.state,
                "contract_status": response.contract_status,
                "missing_artifact_requirement_ids": tuple(compact_missing_ids),
                "missing_artifact_requirement_ids_total_count": len(full_missing_ids),
                "missing_artifact_requirement_ids_truncated": len(compact_missing_ids) < len(full_missing_ids),
                "missing_artifact_requirement_ids_sha256": missing_ids_sha256,
                "task": response.task,
                "model": response.model,
                "tuning": response.tuning,
                "output_directory": response.output_directory,
                "cli_exit_code": response.cli_exit_code,
                "cli_started_at": response.cli_started_at,
                "cli_finished_at": response.cli_finished_at,
                "cli_execution_duration_seconds": response.cli_execution_duration_seconds,
                "cli_version": response.cli_version,
                "input_sha256": response.input_sha256,
                "input_hash_verified": response.input_hash_verified,
                "source_input_sha256": response.source_input_sha256,
                "dataset_preparation": _compact_dataset_preparation(
                    response.dataset_preparation,
                    fallback_source_row_count=response.source_row_count,
                ).model_dump(mode="json"),
                "environment_identity_sha256": response.environment_identity_sha256,
                "effective_seeds": response.effective_seeds,
                "source_row_count": response.source_row_count,
                "row_identity_scheme": response.row_identity_scheme,
                "row_identity_sha256": response.row_identity_sha256,
                "source_row_pairing_verified": response.source_row_pairing_verified,
                "source_row_pairing_sha256": response.source_row_pairing_sha256,
                "preprocessing_summary": (response.preprocessing_summary.model_dump(mode="json") if response.preprocessing_summary is not None else None),
                "application_input_sha256": response.application_input_sha256,
                "application_input_hash_verified": response.application_input_hash_verified,
                "event_input_sha256": response.event_input_sha256,
                "event_input_hash_verified": response.event_input_hash_verified,
                "reported_metrics": metrics,
                "reported_metrics_truncated": omitted > 0,
                "reported_metrics_original_bytes": original_size,
                "reported_metric_groups_omitted": omitted,
                "required_tabular_observations": response.required_tabular_observations.model_dump(mode="json"),
                "artifact_count": response.artifact_count,
                "canonical_artifact_count": response.canonical_artifact_count,
                "summary_mirror_count": response.summary_mirror_count,
                "artifact_index_path": response.artifact_index_path,
                "artifact_index_sha256": response.artifact_index_sha256,
                "artifact_view": response.artifact_view,
                "artifact_view_count": response.artifact_view_count,
                "artifact_offset": response.artifact_offset,
                "returned_artifact_count": 0,
                "next_artifact_offset": None,
                "artifacts": (),
                "artifacts_truncated": False,
                "aggregate_state": response.aggregate_state,
                "aggregate_summary": (response.aggregate_summary.model_dump(mode="json") if response.aggregate_summary is not None else None),
                "children": tuple(child.model_dump(mode="json") for child in compact_children),
                "children_total_count": len(full_children),
                "children_truncated": len(compact_children) < len(full_children),
                "children_sha256": children_sha256,
                "limitations": tuple(compact_limitations),
                "limitations_total_count": limitations_total_count,
                "limitations_truncated": limitation_text_truncated or len(compact_limitations) < limitations_total_count,
                "limitations_sha256": limitations_sha256,
            }

        artifact_reserve = _MAX_COMPACT_ARTIFACT_RESERVE_BYTES if compact_artifact_candidates else 0
        while True:
            payload = build_payload()
            if _json_size_bytes(payload) <= _MAX_COMPACT_RESULT_JSON_BYTES - artifact_reserve:
                break
            if metrics_budget:
                metrics_budget = metrics_budget // 2 if metrics_budget > 2 else 0
            elif compact_limitations:
                compact_limitations.pop()
            elif compact_children:
                compact_children.pop()
            elif compact_missing_ids:
                compact_missing_ids.pop()
            else:
                raise ValueError("compact result identity metadata cannot fit the 64 KiB structured JSON budget")

        accepted_artifacts: list[CompactArtifactReference] = []
        for artifact in compact_artifact_candidates:
            trial_artifacts = [*accepted_artifacts, artifact]
            page_has_unreturned_artifacts = len(trial_artifacts) < len(response.artifacts)
            next_offset = response.artifact_offset + len(trial_artifacts) if page_has_unreturned_artifacts else response.next_artifact_offset
            trial = {
                **payload,
                "returned_artifact_count": len(trial_artifacts),
                "next_artifact_offset": next_offset,
                "artifacts": tuple(item.model_dump(mode="json") for item in trial_artifacts),
                "artifacts_truncated": next_offset is not None,
            }
            if _json_size_bytes(trial) > _MAX_COMPACT_RESULT_JSON_BYTES:
                break
            accepted_artifacts.append(artifact)
            payload = trial

        if compact_artifact_candidates and not accepted_artifacts:
            raise ValueError("one compact artifact receipt cannot fit the 64 KiB structured JSON budget")
        if not compact_artifact_candidates:
            payload["next_artifact_offset"] = response.next_artifact_offset
            payload["artifacts_truncated"] = response.next_artifact_offset is not None
        return cls.model_validate(payload)


class CompactRunArtifactPageResponse(StrictModel):
    """Additive compact artifact page that never replays terminal scientific core fields."""

    response_detail: Literal["artifact_page"] = "artifact_page"
    additive: Literal[True] = True
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    state: Literal["succeeded", "partial_failure"]
    result_record_path: str = Field(min_length=1, max_length=4096)
    result_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_index_path: str = Field(min_length=1, max_length=4096)
    artifact_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_count: int = Field(ge=0)
    canonical_artifact_count: int = Field(ge=0)
    summary_mirror_count: int = Field(ge=0)
    artifact_view: Literal["canonical", "all"]
    artifact_view_count: int = Field(ge=0)
    artifact_page_number: int = Field(ge=1)
    artifact_offset: int = Field(ge=0)
    artifact_limit: int = Field(ge=1, le=200)
    returned_artifact_count: int = Field(ge=0, le=_MAX_COMPACT_ARTIFACT_REFERENCES)
    next_artifact_offset: int | None = Field(None, ge=0)
    artifacts: tuple[CompactArtifactReference, ...] = Field(max_length=_MAX_COMPACT_ARTIFACT_REFERENCES)
    artifacts_truncated: bool
    artifact_page_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact_page(self) -> "CompactRunArtifactPageResponse":
        if self.returned_artifact_count != len(self.artifacts):
            raise ValueError("returned_artifact_count must equal the additive artifact page length")
        if self.returned_artifact_count > self.artifact_limit:
            raise ValueError("an additive artifact page cannot exceed artifact_limit")
        if self.artifact_offset > self.artifact_view_count:
            raise ValueError("artifact_offset cannot exceed the selected artifact view")
        if self.artifact_offset + self.returned_artifact_count > self.artifact_view_count:
            raise ValueError("an additive artifact page cannot extend beyond its selected view")
        if self.artifact_view == "all" and self.artifact_view_count != self.artifact_count:
            raise ValueError("the all-artifact view count must equal artifact_count")
        if self.artifact_view == "canonical" and self.artifact_view_count != self.canonical_artifact_count:
            raise ValueError("the canonical view count must equal canonical_artifact_count")
        if self.canonical_artifact_count + self.summary_mirror_count != self.artifact_count:
            raise ValueError("canonical artifacts plus proven summary mirrors must equal artifact_count")
        expected_page_number = self.artifact_offset // self.artifact_limit + 1
        if self.artifact_page_number != expected_page_number:
            raise ValueError("artifact_page_number must be the one-based page for artifact_offset and artifact_limit")
        if self.next_artifact_offset is None:
            if self.artifact_offset + self.returned_artifact_count != self.artifact_view_count:
                raise ValueError("a final additive artifact page must end at the selected view count")
        elif self.next_artifact_offset != self.artifact_offset + self.returned_artifact_count:
            raise ValueError("next_artifact_offset must immediately follow the returned artifact page")
        if self.artifacts_truncated != (self.next_artifact_offset is not None):
            raise ValueError("artifacts_truncated must agree with next_artifact_offset")
        page_payload = [artifact.model_dump(mode="json") for artifact in self.artifacts]
        if self.artifact_page_sha256 != _canonical_json_sha256(page_payload):
            raise ValueError("artifact_page_sha256 must bind the complete compact artifact page")
        if _json_size_bytes(self.model_dump(mode="json")) > _MAX_COMPACT_RESULT_JSON_BYTES:
            raise ValueError("compact artifact page exceeds the 64 KiB structured JSON budget")
        return self

    @classmethod
    def from_full(
        cls,
        response: RunResultResponse,
        *,
        artifact_limit: int,
    ) -> "CompactRunArtifactPageResponse":
        if (
            response.result_record_path is None
            or response.result_record_sha256 is None
            or response.artifact_index_path is None
            or response.artifact_index_sha256 is None
            or response.canonical_artifact_count is None
            or response.artifact_view_count is None
        ):
            raise ValueError("an additive artifact page requires complete immutable result and index identities")
        compact_artifacts = [_compact_artifact_reference(artifact) for artifact in response.artifacts[:_MAX_COMPACT_ARTIFACT_REFERENCES]]

        def build_payload(artifacts: list[CompactArtifactReference]) -> dict[str, Any]:
            has_unreturned_page_items = len(artifacts) < len(response.artifacts)
            next_offset = response.artifact_offset + len(artifacts) if has_unreturned_page_items else response.next_artifact_offset
            artifact_payload = [artifact.model_dump(mode="json") for artifact in artifacts]
            return {
                "response_detail": "artifact_page",
                "additive": True,
                "run_id": response.run_id,
                "state": response.state,
                "result_record_path": response.result_record_path,
                "result_record_sha256": response.result_record_sha256,
                "artifact_index_path": response.artifact_index_path,
                "artifact_index_sha256": response.artifact_index_sha256,
                "artifact_count": response.artifact_count,
                "canonical_artifact_count": response.canonical_artifact_count,
                "summary_mirror_count": response.summary_mirror_count,
                "artifact_view": response.artifact_view,
                "artifact_view_count": response.artifact_view_count,
                "artifact_page_number": response.artifact_offset // artifact_limit + 1,
                "artifact_offset": response.artifact_offset,
                "artifact_limit": artifact_limit,
                "returned_artifact_count": len(artifacts),
                "next_artifact_offset": next_offset,
                "artifacts": tuple(artifact_payload),
                "artifacts_truncated": next_offset is not None,
                "artifact_page_sha256": _canonical_json_sha256(artifact_payload),
            }

        payload = build_payload(compact_artifacts)
        while compact_artifacts and _json_size_bytes(payload) > _MAX_COMPACT_RESULT_JSON_BYTES:
            compact_artifacts.pop()
            payload = build_payload(compact_artifacts)
        if response.artifacts and not compact_artifacts:
            raise ValueError("one compact artifact receipt cannot fit the 64 KiB structured JSON budget")
        return cls.model_validate(payload)


class RunResultNotModifiedResponse(StrictModel):
    """Small conditional receipt proving that a terminal result is unchanged."""

    response_detail: Literal["not_modified"] = "not_modified"
    not_modified: Literal[True] = True
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    state: Literal["succeeded", "partial_failure"]
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    canonical_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_contract_id: str = Field(min_length=1, max_length=256)
    scientific_execution_contract_bound: bool
    provenance_manifest_path: str = Field(min_length=1, max_length=4096)
    provenance_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_record_path: str = Field(min_length=1, max_length=4096)
    result_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_index_path: str = Field(min_length=1, max_length=4096)
    artifact_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_count: int = Field(ge=0)
    canonical_artifact_count: int = Field(ge=0)
    summary_mirror_count: int = Field(ge=0)
    artifact_view: Literal["canonical", "all"]
    artifact_view_count: int = Field(ge=0)
    output_directory: str = Field(min_length=1, max_length=4096)
    requery_required: Literal[False] = False
    message: Literal["Terminal result is unchanged; metrics and artifacts were not replayed."] = "Terminal result is unchanged; metrics and artifacts were not replayed."


class CancelRunResponse(StrictModel):
    """Cancellation state after targeting one wrapper-owned run."""

    run_id: str
    state: Literal["cancelled", "cancellation_requested"]
    message: str
