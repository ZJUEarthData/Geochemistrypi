"""Strict client requests and wrapper responses."""

import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

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

    worksheet: str | None = Field(None, min_length=1, max_length=255)
    worksheets: tuple[str, ...] = Field(default=(), min_length=0, max_length=16)
    union_mode: Literal["rows"] | None = None
    source_sheet_column: ColumnName | None = None
    source_row_column: ColumnName | None = None
    header_row_index: int = Field(0, ge=0, le=1_000_000)
    header_row_indices: tuple[int, ...] = Field(default=(), max_length=16)
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
        if not self.worksheets and (self.source_sheet_column is not None or self.source_row_column is not None):
            raise ValueError("source sheet/row columns are used only by a worksheet row union")
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
        if self.worksheets and not generated <= set(self.selected_columns):
            raise ValueError("selected_columns must retain the generated source sheet and row columns")
        return self


class ExplicitDatasetReference(StrictModel):
    """One explicit local path, preserved for users who already know it."""

    source: Literal["path"] = "path"
    path: Path
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
    method: Literal["impute"] = "impute"
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean"
    fill_value: float | None = None

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

    type: Literal["logistic_regression"] = "logistic_regression"
    penalty: Literal["l1", "l2", "elasticnet"] = "l2"
    regularization_strength: float = Field(1.0, gt=0)
    solver: Literal["liblinear", "newton-cg", "lbfgs", "sag", "saga"] = "lbfgs"
    l1_ratio: float | None = Field(None, ge=0, le=1)
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
    type: Literal["support_vector_machine"] = "support_vector_machine"
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    degree: int = Field(3, ge=1, le=100)
    gamma: float = Field(0.1, gt=0)
    regularization_strength: float = Field(1.0, gt=0)
    shrinking: bool = True


class DecisionTreeSettings(StrictModel):
    type: Literal["decision_tree"] = "decision_tree"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    maximum_depth: int = Field(4, ge=1, le=100_000)
    minimum_samples_split: int = Field(2, ge=2)
    minimum_samples_leaf: int = Field(1, ge=1)
    maximum_features: int = Field(1, ge=1)


class ForestSettings(StrictModel):
    number_of_estimators: int = Field(100, ge=1, le=100_000)
    maximum_depth: int = Field(4, ge=1, le=100_000)
    minimum_samples_split: int = Field(2, ge=2)
    minimum_samples_leaf: int = Field(1, ge=1)
    maximum_features: int = Field(1, ge=1)
    bootstrap: bool = True
    maximum_samples: float | None = Field(0.8, gt=0, le=1)
    out_of_bag_score: bool = True

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
    type: Literal["k_nearest_neighbors"] = "k_nearest_neighbors"
    number_of_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = Field(30, ge=1)
    metric: Literal["euclidean", "manhattan", "minkowski"] = "minkowski"
    power: int = Field(2, ge=1)


class StochasticGradientDescentSettings(StrictModel):
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
    power: int | None = Field(None, ge=1)

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
    bootstrap: bool = False
    maximum_samples: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def validate_bootstrap_samples(self) -> "IsolationForestAnomalyDetectionSettings":
        if self.bootstrap and self.maximum_samples is None:
            raise ValueError("maximum_samples is required when bootstrap is enabled")
        if not self.bootstrap and self.maximum_samples is not None:
            raise ValueError("maximum_samples is only used when bootstrap is enabled")
        return self


class LocalOutlierFactorAnomalyDetectionSettings(StrictModel):
    type: Literal["local_outlier_factor"] = "local_outlier_factor"
    number_of_neighbors: int = Field(20, ge=1)
    leaf_size: int = Field(30, ge=1)
    power: float = Field(2.0, gt=0)
    contamination: float = Field(0.3, gt=0, le=0.5)
    number_of_jobs: int = 1

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
    evaluation_dataset_path: Path | None = None
    evaluation_dataset: DatasetReference | None = None
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
        if self.mode == "cross_validation" and self.folds is None:
            raise ValueError("cross_validation evaluation requires folds")
        if self.mode != "cross_validation" and self.folds is not None:
            raise ValueError("folds is used only by cross_validation evaluation")
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

    task: Literal["classification"] = "classification"
    training_dataset_path: Path | None = None
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    target_column: ColumnName
    application_dataset_path: Path | None = None
    application_dataset: DatasetReference | None = None
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: MissingValueHandling = Field(default_factory=RejectMissingValues)
    engineered_features: tuple[EngineeredFeature, ...] = Field(default=(), max_length=20)
    label_customization: LabelCustomization = Field(default_factory=EncodeOriginalLabels)
    metric_average: Literal["micro", "macro", "weighted"] = "weighted"
    scaling: ScalingMethod = "standardization"
    feature_selection: FeatureSelection = Field(default_factory=NoFeatureSelection)
    sample_balancing: Literal["none"] = "none"
    test_ratio: float = Field(0.2, gt=0, lt=1)
    tuning: Literal["manual", "automl"] = "manual"
    model: ClassificationModelSettings = Field(default_factory=LogisticRegressionSettings)

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

    task: Literal["regression"] = "regression"
    training_dataset_path: Path | None = None
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
    application_dataset_path: Path | None = None
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

    task: Literal["clustering"] = "clustering"
    training_dataset_path: Path | None = None
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: MissingValueHandling = Field(default_factory=RejectMissingValues)
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

    task: Literal["decomposition"] = "decomposition"
    training_dataset_path: Path | None = None
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    metadata_columns: tuple[ColumnName, ...] = Field(default=(), max_length=256)
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: MissingValueHandling = Field(default_factory=RejectMissingValues)
    engineered_features: tuple[EngineeredFeature, ...] = Field(default=(), max_length=20)
    scaling: ScalingMethod = "standardization"
    model: DecompositionModelSettings = Field(default_factory=PCADecompositionSettings)

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

    task: Literal["anomaly_detection"] = "anomaly_detection"
    training_dataset_path: Path | None = None
    training_dataset: DatasetReference | None = None
    experiment_name: str = Field(min_length=1, max_length=40)
    existing_experiment_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    run_name: str = Field(min_length=1, max_length=40)
    identifier_column: ColumnName
    feature_columns: tuple[ColumnName, ...] = Field(min_length=1, max_length=256)
    world_map: WorldMapConfiguration = Field(default_factory=DisabledWorldMap)
    model_selection: ModelSelection = Field(default_factory=SingleModelSelection)
    missing_values: MissingValueHandling = Field(default_factory=RejectMissingValues)
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

    evaluation: TimeSeriesEvaluationContract = Field(default_factory=TimeSeriesEvaluationContract)
    reproducibility: TimeSeriesReproducibilityContract = Field(default_factory=TimeSeriesReproducibilityContract)
    artifact_requirements: tuple[TimeSeriesArtifactRequirement, ...] = Field(default=(), max_length=128)

    task: Literal["time_series"] = Field(
        "time_series",
        description="Select the generic Time Series workflow family.",
    )
    mode: Literal["subaerial_proportion", "element_mean"] = "subaerial_proportion"
    training_dataset_path: Path | None = Field(
        None,
        description="Top-level local-path alternative to training_dataset; provide exactly one of the two.",
    )
    training_dataset: DatasetReference | None = Field(
        None,
        description="Required top-level input reference alternative; use this field, not dataset, and provide exactly one training input form.",
    )
    experiment_name: str = Field("Time Series", min_length=1, max_length=40)
    run_name: str = Field("Subaerial Proportion", min_length=1, max_length=40)
    bin_width: float = Field(
        gt=0,
        description="Required top-level bin width in age_unit; use bin_width, not bin_width_ma or model_parameters.",
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
    maximum_age_column: ColumnName = Field("R_MAX_AGE", description="Top-level comparison/maximum-age column role.")
    probability_column: ColumnName = Field("SBAP", description="Top-level subaerial-proportion probability column role.")
    latitude_column: ColumnName = Field("LATITUDE", description="Top-level latitude column role.")
    longitude_column: ColumnName = Field("LONGITUDE", description="Top-level longitude column role.")
    element_columns: tuple[ColumnName, ...] = Field(
        default=(),
        max_length=128,
        description="One or more numeric value columns for element_mean mode.",
    )
    filter_column: ColumnName | None = Field(
        default=None,
        description="Optional numeric filter role for element_mean mode.",
    )
    filter_minimum: float | None = None
    filter_maximum: float | None = None
    aggregation: Literal["mean"] = "mean"
    uncertainty: Literal["standard_error"] = "standard_error"
    minimum_samples_per_bin: int = Field(1, ge=1)
    age_unit: Literal["Ma", "Ga"] = "Ma"
    fit_curve: bool = True
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
        "maximum_age_column",
        "probability_column",
        "latitude_column",
        "longitude_column",
    )
    @classmethod
    def validate_column_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("must be a non-blank single-line column name")
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

    @field_validator("element_columns")
    @classmethod
    def validate_element_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column or "\n" in column or "\r" in column for column in normalized):
            raise ValueError("must contain non-blank single-line column names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("must not contain duplicate column names")
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
        return (
            self.age_column,
            self.maximum_age_column,
            self.probability_column,
            self.latitude_column,
            self.longitude_column,
        )

    @field_validator("bin_width")
    @classmethod
    def validate_finite_bin_width(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @field_validator("filter_minimum", "filter_maximum")
    @classmethod
    def validate_finite_filter_bound(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @model_validator(mode="after")
    def validate_time_series_contract(self) -> "TimeSeriesRequest":
        _validate_dataset_choice(self.training_dataset_path, self.training_dataset, "training_dataset")
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

    dataset_path: Path | None = None
    dataset: DatasetReference | None = None
    sample_rows: int = Field(5, ge=0, le=10)
    detail: Literal["full", "names"] = "full"

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
    """Read one bounded artifact page, optionally waiting for terminal success."""

    wait_seconds: float = Field(0, ge=0, le=300)
    artifact_offset: int = Field(0, ge=0)
    artifact_limit: int | None = Field(None, ge=1, le=200)


class StartAnalysisByValidationRequest(StrictModel):
    """Start the exact immutable request identified by validate_analysis."""

    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


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


class CapabilitiesResponse(StrictModel):
    """Installed wrapper and supported CLI workflow information."""

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
    unsupported_interactions: tuple[str, ...] = ()
    notes: tuple[str, ...]


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
    ]
    workflow_mode: str = Field(min_length=1, max_length=80)
    method: str = Field(min_length=1, max_length=120)
    scientific_contract_id: str = Field("scientific-contract-v1/legacy", min_length=1, max_length=255)
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
    effective_seeds: dict[str, int] = Field(default_factory=dict, max_length=8)
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
    experiment_mode: Literal["new", "existing", "not_applicable"]
    experiment_name: str
    existing_experiment_id: str | None = None
    interaction_plan: str
    analysis_process_started: Literal[False] = False
    warnings: tuple[str, ...] = ()


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
    error: str | None = None


class ArtifactReference(StrictModel):
    """Reference to an original file produced by the CLI."""

    artifact_id: str
    category: Literal["artifacts", "metrics", "parameters", "summary"]
    relative_path: str
    local_path: str
    size_bytes: int = Field(ge=0)
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirement_id: str | None = Field(None, max_length=120)
    requirement_ids: tuple[str, ...] = ()
    scientific_type: str | None = Field(None, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=32)


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


class AggregateResultSummary(StrictModel):
    """Bounded parent counts so child failures are visible at a glance."""

    expected_model_count: int = Field(ge=1)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)


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

    run_id: str
    request_hash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    validation_id: str | None = Field(None, pattern=r"^val-[0-9a-f]{32}$")
    canonical_contract_hash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    compiled_plan_hash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    provenance_manifest_path: str | None = None
    provenance_manifest_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    contract_status: Literal["complete", "incomplete"] = "complete"
    missing_artifact_requirement_ids: tuple[str, ...] = ()
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
        "subaerial_proportion_bootstrap", "element_mean", "all_models"
    ]
    tuning: Literal["manual", "automl", "not_applicable"] = "manual"
    output_directory: str
    interaction_trace: str
    cli_stdout_log: str
    cli_stderr_log: str
    cli_exit_code: int
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
    reported_metrics: dict[str, Any]
    artifact_count: int = Field(ge=0)
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
        if self.preprocessing_summary is None:
            return self
        if self.task != "time_series":
            raise ValueError("preprocessing_summary is available only for Time Series results")
        if self.source_row_count is None or self.preprocessing_summary.input_row_count != self.source_row_count:
            raise ValueError("preprocessing_summary input rows must match source_row_count")
        return self


class CancelRunResponse(StrictModel):
    """Cancellation state after targeting one wrapper-owned run."""

    run_id: str
    state: Literal["cancelled", "cancellation_requested"]
    message: str
