"""Strict machine configuration for paper-agnostic CLI execution controls.

The interactive CLI remains the default scientific workflow.  The public
``--scientific-config`` option may additionally bind validated seeds,
evaluation semantics, and native estimator parameters in either an interactive
or automated run. MCP uses the same contract rather than a private execution
path. The contract never selects a dataset, a paper, or an expected result.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

SCIENTIFIC_EXECUTION_CONTRACT_VERSION = 4
SCIENTIFIC_EXECUTION_CONTRACT_FIELDS = (
    "schema_version",
    "workflow_family",
    "workflow_mode",
    "method",
    "split_seed",
    "split_strategy",
    "model_seed",
    "cross_validation_folds",
    "evaluation_mode",
    "confusion_matrix_normalization",
    "external_evaluation_identifier_column",
    "external_evaluation_target_columns",
    "target_transformations",
    "classification_metric_average",
    "classification_positive_label",
    "model_parameters",
)
_MAX_CONTRACT_BYTES = 1024 * 1024
_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_EVALUATION_MODES = frozenset(
    {
        "internal_holdout",
        "external_labeled",
        "training_clustering",
        "fit_transform",
        "training_outlier",
        "novelty_detection",
    }
)
_CONFUSION_MATRIX_NORMALIZATIONS = frozenset({"true", "predicted", "all"})
_SPLIT_STRATEGIES = frozenset({"random_holdout", "stratified_holdout"})
_CLASSIFICATION_METRIC_AVERAGES = frozenset({"auto", "binary", "micro", "macro", "weighted"})
_CLASSIFICATION_XGBOOST_OBJECTIVES = frozenset({"auto", "binary:logistic", "multi:softprob", "multi:softmax"})
_XGBOOST_IMPORTANCE_TYPES = frozenset({"gain", "weight", "cover", "total_gain", "total_cover"})
_CLASSIFICATION_METHODS = frozenset(
    {
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
    }
)
_WORKFLOW_METHODS = {
    ("supervised_learning", "classification"): _CLASSIFICATION_METHODS,
    ("supervised_learning", "regression"): frozenset(
        {
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "multi_layer_perceptron",
            "lasso_regression",
            "elastic_net",
            "stochastic_gradient_descent",
        }
    ),
    ("clustering", "clustering"): frozenset({"kmeans", "affinity_propagation"}),
    ("dimension_reduction", "embedding"): frozenset({"pca", "tsne", "mds"}),
    ("anomaly_detection", "outlier_detection"): frozenset({"isolation_forest", "local_outlier_factor"}),
}

_ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS = frozenset(
    {
        ("classification", "support_vector_machine"),
        ("classification", "decision_tree"),
        ("classification", "random_forest"),
        ("classification", "extra_trees"),
        ("classification", "xgboost"),
        ("classification", "multi_layer_perceptron"),
        ("classification", "gradient_boosting"),
        ("classification", "adaboost"),
        ("regression", "decision_tree"),
        ("regression", "random_forest"),
        ("regression", "extra_trees"),
        ("regression", "gradient_boosting"),
        ("regression", "xgboost"),
        ("regression", "multi_layer_perceptron"),
        ("clustering", "kmeans"),
        ("clustering", "affinity_propagation"),
        ("embedding", "tsne"),
        ("embedding", "mds"),
        ("outlier_detection", "isolation_forest"),
    }
)
_CONDITIONAL_MODEL_SEEDED_WORKFLOW_METHODS = frozenset(
    {
        ("classification", "logistic_regression"),
        ("classification", "stochastic_gradient_descent"),
        ("regression", "lasso_regression"),
        ("regression", "elastic_net"),
        ("regression", "stochastic_gradient_descent"),
        ("embedding", "pca"),
    }
)
_MODEL_SEED_CAPABLE_WORKFLOW_METHODS = _ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS | _CONDITIONAL_MODEL_SEEDED_WORKFLOW_METHODS
_CLASSIFICATION_ESTIMATOR_IDENTITIES = {
    "logistic_regression": ("sklearn", "LogisticRegression"),
    "support_vector_machine": ("sklearn", "SVC"),
    "decision_tree": ("sklearn", "DecisionTreeClassifier"),
    "random_forest": ("sklearn", "RandomForestClassifier"),
    "extra_trees": ("sklearn", "ExtraTreesClassifier"),
    "xgboost": ("xgboost", "XGBClassifier"),
    "multi_layer_perceptron": ("sklearn", "MLPClassifier"),
    "gradient_boosting": ("sklearn", "GradientBoostingClassifier"),
    "k_nearest_neighbors": ("sklearn", "KNeighborsClassifier"),
    "stochastic_gradient_descent": ("sklearn", "SGDClassifier"),
    "adaboost": ("sklearn", "AdaBoostClassifier"),
}
_ESTIMATOR_IDENTITIES = {
    **{("classification", method): identity for method, identity in _CLASSIFICATION_ESTIMATOR_IDENTITIES.items()},
    ("regression", "decision_tree"): ("sklearn", "DecisionTreeRegressor"),
    ("regression", "random_forest"): ("sklearn", "RandomForestRegressor"),
    ("regression", "extra_trees"): ("sklearn", "ExtraTreesRegressor"),
    ("regression", "gradient_boosting"): ("sklearn", "GradientBoostingRegressor"),
    ("regression", "xgboost"): ("xgboost", "XGBRegressor"),
    ("regression", "multi_layer_perceptron"): ("sklearn", "MLPRegressor"),
    ("regression", "lasso_regression"): ("sklearn", "Lasso"),
    ("regression", "elastic_net"): ("sklearn", "ElasticNet"),
    ("regression", "stochastic_gradient_descent"): ("sklearn", "SGDRegressor"),
    ("clustering", "kmeans"): ("sklearn", "KMeans"),
    ("clustering", "affinity_propagation"): ("sklearn", "AffinityPropagation"),
    ("embedding", "pca"): ("sklearn", "PCA"),
    ("embedding", "tsne"): ("sklearn", "TSNE"),
    ("embedding", "mds"): ("sklearn", "MDS"),
    ("outlier_detection", "isolation_forest"): ("sklearn", "IsolationForest"),
    ("outlier_detection", "local_outlier_factor"): ("sklearn", "LocalOutlierFactor"),
}
_REGISTERED_WORKFLOW_METHOD_IDENTITIES = {(workflow_mode, method) for (_, workflow_mode), methods in _WORKFLOW_METHODS.items() for method in methods}
if set(_ESTIMATOR_IDENTITIES) != _REGISTERED_WORKFLOW_METHOD_IDENTITIES:
    raise RuntimeError("Scientific estimator identities must cover every registered workflow method exactly.")
_ALLOWED_MODEL_PARAMETERS = {
    "xgboost": frozenset(
        {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "verbosity",
            "objective",
            "booster",
            "tree_method",
            "n_jobs",
            "gamma",
            "min_child_weight",
            "max_delta_step",
            "subsample",
            "colsample_bytree",
            "colsample_bylevel",
            "colsample_bynode",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
            "base_score",
            "missing",
            "num_parallel_tree",
            "importance_type",
            "validate_parameters",
            "predictor",
            "eval_metric",
            "early_stopping_rounds",
        }
    ),
    "extra_trees": frozenset(
        {
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "bootstrap",
            "oob_score",
            "n_jobs",
            "verbose",
            "warm_start",
            "ccp_alpha",
            "max_samples",
        }
    ),
    "isolation_forest": frozenset(
        {
            "n_estimators",
            "contamination",
            "max_features",
            "bootstrap",
            "max_samples",
        }
    ),
    "local_outlier_factor": frozenset(
        {
            "n_neighbors",
            "algorithm",
            "leaf_size",
            "metric",
            "p",
            "contamination",
            "n_jobs",
        }
    ),
    **{method: frozenset() for method in _CLASSIFICATION_METHODS if method not in {"xgboost", "extra_trees"}},
    **{
        method: frozenset()
        for method in {
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "multi_layer_perceptron",
            "lasso_regression",
            "elastic_net",
            "stochastic_gradient_descent",
            "kmeans",
            "affinity_propagation",
            "pca",
            "tsne",
            "mds",
            "isolation_forest",
        }
        if method not in {"xgboost", "extra_trees", "isolation_forest"}
    },
}


class ScientificExecutionContractError(RuntimeError):
    """Raised when machine-supplied scientific controls are unsafe or false."""


def _model_seed_is_applicable(
    workflow_mode: str,
    method: str,
    parameters: Mapping[str, Any],
) -> bool:
    """Return whether the fitted algorithm actually consumes ``random_state``."""

    identity = (workflow_mode, method)
    if identity in _ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS:
        return True
    if identity not in _CONDITIONAL_MODEL_SEEDED_WORKFLOW_METHODS:
        return False
    if identity == ("classification", "logistic_regression"):
        return parameters.get("solver") in {"liblinear", "sag", "saga"}
    if identity == ("classification", "stochastic_gradient_descent"):
        return bool(parameters.get("shuffle")) or bool(parameters.get("early_stopping"))
    if identity in {
        ("regression", "lasso_regression"),
        ("regression", "elastic_net"),
    }:
        return parameters.get("selection") == "random"
    if identity == ("regression", "stochastic_gradient_descent"):
        return bool(parameters.get("shuffle"))
    if identity == ("embedding", "pca"):
        # ``auto`` is intentionally not accepted here: the effective solver is
        # data-shape dependent, so pre-execution seed attestation would be false.
        return parameters.get("svd_solver") in {"arpack", "randomized"}
    raise ScientificExecutionContractError(f"No model-seed applicability rule is registered for {identity!r}.")


def _require_exact_fields(value: Mapping[str, Any], expected: set, location: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ScientificExecutionContractError(f"Unknown {location} fields: {unknown}")
    if missing:
        raise ScientificExecutionContractError(f"Missing {location} fields: {missing}")


def _validate_token(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ScientificExecutionContractError(f"{field_name} must be a bounded identifier token.")
    return value


def _validate_seed(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not float(value).is_integer() or value < 0 or value > 2**32 - 1:
        raise ScientificExecutionContractError(f"{field_name} must be null or an integer between 0 and 2^32-1.")
    # JSON Schema defines ``integer`` mathematically, so JSON values such as
    # ``2025.0`` are integers too. Canonicalize them before they reach sklearn.
    return int(value)


def _validate_parameter_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScientificExecutionContractError(f"model_parameters.{field_name} must be finite.")
        return value
    if isinstance(value, list):
        if len(value) > 64:
            raise ScientificExecutionContractError(f"model_parameters.{field_name} has too many entries.")
        return tuple(_validate_parameter_value(item, field_name) for item in value)
    raise ScientificExecutionContractError(f"model_parameters.{field_name} must be a JSON scalar, null, or bounded scalar array.")


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ScientificExecutionContractError(f"Scientific execution contract is not strict JSON: {value!r} is not finite.")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_label_identity(value: Any) -> Dict[str, Any]:
    """Encode one semantic label without collapsing values such as 1 and "1"."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, int):
        kind = "integer"
    elif isinstance(value, float) and math.isfinite(value):
        kind = "number"
    elif isinstance(value, str):
        kind = "string"
    else:
        raise ScientificExecutionContractError("Semantic labels must be finite JSON strings, booleans, integers, or numbers.")
    return {"type": kind, "value": value}


def _validated_semantic_label_identity(value: Any, field_name: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ScientificExecutionContractError(f"{field_name} must be null or a typed semantic-label object.")
    _require_exact_fields(value, {"type", "value"}, field_name)
    semantic_type = value["type"]
    semantic_value = value["value"]
    if semantic_type == "boolean":
        valid = isinstance(semantic_value, bool)
    elif semantic_type == "integer":
        valid = not isinstance(semantic_value, bool) and isinstance(semantic_value, (int, float)) and math.isfinite(semantic_value) and float(semantic_value).is_integer()
    elif semantic_type == "number":
        valid = not isinstance(semantic_value, bool) and isinstance(semantic_value, (int, float)) and math.isfinite(semantic_value)
    elif semantic_type == "string":
        valid = isinstance(semantic_value, str)
    else:
        valid = False
    if not valid:
        raise ScientificExecutionContractError(f"{field_name} type does not match its JSON value.")
    return {"type": semantic_type, "value": semantic_value}


def resolve_classification_metric_configuration(
    label_config: Mapping[str, Any],
    requested_average: Optional[str] = None,
    requested_positive_label: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a typed semantic class to its actual encoded metric label."""
    average = requested_average or "auto"
    if average not in _CLASSIFICATION_METRIC_AVERAGES:
        raise ScientificExecutionContractError(f"Unsupported classification metric average: {average!r}.")
    raw_records = label_config.get("typed_label_records", ())
    records = []
    if raw_records:
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ScientificExecutionContractError("typed_label_records entries must be objects.")
            identity = _validated_semantic_label_identity(raw_record.get("semantic_label"), "typed_label_records.semantic_label")
            encoded = raw_record.get("encoded_label")
            if identity is None or isinstance(encoded, bool) or not isinstance(encoded, int):
                raise ScientificExecutionContractError("Each typed label record requires a semantic label and integer encoded label.")
            records.append({"semantic_label": identity, "encoded_label": encoded})
    else:
        for label, encoded in label_config.get("custom_label_to_code", {}).items():
            records.append({"semantic_label": semantic_label_identity(label), "encoded_label": int(encoded)})
    if len(records) < 2 or len({record["encoded_label"] for record in records}) != len(records):
        raise ScientificExecutionContractError("Classification metric resolution requires at least two uniquely encoded semantic labels.")
    class_count = len(records)
    if average == "binary" and class_count != 2:
        raise ScientificExecutionContractError("Binary metric averaging requires exactly two final classes.")
    effective_average = "binary" if average == "auto" and class_count == 2 else "weighted" if average == "auto" else average
    positive_identity = _validated_semantic_label_identity(requested_positive_label, "classification_positive_label")
    if effective_average != "binary" and positive_identity is not None:
        raise ScientificExecutionContractError("A positive label is valid only for effective binary metrics.")
    aggregate_positive_record = None
    if positive_identity is not None:
        key = _canonical_json(positive_identity)
        aggregate_positive_record = next((record for record in records if _canonical_json(record["semantic_label"]) == key), None)
        if aggregate_positive_record is None:
            raise ScientificExecutionContractError("The requested positive label is absent after label customization.")
    elif effective_average == "binary":
        aggregate_positive_record = next((record for record in records if record["encoded_label"] == 1), None)
        if aggregate_positive_record is None:
            raise ScientificExecutionContractError("Legacy automatic binary metrics require an encoded class 1.")
    curve_positive_record = None
    if class_count == 2:
        curve_positive_record = aggregate_positive_record or next(
            (record for record in records if record["encoded_label"] == 1),
            None,
        )
        if curve_positive_record is None:
            raise ScientificExecutionContractError("Binary classification curves require an encoded class 1 or an explicit binary positive class.")
    return {
        "schema_version": 2,
        "requested_average": average,
        "effective_average": effective_average,
        "requested_positive_label": positive_identity,
        "aggregate_semantic_positive_label": aggregate_positive_record["semantic_label"] if aggregate_positive_record is not None else None,
        "aggregate_encoded_positive_label": aggregate_positive_record["encoded_label"] if aggregate_positive_record is not None else None,
        "curve_semantic_positive_label": curve_positive_record["semantic_label"] if curve_positive_record is not None else None,
        "curve_encoded_positive_label": curve_positive_record["encoded_label"] if curve_positive_record is not None else None,
        "curve_probability_column_index": None,
        "consumers": {},
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(str(temporary_path), str(path))
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class ScientificExecutionContract:
    """Validated controls bound to one interactive or automated CLI process."""

    schema_version: int
    workflow_family: str
    workflow_mode: str
    method: str
    split_seed: Optional[int]
    split_strategy: Optional[str]
    model_seed: Optional[int]
    cross_validation_folds: int
    evaluation_mode: str
    confusion_matrix_normalization: Optional[str]
    external_evaluation_identifier_column: Optional[str]
    external_evaluation_target_columns: Tuple[str, ...]
    target_transformation_entries: Tuple[Tuple[str, float, float], ...]
    classification_metric_average: Optional[str]
    classification_positive_label: Optional[Dict[str, Any]]
    model_parameter_entries: Tuple[Tuple[str, Any], ...]
    source_sha256: str

    @property
    def model_parameters(self) -> Dict[str, Any]:
        return dict(self.model_parameter_entries)

    @classmethod
    def load(cls, path: Path) -> "ScientificExecutionContract":
        source = Path(path).expanduser()
        if not source.is_absolute():
            raise ScientificExecutionContractError("--scientific-config must be an absolute path.")
        try:
            resolved = source.resolve(strict=True)
            if resolved.stat().st_size > _MAX_CONTRACT_BYTES:
                raise ScientificExecutionContractError(f"Scientific execution contract exceeds the {_MAX_CONTRACT_BYTES}-byte safety limit.")
            raw = resolved.read_bytes()
            value = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_nonfinite_json_constant,
            )
        except ScientificExecutionContractError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ScientificExecutionContractError(f"Cannot read scientific execution contract: {source}") from exc
        if not isinstance(value, dict):
            raise ScientificExecutionContractError("Scientific execution contract must be a JSON object.")
        _require_exact_fields(
            value,
            set(SCIENTIFIC_EXECUTION_CONTRACT_FIELDS),
            "scientific execution contract",
        )
        if value["schema_version"] != SCIENTIFIC_EXECUTION_CONTRACT_VERSION:
            raise ScientificExecutionContractError("Unsupported scientific execution schema " f"{value['schema_version']!r}; expected {SCIENTIFIC_EXECUTION_CONTRACT_VERSION}.")
        method = _validate_token(value["method"], "method")
        if method not in _ALLOWED_MODEL_PARAMETERS:
            raise ScientificExecutionContractError(f"No generic CLI parameter binding is registered for method {method!r}.")
        workflow_family = _validate_token(value["workflow_family"], "workflow_family")
        workflow_mode = _validate_token(value["workflow_mode"], "workflow_mode")
        allowed_workflow_methods = _WORKFLOW_METHODS.get((workflow_family, workflow_mode))
        if allowed_workflow_methods is None or method not in allowed_workflow_methods:
            raise ScientificExecutionContractError(f"Scientific execution method {method!r} is not registered for " f"workflow {workflow_family!r}/{workflow_mode!r}.")
        raw_parameters = value["model_parameters"]
        if not isinstance(raw_parameters, dict) or len(raw_parameters) > 64:
            raise ScientificExecutionContractError("model_parameters must be a JSON object with at most 64 entries.")
        unknown_parameters = sorted(set(raw_parameters) - _ALLOWED_MODEL_PARAMETERS[method])
        if unknown_parameters:
            raise ScientificExecutionContractError(f"Unsupported model parameters for {method!r}: {unknown_parameters}")
        parameters = []
        for name, parameter_value in sorted(raw_parameters.items()):
            if not isinstance(name, str) or _PARAMETER_NAME.fullmatch(name) is None:
                raise ScientificExecutionContractError(f"Invalid model parameter name: {name!r}")
            parameters.append((name, _validate_parameter_value(parameter_value, name)))
        folds = value["cross_validation_folds"]
        if isinstance(folds, bool) or not isinstance(folds, (int, float)) or not math.isfinite(folds) or not float(folds).is_integer() or folds < 2 or folds > 100:
            raise ScientificExecutionContractError("cross_validation_folds must be an integer between 2 and 100.")
        folds = int(folds)
        evaluation_mode = value["evaluation_mode"]
        if evaluation_mode not in _EVALUATION_MODES:
            raise ScientificExecutionContractError(f"evaluation_mode must be one of {sorted(_EVALUATION_MODES)}.")
        allowed_evaluation_modes = {
            "supervised_learning": {"internal_holdout", "external_labeled"},
            "clustering": {"training_clustering"},
            "dimension_reduction": {"fit_transform"},
            "anomaly_detection": ({"training_outlier", "novelty_detection"} if method == "local_outlier_factor" else {"training_outlier"}),
        }[workflow_family]
        if evaluation_mode not in allowed_evaluation_modes:
            alternatives = "training_outlier or novelty_detection" if workflow_family == "anomaly_detection" and method == "local_outlier_factor" else " or ".join(sorted(allowed_evaluation_modes))
            raise ScientificExecutionContractError(f"evaluation_mode {evaluation_mode!r} is not registered for " f"workflow {workflow_family!r} and method {method!r}; use " f"{alternatives}.")
        confusion_matrix_normalization = value["confusion_matrix_normalization"]
        if confusion_matrix_normalization is not None and confusion_matrix_normalization not in _CONFUSION_MATRIX_NORMALIZATIONS:
            raise ScientificExecutionContractError("confusion_matrix_normalization must be null, true, predicted, or all.")
        if evaluation_mode == "external_labeled" and (workflow_family != "supervised_learning" or workflow_mode != "regression"):
            raise ScientificExecutionContractError("external_labeled evaluation is registered only for supervised regression.")
        if workflow_family != "supervised_learning" and confusion_matrix_normalization is not None:
            raise ScientificExecutionContractError("Confusion-matrix normalization is available only for supervised learning.")
        if confusion_matrix_normalization is not None and workflow_mode != "classification":
            raise ScientificExecutionContractError("Confusion-matrix normalization is available only for classification.")
        external_identifier = value["external_evaluation_identifier_column"]
        if external_identifier is not None and (
            not isinstance(external_identifier, str) or not external_identifier.strip() or len(external_identifier) > 128 or "\n" in external_identifier or "\r" in external_identifier
        ):
            raise ScientificExecutionContractError("External evaluation identifier must be null or a bounded single-line string.")
        if evaluation_mode != "external_labeled" and external_identifier is not None:
            raise ScientificExecutionContractError("An external evaluation identifier is valid only for external_labeled evaluation.")
        raw_external_targets = value["external_evaluation_target_columns"]
        if not isinstance(raw_external_targets, list) or len(raw_external_targets) > 256:
            raise ScientificExecutionContractError("external_evaluation_target_columns must be a bounded string array.")
        external_targets = []
        for column in raw_external_targets:
            if not isinstance(column, str) or not column.strip() or len(column) > 128 or "\n" in column or "\r" in column:
                raise ScientificExecutionContractError("External evaluation target names must be bounded single-line strings.")
            external_targets.append(column)
        if len(external_targets) != len(set(external_targets)):
            raise ScientificExecutionContractError("External evaluation target names must be unique.")
        if evaluation_mode == "external_labeled" and not external_targets:
            raise ScientificExecutionContractError("external_labeled evaluation requires target columns.")
        if evaluation_mode != "external_labeled" and external_targets:
            raise ScientificExecutionContractError("External evaluation targets are valid only for external_labeled evaluation.")
        raw_transformations = value["target_transformations"]
        if not isinstance(raw_transformations, dict) or len(raw_transformations) > 256:
            raise ScientificExecutionContractError("target_transformations must be a bounded JSON object.")
        transformations = []
        for column, transformation in sorted(raw_transformations.items()):
            if not isinstance(column, str) or not column.strip() or len(column) > 128 or "\n" in column or "\r" in column:
                raise ScientificExecutionContractError("Target transformation names must be bounded non-blank strings.")
            if not isinstance(transformation, dict):
                raise ScientificExecutionContractError(f"Target transformation for {column!r} must be an object.")
            _require_exact_fields(
                transformation,
                {"scale", "offset"},
                "target transformation",
            )
            scale = transformation["scale"]
            offset = transformation["offset"]
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(scale)
                or scale == 0
                or isinstance(offset, bool)
                or not isinstance(offset, (int, float))
                or not math.isfinite(offset)
            ):
                raise ScientificExecutionContractError(f"Target transformation for {column!r} requires finite scale/offset and non-zero scale.")
            transformations.append((column, float(scale), float(offset)))
        if transformations and workflow_mode != "regression":
            raise ScientificExecutionContractError("Target transformations are available only for regression.")
        classification_metric_average = value["classification_metric_average"]
        classification_positive_label = _validated_semantic_label_identity(
            value["classification_positive_label"],
            "classification_positive_label",
        )
        if workflow_mode == "classification":
            if classification_metric_average not in _CLASSIFICATION_METRIC_AVERAGES:
                raise ScientificExecutionContractError(f"classification_metric_average must be one of {sorted(_CLASSIFICATION_METRIC_AVERAGES)}.")
            if classification_metric_average == "binary" and classification_positive_label is None:
                raise ScientificExecutionContractError("Explicit binary metric averaging requires classification_positive_label.")
            if classification_metric_average in {"micro", "macro", "weighted"} and classification_positive_label is not None:
                raise ScientificExecutionContractError("Non-binary metric averaging cannot declare classification_positive_label.")
        elif classification_metric_average is not None or classification_positive_label is not None:
            raise ScientificExecutionContractError("Classification metric controls are valid only for classification workflows.")
        resolved_parameters = dict(parameters)
        if method == "xgboost" and workflow_mode == "classification":
            objective = resolved_parameters.get("objective")
            if objective is not None and objective not in _CLASSIFICATION_XGBOOST_OBJECTIVES:
                raise ScientificExecutionContractError("Classification XGBoost objective must be auto, binary:logistic, multi:softprob, or multi:softmax.")
            importance_type = resolved_parameters.get("importance_type")
            if importance_type is not None and importance_type not in _XGBOOST_IMPORTANCE_TYPES:
                raise ScientificExecutionContractError("XGBoost importance_type must be gain, weight, cover, total_gain, or total_cover.")
        split_seed = _validate_seed(value["split_seed"], "split_seed")
        model_seed = _validate_seed(value["model_seed"], "model_seed")
        seed_identity = (workflow_mode, method)
        if model_seed is not None and seed_identity not in _MODEL_SEED_CAPABLE_WORKFLOW_METHODS:
            raise ScientificExecutionContractError(f"model_seed is not applicable to method {method!r} in " f"workflow mode {workflow_mode!r}.")
        if seed_identity in _ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS and model_seed is None:
            raise ScientificExecutionContractError(f"method {method!r} requires an attested model_seed.")
        split_strategy = value["split_strategy"]
        if split_strategy is not None and split_strategy not in _SPLIT_STRATEGIES:
            raise ScientificExecutionContractError(f"split_strategy must be null or one of {sorted(_SPLIT_STRATEGIES)}.")
        if evaluation_mode == "external_labeled" and split_seed is not None:
            raise ScientificExecutionContractError("external_labeled evaluation fits the complete training cohort and must not declare a split_seed.")
        if evaluation_mode == "external_labeled" and split_strategy is not None:
            raise ScientificExecutionContractError("external_labeled evaluation must not declare a split_strategy.")
        if evaluation_mode == "internal_holdout" and workflow_mode == "classification" and split_strategy is None:
            raise ScientificExecutionContractError("Classification holdout evaluation requires an explicit split_strategy.")
        if evaluation_mode == "internal_holdout" and workflow_mode == "regression" and split_strategy != "random_holdout":
            raise ScientificExecutionContractError("Regression holdout evaluation requires split_strategy='random_holdout'.")
        if workflow_family != "supervised_learning" and split_strategy is not None:
            raise ScientificExecutionContractError("split_strategy is available only for supervised learning.")
        if workflow_family != "supervised_learning" and split_seed is not None:
            raise ScientificExecutionContractError("split_seed is available only for supervised learning.")
        return cls(
            schema_version=SCIENTIFIC_EXECUTION_CONTRACT_VERSION,
            workflow_family=workflow_family,
            workflow_mode=workflow_mode,
            method=method,
            split_seed=split_seed,
            split_strategy=split_strategy,
            model_seed=model_seed,
            cross_validation_folds=folds,
            evaluation_mode=evaluation_mode,
            confusion_matrix_normalization=confusion_matrix_normalization,
            external_evaluation_identifier_column=external_identifier,
            external_evaluation_target_columns=tuple(external_targets),
            target_transformation_entries=tuple(transformations),
            classification_metric_average=classification_metric_average,
            classification_positive_label=classification_positive_label,
            model_parameter_entries=tuple(parameters),
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def resolved_model_parameters(self, class_count: Optional[int] = None) -> Dict[str, Any]:
        """Resolve typed context-dependent parameters before estimator construction."""

        parameters = self.model_parameters
        if self.method != "xgboost" or self.workflow_mode != "classification":
            return parameters
        objective = parameters.get("objective")
        if objective == "auto":
            if class_count is None or class_count < 2:
                raise ScientificExecutionContractError("XGBoost classification objective='auto' requires at least two observed classes.")
            parameters["objective"] = "binary:logistic" if class_count == 2 else "multi:softprob"
        elif class_count is not None:
            if class_count == 2 and objective in {"multi:softprob", "multi:softmax"}:
                raise ScientificExecutionContractError(f"XGBoost objective {objective!r} is incompatible with two-class data.")
            if class_count > 2 and objective == "binary:logistic":
                raise ScientificExecutionContractError("XGBoost objective 'binary:logistic' is incompatible with multiclass data.")
        return parameters

    def validate_selected_workflow(
        self,
        workflow_family: str,
        workflow_mode: str,
        method: Optional[str] = None,
    ) -> None:
        """Bind the sidecar to the actual public-CLI selection before fitting."""

        selected_identity = (workflow_family, workflow_mode)
        configured_identity = (self.workflow_family, self.workflow_mode)
        if selected_identity != configured_identity:
            raise ScientificExecutionContractError(
                "Scientific execution workflow "
                f"{configured_identity[0]!r}/{configured_identity[1]!r} cannot "
                "configure selected CLI workflow "
                f"{selected_identity[0]!r}/{selected_identity[1]!r}."
            )
        if method is not None and self.method != method:
            raise ScientificExecutionContractError(f"Scientific execution method {self.method!r} cannot configure " f"selected CLI method {method!r}.")

    def constructor_parameters(
        self,
        method: str,
        legacy: Mapping[str, Any],
        *,
        workflow_family: str,
        workflow_mode: str,
        class_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.validate_selected_workflow(
            workflow_family,
            workflow_mode,
            method,
        )
        parameters = dict(legacy)
        parameters.update(self.resolved_model_parameters(class_count))
        seed_applicable = _model_seed_is_applicable(
            self.workflow_mode,
            method,
            parameters,
        )
        if seed_applicable and self.model_seed is None:
            raise ScientificExecutionContractError(f"method {method!r} requires a model_seed for this parameterization.")
        if not seed_applicable and self.model_seed is not None:
            raise ScientificExecutionContractError(f"model_seed is not effective for method {method!r} with the " "selected parameters.")
        if seed_applicable:
            parameters["random_state"] = self.model_seed
        if method == "local_outlier_factor":
            parameters["novelty"] = self.evaluation_mode == "novelty_detection"
        return parameters

    def transform_targets(self, values: Any) -> Any:
        """Apply declared affine target transformations without changing row identity."""

        transformed = values.copy()
        for column, scale, offset in self.target_transformation_entries:
            if column not in transformed.columns:
                raise ScientificExecutionContractError(f"Target transformation column {column!r} is absent from the selected data.")
            transformed[column] = transformed[column].astype(float) * scale + offset
        return transformed

    def as_record(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_family": self.workflow_family,
            "workflow_mode": self.workflow_mode,
            "method": self.method,
            "split_seed": self.split_seed,
            "split_strategy": self.split_strategy,
            "model_seed": self.model_seed,
            "cross_validation_folds": self.cross_validation_folds,
            "evaluation_mode": self.evaluation_mode,
            "confusion_matrix_normalization": self.confusion_matrix_normalization,
            "external_evaluation_identifier_column": self.external_evaluation_identifier_column,
            "external_evaluation_target_columns": list(self.external_evaluation_target_columns),
            "target_transformations": {column: {"scale": scale, "offset": offset} for column, scale, offset in self.target_transformation_entries},
            "classification_metric_average": self.classification_metric_average,
            "classification_positive_label": _json_safe(self.classification_positive_label),
            "model_parameters": _json_safe(self.model_parameters),
            "source_sha256": self.source_sha256,
        }


_ACTIVE_CONTRACT: ContextVar[Optional[ScientificExecutionContract]] = ContextVar(
    "geochemistrypi_scientific_execution_contract",
    default=None,
)
_SCIENTIFIC_ATTESTATION_COUNT: ContextVar[int] = ContextVar(
    "geochemistrypi_scientific_attestation_count",
    default=0,
)


def active_scientific_execution() -> Optional[ScientificExecutionContract]:
    """Return the contract active in this CLI process, if any."""

    return _ACTIVE_CONTRACT.get()


@contextmanager
def scientific_execution_context(path: Path) -> Iterator[ScientificExecutionContract]:
    """Activate one strict contract for the duration of a CLI workflow."""

    contract = ScientificExecutionContract.load(path)
    if _ACTIVE_CONTRACT.get() is not None:
        raise ScientificExecutionContractError("Another scientific execution contract is already active.")
    token = _ACTIVE_CONTRACT.set(contract)
    attestation_token = _SCIENTIFIC_ATTESTATION_COUNT.set(0)
    try:
        yield contract
        attestation_count = _SCIENTIFIC_ATTESTATION_COUNT.get()
        if attestation_count != 1:
            raise ScientificExecutionContractError("A successful scientific execution must generate exactly one " "Scientific Execution Attestation.json; observed " f"{attestation_count}.")
    finally:
        _SCIENTIFIC_ATTESTATION_COUNT.reset(attestation_token)
        _ACTIVE_CONTRACT.reset(token)


def _attestation_estimators(
    estimator: Any,
    workflow_mode: str,
) -> Tuple[Tuple[Any, ...], Optional[Dict[str, Any]]]:
    """Return the fitted scientific estimators and any supported wrapper identity."""

    from sklearn.multioutput import MultiOutputRegressor

    if not isinstance(estimator, MultiOutputRegressor):
        return (estimator,), None
    if workflow_mode != "regression":
        raise ScientificExecutionContractError("MultiOutputRegressor is only valid for a regression scientific contract.")
    fitted_estimators = tuple(getattr(estimator, "estimators_", ()))
    if not fitted_estimators:
        raise ScientificExecutionContractError("Multi-output regression attestation requires fitted child estimators.")
    estimator_type = type(estimator)
    return fitted_estimators, {
        "module": estimator_type.__module__,
        "qualname": estimator_type.__qualname__,
        "fitted_estimator_count": len(fitted_estimators),
    }


def save_scientific_execution_attestation(
    estimator: Any,
    output_directory: Optional[str],
    classification_metric_configuration: Optional[Mapping[str, Any]] = None,
) -> None:
    """Fail closed unless the fitted estimator contains every bound parameter."""

    contract = active_scientific_execution()
    if contract is None:
        return
    if not output_directory:
        raise ScientificExecutionContractError("GEOPI_OUTPUT_PARAMETERS_PATH is required for scientific attestation.")
    scientific_estimators, wrapper_identity = _attestation_estimators(
        estimator,
        contract.workflow_mode,
    )
    observed_parameter_sets = []
    for scientific_estimator in scientific_estimators:
        if not hasattr(scientific_estimator, "get_params"):
            raise ScientificExecutionContractError("The selected estimator cannot attest effective parameters.")
        observed_parameter_sets.append(_json_safe(scientific_estimator.get_params(deep=False)))
    observed = observed_parameter_sets[0]
    if any(_canonical_json(parameters) != _canonical_json(observed) for parameters in observed_parameter_sets[1:]):
        raise ScientificExecutionContractError("Multi-output regression child estimators do not share one effective parameterization.")
    class_count = None
    scientific_estimator = scientific_estimators[0]
    if contract.workflow_mode == "classification" and hasattr(scientific_estimator, "classes_"):
        class_count = len(scientific_estimator.classes_)
    expected = contract.resolved_model_parameters(class_count)
    seed_applicable = _model_seed_is_applicable(
        contract.workflow_mode,
        contract.method,
        observed,
    )
    if seed_applicable and contract.model_seed is None:
        raise ScientificExecutionContractError(f"The fitted {contract.method!r} parameterization is stochastic but " "the scientific execution contract omitted model_seed.")
    if not seed_applicable and contract.model_seed is not None:
        raise ScientificExecutionContractError(f"The scientific execution contract declared model_seed for a " f"non-stochastic {contract.method!r} parameterization.")
    if seed_applicable:
        expected = {**expected, "random_state": contract.model_seed}
    if contract.method == "local_outlier_factor":
        expected = {
            **expected,
            "novelty": contract.evaluation_mode == "novelty_detection",
        }
    mismatches = {}
    for name, expected_value in expected.items():
        observed_value = observed.get(name, "<missing>")
        if _canonical_json(_json_safe(observed_value)) != _canonical_json(_json_safe(expected_value)):
            mismatches[name] = {
                "expected": _json_safe(expected_value),
                "observed": observed_value,
            }
    if mismatches:
        raise ScientificExecutionContractError("The fitted estimator does not match the scientific execution contract: " + _canonical_json(mismatches))
    try:
        expected_module_root, expected_class_name = _ESTIMATOR_IDENTITIES[(contract.workflow_mode, contract.method)]
    except KeyError as exc:
        raise ScientificExecutionContractError("The selected scientific method has no registered estimator identity.") from exc
    expected_estimator_identity = {
        "module_root": expected_module_root,
        "class_name": expected_class_name,
    }
    observed_estimator_identities = [
        {
            "module": type(value).__module__,
            "qualname": type(value).__qualname__,
        }
        for value in scientific_estimators
    ]
    identity_mismatches = [
        identity
        for value, identity in zip(
            scientific_estimators,
            observed_estimator_identities,
        )
        if type(value).__module__.split(".", 1)[0] != expected_module_root or type(value).__name__ != expected_class_name
    ]
    if identity_mismatches:
        raise ScientificExecutionContractError(
            "The fitted estimator identity does not match the selected scientific method: "
            + _canonical_json(
                {
                    "expected": expected_estimator_identity,
                    "observed": identity_mismatches,
                }
            )
        )
    observed_estimator_identity = dict(observed_estimator_identities[0])
    if wrapper_identity is not None:
        observed_estimator_identity["wrapper"] = wrapper_identity
    metric_record = None
    if contract.workflow_mode == "classification":
        if not hasattr(estimator, "classes_") or class_count is None or class_count < 2:
            raise ScientificExecutionContractError("Classification scientific execution requires a fitted estimator with at least two observed classes.")
        if not isinstance(classification_metric_configuration, Mapping):
            raise ScientificExecutionContractError("Classification scientific execution requires consumed metric semantics for attestation.")
        metric_record = _json_safe(classification_metric_configuration)
        if metric_record.get("schema_version") != 2:
            raise ScientificExecutionContractError("Unsupported consumed classification metric semantics schema.")
        requested_average = metric_record.get("requested_average")
        requested_positive = metric_record.get("requested_positive_label")
        if requested_average != contract.classification_metric_average:
            raise ScientificExecutionContractError("Consumed metric averaging does not match the scientific execution contract.")
        if _canonical_json(requested_positive) != _canonical_json(contract.classification_positive_label):
            raise ScientificExecutionContractError("Consumed positive-label semantics do not match the scientific execution contract.")
        effective_average = metric_record.get("effective_average")
        expected_average = (
            "binary"
            if contract.classification_metric_average == "auto" and class_count == 2
            else "weighted"
            if contract.classification_metric_average == "auto"
            else contract.classification_metric_average
        )
        if effective_average != expected_average:
            raise ScientificExecutionContractError("Effective classification metric averaging was not consumed as declared.")
        aggregate_positive = metric_record.get("aggregate_encoded_positive_label")
        curve_positive = metric_record.get("curve_encoded_positive_label")
        curve_probability_index = metric_record.get("curve_probability_column_index")
        if effective_average == "binary":
            if aggregate_positive is None or _canonical_json(metric_record.get("aggregate_semantic_positive_label")) != _canonical_json(metric_record.get("curve_semantic_positive_label")):
                raise ScientificExecutionContractError("Binary aggregate metrics and binary curves did not consume the same positive class.")
            if _canonical_json(aggregate_positive) != _canonical_json(curve_positive):
                raise ScientificExecutionContractError("Binary aggregate metrics and binary curves used inconsistent encoded positive classes.")
        elif aggregate_positive is not None or metric_record.get("aggregate_semantic_positive_label") is not None:
            raise ScientificExecutionContractError("Non-binary aggregate metrics cannot consume a positive class.")
        consumers = metric_record.get("consumers")
        required_consumers = {"holdout_score", "cross_validation"}
        if class_count == 2:
            required_consumers.update({"precision_recall", "precision_recall_threshold", "roc"})
            classes = [_json_safe(item) for item in getattr(estimator, "classes_", ())]
            matching_indexes = [index for index, item in enumerate(classes) if type(item) is type(curve_positive) and item == curve_positive]
            if len(matching_indexes) != 1 or curve_probability_index != matching_indexes[0]:
                raise ScientificExecutionContractError("The attested positive class does not match the estimator probability column.")
        elif any(
            metric_record.get(name) is not None
            for name in (
                "curve_semantic_positive_label",
                "curve_encoded_positive_label",
                "curve_probability_column_index",
            )
        ):
            raise ScientificExecutionContractError("Multiclass execution cannot attest binary-curve positive-class semantics.")
        if not isinstance(consumers, Mapping) or not required_consumers.issubset(consumers):
            missing_consumers = sorted(required_consumers - set(consumers or {}))
            raise ScientificExecutionContractError(f"Classification metric semantics were not consumed by: {missing_consumers}")
        for consumer in required_consumers:
            consumption = consumers[consumer]
            if not isinstance(consumption, Mapping):
                raise ScientificExecutionContractError(f"Classification metric consumer {consumer!r} did not publish structured semantics.")
            if consumer in {"holdout_score", "cross_validation"}:
                if (
                    consumption.get("consumer_kind") != "aggregate_metric"
                    or consumption.get("effective_average") != effective_average
                    or _canonical_json(consumption.get("aggregate_encoded_positive_label")) != _canonical_json(aggregate_positive)
                ):
                    raise ScientificExecutionContractError(f"Classification aggregate metric consumer {consumer!r} used inconsistent semantics.")
            elif (
                consumption.get("consumer_kind") != "binary_curve"
                or _canonical_json(consumption.get("curve_encoded_positive_label")) != _canonical_json(curve_positive)
                or consumption.get("probability_column_index") != curve_probability_index
            ):
                raise ScientificExecutionContractError(f"Classification curve consumer {consumer!r} used inconsistent positive-class semantics or an inconsistent probability column.")
    elif classification_metric_configuration is not None:
        raise ScientificExecutionContractError("Classification metric attestation is invalid for a non-classification workflow.")
    record = {
        "schema_version": 2,
        "contract": contract.as_record(),
        "effective_model_parameters": observed,
        "verified_parameter_names": sorted(expected),
        "estimator_identity": {
            "expected": expected_estimator_identity,
            "observed": observed_estimator_identity,
        },
        "classification_metric_semantics": metric_record,
        "verification_status": "matched",
    }
    record["attestation_sha256"] = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
    _atomic_write_json(
        Path(output_directory) / "Scientific Execution Attestation.json",
        record,
    )
    _SCIENTIFIC_ATTESTATION_COUNT.set(_SCIENTIFIC_ATTESTATION_COUNT.get() + 1)
