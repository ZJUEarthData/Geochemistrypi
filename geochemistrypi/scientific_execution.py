"""Strict machine configuration for paper-agnostic CLI execution controls.

The interactive CLI remains the default scientific workflow.  MCP automation
may additionally bind already validated seeds and native estimator parameters
through this versioned contract.  The contract never selects a dataset, a
paper, or an expected result.
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

SCIENTIFIC_EXECUTION_CONTRACT_VERSION = 3
_MAX_CONTRACT_BYTES = 1024 * 1024
_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_EVALUATION_MODES = frozenset(
    {
        "internal_holdout",
        "external_labeled",
        "training_outlier",
        "novelty_detection",
    }
)
_CONFUSION_MATRIX_NORMALIZATIONS = frozenset({"true", "predicted", "all"})
_SPLIT_STRATEGIES = frozenset({"random_holdout", "stratified_holdout"})
_CLASSIFICATION_XGBOOST_OBJECTIVES = frozenset({"auto", "binary:logistic", "multi:softprob", "multi:softmax"})
_XGBOOST_IMPORTANCE_TYPES = frozenset({"gain", "weight", "cover", "total_gain", "total_cover"})
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
}


class ScientificExecutionContractError(RuntimeError):
    """Raised when machine-supplied scientific controls are unsafe or false."""


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
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**32 - 1:
        raise ScientificExecutionContractError(f"{field_name} must be null or an integer between 0 and 2^32-1.")
    return value


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    """Validated controls bound to one automated CLI process."""

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
            value = json.loads(raw.decode("utf-8"))
        except ScientificExecutionContractError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ScientificExecutionContractError(f"Cannot read scientific execution contract: {source}") from exc
        if not isinstance(value, dict):
            raise ScientificExecutionContractError("Scientific execution contract must be a JSON object.")
        _require_exact_fields(
            value,
            {
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
                "model_parameters",
            },
            "scientific execution contract",
        )
        if value["schema_version"] != SCIENTIFIC_EXECUTION_CONTRACT_VERSION:
            raise ScientificExecutionContractError("Unsupported scientific execution schema " f"{value['schema_version']!r}; expected {SCIENTIFIC_EXECUTION_CONTRACT_VERSION}.")
        method = _validate_token(value["method"], "method")
        if method not in _ALLOWED_MODEL_PARAMETERS:
            raise ScientificExecutionContractError(f"No generic CLI parameter binding is registered for method {method!r}.")
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
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2 or folds > 100:
            raise ScientificExecutionContractError("cross_validation_folds must be an integer between 2 and 100.")
        evaluation_mode = value["evaluation_mode"]
        if evaluation_mode not in _EVALUATION_MODES:
            raise ScientificExecutionContractError(f"evaluation_mode must be one of {sorted(_EVALUATION_MODES)}.")
        if method != "local_outlier_factor" and evaluation_mode not in {
            "internal_holdout",
            "external_labeled",
        }:
            raise ScientificExecutionContractError(f"evaluation_mode {evaluation_mode!r} is not registered for method {method!r}.")
        if method == "local_outlier_factor" and evaluation_mode not in {
            "training_outlier",
            "novelty_detection",
        }:
            raise ScientificExecutionContractError("Local Outlier Factor must declare training_outlier or novelty_detection evaluation semantics.")
        confusion_matrix_normalization = value["confusion_matrix_normalization"]
        if confusion_matrix_normalization is not None and confusion_matrix_normalization not in _CONFUSION_MATRIX_NORMALIZATIONS:
            raise ScientificExecutionContractError("confusion_matrix_normalization must be null, true, predicted, or all.")
        workflow_family = _validate_token(value["workflow_family"], "workflow_family")
        workflow_mode = _validate_token(value["workflow_mode"], "workflow_mode")
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
            if not isinstance(column, str) or not column.strip() or len(column) > 128:
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
        resolved_parameters = dict(parameters)
        if method == "xgboost" and workflow_mode == "classification":
            objective = resolved_parameters.get("objective")
            if objective not in _CLASSIFICATION_XGBOOST_OBJECTIVES:
                raise ScientificExecutionContractError("Classification XGBoost objective must be auto, binary:logistic, multi:softprob, or multi:softmax.")
            importance_type = resolved_parameters.get("importance_type")
            if importance_type not in _XGBOOST_IMPORTANCE_TYPES:
                raise ScientificExecutionContractError("XGBoost importance_type must be gain, weight, cover, total_gain, or total_cover.")
        split_seed = _validate_seed(value["split_seed"], "split_seed")
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
        return cls(
            schema_version=SCIENTIFIC_EXECUTION_CONTRACT_VERSION,
            workflow_family=workflow_family,
            workflow_mode=workflow_mode,
            method=method,
            split_seed=split_seed,
            split_strategy=split_strategy,
            model_seed=_validate_seed(value["model_seed"], "model_seed"),
            cross_validation_folds=folds,
            evaluation_mode=evaluation_mode,
            confusion_matrix_normalization=confusion_matrix_normalization,
            external_evaluation_identifier_column=external_identifier,
            external_evaluation_target_columns=tuple(external_targets),
            target_transformation_entries=tuple(transformations),
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

    def constructor_parameters(
        self,
        method: str,
        legacy: Mapping[str, Any],
        *,
        class_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.method != method:
            raise ScientificExecutionContractError(f"Scientific execution method {self.method!r} cannot configure selected CLI method {method!r}.")
        parameters = dict(legacy)
        parameters.update(self.resolved_model_parameters(class_count))
        if self.model_seed is not None and method in {"xgboost", "extra_trees"}:
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
            "model_parameters": _json_safe(self.model_parameters),
            "source_sha256": self.source_sha256,
        }


_ACTIVE_CONTRACT: ContextVar[Optional[ScientificExecutionContract]] = ContextVar(
    "geochemistrypi_scientific_execution_contract",
    default=None,
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
    try:
        yield contract
    finally:
        _ACTIVE_CONTRACT.reset(token)


def save_scientific_execution_attestation(estimator: Any, output_directory: Optional[str]) -> None:
    """Fail closed unless the fitted estimator contains every bound parameter."""

    contract = active_scientific_execution()
    if contract is None:
        return
    if not output_directory:
        raise ScientificExecutionContractError("GEOPI_OUTPUT_PARAMETERS_PATH is required for scientific attestation.")
    if not hasattr(estimator, "get_params"):
        raise ScientificExecutionContractError("The selected estimator cannot attest effective parameters.")
    observed = _json_safe(estimator.get_params(deep=False))
    class_count = None
    if contract.workflow_mode == "classification" and hasattr(estimator, "classes_"):
        class_count = len(estimator.classes_)
    expected = contract.resolved_model_parameters(class_count)
    if contract.model_seed is not None and contract.method in {
        "xgboost",
        "extra_trees",
    }:
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
    record = {
        "schema_version": 1,
        "contract": contract.as_record(),
        "effective_model_parameters": observed,
        "verified_parameter_names": sorted(expected),
        "verification_status": "matched",
    }
    record["attestation_sha256"] = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
    _atomic_write_json(
        Path(output_directory) / "Scientific Execution Attestation.json",
        record,
    )
