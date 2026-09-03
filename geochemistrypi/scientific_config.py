"""Public discovery and generation helpers for scientific execution controls.

The strict runtime validator in :mod:`geochemistrypi.scientific_execution`
remains the source of truth.  This module projects that registry into a
versioned JSON Schema, complete editable templates, and named examples without
maintaining a second, task-specific list of supported workflows.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .scientific_execution import (
    _ALLOWED_MODEL_PARAMETERS,
    _ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS,
    _CLASSIFICATION_METRIC_AVERAGES,
    _CLASSIFICATION_XGBOOST_OBJECTIVES,
    _CONDITIONAL_MODEL_SEEDED_WORKFLOW_METHODS,
    _CONFUSION_MATRIX_NORMALIZATIONS,
    _EVALUATION_MODES,
    _SPLIT_STRATEGIES,
    _WORKFLOW_METHODS,
    _XGBOOST_IMPORTANCE_TYPES,
    SCIENTIFIC_EXECUTION_CONTRACT_FIELDS,
    SCIENTIFIC_EXECUTION_CONTRACT_VERSION,
    ScientificExecutionContract,
    ScientificExecutionContractError,
)

SCIENTIFIC_CONFIG_GENERATOR_SCHEMA_VERSION = 1
SCIENTIFIC_CONFIG_JSON_SCHEMA_ID = "https://geochemistrypi.org/schemas/scientific-execution-contract-" f"v{SCIENTIFIC_EXECUTION_CONTRACT_VERSION}.json"

_CONTRACT_FIELDS = SCIENTIFIC_EXECUTION_CONTRACT_FIELDS


def _evaluation_modes_for(
    workflow_family: str,
    workflow_mode: str,
    method: str,
) -> Tuple[str, ...]:
    if workflow_family == "supervised_learning":
        if workflow_mode == "regression":
            return ("external_labeled", "internal_holdout")
        return ("internal_holdout",)
    if workflow_family == "clustering":
        return ("training_clustering",)
    if workflow_family == "dimension_reduction":
        return ("fit_transform",)
    if workflow_family == "anomaly_detection":
        if method == "local_outlier_factor":
            return ("novelty_detection", "training_outlier")
        return ("training_outlier",)
    raise ScientificExecutionContractError(f"No evaluation-mode registry exists for {workflow_family!r}/{workflow_mode!r}.")


def registered_scientific_workflows() -> Tuple[Dict[str, Any], ...]:
    """Return every runtime-registered workflow family, mode, and method."""

    return tuple(
        {
            "workflow_family": workflow_family,
            "workflow_mode": workflow_mode,
            "method": method,
            "evaluation_modes": list(_evaluation_modes_for(workflow_family, workflow_mode, method)),
            "model_seed_policy": (
                "required"
                if (workflow_mode, method) in _ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS
                else "conditional"
                if (workflow_mode, method) in _CONDITIONAL_MODEL_SEEDED_WORKFLOW_METHODS
                else "not_applicable"
            ),
            "allowed_model_parameters": sorted(_ALLOWED_MODEL_PARAMETERS[method]),
        }
        for (workflow_family, workflow_mode), methods in sorted(_WORKFLOW_METHODS.items())
        for method in sorted(methods)
    )


def scientific_config_registry_document() -> Dict[str, Any]:
    """Return the versioned, machine-readable scientific workflow registry."""

    workflows = registered_scientific_workflows()
    return {
        "schema_version": SCIENTIFIC_CONFIG_GENERATOR_SCHEMA_VERSION,
        "scientific_execution_contract_version": (SCIENTIFIC_EXECUTION_CONTRACT_VERSION),
        "description": (
            "Every workflow identity accepted by the strict public "
            "--scientific-config runtime contract. Entries retain evaluation "
            "semantics, seed policy, and all registered model-parameter names."
        ),
        "available_examples": [
            {
                "name": "isolation_forest",
                "workflow_family": "anomaly_detection",
                "workflow_mode": "outlier_detection",
                "method": "isolation_forest",
            }
        ],
        "workflow_count": len(workflows),
        "workflows": list(workflows),
    }


def _parameter_value_schema(name: str, workflow_mode: str) -> Dict[str, Any]:
    if name == "max_samples" and workflow_mode == "outlier_detection":
        return {
            "oneOf": [
                {"const": "auto"},
                {"type": "integer", "minimum": 1},
            ],
            "description": ("Isolation Forest sample count. 'auto' resolves to min(256, n_samples) " "independently of whether sampling uses replacement."),
        }
    if name == "objective" and workflow_mode == "classification":
        return {
            "type": ["null", "string"],
            "enum": [None, *sorted(_CLASSIFICATION_XGBOOST_OBJECTIVES)],
            "description": ("XGBoost classification objective. 'auto' is resolved from the " "observed class count and attested before result publication."),
        }
    if name == "importance_type" and workflow_mode == "classification":
        return {
            "type": ["null", "string"],
            "enum": [None, *sorted(_XGBOOST_IMPORTANCE_TYPES)],
            "description": "Native XGBoost feature-importance calculation.",
        }
    return {
        "$ref": "#/$defs/modelParameterValue",
        "description": (
            f"Optional native estimator constructor parameter '{name}'. "
            "The selected public CLI adapter must consume and attest the exact "
            "effective value; the runtime rejects unsupported or non-finite values."
        ),
    }


def _model_parameters_schema(method: str, workflow_mode: str) -> Dict[str, Any]:
    allowed = sorted(_ALLOWED_MODEL_PARAMETERS[method])
    return {
        "type": "object",
        "default": {},
        "maxProperties": 64,
        "additionalProperties": False,
        "properties": {name: _parameter_value_schema(name, workflow_mode) for name in allowed},
        "description": (
            "Explicit native estimator parameters. Omitted parameters retain the " "normal interactive CLI selection/default behavior; supplied values " "must be consumed and attested exactly."
        ),
    }


def _identity_schema(
    workflow_family: str,
    workflow_mode: str,
    method: str,
) -> Dict[str, Any]:
    evaluation_modes = _evaluation_modes_for(
        workflow_family,
        workflow_mode,
        method,
    )
    template = build_scientific_execution_template(
        workflow_family,
        workflow_mode,
        method,
        _validate=False,
    )
    seed_identity = (workflow_mode, method)
    if seed_identity in _ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS:
        model_seed_schema: Dict[str, Any] = {
            "type": "integer",
            "default": template["model_seed"],
        }
    elif seed_identity in _CONDITIONAL_MODEL_SEEDED_WORKFLOW_METHODS:
        model_seed_schema = {
            "type": ["null", "integer"],
            "default": template["model_seed"],
        }
    else:
        model_seed_schema = {
            "type": "null",
            "const": None,
            "default": None,
        }
    return {
        "title": f"{workflow_family}/{workflow_mode}/{method}",
        "type": "object",
        "properties": {
            "workflow_family": {"const": workflow_family},
            "workflow_mode": {"const": workflow_mode},
            "method": {"const": method},
            "evaluation_mode": {
                "enum": list(evaluation_modes),
                "default": template["evaluation_mode"],
            },
            "split_seed": {"default": template["split_seed"]},
            "split_strategy": {"default": template["split_strategy"]},
            "model_seed": model_seed_schema,
            "classification_metric_average": {"default": template["classification_metric_average"]},
            "model_parameters": _model_parameters_schema(
                method,
                workflow_mode,
            ),
        },
    }


def _cross_field_constraints() -> Tuple[Dict[str, Any], ...]:
    """Share workflow conditions once instead of repeating them in 27 branches."""

    return (
        {
            "if": {
                "properties": {
                    "workflow_family": {
                        "enum": [
                            "anomaly_detection",
                            "clustering",
                            "dimension_reduction",
                        ]
                    }
                },
                "required": ["workflow_family"],
            },
            "then": {
                "properties": {
                    "split_seed": {"type": "null", "const": None},
                    "split_strategy": {"type": "null", "const": None},
                }
            },
        },
        {
            "if": {
                "properties": {"workflow_mode": {"const": "classification"}},
                "required": ["workflow_mode"],
            },
            "then": {
                "properties": {
                    "split_strategy": {"enum": sorted(_SPLIT_STRATEGIES)},
                    "classification_metric_average": {
                        "type": "string",
                        "enum": sorted(_CLASSIFICATION_METRIC_AVERAGES),
                    },
                }
            },
            "else": {
                "properties": {
                    "confusion_matrix_normalization": {
                        "type": "null",
                        "const": None,
                    },
                    "classification_metric_average": {
                        "type": "null",
                        "const": None,
                    },
                    "classification_positive_label": {
                        "type": "null",
                        "const": None,
                    },
                }
            },
        },
        {
            "if": {
                "properties": {"workflow_mode": {"const": "regression"}},
                "required": ["workflow_mode"],
            },
            "else": {"properties": {"target_transformations": {"maxProperties": 0}}},
        },
        {
            "if": {
                "properties": {"evaluation_mode": {"const": "external_labeled"}},
                "required": ["evaluation_mode"],
            },
            "then": {
                "properties": {
                    "split_seed": {"type": "null", "const": None},
                    "split_strategy": {"type": "null", "const": None},
                    "external_evaluation_target_columns": {"minItems": 1},
                }
            },
            "else": {
                "properties": {
                    "external_evaluation_identifier_column": {
                        "type": "null",
                        "const": None,
                    },
                    "external_evaluation_target_columns": {"maxItems": 0},
                }
            },
        },
        {
            "if": {
                "properties": {
                    "workflow_mode": {"const": "regression"},
                    "evaluation_mode": {"const": "internal_holdout"},
                },
                "required": ["workflow_mode", "evaluation_mode"],
            },
            "then": {"properties": {"split_strategy": {"const": "random_holdout"}}},
        },
        {
            "if": {
                "properties": {"classification_metric_average": {"const": "binary"}},
                "required": ["classification_metric_average"],
            },
            "then": {"properties": {"classification_positive_label": {"not": {"type": "null"}}}},
        },
        {
            "if": {
                "properties": {"classification_metric_average": {"enum": ["macro", "micro", "weighted"]}},
                "required": ["classification_metric_average"],
            },
            "then": {
                "properties": {
                    "classification_positive_label": {
                        "type": "null",
                        "const": None,
                    }
                }
            },
        },
    )


def scientific_execution_json_schema() -> Dict[str, Any]:
    """Build the complete versioned JSON Schema from the runtime registry."""

    semantic_label_variants = []
    for semantic_type, value_schema in (
        ("boolean", {"type": "boolean"}),
        ("integer", {"type": "integer"}),
        ("number", {"type": "number"}),
        ("string", {"type": "string"}),
    ):
        semantic_label_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "value"],
                "properties": {
                    "type": {"const": semantic_type},
                    "value": value_schema,
                },
            }
        )
    properties: Dict[str, Any] = {
        "schema_version": {
            "type": "integer",
            "const": SCIENTIFIC_EXECUTION_CONTRACT_VERSION,
            "default": SCIENTIFIC_EXECUTION_CONTRACT_VERSION,
            "description": "Exact scientific execution contract version.",
        },
        "workflow_family": {
            "type": "string",
            "enum": sorted({key[0] for key in _WORKFLOW_METHODS}),
            "description": "Top-level scientific workflow family.",
        },
        "workflow_mode": {
            "type": "string",
            "enum": sorted({key[1] for key in _WORKFLOW_METHODS}),
            "description": "Scientific mode within the selected workflow family.",
        },
        "method": {
            "type": "string",
            "enum": sorted({method for methods in _WORKFLOW_METHODS.values() for method in methods}),
            "description": "Native estimator method selected in the interactive or automated CLI.",
        },
        "split_seed": {
            "type": ["null", "integer"],
            "minimum": 0,
            "maximum": 2**32 - 1,
            "description": ("Train/test split seed. Use null when the workflow has no internal " "holdout split, including external labelled evaluation."),
        },
        "split_strategy": {
            "type": ["null", "string"],
            "enum": [None, *sorted(_SPLIT_STRATEGIES)],
            "description": ("Explicit supervised holdout strategy; classification requires a " "declared strategy and regression uses random_holdout."),
        },
        "model_seed": {
            "type": ["null", "integer"],
            "minimum": 0,
            "maximum": 2**32 - 1,
            "description": ("Estimator random seed. Required for registered stochastic methods, " "forbidden when it cannot affect the selected parameterization."),
        },
        "cross_validation_folds": {
            "type": "integer",
            "minimum": 2,
            "maximum": 100,
            "default": 10,
            "description": ("Cross-validation fold count for workflows that consume cross-validation. " "The complete contract retains this field for every workflow."),
        },
        "evaluation_mode": {
            "type": "string",
            "enum": sorted(_EVALUATION_MODES),
            "description": "Evaluation semantics; permitted values are narrowed by workflow identity.",
        },
        "confusion_matrix_normalization": {
            "type": ["null", "string"],
            "enum": [None, *sorted(_CONFUSION_MATRIX_NORMALIZATIONS)],
            "default": None,
            "description": ("Classification confusion-matrix normalization. Null preserves raw counts; " "the other values normalize by truth, prediction, or all cells."),
        },
        "external_evaluation_identifier_column": {
            "type": ["null", "string"],
            "minLength": 1,
            "maxLength": 128,
            "pattern": r"^(?=.*\S)[^\r\n]+$",
            "default": None,
            "description": ("Identifier column for external-labelled regression evaluation; null " "for every other evaluation mode."),
        },
        "external_evaluation_target_columns": {
            "type": "array",
            "maxItems": 256,
            "uniqueItems": True,
            "default": [],
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "pattern": r"^(?=.*\S)[^\r\n]+$",
            },
            "description": ("Ordered target columns for external-labelled regression evaluation; " "empty for every other evaluation mode."),
        },
        "target_transformations": {
            "type": "object",
            "maxProperties": 256,
            "default": {},
            "propertyNames": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "pattern": r"^(?=.*\S)[^\r\n]+$",
            },
            "additionalProperties": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scale", "offset"],
                "properties": {
                    "scale": {
                        "type": "number",
                        "not": {"const": 0},
                        "description": "Finite non-zero affine scale.",
                    },
                    "offset": {
                        "type": "number",
                        "description": "Finite affine offset.",
                    },
                },
            },
            "description": "Optional per-target affine transformations for regression only.",
        },
        "classification_metric_average": {
            "type": ["null", "string"],
            "enum": [None, *sorted(_CLASSIFICATION_METRIC_AVERAGES)],
            "description": ("Classification metric averaging. Classification requires one declared " "value; every non-classification workflow uses null."),
        },
        "classification_positive_label": {
            "default": None,
            "description": ("Typed semantic positive label for explicit binary classification metrics. " "The type discriminator prevents numeric and textual labels from collapsing."),
            "oneOf": [
                {"type": "null"},
                *semantic_label_variants,
            ],
        },
        "model_parameters": {
            "type": "object",
            "default": {},
            "description": "Narrowed to the exact registered names by workflow identity.",
        },
    }
    identities = [
        _identity_schema(
            item["workflow_family"],
            item["workflow_mode"],
            item["method"],
        )
        for item in registered_scientific_workflows()
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCIENTIFIC_CONFIG_JSON_SCHEMA_ID,
        "title": "GeochemistryPi Scientific Execution Contract",
        "description": (
            "Complete fail-closed controls for one normal interactive or automated "
            "GeochemistryPi data-mining run. The configuration constrains scientific "
            "execution; it does not answer prompts or select a dataset."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": list(_CONTRACT_FIELDS),
        "properties": properties,
        "allOf": [{"oneOf": identities}, *_cross_field_constraints()],
        "$defs": {
            "modelParameterValue": {
                "oneOf": [
                    {
                        "type": [
                            "null",
                            "string",
                            "boolean",
                            "integer",
                            "number",
                        ]
                    },
                    {
                        "type": "array",
                        "maxItems": 64,
                        "items": {"$ref": "#/$defs/modelParameterValue"},
                    },
                ]
            }
        },
        "x-geochemistrypi-generator-schema-version": (SCIENTIFIC_CONFIG_GENERATOR_SCHEMA_VERSION),
        "x-geochemistrypi-registered-workflows": list(registered_scientific_workflows()),
    }


def build_scientific_execution_template(
    workflow_family: str,
    workflow_mode: str,
    method: str,
    *,
    _validate: bool = True,
) -> Dict[str, Any]:
    """Build a complete editable contract for any registered method."""

    methods = _WORKFLOW_METHODS.get((workflow_family, workflow_mode))
    if methods is None:
        raise ScientificExecutionContractError(f"Unknown scientific workflow {workflow_family!r}/{workflow_mode!r}.")
    if method not in methods:
        raise ScientificExecutionContractError(f"Method {method!r} is not registered for " f"{workflow_family!r}/{workflow_mode!r}.")
    supervised = workflow_family == "supervised_learning"
    classification = workflow_mode == "classification"
    document: Dict[str, Any] = {
        "schema_version": SCIENTIFIC_EXECUTION_CONTRACT_VERSION,
        "workflow_family": workflow_family,
        "workflow_mode": workflow_mode,
        "method": method,
        "split_seed": 42 if supervised else None,
        "split_strategy": ("stratified_holdout" if classification else "random_holdout" if supervised else None),
        "model_seed": (42 if (workflow_mode, method) in _ALWAYS_MODEL_SEEDED_WORKFLOW_METHODS else None),
        "cross_validation_folds": 10,
        "evaluation_mode": _evaluation_modes_for(
            workflow_family,
            workflow_mode,
            method,
        )[-1],
        "confusion_matrix_normalization": None,
        "external_evaluation_identifier_column": None,
        "external_evaluation_target_columns": [],
        "target_transformations": {},
        "classification_metric_average": "auto" if classification else None,
        "classification_positive_label": None,
        "model_parameters": {},
    }
    if _validate:
        _validate_generated_contract(document)
    return document


def scientific_execution_example(name: str) -> Dict[str, Any]:
    """Return a complete named example that passes the runtime validator."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized != "isolation_forest":
        raise ScientificExecutionContractError("Unknown scientific-config example. Available examples: isolation_forest.")
    document = build_scientific_execution_template(
        "anomaly_detection",
        "outlier_detection",
        "isolation_forest",
        _validate=False,
    )
    _validate_generated_contract(document)
    return document


def _validate_generated_contract(document: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="geopi-scientific-config-") as directory:
        path = Path(directory).resolve() / "scientific-config.json"
        path.write_bytes(serialized)
        ScientificExecutionContract.load(path)


def write_scientific_config_document(
    path: Path,
    document: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a generated document without implicit replacement."""

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        raise ScientificExecutionContractError("--output must be an absolute path.")
    if destination.suffix.lower() != ".json":
        raise ScientificExecutionContractError("--output must use the .json suffix.")
    try:
        parent = destination.parent.resolve(strict=True)
    except OSError as exc:
        raise ScientificExecutionContractError("--output parent directory must already exist.") from exc
    destination = parent / destination.name
    if destination.is_symlink():
        raise ScientificExecutionContractError("--output cannot replace a symbolic link.")
    if destination.exists() and not overwrite:
        raise ScientificExecutionContractError("--output already exists; pass --force only when replacement is intentional.")
    serialized = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if overwrite:
            os.replace(str(temporary_path), str(destination))
        else:
            try:
                os.link(str(temporary_path), str(destination))
            except FileExistsError as exc:
                raise ScientificExecutionContractError("--output already exists; pass --force only when replacement is intentional.") from exc
    except ScientificExecutionContractError:
        raise
    except OSError as exc:
        raise ScientificExecutionContractError(f"Cannot write scientific-config output: {destination}") from exc
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return destination


def scientific_config_document_json(document: Mapping[str, Any]) -> str:
    """Serialize one public document deterministically for stdout."""

    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
