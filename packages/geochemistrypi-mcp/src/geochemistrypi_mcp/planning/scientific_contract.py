"""Canonical, paper-agnostic scientific identities for MCP analysis requests."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..api.schemas import ArtifactRequirement
from .artifact_mapping import AdapterArtifactMapping
from .interaction_plan import InteractionPlan


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the stable SHA-256 identity of one JSON-compatible value."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _decoded_parameter_entries(entries: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    return {name: json.loads(value) for name, value in entries}


def _semantic_label_identity(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, int):
        kind = "integer"
    elif isinstance(value, float):
        kind = "number"
    elif isinstance(value, str):
        kind = "string"
    else:
        raise TypeError(f"Unsupported semantic label type: {type(value).__name__}")
    return {"type": kind, "value": value}


def resolved_environment_profile(request: Any) -> dict[str, Any]:
    """Normalize the named profile or legacy inline environment without merging them."""
    profile = getattr(request, "environment_profile", None)
    if profile is not None:
        value = profile.model_dump(mode="json")
        return {
            "source": "environment_profile",
            "profile_id": profile.profile_id,
            "profile_identity_sha256": canonical_sha256(value),
            "expected_identity_sha256": profile.expected_identity_sha256,
            "python": profile.python,
            "geochemistrypi": profile.geochemistrypi,
            "mcp": profile.mcp,
            "platform": None,
            "runtime": None,
            "dependency_versions": dict(profile.package_versions),
            "runtime_constraints": dict(profile.runtime_constraints),
        }
    environment = request.reproducibility.environment
    value = environment.model_dump(mode="json")
    specified = any(item not in (None, {}, (), []) for item in value.values())
    return {
        "source": "reproducibility.environment",
        "profile_id": (getattr(request.reproducibility, "dependency_profile", None) or "legacy-inline-environment" if specified else None),
        "profile_identity_sha256": canonical_sha256(value) if specified else None,
        "expected_identity_sha256": environment.expected_identity_sha256,
        "python": getattr(environment, "python", None),
        "geochemistrypi": getattr(environment, "geochemistrypi", None),
        "mcp": getattr(environment, "mcp", None),
        "platform": getattr(environment, "platform", None),
        "runtime": getattr(environment, "runtime", None),
        "dependency_versions": dict(environment.dependency_versions),
        "runtime_constraints": {},
    }


def _column_roles(request: Any) -> dict[str, Any]:
    if request.task == "classification":
        return {
            "identifier": request.identifier_column,
            "features": list(request.feature_columns),
            "target": [request.target_column],
        }
    if request.task == "regression":
        return {
            "identifier": request.identifier_column,
            "features": list(request.feature_columns),
            "target": list(request.resolved_target_columns),
        }
    if request.task == "decomposition" and request.mode == "embedding_label_overlay":
        return {
            "coordinate_identifier": request.identifier_column,
            "coordinates": list(request.feature_columns),
            "label_identifier": request.label_identifier_column,
            "label": request.label_column,
        }
    if request.task in {"clustering", "decomposition", "anomaly_detection"}:
        return {
            "identifier": request.identifier_column,
            "features": list(request.feature_columns),
            "metadata": list(getattr(request, "metadata_columns", ())),
        }
    if request.mode == "reference_anomaly_series":
        return {
            "time": request.time_column,
            "signals": list(request.signal_columns),
            "reference_label": request.reference_label_column,
            "comparison_label": request.comparison_label_column,
            "event_time": request.event_time_column,
            "event_identifier": request.event_identifier_column,
            "event_filter": request.event_filter_column,
        }
    if request.mode == "element_mean":
        return {
            "time": request.age_column,
            "values": list(request.element_columns),
            "filter": request.filter_column,
            "identifier": request.identifier_column,
        }
    if request.mode == "continuous":
        return {
            "time": request.age_column,
            "minimum_time": request.minimum_age_column,
            "maximum_time": request.maximum_age_column,
            "value": request.value_column,
            "latitude": request.latitude_column,
            "longitude": request.longitude_column,
            "filter": request.filter_column,
            "identifier": request.identifier_column,
        }
    return {
        "time": request.age_column,
        "comparison_time": request.maximum_age_column,
        "probability": request.probability_column,
        "latitude": request.latitude_column,
        "longitude": request.longitude_column,
        "identifier": request.identifier_column,
    }


def _preprocessing(request: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "missing_values": request.missing_values.model_dump(mode="json"),
        "selected_columns": list(getattr(request, "resolved_selected_columns", ())),
        "engineered_features": [feature.model_dump(mode="json") for feature in getattr(request, "engineered_features", ())],
        "scaling": getattr(request, "scaling", "none"),
    }
    if hasattr(request, "feature_selection"):
        value["feature_selection"] = request.feature_selection.model_dump(mode="json")
    if request.task == "time_series":
        value["feature_engineering"] = request.feature_engineering
    return value


def _parameters(request: Any) -> dict[str, Any]:
    if request.task == "time_series":
        if request.mode == "reference_anomaly_series":
            return {
                "time_column": request.time_column,
                "signal_columns": list(request.signal_columns),
                "reference_label_column": request.reference_label_column,
                "reference_positive_values": list(request.reference_positive_values),
                "reference_label_provenance": request.reference_label_provenance,
                "comparison_label_column": request.comparison_label_column,
                "comparison_positive_values": list(request.comparison_positive_values),
                "comparison_label_provenance": (request.comparison_label_provenance if request.comparison_label_column is not None else None),
                "event_time_column": request.event_time_column,
                "event_identifier_column": request.event_identifier_column,
                "event_filter_column": request.event_filter_column,
                "event_filter_values": list(request.event_filter_values),
                "association_window_days": request.association_window_days,
                "association_direction": request.association_direction,
            }
        shared = {
            "bin_width": request.bin_width,
            "age_unit": request.age_unit,
        }
        if request.mode == "element_mean":
            return {
                **shared,
                "aggregation": request.aggregation,
                "uncertainty": request.uncertainty,
                "minimum_samples_per_bin": request.minimum_samples_per_bin,
                "filter_minimum": request.filter_minimum,
                "filter_maximum": request.filter_maximum,
            }
        if request.mode == "continuous":
            return {
                **shared,
                "iterations": request.iterations,
                "seed": request.seed,
                "fit_curve": request.fit_curve,
                "relative_value_two_sigma": request.relative_value_two_sigma,
                "minimum_samples_per_bin": request.minimum_samples_per_bin,
                "filter_minimum": request.filter_minimum,
                "filter_maximum": request.filter_maximum,
                "compact_y_axis": request.compact_y_axis,
            }
        return {
            **shared,
            "iterations": request.iterations,
            "seed": request.seed,
            "fit_curve": request.fit_curve,
        }
    if request.task == "decomposition" and request.mode == "embedding_label_overlay":
        return {
            "mode": request.mode,
            "join_policy": "exact_identifier_set_one_to_one",
            "positive_label_values": list(request.positive_label_values),
        }
    parameters = {
        "model_selection": request.model_selection.model_dump(mode="json"),
        "model": request.model.model_dump(mode="json"),
        "test_ratio": getattr(request, "test_ratio", None),
        "tuning": getattr(request, "tuning", "not_applicable"),
    }
    if request.task == "classification":
        parameters["metric_average"] = request.metric_average
        parameters["positive_label"] = _semantic_label_identity(request.positive_label)
    return parameters


def _artifact_category(relative_path: str) -> str | None:
    for part in Path(relative_path).parts:
        if part in {"artifacts", "metrics", "parameters", "summary"}:
            return part
    return None


def describe_scientific_output(relative_path: str, workflow_family: str | None = None) -> dict[str, Any]:
    """Describe one declared CLI output using workflow-aware, paper-agnostic roles."""
    normalized = relative_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    lowered = normalized.lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    category = _artifact_category(normalized)
    media_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "application/json",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".joblib": "application/x-joblib",
        ".pdf": "application/pdf",
        ".svg": "image/svg+xml",
        ".png": "image/png",
    }.get(suffix, "application/octet-stream")
    scientific_type = {
        ".csv": "machine_readable_table",
        ".xlsx": "machine_readable_table",
        ".json": "structured_record",
        ".txt": "structured_record",
        ".pdf": "scientific_figure",
        ".svg": "scientific_figure",
        ".png": "scientific_figure",
        ".joblib": "fitted_model",
    }.get(suffix, "scientific_artifact")
    output_role = "scientific.output"
    if "reference anomaly event associations" in name:
        scientific_type = "event_association_table"
        output_role = "reference_anomaly.event_associations"
    elif "reference anomaly artifact index" in name:
        scientific_type = "artifact_index"
        output_role = "provenance.artifact_index"
    elif "reference anomaly time series manifest" in name:
        scientific_type = "scientific_manifest"
        output_role = "provenance.scientific_manifest"
    elif "reference anomaly time series metrics" in name:
        scientific_type = "reference_anomaly_metrics"
        output_role = "reference_anomaly.metrics"
    elif "reference anomaly time series parameters" in name:
        scientific_type = "parameter_record"
        output_role = "provenance.parameters"
    elif "reference anomaly time series" in name:
        scientific_type = "reference_anomaly_joined_table" if suffix == ".csv" else "reference_anomaly_figure"
        output_role = "reference_anomaly.joined_observations" if suffix == ".csv" else "reference_anomaly.figure"
    elif category == "metrics":
        scientific_type = "evaluation_metrics"
        output_role = "evaluation.metrics"
        if workflow_family == "clustering":
            scientific_type = "clustering_quality_metrics"
            output_role = "clustering.quality"
        elif workflow_family == "time_series":
            scientific_type = "time_series_statistics"
            output_role = "time_series.statistics"
        elif "cross validation" in name:
            scientific_type = "cross_validation_metrics"
            output_role = "evaluation.cross_validation"
        elif "model score" in name:
            scientific_type = "holdout_metrics"
            output_role = "evaluation.holdout"
        elif "classification report" in name:
            scientific_type = "classification_report"
            output_role = "evaluation.classification_report"
        elif "metric configuration" in name:
            scientific_type = "evaluation_configuration"
            output_role = "evaluation.configuration"
    elif "scientific execution attestation" in name:
        scientific_type = "parameter_attestation"
        output_role = "provenance.parameters.attested"
    elif name == "target label mapping.xlsx":
        scientific_type = "target_label_mapping"
        output_role = "classification.target_label_mapping"
    elif category == "parameters":
        scientific_type = "parameter_record"
        output_role = "provenance.parameters"
    elif "transform pipeline" in lowered:
        scientific_type = "preprocessing_state"
        output_role = "preprocessing.fitted_state"
    elif "x reduced" in name:
        scientific_type = "embedding_coordinates"
        output_role = "dimension_reduction.coordinates"
    elif "explained variance" in name:
        scientific_type = "explained_variance"
        output_role = "dimension_reduction.quality"
    elif ("component" in name or "pc data" in name) and suffix in {".csv", ".xlsx"}:
        scientific_type = "component_loadings"
        output_role = "dimension_reduction.loadings"
    elif "cluster labels" in name:
        scientific_type = "cluster_assignments"
        output_role = "clustering.assignments"
    elif "x abnormal detection" in name:
        scientific_type = "anomaly_assignments"
        output_role = "anomaly_detection.assignments"
    elif "lof score" in name:
        scientific_type = "anomaly_scores" if suffix in {".csv", ".xlsx"} else "anomaly_score_figure"
        output_role = "anomaly_detection.scores"
    elif name.startswith("x normal") or name.startswith("x abnormal"):
        scientific_type = "anomaly_subset"
        output_role = "anomaly_detection.subset"
    elif "confusion matrix" in name:
        scientific_type = "confusion_matrix_table" if suffix in {".csv", ".xlsx"} else "confusion_matrix_figure"
        output_role = "evaluation.confusion_matrix"
    elif "residuals diagram" in name:
        scientific_type = "residual_table" if suffix in {".csv", ".xlsx"} else "residual_figure"
        output_role = "evaluation.residuals"
    elif "predicted vs" in name or "model prediction" in name or "y test predict" in name or "application data predicted" in name:
        scientific_type = "prediction_table" if suffix in {".csv", ".xlsx"} else "prediction_figure"
        output_role = "evaluation.predictions"
    elif "subaerial proportion" in name or "continuous time series" in name:
        scientific_type = "time_series_table" if suffix == ".csv" else "time_series_figure"
        output_role = "time_series.estimate" if suffix == ".csv" else "time_series.figure"
    elif suffix == ".joblib" and "model" in lowered:
        scientific_type = "fitted_model"
        output_role = "model.fitted"
    return {
        "relative_path": normalized,
        "category": category,
        "media_type": media_type,
        "scientific_type": scientific_type,
        "output_role": output_role,
    }


def _mapping_descriptor(mapping: AdapterArtifactMapping, workflow_family: str) -> dict[str, Any]:
    if mapping.relative_path is None:
        raise ValueError("Unavailable adapter mappings have no artifact descriptor.")
    descriptor = describe_scientific_output(mapping.relative_path, workflow_family)
    descriptor.update(
        {
            "mapping_id": mapping.mapping_id,
            "scientific_type": mapping.scientific_type,
            "output_role": mapping.output_role,
            "adapter_availability": mapping.availability,
        }
    )
    return descriptor


def adapter_output_descriptors(
    plan: InteractionPlan,
    *,
    include_legacy_fallbacks: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return explicit adapter outputs, optionally retaining legacy path semantics."""
    descriptors: list[dict[str, Any]] = []
    mapped_paths: set[str] = set()
    for mapping in plan.artifact_mappings:
        if mapping.availability != "available" or mapping.relative_path is None:
            continue
        normalized = mapping.relative_path.replace("\\", "/")
        mapped_paths.add(normalized)
        descriptor = _mapping_descriptor(mapping, plan.workflow_family)
        descriptors.append(descriptor)
        if include_legacy_fallbacks:
            fallback = describe_scientific_output(normalized, plan.workflow_family)
            if (fallback["scientific_type"], fallback["output_role"]) != (
                descriptor["scientific_type"],
                descriptor["output_role"],
            ):
                descriptors.append(fallback)
    for relative_path in plan.expected_output_relative_paths:
        normalized = relative_path.replace("\\", "/")
        if normalized not in mapped_paths:
            descriptors.append(describe_scientific_output(normalized, plan.workflow_family))
    unique: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for descriptor in descriptors:
        identity = (
            descriptor["relative_path"],
            descriptor["scientific_type"],
            descriptor["output_role"],
        )
        if identity not in identities:
            identities.add(identity)
            unique.append(descriptor)
    return tuple(unique)


def artifact_requirement_matches(requirement: ArtifactRequirement, descriptor: dict[str, Any]) -> bool:
    """Return whether a planned or produced artifact satisfies one requirement."""
    relative_path = str(descriptor["relative_path"]).replace("\\", "/")
    expected = getattr(requirement, "expected_relative_path", None)
    if expected is not None and expected != relative_path and not expected.endswith(f"/{relative_path}") and not relative_path.endswith(f"/{expected}"):
        return False
    if requirement.path_pattern is not None:
        pattern = requirement.path_pattern
        if not PurePosixPath(relative_path).match(pattern) and not PurePosixPath(relative_path).match(f"**/{pattern}"):
            return False
    category = getattr(requirement, "category", None)
    if category is not None and descriptor.get("category") != category:
        return False
    media_types = getattr(requirement, "media_types", ())
    if media_types and descriptor.get("media_type") not in media_types:
        return False
    if requirement.scientific_type != descriptor.get("scientific_type"):
        return False
    output_role = getattr(requirement, "output_role", None)
    if output_role is not None and output_role != descriptor.get("output_role"):
        return False
    return True


def planned_artifact_requirements(request: Any, plan: InteractionPlan) -> tuple[ArtifactRequirement, ...]:
    """Bind explicit requirements or derive stable requirements from the CLI plan."""
    requirements = []
    for descriptor in adapter_output_descriptors(plan):
        identity = {
            "relative_path": descriptor["relative_path"],
            "scientific_type": descriptor["scientific_type"],
            "output_role": descriptor["output_role"],
        }
        required_json_keys = (
            (
                "schema_version",
                "contract.source_sha256",
                "effective_model_parameters",
                "verified_parameter_names",
                "estimator_identity.expected.module_root",
                "estimator_identity.expected.class_name",
                "estimator_identity.observed.module",
                "estimator_identity.observed.qualname",
                "verification_status",
                "attestation_sha256",
            )
            if descriptor["scientific_type"] == "parameter_attestation"
            else ()
        )
        requirements.append(
            ArtifactRequirement(
                requirement_id=f"planned.{canonical_sha256(identity)[:16]}",
                scientific_type=descriptor["scientific_type"],
                output_role=descriptor["output_role"],
                category=descriptor["category"],
                media_types=(descriptor["media_type"],),
                expected_relative_path=descriptor["relative_path"],
                required_json_keys=required_json_keys,
            )
        )
    derived = tuple(requirements)
    if not request.artifact_requirements:
        return derived
    combined = list(request.artifact_requirements)
    system_requirements = tuple(requirement for requirement in derived if plan.scientific_execution_contract_json is not None and requirement.scientific_type == "parameter_attestation")
    for system_requirement in system_requirements:
        collision = next(
            (requirement for requirement in combined if requirement.requirement_id == system_requirement.requirement_id),
            None,
        )
        if collision is None:
            combined.append(system_requirement)
        elif collision != system_requirement:
            raise ValueError("A caller-declared artifact requirement conflicts with the immutable " "scientific execution attestation requirement.")
    return tuple(combined)


def canonical_scientific_contract(request: Any, plan: InteractionPlan) -> dict[str, Any]:
    """Normalize a v1 task request into the additive v2 scientific contract."""
    requirements = planned_artifact_requirements(request, plan)
    training_reference = request.training_dataset.model_dump(mode="json") if request.training_dataset is not None else {"source": "path", "path": str(request.training_dataset_path)}
    application_reference = None
    if getattr(request, "application_dataset", None) is not None:
        application_reference = request.application_dataset.model_dump(mode="json")
    elif getattr(request, "application_dataset_path", None) is not None:
        application_reference = {"source": "path", "path": str(request.application_dataset_path)}
    datasets: dict[str, Any] = {"training": training_reference}
    if application_reference is not None:
        datasets["application"] = application_reference
    evaluation_dataset = getattr(request.evaluation, "evaluation_dataset", None)
    evaluation_dataset_path = getattr(request.evaluation, "evaluation_dataset_path", None)
    if evaluation_dataset is not None:
        datasets["evaluation"] = evaluation_dataset.model_dump(mode="json")
    elif evaluation_dataset_path is not None:
        datasets["evaluation"] = {"source": "path", "path": str(evaluation_dataset_path)}
    event_dataset_path = getattr(request, "event_dataset_path", None)
    if event_dataset_path is not None:
        datasets["events"] = {"source": "path", "path": str(event_dataset_path)}
    return {
        "contract_version": 2,
        "workflow": {
            "family": plan.workflow_family,
            "mode": plan.workflow_mode,
            "method": plan.method,
        },
        "datasets": datasets,
        "column_roles": _column_roles(request),
        "preprocessing": _preprocessing(request),
        "parameters": _parameters(request),
        "parameter_binding": {
            "requested_model": _decoded_parameter_entries(plan.requested_model_parameters),
            "effective_model": _decoded_parameter_entries(plan.effective_model_parameters),
            "model_binding": plan.model_parameter_binding,
            "requested_preprocessing": _decoded_parameter_entries(plan.requested_preprocessing_parameters),
            "effective_preprocessing": _decoded_parameter_entries(plan.effective_preprocessing_parameters),
            "preprocessing_binding": plan.preprocessing_parameter_binding,
        },
        "requested_seeds": dict(plan.requested_seeds),
        "effective_seeds": dict(plan.effective_seeds),
        "seed_binding": plan.seed_binding,
        "evaluation": request.evaluation.model_dump(mode="json"),
        "reproducibility": request.reproducibility.model_dump(mode="json"),
        "environment_profile": resolved_environment_profile(request),
        "adapter_artifact_mappings": [
            {
                "mapping_id": mapping.mapping_id,
                "scientific_type": mapping.scientific_type,
                "output_role": mapping.output_role,
                "relative_path": mapping.relative_path,
                "availability": mapping.availability,
                "reason": mapping.reason,
            }
            for mapping in plan.artifact_mappings
        ],
        "artifact_requirements": [item.model_dump(mode="json") for item in requirements],
    }


@dataclass(frozen=True)
class ScientificCompatibilityAssessment:
    """Multi-dimensional readiness result for one compiled request."""

    execution_ready: bool
    comparison_ready: bool
    claim_ready: bool
    scientific_status: str
    adapter_status: str
    artifact_status: str
    environment_status: str
    blocking_issues: tuple[str, ...]


def assess_scientific_compatibility(
    request: Any,
    plan: InteractionPlan,
    requirements: tuple[ArtifactRequirement, ...],
    environment_snapshot: Any | None = None,
) -> ScientificCompatibilityAssessment:
    """Check exact adapter/evidence support without weakening requested science."""
    blockers = list(plan.blocking_issues)
    scientific_unmet = False
    artifact_unmet = False
    environment_mismatch = False
    evaluation = request.evaluation
    if evaluation.mode == "external_labeled":
        configured_evaluation_mode = None
        if plan.scientific_execution_contract_json is not None:
            configured_evaluation_mode = json.loads(plan.scientific_execution_contract_json).get("evaluation_mode")
        if request.task != "regression" or configured_evaluation_mode != "external_labeled":
            blockers.append(f"The current CLI adapter does not implement the requested {evaluation.mode} " "evaluation contract as a distinct auditable stage.")
            scientific_unmet = True
    requested_folds = getattr(evaluation, "folds", None)
    if requested_folds is not None:
        configured_folds = None
        if plan.scientific_execution_contract_json is not None:
            configured_folds = json.loads(plan.scientific_execution_contract_json).get("cross_validation_folds")
        elif request.task == "regression":
            configured_folds = 10
        if configured_folds != requested_folds:
            blockers.append("The requested cross-validation folds are not bound into the selected CLI adapter.")
            scientific_unmet = True
    if evaluation.mode == "holdout" and request.task not in {"classification", "regression"}:
        blockers.append("Holdout evaluation is available only for supervised-learning adapters.")
        scientific_unmet = True
    if evaluation.mode == "holdout" and request.task in {"classification", "regression"}:
        configured_execution = json.loads(plan.scientific_execution_contract_json) if plan.scientific_execution_contract_json is not None else {}
        effective_split = configured_execution.get(
            "split_strategy",
            "stratified_holdout" if request.task == "classification" else "random_holdout",
        )
        if evaluation.split_strategy not in {"cli_default", effective_split}:
            blockers.append(f"Requested split strategy {evaluation.split_strategy!r} does not match the CLI adapter's " f"effective {effective_split!r} strategy.")
            scientific_unmet = True
    if request.task == "classification":
        configured_execution = json.loads(plan.scientific_execution_contract_json) if plan.scientific_execution_contract_json is not None else {}
        configured_average = configured_execution.get("classification_metric_average")
        configured_positive = configured_execution.get("classification_positive_label")
        expected_positive = _semantic_label_identity(request.positive_label)
        if plan.scientific_execution_contract_json is not None and (configured_average != request.metric_average or configured_positive != expected_positive):
            blockers.append("Classification metric averaging and positive-label semantics were not preserved in the scientific execution contract.")
            scientific_unmet = True
        interaction_binds_average = any(step.id == "metric_average" for step in plan.steps)
        if plan.scientific_execution_contract_json is None and (
            request.positive_label is not None or request.metric_average == "binary" or (request.metric_average != "auto" and not interaction_binds_average)
        ):
            blockers.append("The selected CLI adapter cannot bind the requested classification metric/positive-label contract.")
            scientific_unmet = True
    has_metric_output = any(_artifact_category(relative_path) == "metrics" for relative_path in plan.expected_output_relative_paths)
    evaluation_metrics = getattr(evaluation, "metrics", ())
    comparison_is_post_run = evaluation.mode == "reference_comparison"
    if evaluation_metrics and not comparison_is_post_run and not has_metric_output:
        blockers.append("The compiled adapter does not declare a metric artifact for the requested evaluation metrics.")
        artifact_unmet = True
    requirement_ids = {requirement.requirement_id for requirement in requirements}
    if evaluation.mode not in {"cli_default", "reference_comparison"}:
        for metric in evaluation_metrics:
            artifact_requirement_id = getattr(evaluation, "metric_artifact_bindings", {}).get(metric)
            if artifact_requirement_id is None:
                blockers.append(f"Requested metric {metric!r} is not bound to an artifact requirement.")
                artifact_unmet = True
            elif artifact_requirement_id not in requirement_ids:
                blockers.append(f"Metric {metric!r} references an unavailable artifact requirement.")
                artifact_unmet = True

    reproducibility = request.reproducibility
    requested_seeds = (
        {"model": request.seed}
        if request.task == "time_series" and request.mode in {"subaerial_proportion", "continuous"}
        else {
            "split": getattr(reproducibility, "split_seed", None),
            "model": getattr(reproducibility, "model_seed", None),
            "tuning": getattr(reproducibility, "tuning_seed", None),
        }
    )
    requested_seeds = {role: value for role, value in requested_seeds.items() if value is not None}
    if dict(plan.requested_seeds) != requested_seeds:
        blockers.append("Requested random seeds were not preserved at the CLI adapter boundary.")
        scientific_unmet = True
    effective_seeds = dict(plan.effective_seeds)
    for role, requested in requested_seeds.items():
        if role not in effective_seeds:
            blockers.append(f"The CLI adapter does not expose an effective {role} seed for attestation.")
            scientific_unmet = True
        elif effective_seeds[role] != requested:
            blockers.append(f"Requested {role} seed {requested} does not match the CLI adapter's effective value {effective_seeds[role]}.")
            scientific_unmet = True
    deterministic_policy = getattr(reproducibility, "deterministic_policy", "adapter_default")
    if deterministic_policy in {"fixed_seed_required", "fixed_seed_and_dependency_required"} and plan.seed_binding == "unbound":
        blockers.append(f"{deterministic_policy} cannot be satisfied because one or more stochastic CLI stages do not expose an effective seed for attestation.")
        scientific_unmet = True

    requested_model = _decoded_parameter_entries(plan.requested_model_parameters)
    effective_model = _decoded_parameter_entries(plan.effective_model_parameters)
    if plan.model_parameter_binding == "interaction_plan":
        mismatched_requested_model = {parameter: value for parameter, value in requested_model.items() if parameter not in effective_model or effective_model[parameter] != value}
        if mismatched_requested_model:
            blockers.append("Requested model parameters do not match the values bound into the CLI interaction plan.")
            scientific_unmet = True
    for parameter, expected in getattr(reproducibility, "model_parameter_assertions", {}).items():
        if parameter not in effective_model:
            blockers.append(f"Required model parameter {parameter!r} is not exposed by the selected CLI adapter.")
            scientific_unmet = True
        elif effective_model[parameter] != expected:
            blockers.append(f"Required model parameter {parameter!r} does not match the effective adapter value.")
            scientific_unmet = True

    requested_preprocessing = _decoded_parameter_entries(plan.requested_preprocessing_parameters)
    effective_preprocessing = _decoded_parameter_entries(plan.effective_preprocessing_parameters)
    if requested_preprocessing != effective_preprocessing:
        blockers.append("Requested preprocessing parameters do not match the values bound into the CLI interaction plan.")
        scientific_unmet = True

    observed_identity = getattr(environment_snapshot, "identity_sha256", None)
    observed_record = getattr(environment_snapshot, "record", {}) if environment_snapshot is not None else {}
    environment = resolved_environment_profile(request)
    dependency_constraints = getattr(reproducibility, "dependency_constraints", {})
    environment_specified = bool(
        environment["expected_identity_sha256"]
        or environment["python"]
        or environment["geochemistrypi"]
        or environment["mcp"]
        or environment["platform"]
        or environment["runtime"]
        or environment["dependency_versions"]
        or environment["runtime_constraints"]
        or dependency_constraints
    )
    if plan.environment_profile_id != environment["profile_id"] or plan.environment_profile_identity_sha256 != environment["profile_identity_sha256"]:
        blockers.append("The environment profile identity was not preserved in the compiled execution plan.")
        scientific_unmet = True
        environment_mismatch = True
    if environment["expected_identity_sha256"] is not None and observed_identity != environment["expected_identity_sha256"]:
        blockers.append("The observed CLI environment identity does not match the requested frozen identity.")
        scientific_unmet = True
        environment_mismatch = True
    observed_python = observed_record.get("python", {}) if isinstance(observed_record, dict) else {}
    observed_geochemistrypi = observed_record.get("geochemistrypi", {}) if isinstance(observed_record, dict) else {}
    observed_mcp = observed_record.get("mcp", {}) if isinstance(observed_record, dict) else {}
    observed_runtime = observed_record.get("runtime", {}) if isinstance(observed_record, dict) else {}
    environment_checks = (
        ("Python version", environment["python"], observed_python.get("version")),
        ("GeochemistryPi version", environment["geochemistrypi"], observed_geochemistrypi.get("version")),
        ("MCP version", environment["mcp"], observed_mcp.get("version")),
        ("platform", environment["platform"], observed_record.get("platform") if isinstance(observed_record, dict) else None),
        ("runtime", environment["runtime"], observed_runtime.get("kind")),
    )
    for label, requested, observed in environment_checks:
        if requested is not None and requested != observed:
            blockers.append(f"Requested {label} does not match the observed CLI environment.")
            scientific_unmet = True
            environment_mismatch = True
    observed_dependencies = observed_record.get("dependencies", {}) if isinstance(observed_record, dict) else {}
    runtime_observed = {
        "kind": observed_runtime.get("kind"),
        "python_implementation": observed_python.get("implementation"),
        "platform": observed_record.get("platform") if isinstance(observed_record, dict) else None,
        "cli_executable_sha256": (observed_record.get("cli_executable", {}).get("sha256") if isinstance(observed_record, dict) else None),
    }
    for constraint, requested in environment["runtime_constraints"].items():
        if runtime_observed.get(constraint) != requested:
            blockers.append(f"Runtime constraint {constraint!r} does not match the observed CLI environment.")
            scientific_unmet = True
            environment_mismatch = True
    requested_dependencies = dict(environment["dependency_versions"])
    requested_dependencies.update(dependency_constraints)
    for package, requested in requested_dependencies.items():
        normalized_package = package.strip().lower().replace("_", "-")
        exact = requested[2:] if requested.startswith("==") else requested
        if any(token in exact for token in "<>=!~,*"):
            blockers.append(f"Dependency constraint for {package!r} is not an exact version and cannot freeze the environment.")
            scientific_unmet = True
            environment_mismatch = True
        elif observed_dependencies.get(normalized_package) != exact:
            blockers.append(f"Dependency {package!r} does not match the requested exact version {exact!r}.")
            scientific_unmet = True
            environment_mismatch = True
    if deterministic_policy == "fixed_seed_and_dependency_required" and environment["expected_identity_sha256"] is None:
        blockers.append("fixed_seed_and_dependency_required needs an expected CLI environment identity.")
        scientific_unmet = True
        environment_mismatch = True

    planned_descriptors = adapter_output_descriptors(plan, include_legacy_fallbacks=True)
    for requirement in requirements:
        if not getattr(requirement, "required", True):
            continue
        matched_count = sum(artifact_requirement_matches(requirement, descriptor) for descriptor in planned_descriptors)
        minimum_count = getattr(requirement, "minimum_count", getattr(requirement, "count", 1))
        maximum_count = getattr(requirement, "maximum_count", getattr(requirement, "count", None))
        if matched_count < minimum_count or (maximum_count is not None and matched_count > maximum_count):
            unavailable = [
                mapping
                for mapping in plan.artifact_mappings
                if mapping.availability == "unavailable"
                and mapping.scientific_type == requirement.scientific_type
                and (getattr(requirement, "output_role", None) is None or mapping.output_role == getattr(requirement, "output_role", None))
            ]
            unavailable_reason = f" Adapter limitation: {unavailable[0].reason}" if unavailable else ""
            blockers.append(
                f"Required artifact {requirement.requirement_id!r} is not satisfied by the compiled CLI plan "
                f"(planned matches={matched_count}, required={minimum_count}..{maximum_count or 'unbounded'})."
                f"{unavailable_reason}"
            )
            artifact_unmet = True

    execution_ready = plan.execution_ready and not scientific_unmet and not artifact_unmet
    comparison_ready = evaluation.mode != "reference_comparison"
    if plan.adapter_status == "unavailable":
        adapter_status = "unavailable"
    elif not execution_ready:
        adapter_status = "requirements_unmet"
    else:
        adapter_status = "available"
    environment_status = "MISMATCH" if environment_mismatch else "READY" if environment_specified else "UNSPECIFIED"
    return ScientificCompatibilityAssessment(
        execution_ready=execution_ready,
        comparison_ready=comparison_ready,
        claim_ready=False,
        scientific_status="requirements_unmet" if scientific_unmet else "valid",
        adapter_status=adapter_status,
        artifact_status="requirements_unmet" if artifact_unmet else "planned",
        environment_status=environment_status,
        blocking_issues=tuple(dict.fromkeys(blockers)),
    )
