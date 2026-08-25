"""Explicit scientific roles exposed by original GeochemistryPi CLI outputs."""

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal


@dataclass(frozen=True)
class AdapterArtifactMapping:
    """One auditable adapter capability, including deliberate unavailability."""

    mapping_id: str
    scientific_type: str
    output_role: str
    relative_path: str | None
    availability: Literal["available", "unavailable"] = "available"
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.mapping_id or not self.scientific_type or not self.output_role:
            raise ValueError("Artifact mappings require an identity, scientific type, and output role.")
        if self.availability == "available" and not self.relative_path:
            raise ValueError("An available artifact mapping requires a CLI relative path.")
        if self.availability == "unavailable" and not self.reason:
            raise ValueError("An unavailable artifact mapping must explain the adapter limitation.")


def _mapping_id(scientific_type: str, output_role: str, relative_path: str | None) -> str:
    identity = f"{scientific_type}\n{output_role}\n{relative_path or '<unavailable>'}"
    return f"adapter-output-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def _available(relative_path: str, scientific_type: str, output_role: str) -> AdapterArtifactMapping:
    normalized = relative_path.replace("\\", "/")
    return AdapterArtifactMapping(
        mapping_id=_mapping_id(scientific_type, output_role, normalized),
        scientific_type=scientific_type,
        output_role=output_role,
        relative_path=normalized,
    )


def _unavailable(scientific_type: str, output_role: str, reason: str) -> AdapterArtifactMapping:
    return AdapterArtifactMapping(
        mapping_id=_mapping_id(scientific_type, output_role, None),
        scientific_type=scientific_type,
        output_role=output_role,
        relative_path=None,
        availability="unavailable",
        reason=reason,
    )


def unavailable_artifact_mapping(
    scientific_type: str,
    output_role: str,
    reason: str,
) -> AdapterArtifactMapping:
    """Create an explicit non-capability without exposing private mapping internals."""
    return _unavailable(scientific_type, output_role, reason)


def _semantic_override(relative_path: str) -> tuple[str, str] | None:
    name = PurePosixPath(relative_path.replace("\\", "/")).name.lower()
    suffix = PurePosixPath(name).suffix.lower()
    if name == "y test.xlsx":
        return "evaluation_labels", "evaluation.true_labels"
    if name == "roc curve - probabilities.xlsx":
        return "score_table", "evaluation.scores"
    if "feature importance diagram" in name:
        return (
            "feature_importance_table" if suffix in {".csv", ".xlsx"} else "feature_importance_figure",
            "model.feature_importance",
        )
    if "confusion matrix" in name:
        return (
            "confusion_matrix_table" if suffix in {".csv", ".xlsx"} else "confusion_matrix_figure",
            "evaluation.confusion_matrix",
        )
    if "y test predict" in name or "application data predicted" in name:
        return "prediction_table", "evaluation.predictions"
    if "model score" in name:
        return "holdout_metrics", "evaluation.holdout"
    if "classification report" in name:
        return "classification_report", "evaluation.classification_report"
    if "residuals diagram" in name:
        return (
            "residual_table" if suffix in {".csv", ".xlsx"} else "residual_figure",
            "evaluation.residuals",
        )
    if "predicted vs" in name:
        return (
            "prediction_table" if suffix in {".csv", ".xlsx"} else "prediction_figure",
            "evaluation.predictions",
        )
    if "subaerial proportion" in name:
        return (
            "time_series_bin_table" if suffix == ".csv" else "time_series_figure",
            "time_series.bins" if suffix == ".csv" else "time_series.figure",
        )
    return None


def _default_semantics(relative_path: str, workflow_family: str) -> tuple[str, str]:
    path = PurePosixPath(relative_path.replace("\\", "/"))
    name = path.name.lower()
    suffix = path.suffix.lower()
    parts = set(path.parts)
    if "metrics" in parts:
        if workflow_family == "clustering":
            return "clustering_quality_metrics", "clustering.quality"
        if workflow_family == "time_series":
            return "time_series_statistics", "time_series.statistics"
        return "evaluation_metrics", "evaluation.metrics"
    if "parameters" in parts:
        return "parameter_record", "provenance.parameters"
    if "transform pipeline" in name:
        return "preprocessing_state", "preprocessing.fitted_state"
    if "x reduced" in name:
        return "embedding_coordinates", "dimension_reduction.coordinates"
    if "explained variance" in name:
        return "explained_variance", "dimension_reduction.quality"
    if ("component" in name or "pc data" in name) and suffix in {".csv", ".xlsx"}:
        return "component_loadings", "dimension_reduction.loadings"
    if "cluster labels" in name:
        return "cluster_assignments", "clustering.assignments"
    if "x abnormal detection" in name:
        return "anomaly_assignments", "anomaly_detection.assignments"
    if "lof score" in name:
        return (
            "anomaly_scores" if suffix in {".csv", ".xlsx"} else "anomaly_score_figure",
            "anomaly_detection.scores",
        )
    if name.startswith("x normal") or name.startswith("x abnormal"):
        return "anomaly_subset", "anomaly_detection.subset"
    if suffix == ".joblib":
        return "fitted_model", "model.fitted"
    if suffix in {".csv", ".xlsx"}:
        return "machine_readable_table", "scientific.output"
    if suffix in {".pdf", ".svg", ".png"}:
        return "scientific_figure", "scientific.output"
    if suffix in {".json", ".txt"}:
        return "structured_record", "scientific.output"
    return "scientific_artifact", "scientific.output"


def build_adapter_artifact_mappings(
    workflow_family: str,
    workflow_mode: str,
    expected_output_relative_paths: tuple[str, ...],
    method: str | None = None,
) -> tuple[AdapterArtifactMapping, ...]:
    """Bind generic scientific roles to paths the selected CLI adapter emits."""
    mappings = []
    for relative_path in expected_output_relative_paths:
        override = _semantic_override(relative_path) or _default_semantics(relative_path, workflow_family)
        mappings.append(_available(relative_path, *override))
        if override == ("time_series_bin_table", "time_series.bins"):
            mappings.append(_available(relative_path, "time_series_uncertainty", "time_series.uncertainty"))
        if override == ("embedding_coordinates", "dimension_reduction.coordinates"):
            mappings.append(_available(relative_path, "embedding_coordinates", "decomposition.coordinates"))
        if override == ("cluster_assignments", "clustering.assignments"):
            mappings.append(_available(relative_path, "cluster_assignments", "clustering.labels"))
        if override == ("anomaly_assignments", "anomaly_detection.assignments"):
            mappings.append(_available(relative_path, "anomaly_labels", "anomaly_detection.labels"))
    if workflow_family == "supervised_learning" and workflow_mode == "classification":
        reason = "The public CLI emits the raw confusion matrix only; MCP does not recalculate scientific artifacts."
        mappings.extend(
            (
                _unavailable(
                    "normalized_confusion_matrix_table",
                    "evaluation.confusion_matrix.normalized",
                    reason,
                ),
                _unavailable(
                    "confusion_matrix_table",
                    "evaluation.confusion_matrix.normalized",
                    reason,
                ),
            )
        )
    normalized_method = (method or "").strip().lower().replace("-", "_")
    if workflow_family == "dimension_reduction" and normalized_method in {"tsne", "t_sne"}:
        mappings.append(
            _unavailable(
                "decomposition_quality_report",
                "dimension_reduction.quality",
                "The public CLI does not emit an independent machine-readable t-SNE quality report.",
            )
        )
    if workflow_family == "anomaly_detection" and normalized_method == "isolation_forest":
        mappings.append(
            _unavailable(
                "anomaly_scores",
                "anomaly_detection.scores",
                "The public Isolation Forest CLI adapter emits assignments and subsets, not numeric decision scores.",
            )
        )
    return tuple(mappings)
