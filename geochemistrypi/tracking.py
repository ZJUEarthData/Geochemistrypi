"""Bounded, machine-readable access to the local MLflow tracking store."""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class TrackingStoreError(ValueError):
    """Raised when a tracking-store request is invalid or cannot be satisfied."""


def resolve_tracking_root(value: str) -> Path:
    """Resolve one explicit absolute tracking root without accepting URI input."""
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise TrackingStoreError("tracking_root must be an absolute local path")
    return root.resolve()


def tracking_uri(root: Path) -> str:
    """Return the portable file URI understood by MLflow."""
    return root.resolve().as_uri()


def _client(root: Path, create: bool) -> MlflowClient:
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise TrackingStoreError(f"MLflow tracking root does not exist: {root}")
    mlflow.set_tracking_uri(tracking_uri(root))
    return MlflowClient()


def _validate_experiment_id(experiment_id: str) -> str:
    normalized = experiment_id.strip()
    if not _EXPERIMENT_ID.fullmatch(normalized):
        raise TrackingStoreError("experiment_id must contain only letters, numbers, '_' or '-'")
    return normalized


def _safe_metric(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _experiment_value(experiment: Any) -> Dict[str, Any]:
    return {
        "experiment_id": str(experiment.experiment_id),
        "name": str(experiment.name),
        "lifecycle_stage": str(experiment.lifecycle_stage),
        "artifact_location": str(experiment.artifact_location),
        "tags": {str(key): str(value) for key, value in sorted((experiment.tags or {}).items())},
    }


def list_experiments(tracking_root: Path, maximum_experiments: int = 100) -> Dict[str, Any]:
    """List active experiments from one local store with a hard response bound."""
    if maximum_experiments < 1 or maximum_experiments > 500:
        raise TrackingStoreError("maximum_experiments must be between 1 and 500")
    client = _client(tracking_root, create=True)
    experiments = client.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=maximum_experiments,
        order_by=["creation_time DESC"],
    )
    values = [_experiment_value(experiment) for experiment in experiments]
    return {
        "schema_version": 1,
        "tracking_root": str(tracking_root),
        "experiment_count": len(values),
        "experiments": values,
    }


def get_experiment(
    tracking_root: Path,
    experiment_id: str,
    maximum_runs: int = 50,
) -> Dict[str, Any]:
    """Return one active experiment and its newest bounded run summaries."""
    if maximum_runs < 0 or maximum_runs > 200:
        raise TrackingStoreError("maximum_runs must be between 0 and 200")
    normalized_id = _validate_experiment_id(experiment_id)
    client = _client(tracking_root, create=True)
    experiment = client.get_experiment(normalized_id)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise TrackingStoreError(f"Active MLflow experiment does not exist: {normalized_id}")
    runs: List[Dict[str, Any]] = []
    if maximum_runs:
        for run in client.search_runs(
            [normalized_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=maximum_runs,
            order_by=["attributes.start_time DESC"],
        ):
            runs.append(
                {
                    "run_id": str(run.info.run_id),
                    "run_name": str(run.data.tags.get("mlflow.runName", "")),
                    "status": str(run.info.status),
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "artifact_uri": str(run.info.artifact_uri),
                    "metrics": {str(key): _safe_metric(value) for key, value in sorted(run.data.metrics.items())},
                    "params": {str(key): str(value)[:500] for key, value in sorted(run.data.params.items())},
                }
            )
    return {
        "schema_version": 1,
        "tracking_root": str(tracking_root),
        "experiment": _experiment_value(experiment),
        "run_count": len(runs),
        "runs": runs,
    }


def require_experiment(tracking_root: Path, experiment_id: str, expected_name: str) -> Dict[str, Any]:
    """Resolve a stable experiment ID and reject output-name mismatches."""
    value = get_experiment(tracking_root, experiment_id, maximum_runs=0)
    experiment = value["experiment"]
    if experiment["name"] != expected_name:
        raise TrackingStoreError(f"experiment_id {experiment_id!r} belongs to {experiment['name']!r}, " f"not requested experiment_name {expected_name!r}")
    return experiment
