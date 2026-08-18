"""Safe discovery and resolution of CLI-owned dataset sources."""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..api.schemas import BuiltInDatasetReference, DatasetCatalogEntry, DatasetReference, DesktopDatasetReference, ExplicitDatasetReference, ListDatasetsRequest, ListDatasetsResponse
from ..config.constants import ISOLATED_CLI_ENVIRONMENT_VARIABLES
from ..config.settings import McpSettings

_SUPPORTED_ANALYSIS_TASKS = {
    "classification",
    "regression",
    "clustering",
    "decomposition",
    "anomaly_detection",
    "time_series",
}
_MAX_CATALOG_OUTPUT_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetCatalogError(ValueError):
    """Raised when a named dataset cannot be discovered or resolved safely."""


@dataclass(frozen=True)
class ResolvedDataset:
    path: Path
    expected_sha256: str | None
    dataset_id: str | None
    source: str


class DatasetCatalog:
    """Query the installed CLI instead of importing its package into MCP."""

    def __init__(self, settings: McpSettings):
        self.settings = settings

    def list(self, request: ListDatasetsRequest) -> ListDatasetsResponse:
        executable, _ = self.settings.require_supported_cli()
        process_environment = os.environ.copy()
        for inherited_name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
            process_environment.pop(inherited_name, None)
        try:
            completed = subprocess.run(
                (
                    str(executable),
                    "datasets",
                    "--source",
                    request.source,
                    "--output",
                    "json",
                ),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=process_environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DatasetCatalogError("Unable to query datasets from the installed GeochemistryPi CLI.") from exc
        stdout = completed.stdout.strip()
        if completed.returncode != 0:
            detail = " ".join(completed.stderr.split())[:500]
            raise DatasetCatalogError("The installed GeochemistryPi CLI could not list datasets" + (f": {detail}" if detail else "."))
        if len(stdout.encode("utf-8")) > _MAX_CATALOG_OUTPUT_BYTES:
            raise DatasetCatalogError("CLI dataset catalog exceeds the 1 MiB safety limit.")
        try:
            value = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise DatasetCatalogError("The installed GeochemistryPi CLI returned an invalid dataset catalog.") from exc
        if not isinstance(value, dict):
            raise DatasetCatalogError("CLI dataset catalog must be a JSON object.")
        expected_fields = {
            "schema_version",
            "source_filter",
            "supported_formats",
            "desktop_root",
            "datasets",
            "warnings",
        }
        unknown = sorted(set(value) - expected_fields)
        missing = sorted(expected_fields - set(value))
        if unknown or missing:
            raise DatasetCatalogError(f"CLI dataset catalog fields are invalid; unknown: {unknown}, missing: {missing}.")
        if value.get("schema_version") != 1 or value.get("source_filter") != request.source:
            raise DatasetCatalogError("CLI dataset catalog version or source filter is inconsistent.")
        raw_datasets = value.get("datasets")
        if not isinstance(raw_datasets, list):
            raise DatasetCatalogError("CLI dataset catalog datasets must be an array.")
        enriched = []
        for raw in raw_datasets:
            if not isinstance(raw, dict):
                raise DatasetCatalogError("Every CLI dataset catalog entry must be an object.")
            item = dict(raw)
            task = item.get("task")
            role = item.get("role")
            blockers = item.get("analysis_blockers")
            if not isinstance(blockers, list) or not all(isinstance(blocker, str) for blocker in blockers):
                raise DatasetCatalogError("Every CLI dataset entry must declare analysis_blockers.")
            item["supported_for_analysis"] = not blockers and (item.get("source") == "desktop" or task in _SUPPORTED_ANALYSIS_TASKS and role in {"training", "application"})
            enriched.append(item)
        value["datasets"] = enriched
        try:
            response = ListDatasetsResponse.model_validate(value)
        except ValidationError as exc:
            raise DatasetCatalogError("The installed CLI dataset catalog does not match schema version 1.") from exc
        identifiers = [item.dataset_id for item in response.datasets]
        if len(identifiers) != len(set(identifiers)):
            raise DatasetCatalogError("CLI dataset catalog contains duplicate dataset IDs.")
        self._validate_paths(response)
        return response

    def _validate_paths(self, response: ListDatasetsResponse) -> None:
        desktop_root = Path(response.desktop_root).resolve() if response.desktop_root is not None else None
        for entry in response.datasets:
            path = Path(entry.path)
            if not path.is_absolute():
                raise DatasetCatalogError(f"CLI dataset catalog returned a non-absolute path for {entry.dataset_id}.")
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise DatasetCatalogError(f"CLI dataset is no longer available: {entry.dataset_id}.") from exc
            if not resolved.is_file():
                raise DatasetCatalogError(f"CLI dataset is not a regular file: {entry.dataset_id}.")
            try:
                metadata = resolved.stat()
                digest = _sha256(resolved)
            except OSError as exc:
                raise DatasetCatalogError(f"CLI dataset became unavailable while it was being verified: {entry.dataset_id}; retry after writes stop.") from exc
            if metadata.st_size != entry.size_bytes or digest != entry.sha256:
                raise DatasetCatalogError(f"CLI dataset changed while it was being listed: {entry.dataset_id}; retry after writes stop.")
            if entry.source == "desktop":
                if desktop_root is None:
                    raise DatasetCatalogError("Desktop dataset catalog omitted its root.")
                try:
                    resolved.relative_to(desktop_root)
                except ValueError as exc:
                    raise DatasetCatalogError(f"Desktop dataset escapes Desktop/geopi_input: {entry.file_name}.") from exc
                if resolved.parent != desktop_root:
                    raise DatasetCatalogError(f"Desktop dataset must be an immediate child of Desktop/geopi_input: {entry.file_name}.")

    def resolve(
        self,
        reference: DatasetReference,
        *,
        task: str | None = None,
        role: str | None = None,
    ) -> ResolvedDataset:
        if isinstance(reference, ExplicitDatasetReference):
            return ResolvedDataset(
                path=reference.path,
                expected_sha256=reference.expected_sha256,
                dataset_id=None,
                source="path",
            )
        source = "builtin" if isinstance(reference, BuiltInDatasetReference) else "desktop"
        response = self.list(ListDatasetsRequest(source=source))
        entry: DatasetCatalogEntry | None = None
        if isinstance(reference, BuiltInDatasetReference):
            entry = next(
                (candidate for candidate in response.datasets if candidate.dataset_id == reference.dataset_id),
                None,
            )
            if entry is None:
                available = [item.dataset_id for item in response.datasets]
                raise DatasetCatalogError(f"Unknown built-in dataset ID {reference.dataset_id!r}; available IDs: {available}")
            if task is not None and entry.task != task:
                raise DatasetCatalogError(f"Built-in dataset {entry.dataset_id!r} is for {entry.task}, not {task}.")
            if role is not None and entry.role != role:
                raise DatasetCatalogError(f"Built-in dataset {entry.dataset_id!r} has role {entry.role}, not {role}.")
            if role == "training" and entry.analysis_blockers:
                raise DatasetCatalogError(
                    f"Built-in dataset {entry.dataset_id!r} is discoverable and inspectable but " f"cannot start this analysis yet; blocking capabilities: {list(entry.analysis_blockers)}."
                )
        elif isinstance(reference, DesktopDatasetReference):
            entry = next(
                (candidate for candidate in response.datasets if candidate.file_name == reference.file_name),
                None,
            )
            if entry is None:
                available = [item.file_name for item in response.datasets]
                raise DatasetCatalogError(f"Desktop dataset {reference.file_name!r} was not found; available files: {available}")
        else:
            raise DatasetCatalogError("Unsupported dataset reference type.")
        expected = reference.expected_sha256
        if expected is not None and expected != entry.sha256:
            raise DatasetCatalogError(f"Dataset {entry.dataset_id!r} changed since it was selected; expected SHA-256 {expected}, found {entry.sha256}.")
        return ResolvedDataset(
            path=Path(entry.path),
            expected_sha256=expected,
            dataset_id=entry.dataset_id,
            source=entry.source,
        )
