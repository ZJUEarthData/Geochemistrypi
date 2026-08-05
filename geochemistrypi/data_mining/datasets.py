"""Read-only discovery metadata for built-in and Desktop CLI datasets."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import load_workbook

from .constants import BUILT_IN_DATASET_PATH

DATASET_CATALOG_SCHEMA_VERSION = 1
SUPPORTED_DATASET_SUFFIXES = {".csv": "csv", ".xlsx": "xlsx"}
_BUILT_IN_DATASETS = {
    "ApplicationData_Classification.xlsx": (
        "builtin:application_classification",
        "classification",
        "application",
    ),
    "ApplicationData_Regression.xlsx": (
        "builtin:application_regression",
        "regression",
        "application",
    ),
    "Data_AnomalyDetection.xlsx": (
        "builtin:anomaly_detection",
        "anomaly_detection",
        "training",
    ),
    "Data_Classification.xlsx": (
        "builtin:classification",
        "classification",
        "training",
    ),
    "Data_Clustering.xlsx": (
        "builtin:clustering",
        "clustering",
        "training",
    ),
    "Data_Decomposition.xlsx": (
        "builtin:decomposition",
        "decomposition",
        "training",
    ),
    "Data_Regression.xlsx": (
        "builtin:regression",
        "regression",
        "training",
    ),
    "Data_Time_Series.xlsx": (
        "builtin:time_series",
        "time_series",
        "training",
    ),
}


class DatasetCatalogError(RuntimeError):
    """Raised when dataset discovery cannot produce trustworthy metadata."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                return sum(1 for _ in reader), len(header)
        except (OSError, UnicodeError, csv.Error, StopIteration):
            return None, None
    try:
        with path.open("rb") as stream:
            workbook = load_workbook(stream, read_only=True, data_only=True)
            try:
                worksheet = workbook.active
                return max(worksheet.max_row - 1, 0), worksheet.max_column
            finally:
                workbook.close()
    except (OSError, ValueError):
        return None, None


def _entry(
    path: Path,
    source: str,
    dataset_id: str,
    task: Optional[str],
    role: str,
) -> Dict[str, Any]:
    metadata = path.stat()
    rows, columns = _shape(path)
    analysis_blockers = []
    return {
        "dataset_id": dataset_id,
        "source": source,
        "role": role,
        "task": task,
        "file_name": path.name,
        "path": str(path),
        "format": SUPPORTED_DATASET_SUFFIXES[path.suffix.lower()],
        "size_bytes": metadata.st_size,
        "sha256": _sha256(path),
        "row_count": rows,
        "column_count": columns,
        "analysis_blockers": analysis_blockers,
    }


def _built_in_entries() -> List[Dict[str, Any]]:
    root = Path(BUILT_IN_DATASET_PATH).resolve(strict=True)
    discovered = {path.name for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES}
    declared = set(_BUILT_IN_DATASETS)
    if discovered != declared:
        raise DatasetCatalogError("Built-in dataset declarations are stale. " f"Undeclared files: {sorted(discovered - declared)}; missing files: {sorted(declared - discovered)}")
    return [_entry(root / name, "builtin", dataset_id, task, role) for name, (dataset_id, task, role) in sorted(_BUILT_IN_DATASETS.items())]


def desktop_input_root() -> Path:
    """Return the same Desktop/geopi_input location used by the human CLI."""
    return (Path.home() / "Desktop" / "geopi_input").resolve()


def _desktop_entries() -> Tuple[List[Dict[str, Any]], List[str], Path]:
    root = desktop_input_root()
    if not root.exists():
        return [], ["Desktop/geopi_input does not exist; discovery did not create it."], root
    if not root.is_dir():
        raise DatasetCatalogError(f"Desktop dataset location is not a directory: {root}")
    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    try:
        candidates = sorted(root.iterdir(), key=lambda item: item.name.casefold())
    except OSError as exc:
        raise DatasetCatalogError(f"Desktop dataset location could not be read: {root}") from exc
    for candidate in candidates:
        if candidate.suffix.lower() not in SUPPORTED_DATASET_SUFFIXES:
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            warnings.append(f"Ignored unsafe Desktop entry: {candidate.name}")
            continue
        if not resolved.is_file():
            warnings.append(f"Ignored non-file Desktop entry: {candidate.name}")
            continue
        stable_name = candidate.name.replace("%", "%25").replace(":", "%3A")
        try:
            entries.append(
                _entry(
                    resolved,
                    "desktop",
                    f"desktop:{stable_name}",
                    None,
                    "unspecified",
                )
            )
        except OSError:
            warnings.append(f"Ignored Desktop entry that changed while being read: {candidate.name}")
    return entries, warnings, root


def dataset_catalog(source: str = "all") -> Dict[str, Any]:
    """Return bounded JSON-ready dataset metadata without changing the filesystem."""
    if source not in {"all", "builtin", "desktop"}:
        raise DatasetCatalogError("source must be one of: all, builtin, desktop")
    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    desktop_root: Optional[Path] = None
    if source in {"all", "builtin"}:
        entries.extend(_built_in_entries())
    if source in {"all", "desktop"}:
        desktop, desktop_warnings, desktop_root = _desktop_entries()
        entries.extend(desktop)
        warnings.extend(desktop_warnings)
    return {
        "schema_version": DATASET_CATALOG_SCHEMA_VERSION,
        "source_filter": source,
        "supported_formats": ["csv", "xlsx"],
        "desktop_root": None if desktop_root is None else str(desktop_root),
        "datasets": entries,
        "warnings": warnings,
    }


def dataset_catalog_json(source: str = "all") -> str:
    return json.dumps(dataset_catalog(source), ensure_ascii=False, sort_keys=True)
