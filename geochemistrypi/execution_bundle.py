"""Validated, paper-agnostic replay bundles for local CLI execution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


EXECUTION_BUNDLE_VERSION = 1
_MAX_BUNDLE_BYTES = 1024 * 1024
_SHA256_LENGTH = 64


class ExecutionBundleError(RuntimeError):
    """Raised when a replay bundle is unsafe, incomplete, or has changed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_fields(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ExecutionBundleError(
            f"Invalid {location}; unknown fields: {unknown}, missing fields: {missing}."
        )


def _bounded_text(value: Any, field: str, *, optional: bool = False) -> str:
    if optional and value in (None, ""):
        return ""
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ExecutionBundleError(f"{field} must be a bounded non-blank string.")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ExecutionBundleError(
            f"{field} must be single-line text without null bytes."
        )
    return value


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str

    @classmethod
    def load(cls, value: Any, base: Path, field: str) -> "BoundFile":
        if not isinstance(value, dict):
            raise ExecutionBundleError(f"{field} must be an object.")
        _exact_fields(value, {"path", "sha256"}, field)
        raw_path = _bounded_text(value["path"], f"{field}.path")
        digest = value["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ExecutionBundleError(
                f"{field}.sha256 must be a lowercase SHA-256 digest."
            )
        candidate = Path(raw_path).expanduser()
        resolved = (
            (base / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        if not resolved.is_file():
            raise ExecutionBundleError(f"{field} is unavailable: {resolved}")
        observed = file_sha256(resolved)
        if observed != digest:
            raise ExecutionBundleError(f"{field} does not match its recorded SHA-256.")
        return cls(resolved, digest)


@dataclass(frozen=True)
class ExecutionBundle:
    plan_name: str
    data_source_name: str
    training_data: Optional[BoundFile]
    application_data: Optional[BoundFile]
    automation_plan: BoundFile
    scientific_config: Optional[BoundFile]
    world_map_config: str
    tracking_root: str
    existing_experiment_id: str
    source_path: Path
    source_sha256: str

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        training_override: Optional[Path] = None,
        application_override: Optional[Path] = None,
    ) -> "ExecutionBundle":
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = source.resolve()
        try:
            if not source.is_file() or source.stat().st_size > _MAX_BUNDLE_BYTES:
                raise ExecutionBundleError(
                    "Execution bundle must be a regular JSON file no larger than 1 MiB."
                )
            payload = source.read_bytes()
            value = json.loads(payload.decode("utf-8"))
        except ExecutionBundleError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExecutionBundleError(
                f"Cannot read execution bundle: {source}"
            ) from exc
        if not isinstance(value, dict):
            raise ExecutionBundleError("Execution bundle must be a JSON object.")
        _exact_fields(
            value,
            {
                "schema_version",
                "plan_name",
                "data_source",
                "training_data",
                "application_data",
                "automation_plan",
                "scientific_config",
                "world_map_config",
                "tracking_root",
                "existing_experiment_id",
            },
            "execution bundle",
        )
        if value["schema_version"] != EXECUTION_BUNDLE_VERSION:
            raise ExecutionBundleError(
                f"Unsupported execution bundle schema: {value['schema_version']!r}."
            )
        data_source = _bounded_text(value["data_source"], "data_source")
        if data_source not in {"ANY_PATH", "BUILT_IN"}:
            raise ExecutionBundleError("data_source must be ANY_PATH or BUILT_IN.")
        base = source.parent

        def load_optional(
            raw: Any, override: Optional[Path], field: str
        ) -> Optional[BoundFile]:
            if raw is None:
                if override is not None:
                    raise ExecutionBundleError(
                        f"{field} override was provided but the bundle declares no {field}."
                    )
                return None
            if override is None:
                return BoundFile.load(raw, base, field)
            if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
                raise ExecutionBundleError(f"{field} must contain path and sha256.")
            return BoundFile.load(
                {
                    "path": str(Path(override).expanduser().resolve()),
                    "sha256": raw["sha256"],
                },
                base,
                field,
            )

        training = load_optional(
            value["training_data"], training_override, "training_data"
        )
        application = load_optional(
            value["application_data"], application_override, "application_data"
        )
        if data_source == "ANY_PATH" and training is None:
            raise ExecutionBundleError("ANY_PATH bundles require training_data.")
        if data_source == "BUILT_IN" and (
            training is not None or application is not None
        ):
            raise ExecutionBundleError(
                "BUILT_IN bundles cannot bind external datasets."
            )
        scientific = (
            None
            if value["scientific_config"] is None
            else BoundFile.load(value["scientific_config"], base, "scientific_config")
        )
        tracking_root = _bounded_text(
            value["tracking_root"], "tracking_root", optional=True
        )
        if tracking_root and not Path(tracking_root).expanduser().is_absolute():
            raise ExecutionBundleError("tracking_root must be absolute when specified.")
        existing_experiment_id = _bounded_text(
            value["existing_experiment_id"], "existing_experiment_id", optional=True
        )
        if (
            existing_experiment_id
            and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", existing_experiment_id) is None
        ):
            raise ExecutionBundleError(
                "existing_experiment_id is not a valid experiment identifier."
            )
        return cls(
            plan_name=_bounded_text(value["plan_name"], "plan_name"),
            data_source_name=data_source,
            training_data=training,
            application_data=application,
            automation_plan=BoundFile.load(
                value["automation_plan"], base, "automation_plan"
            ),
            scientific_config=scientific,
            world_map_config=_bounded_text(
                value["world_map_config"], "world_map_config", optional=True
            ),
            tracking_root=tracking_root,
            existing_experiment_id=existing_experiment_id,
            source_path=source,
            source_sha256=hashlib.sha256(payload).hexdigest(),
        )
