"""Manifest and provenance records for reproducible local runs."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from geochemistrypi_contracts import ArtifactRef

from ._validation import json_mapping, nonempty_string, optional_string, portable_relative_path, require_fields, revision, run_id, sha256, string_tuple, utc_timestamp

RECORD_FORMAT_VERSION = "1.0"


class ProvenanceSection(str, Enum):
    """Stable namespaces for incrementally collected provenance."""

    DATASET = "dataset"
    INPUT = "input"
    SPLIT = "split"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    RESAMPLING = "resampling"
    MODEL = "model"
    EVALUATION = "evaluation"
    ENVIRONMENT = "environment"
    VERSIONS = "versions"
    RESOURCES = "resources"
    TIMING = "timing"
    DETERMINISM = "determinism"
    FAILURE = "failure"


@dataclass(frozen=True)
class ManifestRecord:
    """Index of the durable files and artifacts belonging to a run."""

    run_id: str
    revision: int
    created_at: str
    updated_at: str
    request_sha256: str
    contract_version: str
    request_schema_id: str
    request_schema_sha256: str
    status_path: str
    provenance_path: str
    result_path: Optional[str] = None
    artifacts: Tuple[ArtifactRef, ...] = ()
    warnings: Tuple[str, ...] = ()
    format_version: str = RECORD_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != RECORD_FORMAT_VERSION:
            raise ValueError(f"format_version must be {RECORD_FORMAT_VERSION!r}.")
        object.__setattr__(self, "run_id", run_id(self.run_id))
        object.__setattr__(self, "revision", revision(self.revision))
        object.__setattr__(self, "created_at", utc_timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", utc_timestamp(self.updated_at, "updated_at"))
        object.__setattr__(self, "request_sha256", sha256(self.request_sha256, "request_sha256"))
        object.__setattr__(
            self,
            "contract_version",
            nonempty_string(self.contract_version, "contract_version", 64),
        )
        object.__setattr__(
            self,
            "request_schema_id",
            nonempty_string(self.request_schema_id, "request_schema_id", 1024),
        )
        object.__setattr__(
            self,
            "request_schema_sha256",
            sha256(self.request_schema_sha256, "request_schema_sha256"),
        )
        object.__setattr__(
            self,
            "status_path",
            portable_relative_path(self.status_path, "status_path"),
        )
        object.__setattr__(
            self,
            "provenance_path",
            portable_relative_path(self.provenance_path, "provenance_path"),
        )
        normalized_result = None if self.result_path is None else portable_relative_path(self.result_path, "result_path")
        object.__setattr__(self, "result_path", normalized_result)
        if not isinstance(self.artifacts, (tuple, list)):
            raise TypeError("artifacts must be a tuple or list.")
        artifact_items = []
        for item in self.artifacts:
            if isinstance(item, ArtifactRef):
                artifact_items.append(ArtifactRef.from_dict(item.to_dict()))
            elif isinstance(item, Mapping):
                artifact_items.append(ArtifactRef.from_dict(item))
            else:
                artifact_items.append(item)
        normalized_artifacts = tuple(artifact_items)
        if not all(isinstance(item, ArtifactRef) for item in normalized_artifacts):
            raise TypeError("artifacts must contain ArtifactRef values.")
        artifact_ids = [item.artifact_id for item in normalized_artifacts]
        artifact_paths = [item.relative_path for item in normalized_artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Manifest artifact_id values must be unique.")
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ValueError("Manifest artifact paths must be unique.")
        object.__setattr__(self, "artifacts", normalized_artifacts)
        object.__setattr__(
            self,
            "warnings",
            string_tuple(self.warnings, "warnings"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "run_id": self.run_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request_sha256": self.request_sha256,
            "contract_version": self.contract_version,
            "request_schema_id": self.request_schema_id,
            "request_schema_sha256": self.request_schema_sha256,
            "status_path": self.status_path,
            "provenance_path": self.provenance_path,
            "result_path": self.result_path,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManifestRecord":
        fields = require_fields(
            value,
            required={
                "format_version",
                "run_id",
                "revision",
                "created_at",
                "updated_at",
                "request_sha256",
                "contract_version",
                "request_schema_id",
                "request_schema_sha256",
                "status_path",
                "provenance_path",
                "result_path",
                "artifacts",
                "warnings",
            },
            label="ManifestRecord",
        )
        fields["artifacts"] = tuple(ArtifactRef.from_dict(item) for item in fields["artifacts"])
        return cls(**fields)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Version and process facts needed to explain or reproduce a run."""

    run_id: str
    revision: int
    created_at: str
    updated_at: str
    contract_version: str
    request_schema_id: str
    request_schema_sha256: str
    runtime_version: str
    python_version: str
    implementation: str
    platform: str
    dependency_versions: Mapping[str, str]
    git_commit: Optional[str] = None
    engine_version: Optional[str] = None
    sections: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    format_version: str = RECORD_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != RECORD_FORMAT_VERSION:
            raise ValueError(f"format_version must be {RECORD_FORMAT_VERSION!r}.")
        object.__setattr__(self, "run_id", run_id(self.run_id))
        object.__setattr__(self, "revision", revision(self.revision))
        object.__setattr__(self, "created_at", utc_timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", utc_timestamp(self.updated_at, "updated_at"))
        string_limits = {
            "contract_version": 64,
            "request_schema_id": 1024,
            "runtime_version": 128,
            "python_version": 128,
            "implementation": 128,
            "platform": 2000,
        }
        for field_name, max_length in string_limits.items():
            object.__setattr__(
                self,
                field_name,
                nonempty_string(
                    getattr(self, field_name),
                    field_name,
                    max_length,
                ),
            )
        object.__setattr__(
            self,
            "request_schema_sha256",
            sha256(self.request_schema_sha256, "request_schema_sha256"),
        )
        dependency_versions = json_mapping(self.dependency_versions, "dependency_versions", max_bytes=128 * 1024)
        if not all(isinstance(name, str) and name and isinstance(version, str) and version for name, version in dependency_versions.items()):
            raise ValueError("dependency_versions must map names to version strings.")
        object.__setattr__(self, "dependency_versions", dependency_versions)
        object.__setattr__(
            self,
            "git_commit",
            optional_string(self.git_commit, "git_commit", 128),
        )
        object.__setattr__(
            self,
            "engine_version",
            optional_string(self.engine_version, "engine_version", 128),
        )
        raw_sections = json_mapping(self.sections, "sections")
        normalized_sections: Dict[str, Mapping[str, Any]] = {}
        for name, section in raw_sections.items():
            try:
                section_name = ProvenanceSection(name).value
            except ValueError as exc:
                raise ValueError(f"Unsupported provenance section: {name!r}") from exc
            if not isinstance(section, dict):
                raise TypeError(f"Provenance section {name!r} must be an object.")
            normalized_sections[section_name] = section
        object.__setattr__(self, "sections", normalized_sections)

    @classmethod
    def for_current_process(
        cls,
        *,
        run_id_value: str,
        created_at: str,
        contract_version: str,
        request_schema_id: str,
        request_schema_sha256: str,
        runtime_version: str,
        dependency_versions: Mapping[str, str],
        git_commit: Optional[str] = None,
    ) -> "ProvenanceRecord":
        return cls(
            run_id=run_id_value,
            revision=0,
            created_at=created_at,
            updated_at=created_at,
            contract_version=contract_version,
            request_schema_id=request_schema_id,
            request_schema_sha256=request_schema_sha256,
            runtime_version=runtime_version,
            python_version=platform.python_version(),
            implementation=platform.python_implementation(),
            platform=platform.platform(),
            dependency_versions=dependency_versions,
            git_commit=git_commit,
            engine_version=None,
            sections={
                ProvenanceSection.ENVIRONMENT.value: {
                    "python_executable": sys.executable,
                }
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "run_id": self.run_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "contract_version": self.contract_version,
            "request_schema_id": self.request_schema_id,
            "request_schema_sha256": self.request_schema_sha256,
            "runtime_version": self.runtime_version,
            "python_version": self.python_version,
            "implementation": self.implementation,
            "platform": self.platform,
            "dependency_versions": dict(self.dependency_versions),
            "git_commit": self.git_commit,
            "engine_version": self.engine_version,
            "sections": dict(self.sections),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProvenanceRecord":
        fields = require_fields(
            value,
            required={
                "format_version",
                "run_id",
                "revision",
                "created_at",
                "updated_at",
                "contract_version",
                "request_schema_id",
                "request_schema_sha256",
                "runtime_version",
                "python_version",
                "implementation",
                "platform",
                "dependency_versions",
                "git_commit",
                "engine_version",
                "sections",
            },
            label="ProvenanceRecord",
        )
        return cls(**fields)
