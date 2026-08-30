"""Deterministic, non-scientific source-row identity for MCP orchestration."""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

ROW_IDENTITY_SCHEME = "geochemistrypi-mcp-source-row-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceRowIdentityError(ValueError):
    """Raised when internal source-row lineage cannot be established safely."""


def source_row_identity(dataset_sha256: str, source_row_number: int) -> str:
    """Return one stable opaque identity from source bytes and row position."""
    if _SHA256.fullmatch(dataset_sha256) is None:
        raise SourceRowIdentityError("Source-row identity requires a lowercase SHA-256 dataset fingerprint.")
    if type(source_row_number) is not int or source_row_number < 1:
        raise SourceRowIdentityError("Source-row numbers must be positive integers.")
    payload = f"{ROW_IDENTITY_SCHEME}\0{dataset_sha256}\0{source_row_number}".encode("ascii")
    return f"mcp-row-{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class SourceRowLineage:
    """Compact trace metadata plus reconstructable identities for every source row."""

    dataset_sha256: str
    source_row_count: int
    ordered_identity_sha256: str
    identities: tuple[str, ...] = field(repr=False)
    scheme: str = ROW_IDENTITY_SCHEME

    def as_record(self) -> dict[str, Any]:
        """Return bounded run metadata without serializing every row identity."""
        return {
            "scheme": self.scheme,
            "dataset_sha256": self.dataset_sha256,
            "source_row_count": self.source_row_count,
            "ordered_identity_sha256": self.ordered_identity_sha256,
            "reconstruction": "dataset_sha256 + 1-based source data row number",
        }


def build_source_row_lineage(dataset_sha256: str, source_row_count: int) -> SourceRowLineage:
    """Build and collision-check the internal identities for a dataset snapshot."""
    if type(source_row_count) is not int or source_row_count < 0:
        raise SourceRowIdentityError("Source-row count must be a non-negative integer.")
    identities = tuple(source_row_identity(dataset_sha256, row_number) for row_number in range(1, source_row_count + 1))
    if any(not identity for identity in identities):
        raise SourceRowIdentityError("Internal source-row identity generation produced an empty identity.")
    if len(identities) != len(set(identities)):
        raise SourceRowIdentityError("Internal source-row identity collision detected; the dataset is not safe to execute.")
    ordered_digest = hashlib.sha256()
    for identity in identities:
        ordered_digest.update(identity.encode("ascii"))
        ordered_digest.update(b"\n")
    return SourceRowLineage(
        dataset_sha256=dataset_sha256,
        source_row_count=source_row_count,
        ordered_identity_sha256=ordered_digest.hexdigest(),
        identities=identities,
    )
