"""Stable source identity shared by the Online API and its launcher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_instance_id(project_root: Path = PROJECT_ROOT) -> str:
    """Return a non-reversible fingerprint for the local project directory."""

    normalized_path = str(project_root.resolve()).replace("\\", "/").rstrip("/").casefold()
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:16]


def source_revision(project_root: Path = PROJECT_ROOT) -> str:
    """Return the current Git revision without making Git a runtime requirement."""

    configured_revision = os.getenv("GEOCHEMISTRYPI_SOURCE_REVISION", "").strip()
    if configured_revision:
        return configured_revision

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


INSTANCE_ID = os.getenv("GEOCHEMISTRYPI_ONLINE_INSTANCE_ID", "").strip() or project_instance_id()
SOURCE_REVISION = source_revision()
BUILD_ID = os.getenv("GEOCHEMISTRYPI_BUILD_ID", "").strip() or SOURCE_REVISION
