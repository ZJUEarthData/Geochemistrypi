"""Filesystem boundaries shared by Online downloads and private job exports.

Only server-generated UUID-hex job directories are accepted. Relative artifact
paths use URL-style forward slashes on every host; Windows drives, alternate
data streams and backslash paths are never interpreted as artifact names.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path


def _plain_stat(path: Path):
    """Reject symlinks and Windows junction/reparse points without following."""
    try:
        info = path.lstat()
    except (OSError, ValueError) as exc:
        raise FileNotFoundError("Result path not found") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse_flag:
        raise FileNotFoundError("Linked result paths are not supported")
    return info


def job_directory(jobs_dir: Path, job_id: str) -> Path:
    """Return an existing, unlinked job strictly inside the configured root."""
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise FileNotFoundError("Unknown job")
    root = Path(jobs_dir).resolve()
    requested = root / job_id
    info = _plain_stat(requested)
    directory = requested.resolve()
    if not stat.S_ISDIR(info.st_mode) or directory.parent != root or directory.name != job_id:
        raise FileNotFoundError("Unknown job")
    return directory


def output_directory(jobs_dir: Path, job_id: str) -> Path:
    """Return the unlinked output directory of a validated existing job."""
    directory = job_directory(jobs_dir, job_id)
    requested = directory / "output"
    info = _plain_stat(requested)
    output = requested.resolve()
    if not stat.S_ISDIR(info.st_mode) or output.parent != directory or output.name != "output":
        raise FileNotFoundError("Job output missing")
    return output


def resolve_artifact_path(jobs_dir: Path, job_id: str, file_path: str) -> Path:
    """Resolve a regular artifact, rejecting traversal and links at every level."""
    output = output_directory(jobs_dir, job_id)
    if not isinstance(file_path, str) or not file_path or any(char in file_path for char in "\\\x00:"):
        raise FileNotFoundError("Result file not found")
    parts = file_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise FileNotFoundError("Result file not found")
    candidate = output
    for index, part in enumerate(parts):
        candidate = candidate / part
        info = _plain_stat(candidate)
        expected_type = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
        if not expected_type(info.st_mode):
            raise FileNotFoundError("Result file not found")
    candidate = candidate.resolve()
    try:
        candidate.relative_to(output)
    except ValueError as exc:
        raise FileNotFoundError("Result file not found") from exc
    return candidate
