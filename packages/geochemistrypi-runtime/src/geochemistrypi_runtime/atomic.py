"""Small, dependency-light primitives for durable local files."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

from .exceptions import CorruptedRecordError, UnsafePathError

DEFAULT_MAX_JSON_BYTES = 10 * 1024 * 1024


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize an object deterministically for hashing and persistence."""

    if not isinstance(value, Mapping):
        raise TypeError("Canonical JSON records must be mappings.")
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Record contains a value that is not valid JSON.") from exc
    return (text + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync on platforms that support it."""

    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace a file atomically after flushing its complete new contents."""

    path = Path(path)
    if path.exists() and path.is_symlink():
        raise UnsafePathError(f"Refusing to replace symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        os.replace(str(temporary_path), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one canonical JSON object."""

    atomic_write_bytes(path, canonical_json_bytes(value))


def read_json_object(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> Dict[str, Any]:
    """Read a bounded UTF-8 JSON object from a regular, non-link file."""

    path = Path(path)
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise CorruptedRecordError(f"Required record is missing: {path.name}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise CorruptedRecordError(f"Record must be a regular file: {path.name}")
    if file_stat.st_size > max_bytes:
        raise CorruptedRecordError(f"Record {path.name} exceeds the {max_bytes}-byte safety limit.")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptedRecordError(f"Record is not valid UTF-8 JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise CorruptedRecordError(f"Record must contain a JSON object: {path.name}")
    return value


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""

    path = Path(path)
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise CorruptedRecordError(f"Required file is missing: {path.name}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise UnsafePathError(f"Hash target must be a regular, non-link file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
