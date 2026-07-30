"""Validation helpers shared by runtime records."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Optional, Tuple

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def require_fields(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> Dict[str, Any]:
    """Return a copy after rejecting missing or unknown record fields."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    required_fields = set(required)
    allowed_fields = required_fields | set(optional)
    actual_fields = set(value)
    missing = required_fields - actual_fields
    unknown = actual_fields - allowed_fields
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    return dict(value)


def nonempty_string(value: Any, label: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace.")
    if len(value) > max_length:
        raise ValueError(f"{label} must be at most {max_length} characters.")
    return value


def optional_string(value: Any, label: str, max_length: int) -> Optional[str]:
    if value is None:
        return None
    return nonempty_string(value, label, max_length)


def run_id(value: Any) -> str:
    normalized = nonempty_string(value, "run_id", 128)
    if not RUN_ID_PATTERN.fullmatch(normalized) or normalized in {".", ".."}:
        raise ValueError("run_id contains unsupported characters.")
    return normalized


def identifier(value: Any, label: str) -> str:
    normalized = nonempty_string(value, label, 128)
    if not IDENTIFIER_PATTERN.fullmatch(normalized) or normalized in {".", ".."}:
        raise ValueError(f"{label} contains unsupported characters.")
    return normalized


def revision(value: Any, label: str = "revision") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_timestamp(value: Any, label: str) -> str:
    normalized = nonempty_string(value, label, 64)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone.")
    return normalized


def portable_relative_path(value: Any, label: str) -> str:
    normalized = nonempty_string(value, label, 1024)
    if "\\" in normalized:
        raise ValueError(f"{label} must use forward slashes.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or path.as_posix() != normalized or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe portable relative path.")
    return path.as_posix()


def string_tuple(
    value: Any,
    label: str,
    *,
    max_items: int = 100,
    max_length: int = 1000,
) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of strings.")
    if len(value) > max_items:
        raise ValueError(f"{label} must have at most {max_items} items.")
    return tuple(nonempty_string(item, f"{label} item", max_length) for item in value)


def json_mapping(
    value: Any,
    label: str,
    *,
    max_bytes: int = 1024 * 1024,
) -> Dict[str, Any]:
    """Validate, bound, and detach a JSON mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")

    def inspect(item: Any, depth: int) -> None:
        if depth > 12:
            raise ValueError(f"{label} is nested too deeply.")
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{label} contains a non-finite number.")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{label} object keys must be strings.")
                inspect(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                inspect(child, depth + 1)
            return
        raise TypeError(f"{label} contains a non-JSON value: {type(item).__name__}")

    inspect(value, 0)
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte safety limit.")
    detached = json.loads(encoded)
    if not isinstance(detached, dict):
        raise TypeError(f"{label} must be a mapping.")
    return detached
