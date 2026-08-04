"""Deterministic dataset-header normalization shared by inspection and plans."""

import re
from typing import Any, Sequence

_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


class HeaderValidationError(ValueError):
    """Raised when a dataset header would be ambiguous or unsafe to address."""


def normalize_dataset_header(
    raw_header: Sequence[Any],
    maximum_columns: int,
    *,
    allow_pandas_duplicate_mangling: bool = False,
) -> tuple[str, ...]:
    """Apply pandas-compatible blank names, then reject ambiguous headers.

    pandas names an empty header by its zero-based position (``Unnamed: N``).
    Explicit duplicates remain a hard error because semantic MCP requests must
    identify each column unambiguously instead of relying on pandas' suffixing.
    """
    if not raw_header:
        raise HeaderValidationError("Dataset must contain a header row.")
    if len(raw_header) > maximum_columns:
        raise HeaderValidationError(f"Dataset has {len(raw_header)} columns; the limit is {maximum_columns}.")
    columns = []
    for index, value in enumerate(raw_header):
        column = f"Unnamed: {index}" if value is None or value == "" else str(value)
        if not column.strip():
            raise HeaderValidationError(f"Dataset column {index + 1} contains only whitespace.")
        if column != column.strip():
            raise HeaderValidationError(f"Dataset column {index + 1} has leading or trailing whitespace: {column!r}.")
        if len(column) > 128:
            raise HeaderValidationError(f"Dataset column {index + 1} name must not exceed 128 characters.")
        if _CONTROL_CHARACTER.search(column):
            raise HeaderValidationError(f"Dataset column {index + 1} contains a control character.")
        columns.append(column)
    if len(columns) != len(set(columns)) and not allow_pandas_duplicate_mangling:
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        raise HeaderValidationError(f"Dataset contains duplicate or colliding column names: {duplicates}")
    if allow_pandas_duplicate_mangling:
        raw_names = set(columns)
        seen = set()
        counts = {}
        mangled = []
        for column in columns:
            if column not in seen:
                name = column
                counts.setdefault(column, 1)
            else:
                suffix = counts.get(column, 1)
                name = f"{column}.{suffix}"
                while name in raw_names or name in seen:
                    suffix += 1
                    name = f"{column}.{suffix}"
                counts[column] = suffix + 1
            seen.add(name)
            mangled.append(name)
        columns = mangled
    return tuple(columns)


def header_normalization_warnings(raw_header: Sequence[Any], columns: Sequence[str]) -> tuple[str, ...]:
    warnings = []
    for index, (raw, normalized) in enumerate(zip(raw_header, columns)):
        original = "" if raw is None else str(raw)
        if original != normalized:
            warnings.append(f"Column {index + 1} was normalized from {original!r} to {normalized!r} to match pandas.")
    return tuple(warnings)
