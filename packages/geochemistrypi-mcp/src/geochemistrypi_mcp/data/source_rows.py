"""Source-row boundaries matching the public CLI's pandas readers."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Any


def row_is_empty(row: Sequence[Any]) -> bool:
    """Return whether a worksheet row contains no cell value."""
    return all(value is None or value == "" for value in row)


def iter_cli_excel_rows(rows: Iterable[Sequence[Any]]) -> Iterator[tuple[Any, ...]]:
    """Yield Excel rows while discarding only all-empty trailing rows.

    ``pandas.read_excel`` preserves empty rows inside the used data range but
    does not materialize all-empty rows after the last value.  The public CLI
    uses that reader, so MCP lineage and validation must use the same boundary.
    """
    pending_empty_count = 0
    pending_empty_row: tuple[Any, ...] = ()
    for values in rows:
        row = tuple(values)
        if row_is_empty(row):
            pending_empty_count += 1
            pending_empty_row = row
            continue
        for _ in range(pending_empty_count):
            yield pending_empty_row
        pending_empty_count = 0
        yield row


def iter_cli_csv_rows(rows: Iterable[Sequence[str]]) -> Iterator[tuple[str, ...]]:
    """Yield CSV records while matching pandas' default blank-line skipping."""
    for values in rows:
        row = tuple(values)
        if row:
            yield row
