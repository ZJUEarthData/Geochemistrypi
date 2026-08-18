"""Encoding-safe output for human-facing lifecycle commands.

The MCP server itself must keep stdout reserved for JSON-RPC.  This module is
therefore used only by the setup, doctor, and release administration entry
points, where paths supplied by scientists may contain any Unicode character.
"""

from __future__ import annotations

import sys
from typing import TextIO


def _configure_stream(stream: TextIO) -> None:
    """Prefer UTF-8 and a non-crashing fallback on reconfigurable streams."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError, ValueError):
        # Test captures and embedding hosts can expose immutable text streams.
        # Their own write policy remains in force.
        return


def configure_utf8_console() -> None:
    """Make administrative output safe for Unicode paths on every platform."""
    _configure_stream(sys.stdout)
    _configure_stream(sys.stderr)
