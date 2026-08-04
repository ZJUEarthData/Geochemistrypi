"""Validated access to the static CLI capability inventory."""

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


_ALLOWED_STATUSES = {"implemented", "verified", "known_gap", "not_public"}


class CapabilityManifestError(RuntimeError):
    """Raised when the packaged capability inventory is incomplete or stale."""


@lru_cache(maxsize=1)
def load_capability_manifest() -> dict[str, Any]:
    resource = files("geochemistrypi_mcp").joinpath("cli_capability_manifest_v1.json")
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapabilityManifestError("Cannot read CLI capability manifest v1.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise CapabilityManifestError("Unsupported CLI capability manifest schema.")
    expected = {
        "schema_version",
        "manifest_id",
        "cli_version",
        "automation_contract_version",
        "declarations",
        "capabilities",
    }
    if set(value) != expected:
        raise CapabilityManifestError("CLI capability manifest root fields are invalid.")
    capabilities = value["capabilities"]
    if not isinstance(capabilities, list) or not capabilities:
        raise CapabilityManifestError("CLI capability manifest has no capabilities.")
    identifiers = []
    for index, item in enumerate(capabilities):
        if not isinstance(item, dict) or set(item) != {
            "id",
            "category",
            "status",
            "cli_public",
            "mcp_supported",
            "evidence",
        }:
            raise CapabilityManifestError(f"Capability entry {index} has invalid fields.")
        if item["status"] not in _ALLOWED_STATUSES:
            raise CapabilityManifestError(f"Capability entry {index} has an invalid status.")
        if item["mcp_supported"] and (
            item["status"] != "verified" or not item["evidence"]
        ):
            raise CapabilityManifestError(
                f"MCP capability {item['id']!r} lacks verified parity evidence."
            )
        if item["status"] == "known_gap" and item["mcp_supported"]:
            raise CapabilityManifestError(
                f"Known gap {item['id']!r} cannot be advertised as supported."
            )
        identifiers.append(item["id"])
    if len(identifiers) != len(set(identifiers)):
        raise CapabilityManifestError("CLI capability IDs must be unique.")
    return value


def public_capabilities() -> tuple[dict[str, Any], ...]:
    manifest = load_capability_manifest()
    return tuple(manifest["capabilities"])


def known_gap_ids() -> tuple[str, ...]:
    return tuple(
        item["id"]
        for item in public_capabilities()
        if item["status"] == "known_gap"
    )
