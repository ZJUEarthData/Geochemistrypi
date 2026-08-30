"""Bounded probes for public GeochemistryPi CLI commands and options."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config.constants import ISOLATED_CLI_ENVIRONMENT_VARIABLES


@dataclass(frozen=True)
class CliCapabilityProbe:
    """Observed support for a requested set of public CLI capabilities."""

    available: tuple[str, ...]
    missing: tuple[str, ...]


def _isolated_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    return environment


def _command_help(executable: Path, command: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            (str(executable), command, "--help"),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=_isolated_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    output = "\n".join((completed.stdout, completed.stderr))
    return completed.returncode == 0, output


def probe_cli_capabilities(
    cli_executable: Path,
    requirements: Sequence[str],
) -> CliCapabilityProbe:
    """Probe ``command:<name>`` and ``option:<command>:<flag>`` requirements."""

    executable = Path(cli_executable).expanduser().resolve()
    cached_help: dict[str, tuple[bool, str]] = {}
    available: list[str] = []
    missing: list[str] = []
    for requirement in dict.fromkeys(requirements):
        parts = requirement.split(":", 2)
        if len(parts) == 2 and parts[0] == "command" and parts[1]:
            command = parts[1]
            healthy, _ = cached_help.setdefault(
                command,
                _command_help(executable, command),
            )
        elif len(parts) == 3 and parts[0] == "option" and parts[1] and parts[2].startswith("--"):
            command = parts[1]
            healthy, output = cached_help.setdefault(
                command,
                _command_help(executable, command),
            )
            healthy = healthy and parts[2] in output
        else:
            healthy = False
        (available if healthy else missing).append(requirement)
    return CliCapabilityProbe(tuple(available), tuple(missing))
