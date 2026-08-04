"""Versioned, fail-closed input automation for the existing interactive CLI."""

import builtins
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type


AUTOMATION_CONTRACT_VERSION = 1
_MAX_PLAN_BYTES = 1024 * 1024
_MAX_INPUTS = 512
_MAX_RESPONSE_CHARACTERS = 10_000
_INPUT_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class AutomationContractError(RuntimeError):
    """Raised when machine-supplied CLI input is invalid or incomplete."""


class MissingAutomationInputError(AutomationContractError):
    """Raised when the workflow asks for an input that was not supplied."""


class UnusedAutomationInputError(AutomationContractError):
    """Raised when supplied inputs were not consumed by the workflow."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_exact_fields(value: Dict[str, Any], expected: set, location: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise AutomationContractError(f"Unknown {location} fields: {unknown}")
    if missing:
        raise AutomationContractError(f"Missing {location} fields: {missing}")


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(str(temporary_path), str(path))
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class AutomationInput:
    """One ordered response identified independently from human prompt text."""

    id: str
    response: str


@dataclass(frozen=True)
class AutomationPlan:
    """Strict automation request loaded by the CLI process."""

    schema_version: int
    plan_name: str
    inputs: Tuple[AutomationInput, ...]

    @classmethod
    def load(cls, path: Path) -> "AutomationPlan":
        source = Path(path).expanduser()
        if not source.is_absolute():
            raise AutomationContractError("--automation-plan must be an absolute path.")
        try:
            resolved = source.resolve(strict=True)
            if resolved.stat().st_size > _MAX_PLAN_BYTES:
                raise AutomationContractError(
                    f"Automation plan exceeds the {_MAX_PLAN_BYTES}-byte safety limit."
                )
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except AutomationContractError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AutomationContractError(f"Cannot read automation plan: {source}") from exc
        if not isinstance(value, dict):
            raise AutomationContractError("Automation plan must be a JSON object.")
        _require_exact_fields(value, {"schema_version", "plan_name", "inputs"}, "automation plan")
        if value["schema_version"] != AUTOMATION_CONTRACT_VERSION:
            raise AutomationContractError(
                f"Unsupported automation schema {value['schema_version']!r}; expected {AUTOMATION_CONTRACT_VERSION}."
            )
        plan_name = value["plan_name"]
        if not isinstance(plan_name, str) or not plan_name.strip() or len(plan_name) > 128:
            raise AutomationContractError("plan_name must be a non-blank string of at most 128 characters.")
        raw_inputs = value["inputs"]
        if not isinstance(raw_inputs, list):
            raise AutomationContractError("inputs must be a JSON array.")
        if len(raw_inputs) > _MAX_INPUTS:
            raise AutomationContractError(f"inputs exceeds the {_MAX_INPUTS}-entry safety limit.")
        parsed: List[AutomationInput] = []
        seen = set()
        for index, item in enumerate(raw_inputs):
            if not isinstance(item, dict):
                raise AutomationContractError(f"inputs[{index}] must be a JSON object.")
            _require_exact_fields(item, {"id", "response"}, f"inputs[{index}]")
            input_id = item["id"]
            response = item["response"]
            if not isinstance(input_id, str) or _INPUT_ID.fullmatch(input_id) is None:
                raise AutomationContractError(
                    f"inputs[{index}].id must match {_INPUT_ID.pattern!r}."
                )
            if input_id in seen:
                raise AutomationContractError(f"Duplicate automation input id: {input_id!r}")
            if not isinstance(response, str):
                raise AutomationContractError(f"inputs[{index}].response must be a string.")
            if "\n" in response or "\r" in response:
                raise AutomationContractError(
                    f"inputs[{index}].response must contain exactly one input line."
                )
            if len(response) > _MAX_RESPONSE_CHARACTERS:
                raise AutomationContractError(
                    f"inputs[{index}].response exceeds {_MAX_RESPONSE_CHARACTERS} characters."
                )
            seen.add(input_id)
            parsed.append(AutomationInput(input_id, response))
        return cls(AUTOMATION_CONTRACT_VERSION, plan_name.strip(), tuple(parsed))


class AutomationInputAdapter:
    """Temporarily serve ordered inputs to the unchanged interactive workflow."""

    def __init__(self, plan: AutomationPlan, events_path: Path):
        destination = Path(events_path).expanduser()
        if not destination.is_absolute():
            raise AutomationContractError("--automation-events must be an absolute path.")
        resolved_parent = destination.parent.resolve(strict=True)
        self.events_path = resolved_parent / destination.name
        self.plan = plan
        self._index = 0
        self._events: List[Dict[str, Any]] = []
        self._original_input = builtins.input
        self._started_at: Optional[str] = None

    def __enter__(self) -> "AutomationInputAdapter":
        if builtins.input is not self._original_input:
            raise AutomationContractError("Another input adapter is already active.")
        self._started_at = _utc_now()
        builtins.input = self._input
        return self

    def _input(self, prompt: object = "") -> str:
        prompt_text = str(prompt)
        # Match builtins.input's visible behavior so human-readable CLI logs keep
        # their prompts even though stdin is not used in automation mode.
        sys.stdout.write(prompt_text)
        sys.stdout.flush()
        if self._index >= len(self.plan.inputs):
            raise MissingAutomationInputError(
                "The CLI requested another input after every automation response was consumed. "
                "The workflow or its input contract has changed."
            )
        supplied = self.plan.inputs[self._index]
        self._index += 1
        self._events.append(
            {
                "sequence": self._index,
                "input_id": supplied.id,
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "prompt_length": len(prompt_text),
                "consumed_at": _utc_now(),
            }
        )
        return supplied.response

    def _write_events(self, status: str, error: Optional[BaseException]) -> None:
        payload = {
            "schema_version": AUTOMATION_CONTRACT_VERSION,
            "plan_name": self.plan.plan_name,
            "status": status,
            "started_at": self._started_at,
            "finished_at": _utc_now(),
            "completed_input_ids": [item["input_id"] for item in self._events],
            "unused_input_ids": [item.id for item in self.plan.inputs[self._index :]],
            "events": self._events,
            "error": (
                None
                if error is None
                else {
                    "type": type(error).__name__,
                    "message": " ".join(str(error).split())[:1000],
                }
            ),
        }
        _atomic_write_json(self.events_path, payload)

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: object,
    ) -> bool:
        builtins.input = self._original_input
        if exception is not None:
            self._write_events("failed", exception)
            return False
        unused = [item.id for item in self.plan.inputs[self._index :]]
        if unused:
            error = UnusedAutomationInputError(
                f"CLI completed with {len(unused)} unused automation inputs: {unused}"
            )
            self._write_events("failed", error)
            raise error
        self._write_events("completed", None)
        return False


def automation_input_adapter(plan_path: Path, events_path: Path) -> AutomationInputAdapter:
    """Build the automation adapter without importing any scientific modules."""
    return AutomationInputAdapter(AutomationPlan.load(plan_path), events_path)
