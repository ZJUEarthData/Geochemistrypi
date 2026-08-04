"""Prompt-synchronized, cross-platform subprocess operation for the public CLI."""

import codecs
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import psutil

from ..config.constants import CLI_AUTOMATION_CONTRACT_VERSION, ISOLATED_CLI_ENVIRONMENT_VARIABLES
from ..planning.interaction_plan import InteractionPlan, InteractionStep

ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CAPTURE_DIRECTORY_NAME = ".geochemistrypi-driver"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_console_output(value: str) -> str:
    return ANSI_ESCAPE.sub("", value).replace("\r", "").replace("\f", "\n")


def _write_text_atomically(path: Path, value: str) -> None:
    """Publish a complete capture file without exposing a partial write."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_atomically(path: Path, value: Dict[str, Any]) -> None:
    _write_text_atomically(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _require_exact_fields(value: Dict[str, Any], expected: set[str], location: str, workspace: Path) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise CliProcessError(
            f"Invalid CLI automation {location}; unknown fields: {unknown}, missing fields: {missing}.",
            workspace,
        )


def _load_automation_events(
    path: Path,
    plan: InteractionPlan,
    workspace: Path,
) -> tuple[List[str], List[Dict[str, Any]]]:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise CliProcessError("CLI automation events exceed the 1 MiB safety limit.", workspace)
        value = json.loads(path.read_text(encoding="utf-8"))
    except CliDriverError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliProcessError("CLI did not produce readable automation events.", workspace) from exc
    if not isinstance(value, dict):
        raise CliProcessError("CLI automation events must be a JSON object.", workspace)
    _require_exact_fields(
        value,
        {
            "schema_version",
            "plan_name",
            "status",
            "started_at",
            "finished_at",
            "completed_input_ids",
            "unused_input_ids",
            "events",
            "error",
        },
        "event document",
        workspace,
    )
    if value["schema_version"] != CLI_AUTOMATION_CONTRACT_VERSION:
        raise CliProcessError(
            f"CLI automation event schema {value['schema_version']!r} is unsupported.", workspace
        )
    if value["plan_name"] != plan.name:
        raise CliProcessError("CLI automation events name a different interaction plan.", workspace)
    completed = value["completed_input_ids"]
    unused = value["unused_input_ids"]
    raw_events = value["events"]
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise CliProcessError("CLI automation completed_input_ids must be a string array.", workspace)
    if not isinstance(unused, list) or not all(isinstance(item, str) for item in unused):
        raise CliProcessError("CLI automation unused_input_ids must be a string array.", workspace)
    if not isinstance(raw_events, list):
        raise CliProcessError("CLI automation events must be an array.", workspace)
    expected_ids = [step.id for step in plan.steps]
    if completed != expected_ids[: len(completed)] or unused != expected_ids[len(completed) :]:
        raise CliProcessError(
            "CLI automation consumed input IDs out of order or reported an inconsistent remainder.",
            workspace,
        )
    events: List[Dict[str, Any]] = []
    if len(raw_events) != len(completed):
        raise CliProcessError("CLI automation event count does not match completed inputs.", workspace)
    steps_by_id = {step.id: step for step in plan.steps}
    for index, event in enumerate(raw_events, start=1):
        if not isinstance(event, dict):
            raise CliProcessError(f"CLI automation event {index} must be an object.", workspace)
        _require_exact_fields(
            event,
            {"sequence", "input_id", "prompt_sha256", "prompt_length", "consumed_at"},
            f"event {index}",
            workspace,
        )
        if event["sequence"] != index or event["input_id"] != completed[index - 1]:
            raise CliProcessError("CLI automation events are not in strict sequence order.", workspace)
        if not isinstance(event["prompt_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", event["prompt_sha256"]) is None:
            raise CliProcessError("CLI automation prompt hash is invalid.", workspace)
        if not isinstance(event["prompt_length"], int) or event["prompt_length"] < 0:
            raise CliProcessError("CLI automation prompt length is invalid.", workspace)
        step = steps_by_id[event["input_id"]]
        events.append(
            {
                "step_id": step.id,
                "transport": "cli_automation_v1",
                "response": step.response,
                **event,
            }
        )
    status = value["status"]
    if status == "completed":
        if completed != expected_ids or unused or value["error"] is not None:
            raise CliProcessError("CLI automation completion report is internally inconsistent.", workspace)
        return completed, events
    if status != "failed":
        raise CliProcessError(f"Unknown CLI automation status: {status!r}", workspace)
    error = value["error"]
    message = "The CLI rejected its automation inputs."
    error_type = ""
    if isinstance(error, dict):
        error_type = str(error.get("type", ""))
        message = " ".join(str(error.get("message", message)).split())[:1000]
    if unused:
        raise UnusedResponsesError(message, workspace)
    if error_type == "MissingAutomationInputError":
        raise UnexpectedPromptError(message, workspace)
    raise CliProcessError(message, workspace)


@dataclass(frozen=True)
class CliRunResult:
    """Completed process result and durable evidence locations."""

    workspace: Path
    output_root: Path
    stdout_path: Path
    stderr_path: Path
    trace_path: Path
    returncode: int
    completed_step_ids: Tuple[str, ...]


class CliDriverError(RuntimeError):
    """Base error carrying the isolated workspace for failed-run inspection."""

    def __init__(self, message: str, workspace: Path):
        super().__init__(message)
        self.workspace = workspace
        self.capture_directory = workspace / CAPTURE_DIRECTORY_NAME


class UnexpectedPromptError(CliDriverError):
    """Raised when a later known prompt appears before the expected prompt."""


class PromptTimeoutError(CliDriverError):
    """Raised when the expected prompt does not arrive within the timeout."""


class UnusedResponsesError(CliDriverError):
    """Raised when the CLI exits before consuming the complete interaction plan."""


class CliProcessError(CliDriverError):
    """Raised when the CLI consumes the plan but exits unsuccessfully."""


class CliRunCancelledError(CliDriverError):
    """Raised after cancelling the exact CLI process tree owned by a run."""


class WorkspacePathError(CliDriverError):
    """Raised before execution when legacy Windows libraries cannot save outputs."""


def _ordered_anchor_end(output: str, anchors: Tuple[str, ...], cursor: int) -> Optional[int]:
    position = cursor
    for anchor in anchors:
        found = output.find(anchor, position)
        if found < 0:
            return None
        position = found + len(anchor)
    return position


def _later_step_seen(output: str, plan: InteractionPlan, current_index: int, cursor: int) -> Optional[InteractionStep]:
    for step in plan.steps[slice(current_index + 1, None)]:
        if _ordered_anchor_end(output, step.output_anchors, cursor) is not None:
            return step
    return None


def _read_stream(name: str, stream, events: "queue.Queue[Tuple[str, Optional[bytes]]]") -> None:
    try:
        while True:
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                break
            events.put((name, chunk))
    finally:
        events.put((name, None))


def _terminate_process_tree(process: subprocess.Popen, expected_create_time: Optional[float]) -> None:
    """Terminate only the still-live process tree rooted at the recorded CLI process."""
    if process.poll() is not None:
        return
    try:
        root = psutil.Process(process.pid)
        if expected_create_time is None or abs(root.create_time() - expected_create_time) > 0.01:
            raise RuntimeError("CLI process identity could not be verified before termination.")
        descendants = root.children(recursive=True)
        for child in reversed(descendants):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            root.terminate()
        except psutil.NoSuchProcess:
            pass
        _, alive = psutil.wait_procs([*descendants, root], timeout=5)
        for owned_process in alive:
            try:
                owned_process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
    except psutil.NoSuchProcess:
        pass
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Recorded CLI process {process.pid} did not terminate.") from exc


class CliInteractionDriver:
    """Run a CLI plan through either legacy prompt sync or CLI automation."""

    def __init__(
        self,
        prompt_timeout_seconds: float = 120.0,
        process_timeout_seconds: float = 900.0,
        automation_mode: bool = False,
    ):
        if prompt_timeout_seconds <= 0 or process_timeout_seconds <= 0:
            raise ValueError("Driver timeouts must be positive.")
        self.prompt_timeout_seconds = prompt_timeout_seconds
        self.process_timeout_seconds = process_timeout_seconds
        self.automation_mode = automation_mode

    def run(
        self,
        plan: InteractionPlan,
        workspace_parent: Optional[Path] = None,
        workspace: Optional[Path] = None,
        capture_directory: Optional[Path] = None,
        environment: Optional[Mapping[str, str]] = None,
        cancellation_event: Optional[threading.Event] = None,
        process_started: Optional[Callable[[int, float], None]] = None,
        trace_metadata: Optional[Mapping[str, str]] = None,
    ) -> CliRunResult:
        if workspace_parent is not None and workspace is not None:
            raise ValueError("workspace_parent and workspace are mutually exclusive.")
        if workspace is None:
            parent = Path(workspace_parent).expanduser().resolve() if workspace_parent else None
            if parent is not None:
                parent.mkdir(parents=True, exist_ok=True)
            workspace_path = Path(tempfile.mkdtemp(prefix="gpi-", dir=str(parent) if parent else None)).resolve()
        else:
            workspace_path = Path(workspace).expanduser().resolve()
            workspace_path.mkdir(parents=True, exist_ok=True)
        capture_path = Path(capture_directory).expanduser().resolve() if capture_directory else workspace_path / CAPTURE_DIRECTORY_NAME
        capture_path.mkdir(parents=True, exist_ok=True)
        stdout_path = capture_path / "stdout.log"
        stderr_path = capture_path / "stderr.log"
        trace_path = capture_path / "interaction-trace.json"
        automation_plan_path = capture_path / "automation-plan.json"
        automation_events_path = capture_path / "automation-events.json"
        executed_command = list(plan.public_command)
        if self.automation_mode:
            _write_json_atomically(
                automation_plan_path,
                {
                    "schema_version": CLI_AUTOMATION_CONTRACT_VERSION,
                    "plan_name": plan.name,
                    "inputs": [
                        {"id": step.id, "response": step.response} for step in plan.steps
                    ],
                },
            )
            executed_command.extend(
                (
                    "--automation-plan",
                    str(automation_plan_path),
                    "--automation-events",
                    str(automation_events_path),
                )
            )
        process_environment = os.environ.copy()
        if environment:
            process_environment.update({str(key): str(value) for key, value in environment.items()})
        # The MCP wrapper and the CLI intentionally run in separate Python
        # environments. uv/venv launcher variables must not redirect the CLI
        # executable back into the wrapper interpreter or its standard library.
        for inherited_name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
            process_environment.pop(inherited_name, None)
        process_environment.update(
            {
                "COLUMNS": "200",
                "LINES": "60",
                "MPLBACKEND": "Agg",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "TERM": "dumb",
            }
        )

        started_at = _utc_now()
        started_monotonic = time.monotonic()
        last_stdout_at = started_monotonic
        process: Optional[subprocess.Popen] = None
        process_create_time: Optional[float] = None
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        interaction_events: List[Dict[str, Any]] = []
        completed_step_ids: List[str] = []
        error: Optional[BaseException] = None
        returncode: Optional[int] = None
        step_index = 0
        stdout_cursor = 0

        try:
            if cancellation_event is not None and cancellation_event.is_set():
                raise CliRunCancelledError("The run was cancelled before the CLI process started.", workspace_path)
            if os.name == "nt":
                for relative_path in plan.expected_output_relative_paths:
                    candidate = workspace_path / relative_path
                    if len(str(candidate)) >= 260:
                        raise WorkspacePathError(
                            "The isolated workspace is too deep for GeochemistryPi's Windows plotting dependencies. "
                            f"The projected output path is {len(str(candidate))} characters; choose a shorter workspace parent. "
                            f"Projected path: {candidate}",
                            workspace_path,
                        )
            creation_options: Dict[str, Any] = {}
            if os.name == "nt":
                creation_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                creation_options["start_new_session"] = True
            process = subprocess.Popen(
                executed_command,
                cwd=workspace_path,
                env=process_environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **creation_options,
            )
            process_create_time = psutil.Process(process.pid).create_time()
            if process_started is not None:
                process_started(process.pid, process_create_time)
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            stream_events: "queue.Queue[Tuple[str, Optional[bytes]]]" = queue.Queue()
            readers = [
                threading.Thread(target=_read_stream, args=("stdout", process.stdout, stream_events), daemon=True),
                threading.Thread(target=_read_stream, args=("stderr", process.stderr, stream_events), daemon=True),
            ]
            for reader in readers:
                reader.start()
            decoders = {
                "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
                "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            }
            closed_streams = set()

            while len(closed_streams) < 2 or process.poll() is None:
                now = time.monotonic()
                if cancellation_event is not None and cancellation_event.is_set():
                    raise CliRunCancelledError("The run was cancelled by request.", workspace_path)
                if now - started_monotonic > self.process_timeout_seconds:
                    raise PromptTimeoutError(f"CLI process exceeded the {self.process_timeout_seconds:g}-second total timeout.", workspace_path)
                if not self.automation_mode and step_index < len(plan.steps):
                    expected = plan.steps[step_index]
                    prompt_timeout = expected.timeout_seconds or self.prompt_timeout_seconds
                    if now - last_stdout_at > prompt_timeout:
                        raise PromptTimeoutError(
                            f"Timed out after {prompt_timeout:g} seconds waiting for interaction step " f"{expected.id!r} with anchors {expected.output_anchors!r}.",
                            workspace_path,
                        )
                try:
                    stream_name, chunk = stream_events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if chunk is None:
                    closed_streams.add(stream_name)
                    tail = decoders[stream_name].decode(b"", final=True)
                    if tail:
                        (stdout_parts if stream_name == "stdout" else stderr_parts).append(tail)
                    continue
                decoded = decoders[stream_name].decode(chunk)
                (stdout_parts if stream_name == "stdout" else stderr_parts).append(decoded)
                if stream_name != "stdout":
                    continue
                last_stdout_at = time.monotonic()

                if self.automation_mode:
                    continue

                normalized_stdout = _normalize_console_output("".join(stdout_parts))
                while step_index < len(plan.steps):
                    step = plan.steps[step_index]
                    anchor_end = _ordered_anchor_end(normalized_stdout, step.output_anchors, stdout_cursor)
                    if anchor_end is None:
                        unexpected = _later_step_seen(normalized_stdout, plan, step_index, stdout_cursor)
                        if unexpected is not None:
                            raise UnexpectedPromptError(f"Observed later interaction step {unexpected.id!r} before expected step {step.id!r}.", workspace_path)
                        break
                    process.stdin.write((step.response + "\n").encode("utf-8"))
                    process.stdin.flush()
                    completed_step_ids.append(step.id)
                    interaction_events.append(
                        {
                            "step_id": step.id,
                            "output_anchors": list(step.output_anchors),
                            "response": step.response,
                            "timeout_seconds": step.timeout_seconds or self.prompt_timeout_seconds,
                            "matched_stdout_character": anchor_end,
                            "recorded_at": _utc_now(),
                        }
                    )
                    stdout_cursor = anchor_end
                    step_index += 1

                if step_index == len(plan.steps):
                    unexpected = _later_step_seen(normalized_stdout, plan, -1, stdout_cursor)
                    if unexpected is not None:
                        raise UnexpectedPromptError(f"Observed an additional known prompt after all responses were consumed: {unexpected.id!r}.", workspace_path)

            returncode = process.wait()
            if self.automation_mode:
                completed_step_ids, interaction_events = _load_automation_events(
                    automation_events_path, plan, workspace_path
                )
                step_index = len(completed_step_ids)
            elif step_index != len(plan.steps):
                unused = [step.id for step in plan.steps[step_index:]]
                raise UnusedResponsesError(f"CLI exited with code {returncode} before consuming {len(unused)} planned responses: {unused}", workspace_path)
            if returncode != 0:
                raise CliProcessError(f"CLI consumed the interaction plan but exited with code {returncode}.", workspace_path)
        except BaseException as exc:
            error = exc
            if process is not None:
                _terminate_process_tree(process, process_create_time)
                returncode = process.returncode
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, CliDriverError):
                raise
            raise CliDriverError(f"Unable to operate the GeochemistryPi CLI: {exc}", workspace_path) from exc
        finally:
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            _write_text_atomically(stdout_path, stdout)
            _write_text_atomically(stderr_path, stderr)
            trace = {
                "schema_version": 1,
                "metadata": dict(trace_metadata or {}),
                "plan_name": plan.name,
                "plan_schema_version": plan.schema_version,
                "command": executed_command,
                "input_transport": (
                    "cli_automation_v1" if self.automation_mode else "legacy_prompt_sync_v1"
                ),
                "workspace": str(workspace_path),
                "started_at": started_at,
                "finished_at": _utc_now(),
                "returncode": returncode,
                "status": "cancelled" if isinstance(error, CliRunCancelledError) else "failed" if error else "completed",
                "completed_step_ids": completed_step_ids,
                "unused_step_ids": [step.id for step in plan.steps[step_index:]],
                "events": interaction_events,
                "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
            }
            _write_text_atomically(trace_path, json.dumps(trace, indent=2, ensure_ascii=False) + "\n")

        return CliRunResult(
            workspace=workspace_path,
            output_root=workspace_path / "geopi_output",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            trace_path=trace_path,
            returncode=returncode if returncode is not None else 0,
            completed_step_ids=tuple(completed_step_ids),
        )
