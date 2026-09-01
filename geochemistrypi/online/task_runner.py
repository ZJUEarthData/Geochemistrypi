"""Observable single-slot task queue with cancellation and hard timeouts."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Any, TypeVar
from uuid import UUID, uuid4

from .limits import TASK_TIMEOUT_SECONDS


ResultT = TypeVar("ResultT")
TERMINAL_STATES = {"completed", "failed", "timed_out", "cancelled"}


class TaskTimeoutError(TimeoutError):
    """Raised after a calculation process is forcibly stopped at its deadline."""


class TaskCancelledError(RuntimeError):
    """Raised when a queued or running calculation is cancelled."""


class TaskProcessError(RuntimeError):
    """Raised when a calculation process exits without a usable result."""


@dataclass
class _TaskRecord:
    task_id: str
    label: str
    status: str = "queued"
    progress: int = 0
    queue_position: int | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = "Waiting in the calculation queue."
    error: str | None = None
    ready: threading.Event = field(default_factory=threading.Event, repr=False)
    cancel_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    process: multiprocessing.Process | None = field(default=None, repr=False)


def _task_process(
    connection: Connection,
    operation: Callable[..., Any],
    arguments: dict[str, Any],
) -> None:
    try:
        try:
            payload = ("result", operation(**arguments))
        except Exception as exc:  # Preserve existing API error mapping.
            payload = ("error", exc)

        try:
            serialized = pickle.dumps(payload)
        except Exception as exc:
            serialized = pickle.dumps(
                (
                    "process_error",
                    f"{type(exc).__name__}: the calculation result could not be transferred",
                )
            )
        connection.send_bytes(serialized)
    finally:
        connection.close()


class TaskRunner:
    """Queue calculations, run one at a time, and expose live task state."""

    def __init__(
        self,
        lock_path: Path,
        timeout_seconds: float = TASK_TIMEOUT_SECONDS,
    ):
        if timeout_seconds <= 0:
            raise ValueError("Task timeout must be greater than zero")
        self.timeout_seconds = timeout_seconds
        self.lock_path = lock_path.resolve()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue: deque[str] = deque()
        self._tasks: dict[str, _TaskRecord] = {}
        self._context = multiprocessing.get_context("spawn")

    async def run(
        self,
        operation: Callable[..., ResultT],
        *,
        arguments: dict[str, Any] | None = None,
        tracking_id: str | None = None,
        task_label: str = "Calculation",
    ) -> ResultT:
        task_id = self._register(tracking_id, task_label)
        return await asyncio.to_thread(
            self._run_sync,
            task_id,
            operation,
            arguments or {},
        )

    def get_status(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._tasks.get(task_id)
            return self._snapshot(record) if record else None

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        process = None
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            if record.status in TERMINAL_STATES:
                return self._snapshot(record)

            record.cancel_requested.set()
            if record.status == "queued":
                self._remove_from_queue(record.task_id)
                self._set_terminal(
                    record,
                    "cancelled",
                    "Calculation was cancelled while waiting in the queue.",
                )
                record.ready.set()
                self._wake_next()
            else:
                record.status = "cancelling"
                record.message = "Cancellation requested. Stopping the calculation."
                process = record.process
            snapshot = self._snapshot(record)

        if process is not None and process.is_alive():
            self._stop_process(process)
        return snapshot

    def _register(self, tracking_id: str | None, label: str) -> str:
        task_id = self._normalize_task_id(tracking_id)
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task ID '{task_id}' already exists")
            record = _TaskRecord(task_id=task_id, label=label)
            self._tasks[task_id] = record
            self._queue.append(task_id)
            self._update_queue_positions()
            if self._queue[0] == task_id:
                record.ready.set()
        return task_id

    def _run_sync(
        self,
        task_id: str,
        operation: Callable[..., ResultT],
        arguments: dict[str, Any],
    ) -> ResultT:
        with self._lock:
            record = self._tasks[task_id]
        record.ready.wait()

        lock_file = None
        receive_connection = None
        send_connection = None
        process = None
        process_started = False
        try:
            self._raise_if_cancelled(record)
            lock_file = self._acquire_file_slot(record.cancel_requested)
            self._raise_if_cancelled(record)

            with self._lock:
                record.status = "running"
                record.progress = 50
                record.queue_position = None
                record.started_at = datetime.now(timezone.utc)
                record.message = "Calculation is running."

            receive_connection, send_connection = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_task_process,
                args=(send_connection, operation, arguments),
                name=f"geochemistrypi-online-task-{task_id[:8]}",
            )
            process.start()
            process_started = True
            with self._lock:
                record.process = process
            send_connection.close()

            deadline = time.monotonic() + self.timeout_seconds
            while True:
                self._raise_if_cancelled(record)
                if receive_connection.poll(0.1):
                    break
                if time.monotonic() >= deadline:
                    self._stop_process(process)
                    raise TaskTimeoutError(
                        "Calculation exceeded the 30-minute limit and was stopped."
                    )
            self._raise_if_cancelled(record)

            try:
                kind, payload = pickle.loads(receive_connection.recv_bytes())
            except (EOFError, OSError, pickle.PickleError) as exc:
                self._raise_if_cancelled(record)
                raise TaskProcessError(
                    "The calculation process exited without returning a result."
                ) from exc

            process.join(timeout=5)
            if kind == "result":
                with self._lock:
                    self._set_terminal(record, "completed", "Calculation completed.")
                return payload
            if kind == "error" and isinstance(payload, Exception):
                raise payload
            raise TaskProcessError(str(payload))
        except TaskCancelledError:
            with self._lock:
                if record.status != "cancelled":
                    self._set_terminal(record, "cancelled", "Calculation was cancelled.")
            raise
        except TaskTimeoutError as exc:
            with self._lock:
                self._set_terminal(record, "timed_out", str(exc), str(exc))
            raise
        except Exception as exc:
            with self._lock:
                self._set_terminal(
                    record,
                    "failed",
                    "Calculation failed.",
                    f"{type(exc).__name__}: {exc}",
                )
            raise
        finally:
            if receive_connection is not None:
                receive_connection.close()
            if send_connection is not None:
                send_connection.close()
            if process is not None and process_started:
                if process.is_alive():
                    self._stop_process(process)
                else:
                    process.join(timeout=1)
            if lock_file is not None:
                self._release_file_slot(lock_file)
            with self._lock:
                record.process = None
                self._remove_from_queue(task_id)
                self._wake_next()

    def _acquire_file_slot(self, cancel_requested: threading.Event):
        lock_file = self.lock_path.open("a+b")
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        while True:
            if cancel_requested.is_set():
                lock_file.close()
                raise TaskCancelledError("Calculation was cancelled.")
            try:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    lock_file.close()
                    raise
                time.sleep(0.05)

    def _snapshot(self, record: _TaskRecord) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        elapsed = 0.0
        if record.started_at is not None:
            elapsed = ((record.finished_at or now) - record.started_at).total_seconds()
        return {
            "task_id": record.task_id,
            "label": record.label,
            "status": record.status,
            "progress": record.progress,
            "queue_position": record.queue_position,
            "submitted_at": record.submitted_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "elapsed_seconds": max(0.0, round(elapsed, 1)),
            "timeout_seconds": self.timeout_seconds,
            "cancellable": record.status not in TERMINAL_STATES,
            "message": record.message,
            "error": record.error,
        }

    def _update_queue_positions(self) -> None:
        position = 0
        for task_id in self._queue:
            record = self._tasks[task_id]
            if record.status == "queued":
                position += 1
                record.queue_position = position

    def _remove_from_queue(self, task_id: str) -> None:
        try:
            self._queue.remove(task_id)
        except ValueError:
            pass
        self._update_queue_positions()

    def _wake_next(self) -> None:
        self._update_queue_positions()
        if self._queue:
            self._tasks[self._queue[0]].ready.set()

    @staticmethod
    def _set_terminal(
        record: _TaskRecord,
        status: str,
        message: str,
        error: str | None = None,
    ) -> None:
        record.status = status
        record.progress = 100
        record.queue_position = None
        record.finished_at = datetime.now(timezone.utc)
        record.message = message
        record.error = error

    @staticmethod
    def _raise_if_cancelled(record: _TaskRecord) -> None:
        if record.cancel_requested.is_set():
            raise TaskCancelledError("Calculation was cancelled.")

    @staticmethod
    def _normalize_task_id(task_id: str | None) -> str:
        if task_id is None:
            return str(uuid4())
        try:
            return str(UUID(task_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("X-Task-ID must be a valid UUID") from exc

    @staticmethod
    def _release_file_slot(lock_file) -> None:
        try:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    @staticmethod
    def _stop_process(process: multiprocessing.Process) -> None:
        if not process.is_alive():
            process.join(timeout=1)
            return
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
