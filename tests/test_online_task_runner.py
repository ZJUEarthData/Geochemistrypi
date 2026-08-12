"""Process isolation, timeout, and single-slot tests for Online calculations."""

import asyncio
from pathlib import Path
import time

import pytest

from geochemistrypi.online.task_runner import (
    TaskCancelledError,
    TaskRunner,
    TaskTimeoutError,
)


def _return_after(delay: float, value: str) -> str:
    time.sleep(delay)
    return value


def _write_after(delay: float, marker: Path) -> str:
    time.sleep(delay)
    marker.write_text("finished", encoding="utf-8")
    return "finished"


def _record_then_wait(delay: float, record: Path, label: str) -> str:
    with record.open("a", encoding="utf-8") as stream:
        stream.write(f"{label}\n")
    time.sleep(delay)
    return label


def test_runner_transfers_a_result_from_the_child_process(tmp_path):
    runner = TaskRunner(tmp_path / "task.lock", timeout_seconds=10)

    result = asyncio.run(
        runner.run(_return_after, arguments={"delay": 0, "value": "ok"})
    )

    assert result == "ok"


def test_shared_lock_queues_a_second_calculation(tmp_path):
    lock_path = tmp_path / "task.lock"
    first_runner = TaskRunner(lock_path, timeout_seconds=10)
    second_runner = TaskRunner(lock_path, timeout_seconds=10)

    async def scenario():
        first = asyncio.create_task(
            first_runner.run(
                _return_after,
                arguments={"delay": 0.8, "value": "first"},
            )
        )
        await asyncio.sleep(0.2)
        second = asyncio.create_task(
            second_runner.run(
                _return_after,
                arguments={"delay": 0, "value": "second"},
            )
        )
        await asyncio.sleep(0.2)
        assert not second.done()
        return await first, await second

    assert asyncio.run(scenario()) == ("first", "second")


def test_three_calculations_run_in_submission_order(tmp_path):
    runner = TaskRunner(tmp_path / "task.lock", timeout_seconds=10)
    record = tmp_path / "execution-order.txt"

    async def wait_for_queue_size(expected: int) -> None:
        for _ in range(200):
            with runner._lock:
                if len(runner._queue) == expected:
                    return
            await asyncio.sleep(0.01)
        raise AssertionError(f"Queue did not reach {expected} tasks")

    async def scenario():
        first = asyncio.create_task(
            runner.run(
                _record_then_wait,
                arguments={"delay": 0.5, "record": record, "label": "first"},
            )
        )
        await wait_for_queue_size(1)
        second = asyncio.create_task(
            runner.run(
                _record_then_wait,
                arguments={"delay": 0, "record": record, "label": "second"},
            )
        )
        await wait_for_queue_size(2)
        third = asyncio.create_task(
            runner.run(
                _record_then_wait,
                arguments={"delay": 0, "record": record, "label": "third"},
            )
        )
        await wait_for_queue_size(3)
        return await asyncio.gather(first, second, third)

    assert asyncio.run(scenario()) == ["first", "second", "third"]
    assert record.read_text(encoding="utf-8").splitlines() == [
        "first",
        "second",
        "third",
    ]


def test_timeout_terminates_the_calculation_process(tmp_path):
    marker = tmp_path / "should-not-exist.txt"
    runner = TaskRunner(tmp_path / "task.lock", timeout_seconds=0.2)

    with pytest.raises(TaskTimeoutError, match="30-minute limit"):
        asyncio.run(
            runner.run(
                _write_after,
                arguments={"delay": 2, "marker": marker},
            )
        )

    time.sleep(0.3)
    assert not marker.exists()


def test_queued_task_can_be_cancelled_without_running(tmp_path):
    runner = TaskRunner(tmp_path / "task.lock", timeout_seconds=10)
    record = tmp_path / "cancelled-task.txt"
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"

    async def scenario():
        first = asyncio.create_task(
            runner.run(
                _return_after,
                arguments={"delay": 0.8, "value": "first"},
                tracking_id=first_id,
            )
        )
        for _ in range(200):
            first_status = runner.get_status(first_id)
            if first_status and first_status["status"] == "running":
                break
            await asyncio.sleep(0.01)
        second = asyncio.create_task(
            runner.run(
                _write_after,
                arguments={"delay": 0, "marker": record},
                tracking_id=second_id,
            )
        )
        for _ in range(200):
            status = runner.get_status(second_id)
            if status and status["queue_position"] == 2:
                break
            await asyncio.sleep(0.01)
        cancelled = runner.cancel(second_id)
        assert cancelled and cancelled["status"] == "cancelled"
        with pytest.raises(TaskCancelledError):
            await second
        assert await first == "first"

    asyncio.run(scenario())
    assert not record.exists()


def test_running_task_can_be_cancelled_and_releases_queue(tmp_path):
    runner = TaskRunner(tmp_path / "task.lock", timeout_seconds=10)
    marker = tmp_path / "running-task.txt"
    task_id = "33333333-3333-4333-8333-333333333333"

    async def scenario():
        running = asyncio.create_task(
            runner.run(
                _write_after,
                arguments={"delay": 2, "marker": marker},
                tracking_id=task_id,
            )
        )
        for _ in range(200):
            status = runner.get_status(task_id)
            if status and status["status"] == "running":
                break
            await asyncio.sleep(0.01)
        requested = runner.cancel(task_id)
        assert requested and requested["status"] == "cancelling"
        with pytest.raises(TaskCancelledError):
            await running

    asyncio.run(scenario())
    status = runner.get_status(task_id)
    assert status and status["status"] == "cancelled"
    time.sleep(0.3)
    assert not marker.exists()
