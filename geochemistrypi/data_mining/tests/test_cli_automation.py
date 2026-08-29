import builtins
import json
from pathlib import Path

import pytest
from click import unstyle
from rich.prompt import Confirm, Prompt
from typer.testing import CliRunner

import geochemistrypi.cli as cli_module
from geochemistrypi.automation import AutomationContractError, AutomationPlan, MissingAutomationInputError, UnusedAutomationInputError, automation_input_adapter


def _write_plan(path: Path, inputs, **extra) -> None:
    value = {
        "schema_version": 1,
        "plan_name": "core-automation-test",
        "inputs": inputs,
        **extra,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_automation_adapter_preserves_rich_prompts_and_writes_strict_events(tmp_path: Path, capsys) -> None:
    plan_path = tmp_path / "plan.json"
    events_path = tmp_path / "events.json"
    _write_plan(
        plan_path,
        [
            {"id": "use_previous_experiment", "response": "n"},
            {"id": "experiment_name", "response": "Automation Contract"},
        ],
    )
    original_input = builtins.input

    with automation_input_adapter(plan_path.resolve(), events_path.resolve()):
        assert Confirm.ask("Use Previous Experiment", default=False) is False
        assert Prompt.ask("New Experiment") == "Automation Contract"

    assert builtins.input is original_input
    assert "Use Previous Experiment" in capsys.readouterr().out
    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert events["status"] == "completed"
    assert events["completed_input_ids"] == [
        "use_previous_experiment",
        "experiment_name",
    ]
    assert events["unused_input_ids"] == []
    assert all(len(event["prompt_sha256"]) == 64 for event in events["events"])


def test_automation_plan_rejects_unknown_duplicate_and_multiline_inputs(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, [{"id": "first", "response": "x"}], surprise=True)
    with pytest.raises(AutomationContractError, match="Unknown automation plan fields"):
        AutomationPlan.load(plan_path.resolve())

    _write_plan(
        plan_path,
        [
            {"id": "first", "response": "x"},
            {"id": "first", "response": "y"},
        ],
    )
    with pytest.raises(AutomationContractError, match="Duplicate automation input id"):
        AutomationPlan.load(plan_path.resolve())

    _write_plan(plan_path, [{"id": "first", "response": "x\ny"}])
    with pytest.raises(AutomationContractError, match="exactly one input line"):
        AutomationPlan.load(plan_path.resolve())


def test_automation_adapter_fails_closed_on_missing_and_unused_inputs(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    events_path = tmp_path / "events.json"
    _write_plan(plan_path, [{"id": "only", "response": "x"}])
    with pytest.raises(MissingAutomationInputError, match="requested another input"):
        with automation_input_adapter(plan_path.resolve(), events_path.resolve()):
            input("first")
            input("unexpected")
    assert json.loads(events_path.read_text(encoding="utf-8"))["status"] == "failed"

    _write_plan(
        plan_path,
        [
            {"id": "first", "response": "x"},
            {"id": "unused", "response": "y"},
        ],
    )
    with pytest.raises(UnusedAutomationInputError, match="unused automation inputs"):
        with automation_input_adapter(plan_path.resolve(), events_path.resolve()):
            input("first")
    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert events["unused_input_ids"] == ["unused"]


def test_automation_paths_must_be_absolute(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, [{"id": "first", "response": "x"}])
    with pytest.raises(AutomationContractError, match="absolute path"):
        AutomationPlan.load(Path("plan.json"))


def test_automation_plan_supports_noninteractive_commands(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    events_path = tmp_path / "events.json"
    _write_plan(plan_path, [])

    plan = AutomationPlan.load(plan_path.resolve())
    assert plan.inputs == ()
    with automation_input_adapter(plan_path.resolve(), events_path.resolve()):
        pass

    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert events["status"] == "completed"
    assert events["completed_input_ids"] == []
    assert events["events"] == []


def test_scientific_import_failures_are_recorded_by_the_automation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    events_path = tmp_path / "events.json"
    _write_plan(plan_path, [])
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.endswith("data_mining.cli_pipeline"):
            raise RuntimeError("native scientific dependency is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(RuntimeError, match="native scientific dependency is unavailable"):
        cli_module._run_cli_pipeline(
            training_data_path="unused.csv",
            application_data_path="",
            data_source_name="DATA",
            automation_plan=str(plan_path.resolve()),
            automation_events=str(events_path.resolve()),
        )

    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert events["status"] == "failed"
    assert events["error"] == {
        "type": "RuntimeError",
        "message": "native scientific dependency is unavailable",
    }


def test_data_mining_tracking_options_require_safe_stable_selection(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    missing_root = runner.invoke(
        cli_module.app,
        ["data-mining", "--data", str(tmp_path / "rocks.csv"), "--existing-experiment-id", "7"],
    )
    assert missing_root.exit_code == 2
    assert "requires --tracking-root" in unstyle(missing_root.output)

    relative_root = runner.invoke(
        cli_module.app,
        ["data-mining", "--data", str(tmp_path / "rocks.csv"), "--tracking-root", "relative"],
    )
    assert relative_root.exit_code == 2
    assert "must be an absolute local path" in unstyle(relative_root.output)

    captured = {}
    monkeypatch.setattr(cli_module, "_run_cli_pipeline", lambda **values: captured.update(values))
    tracking_root = tmp_path / "tracking"
    accepted = runner.invoke(
        cli_module.app,
        [
            "data-mining",
            "--data",
            str(tmp_path / "rocks.csv"),
            "--tracking-root",
            str(tracking_root),
            "--existing-experiment-id",
            "stable-7",
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    assert captured["tracking_root"] == str(tracking_root)
    assert captured["existing_experiment_id"] == "stable-7"


def test_interactive_data_mining_accepts_scientific_config_without_automation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scientific_path = (tmp_path / "scientific-execution.json").resolve()
    scientific_path.write_text("{}", encoding="utf-8")
    observed = {}
    monkeypatch.setattr(
        cli_module, "_run_cli_pipeline", lambda **values: observed.update(values)
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "data-mining",
            "--data",
            str(tmp_path / "rocks.csv"),
            "--scientific-config",
            str(scientific_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["scientific_config"] == str(scientific_path)
    assert observed["automation_plan"] == ""
    assert observed["automation_events"] == ""
