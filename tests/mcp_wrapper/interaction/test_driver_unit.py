import ast
import json
import os
import sys
from pathlib import Path

import geochemistrypi_mcp as cli_driver_package
import pytest
from geochemistrypi_mcp import (
    ClassificationPlanCompiler,
    ClassificationRequest,
    CliInteractionDriver,
    InteractionPlan,
    InteractionStep,
    PlanCompilationError,
    PromptTimeoutError,
    TimeSeriesPlanCompiler,
    TimeSeriesRequest,
    UnexpectedPromptError,
    UnusedResponsesError,
    WorkspacePathError,
)
from geochemistrypi_mcp.api.schemas import BuiltInDatasetReference
from geochemistrypi_mcp.planning.interaction_plan import _console_script_name
from pydantic import ValidationError


def _request(data_path: Path, **overrides) -> ClassificationRequest:
    values = {
        "training_dataset_path": data_path,
        "experiment_name": "Driver Contract",
        "run_name": "Classification V1",
        "identifier_column": "SampleID",
        "feature_columns": ("SIO2(WT%)", "TIO2(WT%)"),
        "target_column": "Label",
    }
    values.update(overrides)
    return ClassificationRequest(**values)


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "classification.csv"
    path.write_text(
        "SampleID,Label,SIO2(WT%),TIO2(WT%)\n"
        "GEO-1,0,50.0,1.0\n"
        "GEO-2,1,60.0,2.0\n"
        "GEO-3,0,52.0,1.2\n"
        "GEO-4,1,62.0,2.2\n"
        "GEO-5,0,54.0,1.4\n"
        "GEO-6,1,64.0,2.4\n"
        "GEO-7,0,56.0,1.6\n"
        "GEO-8,1,66.0,2.6\n",
        encoding="utf-8",
    )
    return path


def _fake_script(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "fake_cli.py"
    path.write_text(source, encoding="utf-8")
    return path


def _plan(script: Path, steps: tuple[InteractionStep, ...]) -> InteractionPlan:
    return InteractionPlan(
        schema_version=1,
        name="fake-cross-platform-cli",
        public_command=(sys.executable, "-u", str(script)),
        steps=steps,
    )


def test_semantic_request_rejects_unknown_fields_and_conflicting_columns(tmp_path: Path) -> None:
    data_path = _dataset(tmp_path)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _request(data_path, unsupported_option=True)
    with pytest.raises(ValidationError, match="target_column must not also be a feature"):
        _request(data_path, feature_columns=("Label", "SIO2(WT%)"))
    with pytest.raises(ValidationError, match="unsafe in an output directory name"):
        _request(data_path, run_name="unsafe/name")


def test_semantic_request_accepts_one_dataset_source_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    data_path = _dataset(tmp_path)
    values = {
        "training_dataset": BuiltInDatasetReference(dataset_id="builtin:classification"),
        "experiment_name": "Driver Contract",
        "run_name": "Classification V1",
        "identifier_column": "SampleID",
        "feature_columns": ("SIO2(WT%)", "TIO2(WT%)"),
        "target_column": "Label",
    }

    assert ClassificationRequest(**values).training_dataset.source == "builtin"
    with pytest.raises(ValidationError, match="provide exactly one"):
        ClassificationRequest(**{**values, "training_dataset_path": data_path})
    with pytest.raises(ValidationError, match="provide exactly one"):
        ClassificationRequest(**{key: value for key, value in values.items() if key != "training_dataset"})


def test_plan_compiler_maps_semantic_column_names_to_cli_indices(tmp_path: Path) -> None:
    data_path = _dataset(tmp_path)
    plan = ClassificationPlanCompiler().compile(_request(data_path), cli_executable=Path(sys.executable))
    responses = {step.id: step.response for step in plan.steps}

    assert plan.schema_version == 1
    assert plan.public_command[:4] == (
        str(Path(sys.executable).resolve()),
        "data-mining",
        "--data",
        str(data_path.resolve()),
    )
    assert plan.public_command[4] == "--world-map-config"
    assert json.loads(plan.public_command[5]) == {
        "schema_version": 1,
        "enabled": False,
        "longitude_column": None,
        "latitude_column": None,
        "value_columns": [],
    }
    assert responses["identifier_column"] == "1"
    assert responses["selected_data_columns"] == "[2,4]"
    assert responses["feature_columns"] == "[2,3]"
    assert responses["target_column"] == "1"
    assert responses["classification_mode"] == "2"
    assert responses["logistic_regression"] == "1"


def test_plan_compiler_rejects_missing_columns_before_starting_cli(tmp_path: Path) -> None:
    data_path = _dataset(tmp_path)
    request = _request(data_path, feature_columns=("SIO2(WT%)", "MISSING"))

    with pytest.raises(PlanCompilationError, match=r"absent from the training dataset: \['MISSING'\]"):
        ClassificationPlanCompiler().compile(request, cli_executable=Path(sys.executable))


def test_coordinate_dataset_can_disable_or_semantically_configure_world_map(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "coordinates.csv"
    data_path.write_text(
        "SampleID,Label,SIO2(WT%),TIO2(WT%),LATITUDE,LONGITUDE\n"
        "A,0,50,1,30,120\n"
        "B,1,60,2,31,121\n"
        "C,0,51,1.1,32,122\n"
        "D,1,61,2.1,33,123\n"
        "E,0,52,1.2,34,124\n"
        "F,1,62,2.2,35,125\n"
        "G,0,53,1.3,36,126\n"
        "H,1,63,2.3,37,127\n"
        "I,0,54,1.4,38,128\n"
        "J,1,64,2.4,39,129\n",
        encoding="utf-8",
    )

    disabled = ClassificationPlanCompiler().compile(_request(data_path), cli_executable=Path(sys.executable))
    configured = ClassificationPlanCompiler().compile(
        _request(
            data_path,
            world_map={
                "enabled": True,
                "longitude_column": "LONGITUDE",
                "latitude_column": "LATITUDE",
                "value_columns": ["SIO2(WT%)"],
            },
        ),
        cli_executable=Path(sys.executable),
    )

    assert json.loads(disabled.public_command[-1])["enabled"] is False
    assert json.loads(configured.public_command[-1])["value_columns"] == ["SIO2(WT%)"]


def test_world_map_values_and_coordinate_ranges_fail_before_cli(tmp_path: Path) -> None:
    data_path = tmp_path / "invalid-coordinates.csv"
    data_path.write_text(
        "SampleID,Label,Value,LATITUDE,LONGITUDE\n" "A,0,not-a-number,30,120\n" "B,1,2,95,121\n",
        encoding="utf-8",
    )
    request = _request(
        data_path,
        feature_columns=("Value",),
        world_map={
            "enabled": True,
            "longitude_column": "LONGITUDE",
            "latitude_column": "LATITUDE",
            "value_columns": ["Value"],
        },
    )

    with pytest.raises(PlanCompilationError, match="non-numeric"):
        ClassificationPlanCompiler().compile(request, cli_executable=Path(sys.executable))


def test_time_series_plan_is_noninteractive_and_semantically_validated(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "time-series.csv"
    data_path.write_text(
        "R_AGE,R_MAX_AGE,SBAP,LATITUDE,LONGITUDE\n" "10,12,0.9,-20,100\n" "20,25,0.1,5,110\n" "35,40,0.8,30,120\n",
        encoding="utf-8",
    )
    request = TimeSeriesRequest(
        training_dataset_path=data_path,
        bin_width=10,
        iterations=7,
        seed=19,
        fit_curve=False,
    )

    plan = TimeSeriesPlanCompiler().compile(request, cli_executable=Path(sys.executable))

    assert plan.steps == ()
    assert plan.public_command[:4] == (
        str(Path(sys.executable).resolve()),
        "time-series",
        "--input",
        str(data_path.resolve()),
    )
    assert plan.public_command[-1] == "--no-fit-curve"
    assert "Time Series Metrics.json" in plan.expected_output_relative_paths[2]


def test_time_series_plan_rejects_invalid_rows_before_cli(tmp_path: Path) -> None:
    data_path = tmp_path / "invalid-time-series.csv"
    data_path.write_text(
        "R_AGE,R_MAX_AGE,SBAP,LATITUDE,LONGITUDE\n" "20,10,1.2,95,181\n",
        encoding="utf-8",
    )
    request = TimeSeriesRequest(
        training_dataset_path=data_path,
        bin_width=10,
    )

    with pytest.raises(PlanCompilationError, match="maximum age"):
        TimeSeriesPlanCompiler().compile(request, cli_executable=Path(sys.executable))


@pytest.mark.parametrize(("os_name", "expected"), [("nt", "geochemistrypi.exe"), ("posix", "geochemistrypi")])
def test_console_script_name_is_cross_platform(os_name: str, expected: str) -> None:
    assert _console_script_name(os_name) == expected


def test_driver_waits_for_prompts_and_captures_durable_evidence(tmp_path: Path) -> None:
    script = _fake_script(
        tmp_path,
        """import os
from pathlib import Path
import sys

print("FIRST PROMPT> ", end="", flush=True)
first = input()
print(f"FIRST={first}", flush=True)
print("diagnostic", file=sys.stderr, flush=True)
print("SECOND PROMPT> ", end="", flush=True)
second = input()
Path("process-cwd.txt").write_text("isolated", encoding="utf-8")
print(f"ENCODING={os.environ['PYTHONIOENCODING']}", flush=True)
print(f"UNBUFFERED={os.environ['PYTHONUNBUFFERED']}", flush=True)
print(f"DATABASE={os.environ.get('SQLALCHEMY_DATABASE_URL', '<missing>')}", flush=True)
print(f"PYTHONHOME={os.environ.get('PYTHONHOME', '<missing>')}", flush=True)
print(f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '<missing>')}", flush=True)
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<missing>')}", flush=True)
print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '<missing>')}", flush=True)
print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '<missing>')}", flush=True)
print(f"NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS', '<missing>')}", flush=True)
print(f"VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS', '<missing>')}", flush=True)
print(f"BLIS_NUM_THREADS={os.environ.get('BLIS_NUM_THREADS', '<missing>')}", flush=True)
print(f"SECOND={second}", flush=True)
""",
    )
    steps = (
        InteractionStep("first", ("FIRST PROMPT>",), "alpha"),
        InteractionStep("second", ("SECOND PROMPT>",), "beta"),
    )

    result = CliInteractionDriver(prompt_timeout_seconds=2, process_timeout_seconds=10).run(
        _plan(script, steps),
        workspace_parent=tmp_path / "runs",
        environment={
            "PYTHONHOME": "C:/wrong-python",
            "PYTHONIOENCODING": "ascii",
            "PYTHONUNBUFFERED": "0",
            "SQLALCHEMY_DATABASE_URL": "sqlite:///wrong.db",
            "VIRTUAL_ENV": "C:/wrong-environment",
            "OMP_NUM_THREADS": "99",
            "OPENBLAS_NUM_THREADS": "99",
            "MKL_NUM_THREADS": "99",
            "NUMEXPR_NUM_THREADS": "99",
            "VECLIB_MAXIMUM_THREADS": "99",
            "BLIS_NUM_THREADS": "99",
        },
    )

    assert result.returncode == 0
    assert result.completed_step_ids == ("first", "second")
    assert result.workspace.parent == (tmp_path / "runs").resolve()
    assert (result.workspace / "process-cwd.txt").read_text(encoding="utf-8") == "isolated"
    stdout = result.stdout_path.read_text(encoding="utf-8")
    assert "FIRST=alpha" in stdout
    assert "SECOND=beta" in stdout
    assert "ENCODING=utf-8" in stdout
    assert "UNBUFFERED=1" in stdout
    assert "DATABASE=<missing>" in stdout
    assert "PYTHONHOME=C:/wrong-python" not in stdout
    assert "VIRTUAL_ENV=<missing>" in stdout
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert f"{variable}=1" in stdout
    assert result.stderr_path.read_text(encoding="utf-8").strip() == "diagnostic"
    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "completed"
    assert trace["completed_step_ids"] == ["first", "second"]
    assert [event["response"] for event in trace["events"]] == ["alpha", "beta"]


def test_driver_uses_cli_automation_contract_without_prompt_matching(
    tmp_path: Path,
) -> None:
    script = _fake_script(
        tmp_path,
        """import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--automation-plan', required=True)
parser.add_argument('--automation-events', required=True)
args = parser.parse_args()
plan = json.loads(Path(args.automation_plan).read_text(encoding='utf-8'))
events = []
for sequence, item in enumerate(plan['inputs'], start=1):
    prompt = 'PROMPT TEXT MAY CHANGE WITHOUT BREAKING THE TRANSPORT'
    events.append({
        'sequence': sequence,
        'input_id': item['id'],
        'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        'prompt_length': len(prompt),
        'consumed_at': datetime.now(timezone.utc).isoformat(),
    })
Path(args.automation_events).write_text(json.dumps({
    'schema_version': 1,
    'plan_name': plan['plan_name'],
    'status': 'completed',
    'started_at': datetime.now(timezone.utc).isoformat(),
    'finished_at': datetime.now(timezone.utc).isoformat(),
    'completed_input_ids': [item['id'] for item in plan['inputs']],
    'unused_input_ids': [],
    'events': events,
    'error': None,
}), encoding='utf-8')
print('automation completed', flush=True)
""",
    )
    steps = (
        InteractionStep("first", ("OLD FIRST PROMPT>",), "alpha"),
        InteractionStep("second", ("OLD SECOND PROMPT>",), "beta"),
    )

    result = CliInteractionDriver(
        prompt_timeout_seconds=0.1,
        process_timeout_seconds=10,
        automation_mode=True,
    ).run(_plan(script, steps), workspace_parent=tmp_path / "runs")

    assert result.completed_step_ids == ("first", "second")
    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert trace["input_transport"] == "cli_automation_v1"
    assert trace["events"][0]["response"] == "alpha"
    assert "--automation-plan" in trace["command"]
    assert "OLD FIRST PROMPT" not in result.stdout_path.read_text(encoding="utf-8")


def test_driver_fails_on_a_known_prompt_arriving_out_of_order(tmp_path: Path) -> None:
    script = _fake_script(tmp_path, 'print("SECOND PROMPT> ", end="", flush=True)\ninput()\n')
    plan = _plan(
        script,
        (
            InteractionStep("first", ("FIRST PROMPT>",), "alpha"),
            InteractionStep("second", ("SECOND PROMPT>",), "beta"),
        ),
    )

    with pytest.raises(UnexpectedPromptError, match="before expected step 'first'") as captured:
        CliInteractionDriver(prompt_timeout_seconds=2, process_timeout_seconds=10).run(plan, workspace_parent=tmp_path / "runs")

    trace = json.loads((captured.value.capture_directory / "interaction-trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["completed_step_ids"] == []


def test_driver_fails_when_cli_exits_with_unused_responses(tmp_path: Path) -> None:
    script = _fake_script(tmp_path, 'print("FIRST PROMPT> ", end="", flush=True)\ninput()\n')
    plan = _plan(
        script,
        (
            InteractionStep("first", ("FIRST PROMPT>",), "alpha"),
            InteractionStep("second", ("SECOND PROMPT>",), "beta"),
        ),
    )

    with pytest.raises(UnusedResponsesError, match="before consuming 1 planned responses"):
        CliInteractionDriver(prompt_timeout_seconds=2, process_timeout_seconds=10).run(plan, workspace_parent=tmp_path / "runs")


def test_driver_fails_closed_when_prompt_text_changes(tmp_path: Path) -> None:
    script = _fake_script(tmp_path, 'print("CHANGED PROMPT> ", end="", flush=True)\ninput()\n')
    plan = _plan(script, (InteractionStep("first", ("FIRST PROMPT>",), "alpha"),))

    with pytest.raises(PromptTimeoutError, match="waiting for interaction step 'first'"):
        CliInteractionDriver(prompt_timeout_seconds=0.25, process_timeout_seconds=2).run(plan, workspace_parent=tmp_path / "runs")


def test_prompt_timeout_is_not_starved_by_continuous_stderr(tmp_path: Path) -> None:
    script = _fake_script(
        tmp_path,
        'import sys, time\nprint("CHANGED PROMPT> ", end="", flush=True)\nwhile True:\n    print("diagnostic" * 100, file=sys.stderr, flush=True)\n    time.sleep(0.001)\n',
    )
    plan = _plan(script, (InteractionStep("first", ("FIRST PROMPT>",), "alpha"),))

    with pytest.raises(PromptTimeoutError, match="waiting for interaction step 'first'"):
        CliInteractionDriver(prompt_timeout_seconds=0.25, process_timeout_seconds=2).run(plan, workspace_parent=tmp_path / "runs")


@pytest.mark.skipif(os.name != "nt", reason="Windows legacy path budget regression")
def test_driver_rejects_an_output_path_that_windows_plotting_cannot_save(tmp_path: Path) -> None:
    script = _fake_script(tmp_path, 'raise SystemExit("must not start")\n')
    plan = InteractionPlan(
        schema_version=1,
        name="windows-path-budget",
        public_command=(sys.executable, "-u", str(script)),
        steps=(InteractionStep("never", ("NEVER>",), ""),),
        expected_output_relative_paths=("x" * 260,),
    )

    with pytest.raises(WorkspacePathError, match="choose a shorter workspace parent"):
        CliInteractionDriver(prompt_timeout_seconds=1, process_timeout_seconds=2).run(plan, workspace_parent=tmp_path / "runs")


@pytest.mark.skipif(os.name != "nt", reason="Windows generated-sidecar path budget regression")
def test_driver_reserves_windows_path_space_for_generated_spreadsheet_sidecars(tmp_path: Path) -> None:
    script = _fake_script(tmp_path, 'raise SystemExit("must not start")\n')
    workspace = tmp_path / "workspace"
    relative_path = "x" * (259 - len(str(workspace)) - 1)
    plan = InteractionPlan(
        schema_version=1,
        name="windows-sidecar-path-budget",
        public_command=(sys.executable, "-u", str(script)),
        steps=(InteractionStep("never", ("NEVER>",), ""),),
        expected_output_relative_paths=(relative_path,),
    )

    assert len(str(workspace / relative_path)) == 259
    with pytest.raises(WorkspacePathError, match="including generated sidecars"):
        CliInteractionDriver(prompt_timeout_seconds=1, process_timeout_seconds=2).run(plan, workspace=workspace)


def test_driver_package_does_not_import_machine_learning_implementations() -> None:
    package_directory = Path(cli_driver_package.__file__).resolve().parent
    forbidden_prefixes = ("geochemistrypi", "sklearn", "xgboost", "ray", "mlflow")
    imported_modules = []
    for path in package_directory.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

    assert not [module for module in imported_modules if module.startswith(forbidden_prefixes)]
