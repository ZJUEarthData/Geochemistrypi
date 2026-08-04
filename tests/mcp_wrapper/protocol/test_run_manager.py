import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
from geochemistrypi_mcp import CliInteractionDriver, InteractionPlan, InteractionStep
from geochemistrypi_mcp.classification_contract import MODEL_DISPLAY_NAMES as CLASSIFICATION_MODEL_DISPLAY_NAMES
from geochemistrypi_mcp.classification_contract import MODEL_ORDER as CLASSIFICATION_MODEL_ORDER
from geochemistrypi_mcp.dataset_catalog import ResolvedDataset
from geochemistrypi_mcp.runs import RunManager, RunStateError
from geochemistrypi_mcp.schemas import BuiltInDatasetReference, ClassificationRequest, ClusteringRequest, DecompositionRequest, RandomForestSettings, TimeSeriesRequest
from geochemistrypi_mcp.settings import McpSettings


def _request(
    dataset: Path,
    run_name: str = "Reference Run",
    application_dataset: Path | None = None,
) -> ClassificationRequest:
    return ClassificationRequest(
        training_dataset_path=dataset,
        experiment_name="MCP Contract",
        run_name=run_name,
        identifier_column="SampleID",
        feature_columns=("SIO2",),
        target_column="Label",
        application_dataset_path=application_dataset,
    )


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "rocks.csv"
    path.write_text("SampleID,SIO2,Label\nA,50.1,basalt\nB,61.0,granite\n", encoding="utf-8")
    return path


class ScriptPlanCompiler:
    def __init__(self, script: Path):
        self.script = script

    def compile(
        self,
        request: ClassificationRequest | ClusteringRequest | DecompositionRequest | TimeSeriesRequest,
        cli_executable: Path | None = None,
    ) -> InteractionPlan:
        return InteractionPlan(
            schema_version=1,
            name="test-cli-plan",
            public_command=(sys.executable, "-u", str(self.script), request.experiment_name, request.run_name),
            steps=(InteractionStep("ready", ("READY>",), "continue"),),
        )


def _manager(
    tmp_path: Path,
    script: Path,
    maximum_pending_runs: int = 8,
    dataset_catalog=None,
) -> RunManager:
    settings = McpSettings(
        runs_root=tmp_path / "runs",
        cli_executable=Path(sys.executable),
        maximum_dataset_bytes=1024 * 1024,
        maximum_pending_runs=maximum_pending_runs,
    )
    return RunManager(
        settings,
        plan_compiler=ScriptPlanCompiler(script),
        driver_factory=lambda: CliInteractionDriver(prompt_timeout_seconds=3, process_timeout_seconds=20),
        cli_resolver=lambda: (Path(sys.executable), "0.8.0"),
        dataset_catalog=dataset_catalog,
    )


class FixedDatasetCatalog:
    def __init__(self, path: Path):
        self.path = path

    def resolve(self, reference, *, task=None, role=None) -> ResolvedDataset:
        assert reference.dataset_id == "builtin:classification"
        assert task == "classification"
        assert role == "training"
        return ResolvedDataset(
            path=self.path,
            expected_sha256=None,
            dataset_id=reference.dataset_id,
            source="builtin",
        )


def _wait_for_state(manager: RunManager, run_id: str, expected: set[str], timeout: float = 10) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = manager.get_status(run_id).state
        if state in expected:
            return state
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not reach {expected}; last state was {manager.get_status(run_id).state}")


def test_validate_analysis_previews_workload_without_creating_run_or_process(tmp_path: Path) -> None:
    script = tmp_path / "never-started.py"
    script.write_text("raise AssertionError('must not start')\n", encoding="utf-8")
    dataset = _dataset(tmp_path)
    manager = _manager(tmp_path, script)
    try:
        analysis_request = _request(dataset).model_copy(update={"model": RandomForestSettings(number_of_estimators=500)})
        preview = manager.validate(analysis_request)
        assert preview.valid is True
        assert preview.task == "classification"
        assert preview.models == ("random_forest",)
        assert preview.estimated_model_count == 1
        assert preview.columns == ("SampleID", "SIO2", "Label")
        assert preview.identifier_column == "SampleID"
        assert preview.feature_columns == ("SIO2",)
        assert preview.target_column == "Label"
        assert preview.resolved_model_parameters["number_of_estimators"] == 500
        assert preview.training_sha256
        assert preview.analysis_process_started is False
        assert "inference will be skipped" in " ".join(preview.warnings)
        assert not manager.settings.runs_root.exists()
    finally:
        manager.close()


def test_run_is_non_blocking_atomic_and_references_only_original_cli_outputs(tmp_path: Path) -> None:
    script = tmp_path / "successful_cli.py"
    script.write_text(
        """import json
import sys
import time
from pathlib import Path

print("READY>", flush=True)
input()
time.sleep(0.4)
root = Path("geopi_output") / sys.argv[1] / sys.argv[2]
for name in ("artifacts", "metrics", "parameters", "summary"):
    (root / name).mkdir(parents=True)
(root / "artifacts" / "model.joblib").write_bytes(b"real-cli-model")
(root / "metrics" / "Model Score.txt").write_text(json.dumps({"accuracy": 0.75, "f1": 0.73}), encoding="utf-8")
(root / "parameters" / "Model Parameters.txt").write_text(json.dumps({"solver": "lbfgs"}), encoding="utf-8")
(root / "summary" / "Model Score.txt").write_text(json.dumps({"accuracy": 0.75, "f1": 0.73}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    dataset = _dataset(tmp_path)
    manager = _manager(tmp_path, script)
    started = time.monotonic()
    acknowledgement = manager.start(_request(dataset))
    elapsed = time.monotonic() - started

    try:
        assert elapsed < 0.5
        assert acknowledgement.state == "queued"
        assert _wait_for_state(manager, acknowledgement.run_id, {"succeeded"}) == "succeeded"
        result = manager.get_result(acknowledgement.run_id)
        assert result.input_hash_verified is True
        assert result.reported_metrics["Model Score.txt"] == {"accuracy": 0.75, "f1": 0.73}
        assert result.artifact_count == 4
        assert all(Path(item.local_path).is_relative_to(Path(result.output_directory)) for item in result.artifacts)
        assert {item.category for item in result.artifacts} == {"artifacts", "metrics", "parameters", "summary"}
        wrapper = tmp_path / "runs" / acknowledgement.run_id / "wrapper"
        assert json.loads((wrapper / "status.json").read_text(encoding="utf-8"))["state"] == "succeeded"
        assert json.loads((wrapper / "artifact-index.json").read_text(encoding="utf-8"))["artifacts"]
        assert not list(wrapper.glob("*.tmp"))

        clustering = manager.start(
            ClusteringRequest(
                task="clustering",
                training_dataset_path=dataset,
                experiment_name="MCP Contract",
                run_name="Clustering Run",
                identifier_column="SampleID",
                feature_columns=("SIO2",),
            )
        )
        assert _wait_for_state(manager, clustering.run_id, {"succeeded"}) == "succeeded"
        clustering_result = manager.get_result(clustering.run_id)
        assert clustering_result.task == "clustering"
        assert clustering_result.model == "kmeans"
        assert clustering_result.tuning == "not_applicable"
        assert clustering_result.application_input_sha256 is None

        decomposition = manager.start(
            DecompositionRequest(
                task="decomposition",
                training_dataset_path=dataset,
                experiment_name="MCP Contract",
                run_name="Decomposition Run",
                identifier_column="SampleID",
                feature_columns=("SIO2",),
            )
        )
        assert _wait_for_state(manager, decomposition.run_id, {"succeeded"}) == "succeeded"
        decomposition_result = manager.get_result(decomposition.run_id)
        assert decomposition_result.task == "decomposition"
        assert decomposition_result.model == "pca"
        assert decomposition_result.tuning == "not_applicable"
        assert decomposition_result.reported_metrics["Model Score.txt"] == {
            "accuracy": 0.75,
            "f1": 0.73,
        }

        time_series = manager.start(
            TimeSeriesRequest(
                training_dataset_path=dataset,
                experiment_name="MCP Contract",
                run_name="Time Series Run",
                bin_width=10,
            )
        )
        assert _wait_for_state(manager, time_series.run_id, {"succeeded"}) == "succeeded"
        time_series_result = manager.get_result(time_series.run_id)
        assert time_series_result.task == "time_series"
        assert time_series_result.model == "subaerial_proportion_bootstrap"
        assert time_series_result.tuning == "not_applicable"
    finally:
        manager.close()


def test_builtin_reference_is_resolved_before_snapshot_and_recorded(
    tmp_path: Path,
) -> None:
    script = tmp_path / "builtin_cli.py"
    script.write_text(
        """import sys
from pathlib import Path
print('READY>', flush=True)
input()
root = Path('geopi_output') / sys.argv[1] / sys.argv[2]
for name in ('artifacts', 'metrics', 'parameters', 'summary'):
    (root / name).mkdir(parents=True)
""",
        encoding="utf-8",
    )
    dataset = _dataset(tmp_path)
    before = dataset.read_bytes()
    manager = _manager(
        tmp_path,
        script,
        dataset_catalog=FixedDatasetCatalog(dataset.resolve()),
    )
    request = ClassificationRequest(
        training_dataset=BuiltInDatasetReference(dataset_id="builtin:classification"),
        experiment_name="MCP Contract",
        run_name="Built-in Source",
        identifier_column="SampleID",
        feature_columns=("SIO2",),
        target_column="Label",
    )

    started = manager.start(request)
    assert _wait_for_state(manager, started.run_id, {"succeeded", "failed"}) == "succeeded"
    record = json.loads((tmp_path / "runs" / started.run_id / "wrapper" / "request.json").read_text(encoding="utf-8"))

    assert record["request"]["training_dataset"]["dataset_id"] == "builtin:classification"
    assert record["input"]["source"] == "builtin"
    assert record["input"]["dataset_id"] == "builtin:classification"
    assert dataset.read_bytes() == before
    manager.close()


def test_all_models_partial_failure_is_terminal_and_result_remains_available(
    tmp_path: Path,
) -> None:
    models = [CLASSIFICATION_MODEL_DISPLAY_NAMES[model] for model in CLASSIFICATION_MODEL_ORDER]
    script = tmp_path / "aggregate_cli.py"
    source = """import json
import sys
from pathlib import Path

print('READY>', flush=True)
input()
root = Path('geopi_output') / sys.argv[1] / sys.argv[2]
for name in ('artifacts', 'metrics', 'parameters', 'summary'):
    (root / name).mkdir(parents=True)
models = __MODELS__
children = []
for index, model in enumerate(models):
    child = root / model
    child.mkdir()
    failed = index == 1
    children.append({
        'model': model,
        'state': 'failed' if failed else 'succeeded',
        'output_relative_path': model,
        'artifact_count': 0,
        'error': 'bounded child failure' if failed else None,
    })
manifest = {
    'schema_version': 1,
    'task': 'classification',
    'selection_mode': 'all',
    'tuning': 'manual',
    'state': 'partial_failure',
    'expected_model_count': len(models),
    'succeeded_count': len(models) - 1,
    'failed_count': 1,
    'children': children,
}
(root / 'summary' / 'Aggregate Model Results.json').write_text(
    json.dumps(manifest), encoding='utf-8'
)
""".replace(
        "__MODELS__", repr(models)
    )
    script.write_text(source, encoding="utf-8")
    manager = _manager(tmp_path, script)
    request = ClassificationRequest(
        training_dataset_path=_dataset(tmp_path),
        experiment_name="MCP Contract",
        run_name="All Models Partial",
        identifier_column="SampleID",
        feature_columns=("SIO2",),
        target_column="Label",
        model_selection={"mode": "all", "tuning": "manual"},
    )

    started = manager.start(request)
    try:
        assert _wait_for_state(manager, started.run_id, {"partial_failure"}) == "partial_failure"
        result = manager.get_result(started.run_id)
        assert result.state == "partial_failure"
        assert result.model == "all_models"
        assert result.aggregate_state == "partial_failure"
        assert result.aggregate_summary.model_dump() == {
            "expected_model_count": len(CLASSIFICATION_MODEL_ORDER),
            "succeeded_count": len(CLASSIFICATION_MODEL_ORDER) - 1,
            "failed_count": 1,
        }
        assert len(result.children) == len(models)
        assert [child.model for child in result.children] == models
        assert result.children[1].state == "failed"
        assert result.children[1].error == "bounded child failure"
    finally:
        manager.close()


def test_cancellation_terminates_recorded_tree_and_preserves_unrelated_process(tmp_path: Path) -> None:
    script = tmp_path / "long_cli.py"
    script.write_text(
        """import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path("child.pid").write_text(str(child.pid), encoding="utf-8")
print("READY>", flush=True)
input()
time.sleep(60)
""",
        encoding="utf-8",
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    manager = _manager(tmp_path, script)
    acknowledgement = manager.start(_request(_dataset(tmp_path)))
    workspace = tmp_path / "runs" / acknowledgement.run_id / "workspace"
    try:
        assert _wait_for_state(manager, acknowledgement.run_id, {"running"}) == "running"
        deadline = time.monotonic() + 5
        while not (workspace / "child.pid").is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        child_pid = int((workspace / "child.pid").read_text(encoding="utf-8"))
        response = manager.cancel(acknowledgement.run_id)
        assert response.state == "cancellation_requested"
        assert _wait_for_state(manager, acknowledgement.run_id, {"cancelled"}) == "cancelled"
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not psutil.pid_exists(child_pid)
        assert unrelated.poll() is None
    finally:
        manager.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_input_change_during_execution_fails_integrity_check(tmp_path: Path) -> None:
    script = tmp_path / "integrity_cli.py"
    script.write_text(
        """import sys
import time
from pathlib import Path

print("READY>", flush=True)
input()
time.sleep(0.5)
root = Path("geopi_output") / sys.argv[1] / sys.argv[2]
for name in ("artifacts", "metrics", "parameters", "summary"):
    (root / name).mkdir(parents=True)
""",
        encoding="utf-8",
    )
    dataset = _dataset(tmp_path)
    manager = _manager(tmp_path, script)
    acknowledgement = manager.start(_request(dataset, run_name="Integrity"))
    try:
        assert _wait_for_state(manager, acknowledgement.run_id, {"running"}) == "running"
        dataset.write_text(dataset.read_text(encoding="utf-8") + "C,55.0,andesite\n", encoding="utf-8")
        assert _wait_for_state(manager, acknowledgement.run_id, {"failed"}) == "failed"
        status = manager.get_status(acknowledgement.run_id)
        assert "changed during CLI execution" in status.error
        with pytest.raises(RunStateError, match="available only after it succeeds"):
            manager.get_result(acknowledgement.run_id)
    finally:
        manager.close()


def test_application_input_change_during_execution_fails_integrity_check(tmp_path: Path) -> None:
    script = tmp_path / "application_integrity_cli.py"
    script.write_text(
        """import sys
import time
from pathlib import Path

print("READY>", flush=True)
input()
time.sleep(0.5)
root = Path("geopi_output") / sys.argv[1] / sys.argv[2]
for name in ("artifacts", "metrics", "parameters", "summary"):
    (root / name).mkdir(parents=True)
""",
        encoding="utf-8",
    )
    training = _dataset(tmp_path)
    application = tmp_path / "application.csv"
    application.write_text("SampleID,SIO2\nP-1,55.0\n", encoding="utf-8")
    manager = _manager(tmp_path, script)
    acknowledgement = manager.start(_request(training, run_name="Application Integrity", application_dataset=application))
    try:
        assert _wait_for_state(manager, acknowledgement.run_id, {"running"}) == "running"
        application.write_text(application.read_text(encoding="utf-8") + "P-2,57.0\n", encoding="utf-8")
        assert _wait_for_state(manager, acknowledgement.run_id, {"failed"}) == "failed"
        status = manager.get_status(acknowledgement.run_id)
        assert "application dataset changed during cli execution" in status.error.lower()
        with pytest.raises(RunStateError, match="available only after it succeeds"):
            manager.get_result(acknowledgement.run_id)
    finally:
        manager.close()


def test_recovery_never_terminates_a_process_from_stale_pid_metadata(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run-0123456789abcdef"
    wrapper = runs_root / run_id / "wrapper"
    wrapper.mkdir(parents=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (wrapper / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "state": "running",
                "created_at": "2026-08-02T12:00:00+00:00",
                "started_at": "2026-08-02T12:00:01+00:00",
                "finished_at": None,
                "cli_pid": unrelated.pid,
                "recorded_cli_pid": unrelated.pid,
                "recorded_process_create_time": psutil.Process(unrelated.pid).create_time(),
                "progress_message": "running",
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    settings = McpSettings(runs_root=runs_root, cli_executable=Path(sys.executable))
    manager = RunManager(settings, cli_resolver=lambda: (Path(sys.executable), "0.8.0"))
    try:
        status = manager.get_status(run_id)
        assert status.state == "failed"
        assert "no stale PID was terminated" in status.error
        assert unrelated.poll() is None
        with pytest.raises(RunStateError, match="already failed"):
            manager.cancel(run_id)
    finally:
        manager.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_run_queue_fails_closed_at_capacity_and_reopens_after_cancellation(tmp_path: Path) -> None:
    script = tmp_path / "queue_cli.py"
    script.write_text(
        """import time

print("READY>", flush=True)
input()
time.sleep(60)
""",
        encoding="utf-8",
    )
    manager = _manager(tmp_path, script, maximum_pending_runs=1)
    dataset = _dataset(tmp_path)
    first = manager.start(_request(dataset, run_name="First"))
    try:
        assert _wait_for_state(manager, first.run_id, {"running"}) == "running"
        with pytest.raises(RunStateError, match="run queue is full"):
            manager.start(_request(dataset, run_name="Rejected"))
        assert len(tuple((tmp_path / "runs").glob("run-*"))) == 1

        manager.cancel(first.run_id)
        assert _wait_for_state(manager, first.run_id, {"cancelled"}) == "cancelled"
        replacement = manager.start(_request(dataset, run_name="Replacement"))
        assert replacement.state == "queued"
    finally:
        manager.close()


def test_default_driver_uses_the_configured_total_process_timeout(tmp_path: Path) -> None:
    settings = McpSettings(
        runs_root=tmp_path / "runs",
        cli_executable=Path(sys.executable),
        maximum_process_seconds=37,
    )
    manager = RunManager(settings, cli_resolver=lambda: (Path(sys.executable), "0.8.0"))
    try:
        assert manager.driver_factory().process_timeout_seconds == 37
        assert manager.driver_factory().automation_mode is True
    finally:
        manager.close()
