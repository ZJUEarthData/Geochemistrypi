"""Summary must contain only existing, unchanged result files."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import stat
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from geochemistrypi.online.app import create_app
from geochemistrypi.online.data_mining_service import DataMiningService
from geochemistrypi.online import summary_service


def dataset():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(40, 3))
    return pd.DataFrame({
        "Sample": [f"S{i:03}" for i in range(40)],
        "A": x[:, 0], "B": x[:, 1], "C": x[:, 2],
        "Target": x[:, 0] * 2 - x[:, 1],
        "Class": np.where(x[:, 0] > 0, "basalt", "andesite"),
    })


def uploaded(frame):
    return frame.to_csv(index=False).encode("utf-8")


def archive_path(service, payload, job_id):
    return service.resolve_artifact(job_id, payload["artifact"]["download_url"].split("/")[-1])


def saved_job(service, files=None):
    job_id = uuid4().hex
    output = service.jobs_dir / job_id / "output"
    output.mkdir(parents=True)
    for name, content in (files if files is not None else {"result.csv": b"A,B\r\n1,2\r\n"}).items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return job_id, output


@pytest.mark.parametrize("mode", [
    "regression", "classification", "clustering", "dimensionality_reduction",
    "anomaly_detection", "time_series",
])
def test_six_modes_package_exact_existing_artifacts(tmp_path, monkeypatch, mode):
    service = DataMiningService(tmp_path)
    frame = dataset()
    if mode == "time_series":
        frame = pd.DataFrame({
            "Age": np.linspace(100, 1000, 20),
            "MaxAge": np.linspace(100, 1000, 20) + 10,
            "Probability": np.linspace(0, 1, 20),
            "Latitude": np.linspace(-40, 40, 20),
            "Longitude": np.linspace(-60, 60, 20),
        })
        result = service.run_time_series(
            filename="input.csv", content=uploaded(frame), age_column="Age",
            age_max_column="MaxAge", probability_column="Probability",
            latitude_column="Latitude", longitude_column="Longitude",
            bin_width=200, bootstrap_iterations=10,
        )
    else:
        arguments = dict(filename="input.csv", content=uploaded(frame), feature_columns=["A", "B", "C"])
        if mode in {"regression", "classification"}:
            arguments["target_column"] = "Target" if mode == "regression" else "Class"
        result = getattr(service, "run_" + mode)(**arguments)

    before = {a.name: service.resolve_artifact(result.job_id, a.name).read_bytes() for a in result.artifacts}
    assert before

    def forbidden(*args, **kwargs):
        pytest.fail("Summary must not load a model or rerun an analysis")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(service, "run_" + mode, forbidden)
    payload = service.generate_summary(result.job_id)
    with ZipFile(archive_path(service, payload, result.job_id)) as archive:
        assert sorted(archive.namelist()) == sorted("summary/" + name for name in before)
        assert archive.testzip() is None
        for name, content in before.items():
            assert archive.read("summary/" + name) == content
            assert service.resolve_artifact(result.job_id, name).read_bytes() == content
    assert sorted(payload["files"]) == sorted(before)
    assert payload["artifact"]["name"] == "summary.zip"
    assert not (service.jobs_dir / result.job_id / "summary_context.joblib").exists()


def test_old_jobs_include_nested_outputs_but_never_private_files_or_other_jobs(tmp_path, monkeypatch):
    service = DataMiningService(tmp_path)
    files = {"预测.csv": b"\xef\xbb\xbfA\n1\n", "figures/a.svg": b"<svg/>",
             "trained_pipeline.joblib": b"opaque bytes, not a valid model",
             "details.xlsx": b"opaque workbook bytes", "optional.zip": b"existing non-Summary archive"}
    job_id, output = saved_job(service, files)
    (output.parent / "input.csv").write_bytes(b"private input")
    (output.parent / "summary_context.joblib").write_bytes(b"obsolete private snapshot")
    saved_job(service, {"other-user.csv": b"unrelated job"})

    def forbidden(*args, **kwargs):
        pytest.fail("Do not unpickle result files")

    monkeypatch.setattr(joblib, "load", forbidden)
    payload = service.generate_summary(job_id)
    with ZipFile(archive_path(service, payload, job_id)) as archive:
        assert sorted(archive.namelist()) == sorted("summary/" + name for name in files)
        for name, content in files.items():
            assert archive.read("summary/" + name) == content


def test_repeated_downloads_do_not_nest_archives_and_changed_outputs_get_new_urls(tmp_path):
    service = DataMiningService(tmp_path)
    job_id, output = saved_job(service)
    (output / "summary.zip").write_bytes(b"legacy Summary")
    (output / ("summary_" + "a" * 64 + ".zip")).write_bytes(b"old generated Summary")
    first = service.generate_summary(job_id)
    path = archive_path(service, first, job_id)
    original = path.read_bytes()
    assert service.generate_summary(job_id) == first
    assert path.read_bytes() == original
    with ZipFile(path) as archive:
        assert archive.namelist() == ["summary/result.csv"]
    (output / "new-report.json").write_bytes(b'{"ok":true}')
    changed = service.generate_summary(job_id)
    assert changed["artifact"]["download_url"] != first["artifact"]["download_url"]
    assert path.read_bytes() == original
    with ZipFile(archive_path(service, changed, job_id)) as archive:
        assert sorted(archive.namelist()) == ["summary/new-report.json", "summary/result.csv"]


def test_concurrent_requests_publish_complete_identical_archives(tmp_path):
    service = DataMiningService(tmp_path)
    job_id, output = saved_job(service, {"large.bin": bytes(range(256)) * 10000})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(service.generate_summary, [job_id] * 4))
    assert all(payload == results[0] for payload in results)
    assert len(list(output.glob("summary_*.zip"))) == 1
    with ZipFile(archive_path(service, results[0], job_id)) as archive:
        assert archive.read("summary/large.bin") == (output / "large.bin").read_bytes()


def test_failed_packaging_never_publishes_partial_archive(tmp_path, monkeypatch):
    service = DataMiningService(tmp_path)
    job_id, output = saved_job(service)
    first = service.generate_summary(job_id)
    original = archive_path(service, first, job_id).read_bytes()
    before = sorted(output.iterdir())

    def broken(source, target, **kwargs):
        target.write(b"partial")
        raise OSError("simulated disk full")

    monkeypatch.setattr(summary_service.shutil, "copyfileobj", broken)
    with pytest.raises(OSError, match="disk full"):
        service.generate_summary(job_id)
    assert sorted(output.iterdir()) == before
    assert archive_path(service, first, job_id).read_bytes() == original
    assert not list(output.parent.glob(".summary-build-*"))


@pytest.mark.parametrize("nested", [False, True])
def test_reparse_points_are_rejected_before_reading_or_descending(tmp_path, monkeypatch, nested):
    service = DataMiningService(tmp_path)
    job_id, output = saved_job(service)
    guarded = output / ("linked" if nested else "result.csv")
    if nested:
        guarded.mkdir()
    original_lstat = Path.lstat
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, raising=False)

    def flagged(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == guarded:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(Path, "lstat", flagged)
    with pytest.raises(FileNotFoundError, match="Linked result paths"):
        service.generate_summary(job_id)
    assert not list(output.glob("summary_*.zip"))


def test_summary_endpoint_download_and_errors_without_snapshots(tmp_path, monkeypatch):
    app = create_app(runtime_dir=tmp_path)
    service = app.state.data_mining_service
    job_id, output = saved_job(service, {"classification_predictions.csv": b"true,predicted\n0,0\n"})
    (output.parent / "private.json").write_bytes(b"secret")
    empty_job, _ = saved_job(service, {})
    with TestClient(app) as client:
        response = client.post(f"/api/data-mining/jobs/{job_id}/summary")
        assert response.status_code == 200, response.text
        payload = response.json()
        download = client.get(payload["artifact"]["download_url"])
        assert download.status_code == 200
        assert 'filename="summary.zip"' in download.headers["content-disposition"]
        assert payload["artifact"]["size_bytes"] == len(download.content)
        with ZipFile(BytesIO(download.content)) as archive:
            assert archive.namelist() == ["summary/classification_predictions.csv"]
        original = client.get(f"/api/data-mining/jobs/{job_id}/files/classification_predictions.csv")
        assert original.content == b"true,predicted\n0,0\n"
        assert 'filename="classification_predictions.csv"' in original.headers["content-disposition"]
        for invalid in ("invalid", uuid4().hex, empty_job):
            assert client.post(f"/api/data-mining/jobs/{invalid}/summary").status_code == 404
        assert client.get(f"/api/data-mining/jobs/{job_id}/files/private.json").status_code == 404
        assert client.get(f"/api/data-mining/jobs/{job_id}/files/..%5Cprivate.json").status_code == 404

        def broken(*args):
            raise OSError("private server path should not be exposed")

        monkeypatch.setattr(service, "generate_summary", broken)
        failed = client.post(f"/api/data-mining/jobs/{job_id}/summary")
        assert failed.status_code == 500
        assert "private server" not in failed.text
        assert "retry" in failed.json()["detail"]


def test_classification_api_result_can_be_packaged_without_changing_contract(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path)) as client:
        result = client.post(
            "/api/data-mining/classification",
            data={"model": "logistic_regression", "feature_columns": json.dumps(["A", "B", "C"]), "target_column": "Class"},
            files={"dataset": ("input.csv", uploaded(dataset()), "text/csv")},
        )
        assert result.status_code == 200, result.text
        payload = result.json()
        response = client.post(f"/api/data-mining/jobs/{payload['job_id']}/summary")
        assert response.status_code == 200, response.text
        archive = client.get(response.json()["artifact"]["download_url"])
        with ZipFile(BytesIO(archive.content)) as zipped:
            for artifact in payload["artifacts"]:
                assert zipped.read("summary/" + artifact["name"]) == client.get(artifact["download_url"]).content
