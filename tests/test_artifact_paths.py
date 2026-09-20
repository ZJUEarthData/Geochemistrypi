"""Downloads must never reinterpret a job ID as an arbitrary host path."""

from pathlib import Path
import stat
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from geochemistrypi.online.artifact_paths import job_directory, output_directory
from geochemistrypi.online.data_mining_service import DataMiningService
from geochemistrypi.online.service import OnlineService
from geochemistrypi.online.summary_service import build_summary


JOB_ID = "a1" * 16
OTHER_JOB_ID = "b2" * 16


@pytest.fixture(params=[DataMiningService, OnlineService], ids=["data-mining", "chemical-modeling"])
def service(request, tmp_path):
    return request.param(tmp_path / "runtime")


def create_output(service, job_id=JOB_ID):
    output = service.jobs_dir / job_id / "output"
    output.mkdir(parents=True)
    artifact = output / "result.txt"
    artifact.write_text("public artifact", encoding="utf-8")
    return output, artifact


def test_normal_flat_and_nested_artifact_download(service):
    output, artifact = create_output(service)
    nested = output / "figures" / "plot 1.svg"
    nested.parent.mkdir()
    nested.write_text("<svg/>", encoding="utf-8")
    assert service.resolve_artifact(JOB_ID, "result.txt") == artifact.resolve()
    assert service.resolve_artifact(JOB_ID, "figures/plot 1.svg") == nested.resolve()
    assert job_directory(service.jobs_dir, JOB_ID) == output.parent.resolve()
    assert output_directory(service.jobs_dir, JOB_ID) == output.resolve()


def test_absolute_job_directory_outside_jobs_root_is_rejected(service, tmp_path):
    outside = tmp_path / "outside"
    (outside / "output").mkdir(parents=True)
    (outside / "output" / "private.txt").write_text("not a job artifact", encoding="utf-8")
    # This exact call returned the outside file before the shared-path fix.
    with pytest.raises(FileNotFoundError):
        service.resolve_artifact(str(outside.resolve()), "private.txt")


@pytest.mark.parametrize("job_id", ["..", "../outside", "..\\outside", "C:\\outside", "C:outside", "/outside", "\\\\server\\share", "not-a-uuid", "a" * 31, "g" * 32])
def test_malformed_or_traversing_job_id_is_rejected(service, job_id):
    create_output(service)
    with pytest.raises(FileNotFoundError):
        service.resolve_artifact(job_id, "result.txt")


@pytest.mark.parametrize("file_path", ["../summary_context.joblib", "..\\summary_context.joblib", "figures/../../summary_context.joblib", "/etc/passwd", "C:\\outside\\private.txt", "C:private.txt", "result.txt:stream", "\\\\server\\share\\private.txt", "", ".", "figures/../result.txt", "result.txt\x00"])
def test_artifact_path_escape_and_windows_aliases_are_rejected(service, file_path):
    output, _ = create_output(service)
    (output.parent / "summary_context.joblib").write_bytes(b"private trusted snapshot")
    with pytest.raises(FileNotFoundError):
        service.resolve_artifact(JOB_ID, file_path)


def test_missing_jobs_files_and_private_snapshot_are_not_served(service):
    output, _ = create_output(service)
    snapshot = output.parent / "summary_context.joblib"
    snapshot.write_bytes(b"private trusted snapshot")
    for job_id, path in ((OTHER_JOB_ID, "result.txt"), (JOB_ID, "missing.txt"), (JOB_ID, "summary_context.joblib"), (JOB_ID, "figures")):
        with pytest.raises(FileNotFoundError):
            service.resolve_artifact(job_id, path)


def _link_or_skip(link: Path, target: Path, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation unavailable on this host: {exc}")


def test_job_directory_symlink_is_rejected_even_when_target_is_another_job(service):
    output, _ = create_output(service)
    alias = service.jobs_dir / OTHER_JOB_ID
    _link_or_skip(alias, output.parent, directory=True)
    with pytest.raises(FileNotFoundError):
        service.resolve_artifact(OTHER_JOB_ID, "result.txt")
    with pytest.raises(FileNotFoundError):
        build_summary(service.jobs_dir, OTHER_JOB_ID)


def test_output_directory_symlink_is_rejected(service, tmp_path):
    job = service.jobs_dir / JOB_ID
    job.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.txt").write_text("outside", encoding="utf-8")
    _link_or_skip(job / "output", outside, directory=True)
    with pytest.raises(FileNotFoundError):
        service.resolve_artifact(JOB_ID, "result.txt")


@pytest.mark.parametrize("directory_link", [False, True], ids=["file-symlink", "nested-dir-symlink"])
def test_artifact_component_symlink_is_rejected(service, tmp_path, directory_link):
    output, _ = create_output(service)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "private.txt"
    target.write_text("outside", encoding="utf-8")
    if directory_link:
        _link_or_skip(output / "linked", outside, directory=True)
        path = "linked/private.txt"
    else:
        _link_or_skip(output / "linked.txt", target)
        path = "linked.txt"
    with pytest.raises(FileNotFoundError):
        service.resolve_artifact(JOB_ID, path)


@pytest.mark.parametrize("component", ["job", "output", "artifact"])
def test_windows_reparse_point_flag_is_rejected_without_symlink_privileges(service, monkeypatch, component):
    output, artifact = create_output(service)
    guarded = {"job": output.parent, "output": output, "artifact": artifact}[component]
    original_lstat = Path.lstat
    reparse_flag = 0x400
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", reparse_flag, raising=False)

    def flagged_lstat(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == guarded:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=reparse_flag)
        return info

    monkeypatch.setattr(Path, "lstat", flagged_lstat)
    with pytest.raises(FileNotFoundError, match="Linked result paths"):
        service.resolve_artifact(JOB_ID, "result.txt")


def test_http_download_preserves_normal_files_and_rejects_absolute_job_ids(service, tmp_path):
    from fastapi.testclient import TestClient
    from geochemistrypi.online.app import create_app

    create_output(service)
    outside = tmp_path / "outside"
    (outside / "output").mkdir(parents=True)
    (outside / "output" / "private.txt").write_text("outside", encoding="utf-8")
    prefix = "/api/data-mining/jobs" if isinstance(service, DataMiningService) else "/api/jobs"
    with TestClient(create_app(service.runtime_dir)) as client:
        response = client.get(f"{prefix}/{JOB_ID}/files/result.txt")
        assert response.status_code == 200 and response.text == "public artifact"
        absolute_job_id = quote(str(outside.resolve()), safe="")
        assert client.get(f"{prefix}/{absolute_job_id}/files/private.txt").status_code == 404
        assert client.get(f"{prefix}/{JOB_ID}/files/..%5Csummary_context.joblib").status_code == 404
