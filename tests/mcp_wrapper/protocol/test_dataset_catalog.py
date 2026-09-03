import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from geochemistrypi_mcp.api.schemas import BuiltInDatasetReference, DesktopDatasetReference, ListDatasetsRequest
from geochemistrypi_mcp.config.constants import CLI_VERSION
from geochemistrypi_mcp.data.catalog import DatasetCatalog, DatasetCatalogError


def _entry(path: Path, **overrides) -> dict:
    value = {
        "dataset_id": "builtin:classification",
        "source": "builtin",
        "role": "training",
        "task": "classification",
        "file_name": path.name,
        "path": str(path.resolve()),
        "format": path.suffix.lstrip("."),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": 1,
        "column_count": 2,
        "analysis_blockers": [],
    }
    value.update(overrides)
    return value


def _catalog(entries, source="builtin", desktop_root=None) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "source_filter": source,
            "supported_formats": ["csv", "xlsx"],
            "desktop_root": desktop_root,
            "datasets": entries,
            "warnings": [],
        }
    )


def _service(monkeypatch, stdout: str, returncode: int = 0) -> DatasetCatalog:
    settings = SimpleNamespace(require_supported_cli=lambda: (Path("C:/fake/geochemistrypi.exe"), CLI_VERSION))
    monkeypatch.setattr(
        "geochemistrypi_mcp.data.catalog.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr="catalog error" if returncode else ""),
    )
    return DatasetCatalog(settings)


def test_catalog_lists_and_resolves_a_task_matched_builtin(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "classification.csv"
    path.write_text("id,value\n1,2\n", encoding="utf-8")
    service = _service(monkeypatch, _catalog([_entry(path)]))

    response = service.list(ListDatasetsRequest(source="builtin"))
    resolved = service.resolve(
        BuiltInDatasetReference(dataset_id="builtin:classification"),
        task="classification",
        role="training",
    )

    assert response.datasets[0].supported_for_analysis is True
    assert resolved.path == path.resolve()
    assert resolved.dataset_id == "builtin:classification"


def test_builtin_resolution_rejects_wrong_task_or_role(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "classification.csv"
    path.write_text("id,value\n1,2\n", encoding="utf-8")
    service = _service(monkeypatch, _catalog([_entry(path)]))
    reference = BuiltInDatasetReference(dataset_id="builtin:classification")

    with pytest.raises(DatasetCatalogError, match="not regression"):
        service.resolve(reference, task="regression", role="training")
    with pytest.raises(DatasetCatalogError, match="not application"):
        service.resolve(reference, task="classification", role="application")
    with pytest.raises(DatasetCatalogError, match="changed since it was selected"):
        service.resolve(
            BuiltInDatasetReference(
                dataset_id="builtin:classification",
                expected_sha256="0" * 64,
            ),
            task="classification",
            role="training",
        )


def test_builtin_resolution_surfaces_known_analysis_blockers(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "classification.csv"
    path.write_text("id,value\n1,2\n", encoding="utf-8")
    entry = _entry(path, analysis_blockers=["branch.world_map"])
    service = _service(monkeypatch, _catalog([entry]))

    with pytest.raises(DatasetCatalogError, match="branch.world_map"):
        service.resolve(
            BuiltInDatasetReference(dataset_id="builtin:classification"),
            task="classification",
            role="training",
        )


def test_desktop_catalog_cannot_escape_or_recurse_from_expected_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "Desktop" / "geopi_input"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_text("id,value\n1,2\n", encoding="utf-8")
    entry = _entry(
        outside,
        dataset_id="desktop:outside.csv",
        source="desktop",
        role="unspecified",
        task=None,
    )
    service = _service(
        monkeypatch,
        _catalog([entry], source="desktop", desktop_root=str(root.resolve())),
    )

    with pytest.raises(DatasetCatalogError, match="escapes Desktop/geopi_input"):
        service.list(ListDatasetsRequest(source="desktop"))


def test_desktop_reference_rejects_path_components() -> None:
    with pytest.raises(ValueError, match="plain file name"):
        DesktopDatasetReference(file_name="../outside.csv")


def test_catalog_surfaces_cli_failure_without_parsing_stdout(monkeypatch) -> None:
    service = _service(monkeypatch, "not json", returncode=2)

    with pytest.raises(DatasetCatalogError, match="catalog error"):
        service.list(ListDatasetsRequest(source="builtin"))


def test_catalog_surfaces_a_dataset_disappearing_during_verification(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "classification.csv"
    path.write_text("id,value\n1,2\n", encoding="utf-8")
    service = _service(monkeypatch, _catalog([_entry(path)]))
    monkeypatch.setattr(
        "geochemistrypi_mcp.data.catalog._sha256",
        lambda candidate: (_ for _ in ()).throw(OSError("file disappeared")),
    )

    with pytest.raises(DatasetCatalogError, match="became unavailable"):
        service.list(ListDatasetsRequest(source="builtin"))


def test_resolve_many_queries_each_source_once_and_hashes_only_selected_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    training = tmp_path / "classification.csv"
    training.write_text("id,value\n1,2\n", encoding="utf-8")
    application = tmp_path / "application.csv"
    application.write_text("id,value\n3,4\n", encoding="utf-8")
    desktop_root = tmp_path / "Desktop" / "geopi_input"
    desktop_root.mkdir(parents=True)
    desktop = desktop_root / "selected.csv"
    desktop.write_text("id,value\n5,6\n", encoding="utf-8")
    unrelated = desktop_root / "unrelated.csv"
    unrelated.write_text("id,value\n7,8\n", encoding="utf-8")

    builtin_entries = [
        _entry(training),
        _entry(
            application,
            dataset_id="builtin:application_classification",
            role="application",
        ),
    ]
    desktop_entry = _entry(
        desktop,
        dataset_id="desktop:selected.csv",
        source="desktop",
        role="unspecified",
        task=None,
    )
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        source = command[command.index("--source") + 1]
        stdout = (
            _catalog(builtin_entries, source="builtin")
            if source == "builtin"
            else _catalog(
                [desktop_entry],
                source="desktop",
                desktop_root=str(desktop_root.resolve()),
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    hashed: list[Path] = []
    original_sha256 = hashlib.sha256

    def recording_sha256(path: Path) -> str:
        hashed.append(path.resolve())
        return original_sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("geochemistrypi_mcp.data.catalog.subprocess.run", fake_run)
    monkeypatch.setattr("geochemistrypi_mcp.data.catalog._sha256", recording_sha256)
    settings = SimpleNamespace(require_supported_cli=lambda: (Path("C:/fake/geochemistrypi.exe"), CLI_VERSION))
    service = DatasetCatalog(settings)

    resolved = service.resolve_many(
        (
            (
                BuiltInDatasetReference(dataset_id="builtin:classification"),
                "classification",
                "training",
            ),
            (
                BuiltInDatasetReference(dataset_id="builtin:application_classification"),
                "classification",
                "application",
            ),
            (DesktopDatasetReference(file_name="selected.csv"), None, None),
        )
    )

    assert [item.path for item in resolved] == [
        training.resolve(),
        application.resolve(),
        desktop.resolve(),
    ]
    assert [command[command.index("--source") + 1] for command in calls] == [
        "builtin",
        "desktop",
    ]
    assert calls[0].count("--dataset-id") == 2
    assert calls[1].count("--file-name") == 1
    assert hashed == [training.resolve(), application.resolve(), desktop.resolve()]
    assert unrelated.resolve() not in hashed
