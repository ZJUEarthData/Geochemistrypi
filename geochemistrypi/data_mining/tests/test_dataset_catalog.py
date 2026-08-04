import hashlib
from pathlib import Path

from geochemistrypi.data_mining import datasets as dataset_module
from geochemistrypi.data_mining.datasets import dataset_catalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_declared_builtin_is_listed_without_modification() -> None:
    root = Path(dataset_module.BUILT_IN_DATASET_PATH)
    before = {path.name: (_sha256(path), path.stat().st_mtime_ns) for path in root.iterdir()}

    result = dataset_catalog("builtin")

    after = {path.name: (_sha256(path), path.stat().st_mtime_ns) for path in root.iterdir()}
    assert before == after
    assert len(result["datasets"]) == 8
    assert {item["dataset_id"] for item in result["datasets"]} == {
        "builtin:application_classification",
        "builtin:application_regression",
        "builtin:anomaly_detection",
        "builtin:classification",
        "builtin:clustering",
        "builtin:decomposition",
        "builtin:regression",
        "builtin:time_series",
    }
    assert all(item["sha256"] == _sha256(Path(item["path"])) for item in result["datasets"])
    assert all("branch.world_map" not in item["analysis_blockers"] for item in result["datasets"])


def test_desktop_discovery_is_read_only_non_recursive_and_format_limited(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    desktop_root = home / "Desktop" / "geopi_input"
    desktop_root.mkdir(parents=True)
    (desktop_root / "valid.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    (desktop_root / "unsupported.xls").write_bytes(b"legacy")
    nested = desktop_root / "nested"
    nested.mkdir()
    (nested / "hidden.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    before = sorted(str(path.relative_to(home)) for path in home.rglob("*"))

    result = dataset_catalog("desktop")

    after = sorted(str(path.relative_to(home)) for path in home.rglob("*"))
    assert before == after
    assert [item["file_name"] for item in result["datasets"]] == ["valid.csv"]


def test_desktop_discovery_does_not_create_a_missing_directory(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "missing-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    result = dataset_catalog("desktop")

    assert result["datasets"] == []
    assert not home.exists()
    assert "did not create it" in result["warnings"][0]


def test_desktop_discovery_warns_when_a_file_changes_during_read(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    desktop_root = home / "Desktop" / "geopi_input"
    desktop_root.mkdir(parents=True)
    (desktop_root / "changing.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        dataset_module,
        "_entry",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("file changed")),
    )

    result = dataset_catalog("desktop")

    assert result["datasets"] == []
    assert "changed while being read" in result["warnings"][0]
