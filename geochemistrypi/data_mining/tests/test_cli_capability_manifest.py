import json
from pathlib import Path

import typer

from geochemistrypi.cli import app
from geochemistrypi.data_mining.constants import ANOMALYDETECTION_MODELS, CLASSIFICATION_MODELS, CLUSTERING_MODELS, DECOMPOSITION_MODELS, MODE_OPTION, REGRESSION_MODELS
from geochemistrypi.data_mining.datasets import _BUILT_IN_DATASETS


def _manifest() -> dict:
    repository = Path(__file__).resolve().parents[3]
    path = repository / "packages" / "geochemistrypi-mcp" / "src" / "geochemistrypi_mcp" / "contracts" / "cli_capability_manifest_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_tracks_every_active_typer_command_and_option() -> None:
    command = typer.main.get_command(app)
    actual = {
        "root": sorted(option for parameter in command.params for option in parameter.opts),
        **{name: sorted(option for parameter in child.params for option in parameter.opts) for name, child in command.commands.items()},
    }

    assert actual == _manifest()["declarations"]["commands"]


def test_manifest_tracks_cli_modes_models_and_bundled_datasets() -> None:
    declarations = _manifest()["declarations"]

    assert declarations["modes"] == MODE_OPTION
    assert declarations["models_by_task"] == {
        "regression": REGRESSION_MODELS,
        "classification": CLASSIFICATION_MODELS,
        "clustering": CLUSTERING_MODELS,
        "decomposition": DECOMPOSITION_MODELS,
        "anomaly_detection": ANOMALYDETECTION_MODELS,
    }
    assert declarations["bundled_dataset_ids"] == sorted(value[0] for value in _BUILT_IN_DATASETS.values())


def test_supported_manifest_entries_have_evidence_and_known_gaps_are_visible() -> None:
    manifest = _manifest()
    capabilities = manifest["capabilities"]
    identifiers = [item["id"] for item in capabilities]

    assert len(identifiers) == len(set(identifiers))
    assert sum(item["category"] == "model" for item in capabilities) == 36
    assert all(item["status"] == "verified" and item["evidence"] for item in capabilities if item["mcp_supported"])
    assert all(not item["mcp_supported"] for item in capabilities if item["status"] == "known_gap")
    repository = Path(__file__).resolve().parents[3]
    for item in capabilities:
        for reference in item["evidence"]:
            if reference.startswith(("parity-scenario:", "parity-group:")):
                continue
            assert (repository / reference.split("::", 1)[0]).is_file(), reference


def test_repository_has_no_non_strict_xfail_escape_hatch() -> None:
    repository = Path(__file__).resolve().parents[3]
    test_sources = [
        *repository.glob("geochemistrypi/**/test_*.py"),
        *repository.glob("tests/**/test_*.py"),
    ]
    offenders = [str(path.relative_to(repository)) for path in test_sources if "pytest.mark.xfail" in path.read_text(encoding="utf-8") and "strict=True" not in path.read_text(encoding="utf-8")]
    assert offenders == []
