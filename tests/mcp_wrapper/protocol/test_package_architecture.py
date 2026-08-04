import ast
from pathlib import Path

import geochemistrypi_mcp


ROOT_MODULES = {
    "__init__.py",
    "__main__.py",
    "doctor.py",
    "release.py",
    "server.py",
    "setup.py",
}
LAYERS = {
    "api",
    "config",
    "contracts",
    "data",
    "lifecycle",
    "planning",
    "runtime",
    "tracking",
}
ALLOWED_LAYER_IMPORTS = {
    "api": {"api", "config", "contracts", "data", "planning", "runtime", "tracking"},
    "config": {"config"},
    "contracts": {"contracts"},
    "data": {"api", "config", "data"},
    "lifecycle": {"config", "lifecycle", "tracking"},
    "planning": {"api", "config", "contracts", "data", "planning"},
    "runtime": {"api", "config", "contracts", "data", "planning", "runtime", "tracking"},
    "tracking": {"api", "config", "tracking"},
}


def _package_directory() -> Path:
    return Path(geochemistrypi_mcp.__file__).resolve().parent


def test_source_package_keeps_only_stable_entry_points_at_the_root() -> None:
    package_directory = _package_directory()

    assert {path.name for path in package_directory.glob("*.py")} == ROOT_MODULES
    assert {
        path.name
        for path in package_directory.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    } == LAYERS


def test_internal_dependencies_follow_the_layered_architecture() -> None:
    package_directory = _package_directory()
    violations = []

    for layer in sorted(LAYERS):
        for path in (package_directory / layer).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level != 2 or not node.module:
                    continue
                target_layer = node.module.split(".", 1)[0]
                if target_layer in LAYERS and target_layer not in ALLOWED_LAYER_IMPORTS[layer]:
                    violations.append(f"{path.relative_to(package_directory)} -> {node.module}")

    assert violations == []
