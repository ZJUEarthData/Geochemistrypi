import ast
import re
from pathlib import Path


RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "geochemistrypi-runtime"
    / "src"
    / "geochemistrypi_runtime"
)

FORBIDDEN_IMPORT_ROOTS = {
    "data_mining",
    "fastapi",
    "geochemistrypi",
    "mcp",
    "pandas",
    "sklearn",
}


def test_runtime_does_not_import_engine_or_service_frameworks() -> None:
    violations = []
    for source_path in RUNTIME_SOURCE.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported_names = [node.module or ""]
            else:
                continue
            for imported_name in imported_names:
                root = imported_name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        f"{source_path.name}:{node.lineno} imports {imported_name}"
                    )

    assert violations == []


def test_runtime_has_only_approved_runtime_dependencies() -> None:
    pyproject_path = RUNTIME_SOURCE.parents[1] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")

    dependency_block = re.search(
        r"(?ms)^dependencies\s*=\s*\[(.*?)^\]",
        pyproject_text,
    )
    assert dependency_block is not None
    dependencies = set(re.findall(r'"([^"]+)"', dependency_block.group(1)))
    assert dependencies == {
        "filelock>=3.13,<4",
        "geochemistrypi-contracts==0.1.0",
    }
