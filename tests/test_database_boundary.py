import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from geochemistrypi.database import DatabaseConfigurationError, get_db


def test_api_routers_import_without_database_configuration(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["SQLALCHEMY_DATABASE_URL"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import geochemistrypi.auth.router; import geochemistrypi.data_mining.router",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_database_access_without_configuration_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQLALCHEMY_DATABASE_URL", "")
    database_dependency = get_db()

    with pytest.raises(DatabaseConfigurationError, match="SQLALCHEMY_DATABASE_URL is required"):
        next(database_dependency)


def test_database_access_uses_explicit_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_path = tmp_path / "database-boundary.sqlite3"
    monkeypatch.setenv("SQLALCHEMY_DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    database_dependency = get_db()
    session = next(database_dependency)

    try:
        assert session.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        database_dependency.close()
