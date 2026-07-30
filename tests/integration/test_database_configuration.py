import importlib

import pytest
from sqlalchemy import inspect


def _load_database_module():
    return importlib.import_module("geochemistrypi.database")


def test_database_module_import_does_not_require_database_url(monkeypatch) -> None:
    monkeypatch.delenv("SQLALCHEMY_DATABASE_URL", raising=False)

    database = importlib.reload(_load_database_module())

    assert database.Base is not None


def test_api_router_import_does_not_initialize_database(monkeypatch) -> None:
    monkeypatch.delenv("SQLALCHEMY_DATABASE_URL", raising=False)
    database = _load_database_module()
    database.get_engine.cache_clear()

    router = importlib.import_module("geochemistrypi.data_mining.router")

    assert router.router.prefix == "/data-mining"


def test_database_access_requires_explicit_configuration(monkeypatch) -> None:
    monkeypatch.delenv("SQLALCHEMY_DATABASE_URL", raising=False)
    database = _load_database_module()
    database.get_engine.cache_clear()

    with pytest.raises(RuntimeError, match="SQLALCHEMY_DATABASE_URL is not configured"):
        database.get_engine()


def test_configured_sqlite_database_creates_session(monkeypatch) -> None:
    monkeypatch.setenv("SQLALCHEMY_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    database = _load_database_module()
    database.get_engine.cache_clear()

    engine = database.get_engine()
    session = database.create_session()
    try:
        assert engine.url.get_backend_name() == "sqlite"
        assert session.bind is engine
    finally:
        session.close()
        engine.dispose()
        database.get_engine.cache_clear()


def test_api_startup_initializes_configured_database(monkeypatch) -> None:
    monkeypatch.setenv("SQLALCHEMY_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    database = _load_database_module()
    database.get_engine.cache_clear()
    api = importlib.import_module("geochemistrypi.start_dash_pipeline")

    engine = database.get_engine()
    try:
        api.create_database_tables()
        assert {"users", "datasets", "diagrams"} <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
        database.get_engine.cache_clear()
