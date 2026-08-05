import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Load the .env file
load_dotenv()

# Create a base class for the models
Base = declarative_base()


class DatabaseConfigurationError(RuntimeError):
    """Raised when database access is requested without a configured URL."""


def get_database_url() -> str:
    """Return the configured database URL at the database access boundary."""
    database_url = os.getenv("SQLALCHEMY_DATABASE_URL", "").strip()
    if not database_url:
        raise DatabaseConfigurationError("SQLALCHEMY_DATABASE_URL is required before database access.")
    return database_url


@lru_cache(maxsize=None)
def _create_engine(database_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def get_engine() -> Engine:
    """Create or reuse the engine for the currently configured database URL."""
    return _create_engine(get_database_url())


def get_session() -> Session:
    """Create a database session after validating database configuration."""
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return session_factory()


def get_db():
    """Get a database session."""
    db = get_session()
    try:
        yield db
    finally:
        db.close()
