import os
from functools import lru_cache
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DATABASE_URL_ENV = "SQLALCHEMY_DATABASE_URL"

# Create a base class for the models without initializing a database connection.
Base = declarative_base()


def get_database_url() -> str:
    """Return the explicitly configured database URL.

    Importing API routes must not require database configuration. The URL is
    therefore resolved only when database access is requested.
    """
    database_url = os.getenv(DATABASE_URL_ENV)
    if not database_url:
        raise RuntimeError(f"{DATABASE_URL_ENV} is not configured. " "Set it before starting the API or opening a database session.")
    return database_url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache the configured SQLAlchemy engine on first use."""
    database_url = get_database_url()
    connect_args = {}
    if make_url(database_url).get_backend_name() == "sqlite":
        connect_args["check_same_thread"] = False
    return create_engine(database_url, connect_args=connect_args)


def create_session() -> Session:
    """Create a database session bound to the lazily initialized engine."""
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return session_factory()


def get_db() -> Generator[Session, None, None]:
    """Get a database session."""
    db = create_session()
    try:
        yield db
    finally:
        db.close()
