import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load the .env file
load_dotenv()

# Get the database URL from the environment
database_url = os.getenv("SQLALCHEMY_DATABASE_URL")

# Create the database engine
engine = create_engine(database_url, connect_args={"check_same_thread": False})

# Create a sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for the models
Base = declarative_base()


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
