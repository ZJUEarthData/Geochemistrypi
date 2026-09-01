from database import Base
from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    upload_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    datasets = relationship("Dataset", back_populates="user")
