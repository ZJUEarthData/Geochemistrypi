from database import Base
from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import relationship


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    # description = Column(Text)
    json_data = Column(Text)
    sequence = Column(Integer)
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="datasets")
    diagrams = relationship("Diagram", back_populates="dataset")


class Diagram(Base):
    __tablename__ = "diagrams"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    # description = Column(Text)
    image = Column(LargeBinary)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))

    dataset = relationship("Dataset", back_populates="diagrams")
