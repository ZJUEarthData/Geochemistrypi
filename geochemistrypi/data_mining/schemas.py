from pydantic import BaseModel


class Dataset(BaseModel):
    id: int
    name: str = None
    # description: str = None
    json_data: str = None
    sequence: int = None
    user_id: int = None

    class Config:
        orm_mode = True


class Diagram(BaseModel):
    id: int
    name: str = None
    # description: str = None
    image: bytes = None
    dataset_id: int = None

    class Config:
        orm_mode = True
