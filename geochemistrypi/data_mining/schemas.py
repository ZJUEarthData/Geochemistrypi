from typing import Optional
from pydantic import BaseModel

try:
    from pydantic import ConfigDict  # v2 才有
    _P2 = True
except Exception:
    _P2 = False


class Dataset(BaseModel):
    id: int
    name: Optional[str] = None
    json_data: Optional[str] = None
    sequence: Optional[int] = None
    user_id: Optional[int] = None

    if _P2:
        model_config = ConfigDict(from_attributes=True)
    else:
        class Config:
            orm_mode = True


class Diagram(BaseModel):
    id: int
    name: Optional[str] = None
    image: Optional[bytes] = None
    dataset_id: Optional[int] = None

    if _P2:
        model_config = ConfigDict(from_attributes=True)
    else:
        class Config:
            orm_mode = True


class BasicDatasetInfo(BaseModel):
    id: int
    name: str
    sequence: int
