from typing import Union

# from data_mining.schemas import Dataset
from pydantic import BaseModel


class UserBase(BaseModel):
    username: str = None
    email: str = None
    is_active: Union[bool, None] = True
    upload_count: int = 0


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    hashed_password: str
    # datasets: list = [Dataset]

    class Config:
        orm_mode = True
