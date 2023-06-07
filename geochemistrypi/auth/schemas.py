from typing import Union

from pydantic import BaseModel


class UserBase(BaseModel):
    username: str = None
    email: str = None
    is_active: Union[bool, None] = True


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    hashed_password: str

    class Config:
        orm_mode = True
