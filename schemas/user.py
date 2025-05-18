from typing import Optional
from pydantic import BaseModel


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str
    age: int
    gender: str
    username: str
    city: str


class UserLogin(UserBase):
    password: str


class UserProfile(UserBase):
    username: Optional[str] = None
    age: Optional[int] = None
    gender: str
    city: str


class Token(BaseModel):
    access_token: str
    token_type: str
