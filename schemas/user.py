from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, validator


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str
    gender: str
    username: str
    dateOfBirth: date

    @validator('dateOfBirth')
    def validate_date_of_birth(cls, v):
        if isinstance(v, str):
            try:
                return datetime.strptime(v, '%d/%m/%Y').date()
            except ValueError:
                raise ValueError('dateOfBirth must be in DD/MM/YYYY format')
        return v


class UserLogin(UserBase):
    password: str


class UserProfile(UserBase):
    username: Optional[str] = None
    age: Optional[int] = None
    gender: str
    dateOfBirth: Optional[str] = None
    profile_image_url: Optional[str] = None

    class Config:
        from_attributes = True

    @validator('dateOfBirth', pre=True)
    def format_date_of_birth(cls, v):
        if v is None:
            return v
        if isinstance(v, date):
            return v.strftime('%d/%m/%Y')
        return v


class UserProfileUpdate(BaseModel):
    username: Optional[str] = None
    profile_image_url: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str
