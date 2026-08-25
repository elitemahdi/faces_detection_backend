from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    MANAGER = "manager"


class UserBase(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: str = Field(..., min_length=7, max_length=20)
    role: UserRole = Field(default=UserRole.USER)


class UserCreate(UserBase):
    pass


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=100)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, min_length=7, max_length=20)
    role: UserRole | None = None
    photo_path: str | None = None


class UserResponse(UserBase):
    id: int
    photo_path: str
    created_at: datetime
    updated_at: datetime | None = None
    created_by: int | None = None

    model_config = ConfigDict(from_attributes=True)