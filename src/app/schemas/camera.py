from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CameraBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, examples=["Main Entrance Camera"])
    source: str = Field(..., min_length=1, max_length=255, examples=["0", "rtsp://192.168.1.100:554/stream1"])
    description: str | None = Field(None, max_length=500, examples=["Primary entry surveillance camera"])
    location: str | None = Field(None, max_length=150, examples=["Building A - North Wing"])
    floor: str | None = Field(None, max_length=50, examples=["2nd Floor"])
    unit: str | None = Field(None, max_length=50, examples=["Unit 204"])
    room: str | None = Field(None, max_length=50, examples=["Server Room"])
    is_active: bool = Field(default=False)


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=100)
    source: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=500)
    location: str | None = Field(None, max_length=150)
    floor: str | None = Field(None, max_length=50)
    unit: str | None = Field(None, max_length=50)
    room: str | None = Field(None, max_length=50)
    is_active: bool | None = None


class CameraResponse(CameraBase):
    id: int
    created_at: datetime
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
