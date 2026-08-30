from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ZoneType(str, Enum):
    BANNED = "banned"          # Zero tolerance: No one is allowed to enter
    ADMIN = "admin"            # Restricted: Only users with admin role permitted
    REGISTERED = "registered"  # Restricted: Any registered user permitted; unknown triggers alert
    DETECTED = "detected"      # Public/Monitored: Any detected face is allowed; face presence verified


class ZoneBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Descriptive zone name (e.g., Vault, Lobby)")
    zone_type: ZoneType = Field(default=ZoneType.REGISTERED, description="Zone access policy type")
    coordinates: list[list[float]] = Field(
        ...,
        description="List of normalized polygon vertices [[x1, y1], [x2, y2], ...] (0.0 to 1.0)",
    )
    is_active: bool = Field(default=True, description="Whether zone rule evaluation is active")

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, coords: list[list[float]]) -> list[list[float]]:
        if len(coords) < 3:
            raise ValueError("A polygon zone requires at least 3 vertices.")
        for idx, pt in enumerate(coords):
            if len(pt) != 2:
                raise ValueError(f"Vertex at index {idx} must be a 2D point [x, y].")
            x, y = pt[0], pt[1]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"Vertex coordinates at index {idx} ({x}, {y}) must be normalized between 0.0 and 1.0.")
        return coords


class ZoneCreate(ZoneBase):
    camera_id: int | None = Field(default=None, description="ID of camera this zone is bound to")


class ZoneUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    zone_type: ZoneType | None = None
    coordinates: list[list[float]] | None = None
    is_active: bool | None = None

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, coords: list[list[float]] | None) -> list[list[float]] | None:
        if coords is None:
            return coords
        if len(coords) < 3:
            raise ValueError("A polygon zone requires at least 3 vertices.")
        for idx, pt in enumerate(coords):
            if len(pt) != 2:
                raise ValueError(f"Vertex at index {idx} must be a 2D point [x, y].")
            x, y = pt[0], pt[1]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"Vertex coordinates at index {idx} ({x}, {y}) must be normalized between 0.0 and 1.0.")
        return coords


class ZoneResponse(ZoneBase):
    id: int
    camera_id: int
    created_at: datetime
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
