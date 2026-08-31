from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class HeatmapColormap(str, Enum):
    JET = "JET"
    HOT = "HOT"
    INFERNO = "INFERNO"
    VIRIDIS = "VIRIDIS"
    TURBO = "TURBO"
    OCEAN = "OCEAN"


class DetectionEventCreate(BaseModel):
    camera_id: int
    user_id: int | None = None
    user_name: str | None = None
    norm_x: float = Field(..., ge=0.0, le=1.0)
    norm_y: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_recognized: bool = False
    zone_id: int | None = None
    timestamp: datetime | None = None


class DetectionEventResponse(BaseModel):
    id: int
    camera_id: int
    user_id: int | None = None
    user_name: str | None = None
    norm_x: float
    norm_y: float
    confidence: float
    is_recognized: bool
    zone_id: int | None = None
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class HeatmapPoint(BaseModel):
    x: float
    y: float
    weight: float = 1.0
    user_id: int | None = None
    user_name: str | None = None
    timestamp: datetime


class UserRankingItem(BaseModel):
    user_id: int | None
    user_name: str
    count: int
    percentage: float


class ZoneRankingItem(BaseModel):
    zone_id: int | None
    zone_name: str
    zone_type: str
    count: int
    percentage: float


class HeatmapStatsResponse(BaseModel):
    camera_id: int
    total_detections: int
    recognized_count: int
    unknown_count: int
    recognition_rate: float
    unique_users_count: int
    top_user_name: str | None = None
    peak_hour: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    start_time_of_day: str | None = None
    end_time_of_day: str | None = None
    hourly_distribution: dict[int, int] = Field(default_factory=dict)
    top_users_ranking: list[UserRankingItem] = Field(default_factory=list)
    zone_traffic_ranking: list[ZoneRankingItem] = Field(default_factory=list)


class HeatmapJsonResponse(BaseModel):
    camera_id: int
    total_points: int
    stats: HeatmapStatsResponse
    points: list[HeatmapPoint]
    grid_matrix: list[list[float]] | None = None
