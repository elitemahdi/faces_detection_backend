from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class VideoProcessResponse(BaseModel):
    video_id: str
    original_filename: str
    annotated_filename: str
    download_url: str
    total_frames: int
    fps: float
    duration_seconds: float
    total_detections: int
    recognized_users: list[str] = Field(default_factory=list)
    file_size_bytes: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VideoSummaryItem(BaseModel):
    video_id: str
    filename: str
    file_size_bytes: int
    download_url: str
    created_at: datetime
