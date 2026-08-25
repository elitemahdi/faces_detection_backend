from pathlib import Path
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.video_service import VideoProcessingService

router = APIRouter(prefix="/videos", tags=["Video Processing"])


def cleanup_file(path: Path):
    """Deletes temporary rendered video after delivery."""
    path.unlink(missing_ok=True)


@router.post(
    "/annotate",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload video and download the annotated MP4 file",
)
async def annotate_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="MP4/AVI video file"),
    sample_rate: int = Query(
        1,
        ge=1,
        le=10,
        description="Run detection every Nth frame (1 = every frame)",
    ),
    db: AsyncSession = Depends(get_db),
):
    service = VideoProcessingService(db)
    output_path = await service.annotate_video_file(
        video_file=video, sample_rate=sample_rate
    )

    # Optional: uncomment to delete rendered file after download finishes
    # background_tasks.add_task(cleanup_file, output_path)

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"annotated_{video.filename}",
    )