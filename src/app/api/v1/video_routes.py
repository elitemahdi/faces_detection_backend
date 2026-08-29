from pathlib import Path
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Query,
    UploadFile,
    Response,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.video import VideoProcessResponse
from app.services.video_service import VideoProcessingService

router = APIRouter(prefix="/videos", tags=["Video Processing"])


@router.post(
    "/process",
    response_model=VideoProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process video for face detection and return detailed recognition metrics",
)
async def process_video(
    video: UploadFile = File(..., description="MP4/AVI video file"),
    sample_rate: int = Query(
        1,
        ge=1,
        le=10,
        description="Run detection every Nth frame (1 = every frame)",
    ),
    match_threshold: float = Query(
        0.38,
        ge=0.1,
        le=1.0,
        description="Cosine similarity threshold for AdaFace face recognition match (typically 0.35 - 0.45)",
    ),
    db: AsyncSession = Depends(get_db),
):
    service = VideoProcessingService(db)
    result = await service.annotate_video_file(
        video_file=video,
        sample_rate=sample_rate,
        match_threshold=match_threshold,
    )
    return result["metadata"]


@router.get(
    "/",
    response_model=list[VideoProcessResponse],
    status_code=status.HTTP_200_OK,
    summary="List all processed video results",
)
async def list_processed_videos(
    db: AsyncSession = Depends(get_db),
):
    service = VideoProcessingService(db)
    return service.list_processed_videos()


@router.get(
    "/{video_id}/download",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Download the annotated MP4 video file",
)
async def download_annotated_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    service = VideoProcessingService(db)
    path = service.get_video_file_path(video_id)
    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


@router.get(
    "/{video_id}/stream",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Stream or preview the annotated video file",
)
async def stream_annotated_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    service = VideoProcessingService(db)
    path = service.get_video_file_path(video_id)
    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=path.name,
    )


@router.delete(
    "/{video_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a processed video result",
)
async def delete_processed_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    service = VideoProcessingService(db)
    service.delete_processed_video(video_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/annotate",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload video and directly download the annotated MP4 file",
)
async def annotate_video(
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
    result = await service.annotate_video_file(
        video_file=video, sample_rate=sample_rate
    )
    output_path = result["output_path"]

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"annotated_{video.filename}",
        headers={"Content-Disposition": f'attachment; filename="annotated_{video.filename}"'},
    )
