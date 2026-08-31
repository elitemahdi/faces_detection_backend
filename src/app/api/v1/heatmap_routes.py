import csv
import io
import random
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Query, Response, status
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.repositories.detection_event_repository import DetectionEventRepository
from app.repositories.user_repository import UserRepository
from app.schemas.heatmap import (
    DetectionEventCreate,
    DetectionEventResponse,
    HeatmapColormap,
    HeatmapJsonResponse,
    HeatmapStatsResponse,
)
from app.services.heatmap_service import HeatmapService

router = APIRouter(prefix="/cameras/{camera_id}/heatmap", tags=["Camera Heatmap"])


def get_heatmap_service(db: AsyncSession = Depends(get_db)) -> HeatmapService:
    return HeatmapService(db)


@router.get(
    "",
    response_model=HeatmapJsonResponse,
    status_code=status.HTTP_200_OK,
    summary="Get heatmap density points, 2D matrix, and detailed intelligence report with custom filters",
)
async def get_camera_heatmap(
    camera_id: int,
    start_time: datetime | None = Query(None, description="Filter start date/time (ISO 8601)"),
    end_time: datetime | None = Query(None, description="Filter end date/time (ISO 8601)"),
    start_time_of_day: str | None = Query(None, description="Time of day filter start (e.g. 10:00)"),
    end_time_of_day: str | None = Query(None, description="Time of day filter end (e.g. 16:00)"),
    user_id: int | None = Query(None, description="Filter by registered user ID"),
    user_name: str | None = Query(None, description="Filter by user full name search"),
    is_recognized: bool | None = Query(None, description="Filter recognized users (True) or unknowns (False)"),
    zone_id: int | None = Query(None, description="Filter by specific security zone ID"),
    grid_size: int = Query(32, ge=8, le=128, description="Density matrix resolution (e.g. 32x32)"),
    service: HeatmapService = Depends(get_heatmap_service),
):
    return await service.get_heatmap_data(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        start_time_of_day=start_time_of_day,
        end_time_of_day=end_time_of_day,
        user_id=user_id,
        user_name=user_name,
        is_recognized=is_recognized,
        zone_id=zone_id,
        grid_size=grid_size,
    )


@router.get(
    "/image",
    summary="Render and download high-resolution JPEG heatmap image overlay",
)
async def get_camera_heatmap_image(
    camera_id: int,
    start_time: datetime | None = Query(None, description="Start date/time"),
    end_time: datetime | None = Query(None, description="End date/time"),
    start_time_of_day: str | None = Query(None, description="e.g. 10:00"),
    end_time_of_day: str | None = Query(None, description="e.g. 16:00"),
    user_id: int | None = Query(None, description="Filter by user ID"),
    user_name: str | None = Query(None, description="Filter by user name"),
    is_recognized: bool | None = Query(None, description="Filter recognition status"),
    zone_id: int | None = Query(None, description="Filter by zone ID"),
    colormap: HeatmapColormap = Query(HeatmapColormap.JET, description="Color palette gradient"),
    radius: int = Query(35, ge=10, le=100, description="Gaussian kernel blur radius"),
    alpha: float = Query(0.55, ge=0.1, le=1.0, description="Overlay opacity blending ratio"),
    service: HeatmapService = Depends(get_heatmap_service),
):
    jpeg_bytes = await service.generate_heatmap_image(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        start_time_of_day=start_time_of_day,
        end_time_of_day=end_time_of_day,
        user_id=user_id,
        user_name=user_name,
        is_recognized=is_recognized,
        zone_id=zone_id,
        colormap=colormap,
        radius=radius,
        alpha=alpha,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get(
    "/export",
    summary="Export heatmap detection logs as CSV report",
)
async def export_heatmap_csv(
    camera_id: int,
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
    start_time_of_day: str | None = Query(None),
    end_time_of_day: str | None = Query(None),
    user_id: int | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    repo = DetectionEventRepository(db)
    events = await repo.query_events(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        start_time_of_day=start_time_of_day,
        end_time_of_day=end_time_of_day,
        user_id=user_id,
        limit=50000,
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Event ID", "Camera ID", "Timestamp (UTC)", "User ID", "User Name", "Is Recognized", "Confidence", "Norm X", "Norm Y", "Zone ID"])

    for e in events:
        writer.writerow([
            e.id,
            e.camera_id,
            e.timestamp.isoformat() if e.timestamp else "",
            e.user_id or "",
            e.user_name or "Unknown Visitor",
            e.is_recognized,
            e.confidence,
            e.norm_x,
            e.norm_y,
            e.zone_id or "",
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="heatmap_report_camera_{camera_id}.csv"'},
    )


@router.post(
    "/generate-sample-data",
    status_code=status.HTTP_200_OK,
    summary="Generate realistic historical spatial detection events with variable daily hours for testing",
)
async def generate_sample_detection_data(
    camera_id: int,
    count: int = Query(250, ge=10, le=5000, description="Number of sample events to generate"),
    days_back: int = Query(7, ge=1, le=60, description="Spread events across past N days"),
    db: AsyncSession = Depends(get_db),
):
    user_repo = UserRepository(db)
    users = await user_repo.list_all(skip=0, limit=20)
    event_repo = DetectionEventRepository(db)

    now = datetime.now(timezone.utc)
    sample_events = []

    clusters = [
        (0.35, 0.45, 0.08),  # Entrance Lobby
        (0.65, 0.55, 0.10),  # Hallway
        (0.50, 0.30, 0.06),  # Front Desk / Cashier
    ]

    for _ in range(count):
        cx, cy, std = random.choice(clusters)
        nx = float(np.clip(random.gauss(cx, std), 0.05, 0.95))
        ny = float(np.clip(random.gauss(cy, std), 0.05, 0.95))

        # Weight time of day towards working hours (09:00 - 18:00)
        day_offset = random.randint(0, days_back - 1)
        if random.random() < 0.75:
            hour = random.randint(9, 17)
        else:
            hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        base_date = now.date() - timedelta(days=day_offset)
        ts = datetime(base_date.year, base_date.month, base_date.day, hour, minute, second, tzinfo=timezone.utc)

        if users and random.random() < 0.70:
            user = random.choice(users)
            sample_events.append({
                "camera_id": camera_id,
                "user_id": user.id,
                "user_name": user.full_name,
                "norm_x": round(nx, 4),
                "norm_y": round(ny, 4),
                "confidence": round(random.uniform(0.75, 0.98), 2),
                "is_recognized": True,
                "timestamp": ts,
            })
        else:
            sample_events.append({
                "camera_id": camera_id,
                "user_id": None,
                "user_name": None,
                "norm_x": round(nx, 4),
                "norm_y": round(ny, 4),
                "confidence": round(random.uniform(0.30, 0.85), 2),
                "is_recognized": False,
                "timestamp": ts,
            })

    created_count = await event_repo.bulk_log_events(sample_events)
    return {
        "message": f"Successfully generated {created_count} sample detection events with daytime distribution for camera #{camera_id}.",
        "camera_id": camera_id,
        "count": created_count,
    }
