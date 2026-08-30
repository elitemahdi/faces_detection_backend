from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.zone import ZoneCreate, ZoneResponse, ZoneUpdate
from app.services.zone_service import ZoneService

router = APIRouter(tags=["Zones"])


def get_zone_service(db: AsyncSession = Depends(get_db)) -> ZoneService:
    return ZoneService(db)


@router.post(
    "/cameras/{camera_id}/zones",
    response_model=ZoneResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new zone bound to a camera",
)
async def create_zone(
    camera_id: int,
    zone_in: ZoneCreate,
    service: ZoneService = Depends(get_zone_service),
):
    return await service.create_zone(camera_id=camera_id, zone_in=zone_in)


@router.get(
    "/cameras/{camera_id}/zones",
    response_model=list[ZoneResponse],
    status_code=status.HTTP_200_OK,
    summary="List all zones for a camera",
)
async def list_zones_for_camera(
    camera_id: int,
    active_only: bool = Query(False, description="Filter only active zones"),
    service: ZoneService = Depends(get_zone_service),
):
    return await service.list_zones_by_camera(
        camera_id=camera_id, active_only=active_only
    )


@router.get(
    "/zones/{zone_id}",
    response_model=ZoneResponse,
    status_code=status.HTTP_200_OK,
    summary="Get zone details by ID",
)
async def get_zone(
    zone_id: int,
    service: ZoneService = Depends(get_zone_service),
):
    return await service.get_zone(zone_id=zone_id)


@router.put(
    "/zones/{zone_id}",
    response_model=ZoneResponse,
    status_code=status.HTTP_200_OK,
    summary="Update zone properties or polygon coordinates",
)
async def update_zone(
    zone_id: int,
    zone_in: ZoneUpdate,
    service: ZoneService = Depends(get_zone_service),
):
    return await service.update_zone(zone_id=zone_id, zone_in=zone_in)


@router.delete(
    "/zones/{zone_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete/deactivate a zone",
)
async def delete_zone(
    zone_id: int,
    service: ZoneService = Depends(get_zone_service),
):
    await service.delete_zone(zone_id=zone_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
