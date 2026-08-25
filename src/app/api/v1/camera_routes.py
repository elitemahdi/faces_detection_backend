from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate
from app.services.camera_service import CameraService

router = APIRouter(prefix="/cameras", tags=["Cameras"])


def get_camera_service(db: AsyncSession = Depends(get_db)) -> CameraService:
    return CameraService(db)


@router.post(
    "/",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new camera",
)
async def create_camera(
        camera_in: CameraCreate,
        verify_connection: bool = Query(
            False, description="Attempt a test frame capture upon registration"
        ),
        service: CameraService = Depends(get_camera_service),
):
    return await service.register_camera(
        camera_in=camera_in, verify_connection=verify_connection
    )


@router.get(
    "/",
    response_model=list[CameraResponse],
    status_code=status.HTTP_200_OK,
    summary="List all registered cameras",
)
async def list_cameras(
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
        active_only: bool = Query(False),
        service: CameraService = Depends(get_camera_service),
):
    return await service.list_cameras(
        skip=skip, limit=limit, active_only=active_only
    )


@router.get(
    "/{camera_id}",
    response_model=CameraResponse,
    status_code=status.HTTP_200_OK,
    summary="Get camera details by ID",
)
async def get_camera(
        camera_id: int,
        service: CameraService = Depends(get_camera_service),
):
    return await service.get_camera(camera_id=camera_id)


@router.put(
    "/{camera_id}",
    response_model=CameraResponse,
    status_code=status.HTTP_200_OK,
    summary="Update camera configuration",
)
async def update_camera(
        camera_id: int,
        camera_in: CameraUpdate,
        service: CameraService = Depends(get_camera_service),
):
    return await service.update_camera(
        camera_id=camera_id, camera_in=camera_in
    )


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate/delete a camera",
)
async def delete_camera(
        camera_id: int,
        service: CameraService = Depends(get_camera_service),
):
    await service.delete_camera(camera_id=camera_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{camera_id}/test",
    status_code=status.HTTP_200_OK,
    summary="Test live feed connectivity for a camera",
)
async def test_camera_connection(
        camera_id: int,
        service: CameraService = Depends(get_camera_service),
):
    return await service.test_camera(camera_id=camera_id)
