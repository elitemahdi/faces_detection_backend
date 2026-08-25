import cv2
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.camera_repository import CameraRepository
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate


class CameraService:

    def __init__(self, session: AsyncSession):
        self.repository = CameraRepository(session)

    @staticmethod
    def _test_camera_connection(source: str) -> bool:
        """Attempts to open the video source (webcam index or stream URL) to verify access."""
        device_src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(device_src)
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret

    async def register_camera(
            self, camera_in: CameraCreate, verify_connection: bool = False
    ) -> CameraResponse:
        existing = await self.repository.get_by_name(camera_in.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Camera with name '{camera_in.name}' already exists.",
            )

        if verify_connection:
            is_connected = self._test_camera_connection(camera_in.source)
            if not is_connected:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not connect to camera source '{camera_in.source}'. Check URL/device index.",
                )

        db_camera = await self.repository.create(camera_in)
        return CameraResponse.model_validate(db_camera)

    async def get_camera(self, camera_id: int) -> CameraResponse:
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )
        return CameraResponse.model_validate(camera)

    async def list_cameras(
            self, skip: int = 0, limit: int = 50, active_only: bool = False
    ) -> list[CameraResponse]:
        cameras = await self.repository.list_all(
            skip=skip, limit=limit, active_only=active_only
        )
        return [CameraResponse.model_validate(cam) for cam in cameras]

    async def update_camera(
            self, camera_id: int, camera_in: CameraUpdate
    ) -> CameraResponse:
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )

        if camera_in.name:
            conflict = await self.repository.get_by_name(
                camera_in.name, exclude_camera_id=camera_id
            )
            if conflict:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Camera with name '{camera_in.name}' already exists.",
                )

        updated_camera = await self.repository.update(camera, camera_in)
        return CameraResponse.model_validate(updated_camera)

    async def delete_camera(self, camera_id: int) -> None:
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )
        await self.repository.soft_delete(camera)

    async def test_camera(self, camera_id: int) -> dict:
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )
        connected = self._test_camera_connection(camera.source)
        return {
            "camera_id": camera.id,
            "name": camera.name,
            "source": camera.source,
            "connected": connected,
        }
