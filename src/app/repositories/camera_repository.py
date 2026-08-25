from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraUpdate


class CameraRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, camera_id: int) -> Camera | None:
        query = select(Camera).where(
            Camera.id == camera_id, Camera.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_name(
            self, name: str, exclude_camera_id: int | None = None
    ) -> Camera | None:
        query = select(Camera).where(
            Camera.name == name, Camera.deleted_at.is_(None)
        )
        if exclude_camera_id is not None:
            query = query.where(Camera.id != exclude_camera_id)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def list_all(
            self, skip: int = 0, limit: int = 50, active_only: bool = False
    ) -> list[Camera]:
        query = select(Camera).where(Camera.deleted_at.is_(None))
        if active_only:
            query = query.where(Camera.is_active.is_(True))
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create(self, camera_in: CameraCreate) -> Camera:
        db_camera = Camera(
            name=camera_in.name,
            source=camera_in.source,
            description=camera_in.description,
            location=camera_in.location,
            floor=camera_in.floor,
            unit=camera_in.unit,
            room=camera_in.room,
            is_active=camera_in.is_active,
        )
        self.session.add(db_camera)
        await self.session.commit()
        await self.session.refresh(db_camera)
        return db_camera

    async def update(self, camera: Camera, camera_in: CameraUpdate) -> Camera:
        update_data = camera_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(camera, field, value)
        await self.session.commit()
        await self.session.refresh(camera)
        return camera

    async def soft_delete(self, camera: Camera) -> None:
        camera.deleted_at = datetime.now(timezone.utc)
        camera.is_active = False
        await self.session.commit()
