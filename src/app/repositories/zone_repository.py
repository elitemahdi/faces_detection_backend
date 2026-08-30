from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.zone import Zone
from app.schemas.zone import ZoneCreate, ZoneUpdate


class ZoneRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, zone_id: int) -> Zone | None:
        query = select(Zone).where(
            Zone.id == zone_id, Zone.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        return result.scalars().first()

    async def list_by_camera(
        self, camera_id: int, active_only: bool = False
    ) -> list[Zone]:
        query = select(Zone).where(
            Zone.camera_id == camera_id, Zone.deleted_at.is_(None)
        )
        if active_only:
            query = query.where(Zone.is_active.is_(True))
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def list_all(
        self, skip: int = 0, limit: int = 100, active_only: bool = False
    ) -> list[Zone]:
        query = select(Zone).where(Zone.deleted_at.is_(None))
        if active_only:
            query = query.where(Zone.is_active.is_(True))
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create(self, camera_id: int, zone_in: ZoneCreate) -> Zone:
        db_zone = Zone(
            camera_id=camera_id,
            name=zone_in.name,
            zone_type=zone_in.zone_type,
            coordinates=zone_in.coordinates,
            is_active=zone_in.is_active,
        )
        self.session.add(db_zone)
        await self.session.commit()
        await self.session.refresh(db_zone)
        return db_zone

    async def update(self, zone: Zone, zone_in: ZoneUpdate) -> Zone:
        update_data = zone_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(zone, field, value)
        await self.session.commit()
        await self.session.refresh(zone)
        return zone

    async def soft_delete(self, zone: Zone) -> None:
        zone.deleted_at = datetime.now(timezone.utc)
        zone.is_active = False
        await self.session.commit()
