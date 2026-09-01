from datetime import datetime, timezone
from sqlalchemy import func, insert, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.detection_event import DetectionEvent
from app.schemas.heatmap import DetectionEventCreate


class DetectionEventRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_event(self, event_in: DetectionEventCreate) -> DetectionEvent:
        db_event = DetectionEvent(
            camera_id=event_in.camera_id,
            user_id=event_in.user_id,
            user_name=event_in.user_name,
            norm_x=event_in.norm_x,
            norm_y=event_in.norm_y,
            confidence=event_in.confidence,
            is_recognized=event_in.is_recognized,
            zone_id=event_in.zone_id,
            timestamp=event_in.timestamp or datetime.now(timezone.utc),
        )
        self.session.add(db_event)
        await self.session.commit()
        await self.session.refresh(db_event)
        return db_event

    async def bulk_log_events(self, events: list[dict]) -> int:
        """High-performance direct multi-value bulk INSERT bypassing ORM unit-of-work overhead."""
        if not events:
            return 0

        now = datetime.now(timezone.utc)
        insert_values = [
            {
                "camera_id": e["camera_id"],
                "user_id": e.get("user_id"),
                "user_name": e.get("user_name"),
                "norm_x": e["norm_x"],
                "norm_y": e["norm_y"],
                "confidence": e.get("confidence", 1.0),
                "is_recognized": e.get("is_recognized", False),
                "zone_id": e.get("zone_id"),
                "timestamp": e.get("timestamp") or now,
            }
            for e in events
        ]

        stmt = insert(DetectionEvent).values(insert_values)
        await self.session.execute(stmt)
        await self.session.commit()
        return len(insert_values)

    async def query_events(
        self,
        camera_id: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        start_time_of_day: str | None = None,  # e.g. "10:00"
        end_time_of_day: str | None = None,    # e.g. "16:00"
        user_id: int | None = None,
        user_name: str | None = None,
        is_recognized: bool | None = None,
        zone_id: int | None = None,
        limit: int = 10000,
    ) -> list[DetectionEvent]:
        query = select(DetectionEvent).where(DetectionEvent.camera_id == camera_id)

        if start_time:
            query = query.where(DetectionEvent.timestamp >= start_time)
        if end_time:
            query = query.where(DetectionEvent.timestamp <= end_time)
        if user_id is not None:
            query = query.where(DetectionEvent.user_id == user_id)
        if user_name:
            query = query.where(DetectionEvent.user_name.ilike(f"%{user_name}%"))
        if is_recognized is not None:
            query = query.where(DetectionEvent.is_recognized == is_recognized)
        if zone_id is not None:
            query = query.where(DetectionEvent.zone_id == zone_id)

        # Parse flexible Time-of-Day boundaries into minutes of the day (0 to 1439 in UTC)
        start_min: int | None = None
        end_min: int | None = None
        if start_time_of_day:
            try:
                sh, sm = map(int, start_time_of_day.split(":"))
                start_min = sh * 60 + sm
            except Exception:
                pass
        if end_time_of_day:
            try:
                eh, em = map(int, end_time_of_day.split(":"))
                end_min = eh * 60 + em
            except Exception:
                pass

        if start_min is not None or end_min is not None:
            utc_ts = func.timezone("UTC", DetectionEvent.timestamp)
            minute_expr = (
                func.extract("hour", utc_ts) * 60
                + func.extract("minute", utc_ts)
            )
            if start_min is not None and end_min is not None:
                if start_min <= end_min:
                    query = query.where(
                        minute_expr >= start_min, minute_expr <= end_min
                    )
                else:
                    # Overnight window (e.g. 22:00 (1320) to 06:00 (360))
                    query = query.where(
                        or_(minute_expr >= start_min, minute_expr <= end_min)
                    )
            elif start_min is not None:
                query = query.where(minute_expr >= start_min)
            elif end_min is not None:
                query = query.where(minute_expr <= end_min)

        query = query.order_by(DetectionEvent.timestamp.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
