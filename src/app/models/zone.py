from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ZoneType(str, Enum):
    BANNED = "banned"          # Zero tolerance: No one is allowed to enter
    ADMIN = "admin"            # Restricted: Only users with admin role permitted
    REGISTERED = "registered"  # Restricted: Any registered user permitted; unknown triggers alert
    DETECTED = "detected"      # Public/Monitored: Any detected face is allowed; face presence verified


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    zone_type: Mapped[ZoneType] = mapped_column(
        SQLEnum(
            ZoneType,
            name="zone_type_enum",
            values_callable=lambda x: [e.value for e in x],
            native_enum=True,
        ),
        nullable=False,
        default=ZoneType.REGISTERED,
    )
    # Coordinates stored as normalized [[x1, y1], [x2, y2], ...] (floats in [0.0, 1.0])
    coordinates: Mapped[list[list[float]]] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=True,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    camera = relationship("Camera", back_populates="zones")
    detection_events = relationship(
        "DetectionEvent",
        back_populates="zone",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
