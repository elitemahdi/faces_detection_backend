from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    user_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    
    # Normalized coordinates in [0.0, 1.0] representing face center
    norm_x: Mapped[float] = mapped_column(Float, nullable=False)
    norm_y: Mapped[float] = mapped_column(Float, nullable=False)
    
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_recognized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    zone_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("zones.id", ondelete="SET NULL"), nullable=True, index=True
    )
    
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    camera = relationship("Camera", back_populates="detection_events")
    user = relationship("User", backref="detection_events")
    zone = relationship("Zone", back_populates="detection_events")
