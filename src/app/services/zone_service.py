from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.zone import Zone, ZoneType
from app.repositories.camera_repository import CameraRepository
from app.repositories.zone_repository import ZoneRepository
from app.schemas.zone import ZoneCreate, ZoneResponse, ZoneUpdate


@dataclass
class ZoneRuleCheckResult:
    is_violation: bool
    status_level: str          # "OK", "WARNING", "BREACH"
    message: str               # Human-readable status / breach label
    color_bgr: tuple[int, int, int]  # BGR color for bounding box / tag
    zone_name: str
    zone_type: str


class ZoneEngine:
    """Core spatial geometry and rule evaluation engine for security zones."""

    # Default BGR theme colors per zone type
    ZONE_COLORS = {
        ZoneType.BANNED: (0, 0, 220),         # Crimson Red
        ZoneType.ADMIN: (220, 100, 0),        # Deep Blue / Navy
        ZoneType.REGISTERED: (0, 200, 240),   # Amber / Gold
        ZoneType.DETECTED: (220, 220, 0),     # Cyan
    }

    # Violation / Alert BGR Colors
    COLOR_VIOLATION = (0, 0, 255)            # Bright Red
    COLOR_WARNING = (0, 140, 255)             # Bright Orange
    COLOR_AUTHORIZED = (0, 220, 0)            # Emerald Green

    @staticmethod
    def denormalize_points(
        norm_coords: list[list[float]], width: int, height: int
    ) -> np.ndarray:
        """Converts normalized [[x, y], ...] coordinates (0.0 to 1.0) into integer pixel coordinates."""
        pts = np.array(
            [[int(pt[0] * width), int(pt[1] * height)] for pt in norm_coords],
            dtype=np.int32,
        )
        return pts.reshape((-1, 1, 2))

    @staticmethod
    def is_point_in_zone(point: tuple[int, int], polygon_contour: np.ndarray) -> bool:
        """Vectorized spatial containment test: returns True if point is inside or on edge of polygon."""
        pt = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(polygon_contour, pt, False) >= 0

    @classmethod
    def evaluate_zone_rule(
        cls,
        zone_type: ZoneType | str,
        matched_user: dict[str, Any] | None,
        is_recognized: bool,
        face_detected: bool = True,
        zone_name: str = "Zone",
    ) -> ZoneRuleCheckResult:
        """Evaluates access policy rules for a detected person inside a designated zone."""
        z_type = ZoneType(zone_type) if isinstance(zone_type, str) else zone_type

        # 1. BANNED ZONE: Absolute zero tolerance. No one is permitted to enter.
        if z_type == ZoneType.BANNED:
            user_label = matched_user["name"] if (is_recognized and matched_user) else "Unknown Visitor"
            return ZoneRuleCheckResult(
                is_violation=True,
                status_level="BREACH",
                message=f"BREACH: Banned Area [{user_label}]",
                color_bgr=cls.COLOR_VIOLATION,
                zone_name=zone_name,
                zone_type=z_type.value,
            )

        # 2. ADMIN ZONE: Only users with role containing 'admin' are permitted.
        elif z_type == ZoneType.ADMIN:
            if is_recognized and matched_user:
                role_str = str(matched_user.get("role", "")).lower()
                if "admin" in role_str:
                    return ZoneRuleCheckResult(
                        is_violation=False,
                        status_level="OK",
                        message=f"ADMIN ACCESS: {matched_user['name']}",
                        color_bgr=cls.COLOR_AUTHORIZED,
                        zone_name=zone_name,
                        zone_type=z_type.value,
                    )
                else:
                    return ZoneRuleCheckResult(
                        is_violation=True,
                        status_level="BREACH",
                        message=f"UNAUTHORIZED: Non-Admin ({matched_user['name']})",
                        color_bgr=cls.COLOR_VIOLATION,
                        zone_name=zone_name,
                        zone_type=z_type.value,
                    )
            else:
                return ZoneRuleCheckResult(
                    is_violation=True,
                    status_level="BREACH",
                    message="BREACH: Unknown in Admin Zone",
                    color_bgr=cls.COLOR_VIOLATION,
                    zone_name=zone_name,
                    zone_type=z_type.value,
                )

        # 3. REGISTERED ZONE: Only registered users (admin or standard user) are permitted.
        elif z_type == ZoneType.REGISTERED:
            if is_recognized and matched_user:
                return ZoneRuleCheckResult(
                    is_violation=False,
                    status_level="OK",
                    message=f"AUTHORIZED: {matched_user['name']}",
                    color_bgr=cls.COLOR_AUTHORIZED,
                    zone_name=zone_name,
                    zone_type=z_type.value,
                )
            else:
                return ZoneRuleCheckResult(
                    is_violation=True,
                    status_level="BREACH",
                    message="BREACH: Unknown in Registered Zone",
                    color_bgr=cls.COLOR_VIOLATION,
                    zone_name=zone_name,
                    zone_type=z_type.value,
                )

        # 4. DETECTED ZONE: Any person (known or unknown) is allowed; verifies face detection presence.
        elif z_type == ZoneType.DETECTED:
            if face_detected:
                if is_recognized and matched_user:
                    return ZoneRuleCheckResult(
                        is_violation=False,
                        status_level="OK",
                        message=f"VERIFIED: {matched_user['name']}",
                        color_bgr=cls.COLOR_AUTHORIZED,
                        zone_name=zone_name,
                        zone_type=z_type.value,
                    )
                else:
                    return ZoneRuleCheckResult(
                        is_violation=False,
                        status_level="OK",
                        message="VERIFIED: Visitor (Face Detected)",
                        color_bgr=(0, 240, 240),  # Yellow / Light Amber
                        zone_name=zone_name,
                        zone_type=z_type.value,
                    )
            else:
                return ZoneRuleCheckResult(
                    is_violation=True,
                    status_level="WARNING",
                    message="ALERT: Human Without Face Detected",
                    color_bgr=cls.COLOR_WARNING,
                    zone_name=zone_name,
                    zone_type=z_type.value,
                )

        # Fallback
        return ZoneRuleCheckResult(
            is_violation=False,
            status_level="OK",
            message="OK",
            color_bgr=cls.COLOR_AUTHORIZED,
            zone_name=zone_name,
            zone_type="general",
        )

    @classmethod
    def render_zones_on_frame(
        cls,
        frame: np.ndarray,
        zones_with_contours: list[tuple[Any, np.ndarray]],
        active_violation_zone_ids: set[int] | None = None,
    ) -> np.ndarray:
        """Renders semi-transparent filled polygons (25% alpha), solid borders, and zone tags on frame."""
        if not zones_with_contours:
            return frame

        fh, fw, _ = frame.shape
        overlay = frame.copy()
        violation_ids = active_violation_zone_ids or set()

        for zone_obj, contour in zones_with_contours:
            z_id = getattr(zone_obj, "id", None) or (zone_obj.get("id") if isinstance(zone_obj, dict) else None)
            z_type_raw = getattr(zone_obj, "zone_type", None) or (zone_obj.get("zone_type") if isinstance(zone_obj, dict) else "registered")
            z_type = ZoneType(z_type_raw) if isinstance(z_type_raw, str) else z_type_raw
            z_name = getattr(zone_obj, "name", None) or (zone_obj.get("name") if isinstance(zone_obj, dict) else "Zone")

            is_breached = z_id in violation_ids
            base_color = (0, 0, 255) if is_breached else cls.ZONE_COLORS.get(z_type, (200, 200, 200))

            # 1. Fill semi-transparent polygon
            cv2.fillPoly(overlay, [contour], base_color)

            # 2. Solid outline border
            cv2.polylines(frame, [contour], isClosed=True, color=base_color, thickness=2, lineType=cv2.LINE_AA)

            # 3. Zone Centroid Tag Label
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                pts = contour.reshape((-1, 2))
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

            # Format tag text
            status_text = "BREACH" if is_breached else z_type.value.upper()
            tag_label = f"[{status_text}] {z_name}"

            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = max(4, min(fw - tw - 12, cx - tw // 2))
            ty = max(20, min(fh - 10, cy))

            # Draw tag badge background & text
            cv2.rectangle(
                frame,
                (tx - 4, ty - th - 6),
                (tx + tw + 6, ty + 4),
                base_color,
                -1,
            )
            cv2.putText(
                frame,
                tag_label,
                (tx, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Apply 25% alpha blending for zone fills
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame


zone_engine = ZoneEngine()


class ZoneService:

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = ZoneRepository(session)
        self.camera_repository = CameraRepository(session)

    async def create_zone(self, camera_id: int, zone_in: ZoneCreate) -> ZoneResponse:
        camera = await self.camera_repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera with ID {camera_id} not found.",
            )

        db_zone = await self.repository.create(camera_id=camera_id, zone_in=zone_in)
        return ZoneResponse.model_validate(db_zone)

    async def get_zone(self, zone_id: int) -> ZoneResponse:
        zone = await self.repository.get_by_id(zone_id)
        if not zone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Zone not found.",
            )
        return ZoneResponse.model_validate(zone)

    async def list_zones_by_camera(
        self, camera_id: int, active_only: bool = False
    ) -> list[ZoneResponse]:
        camera = await self.camera_repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera with ID {camera_id} not found.",
            )

        zones = await self.repository.list_by_camera(
            camera_id=camera_id, active_only=active_only
        )
        return [ZoneResponse.model_validate(z) for z in zones]

    async def update_zone(self, zone_id: int, zone_in: ZoneUpdate) -> ZoneResponse:
        zone = await self.repository.get_by_id(zone_id)
        if not zone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Zone not found.",
            )

        updated_zone = await self.repository.update(zone=zone, zone_in=zone_in)
        return ZoneResponse.model_validate(updated_zone)

    async def delete_zone(self, zone_id: int) -> None:
        zone = await self.repository.get_by_id(zone_id)
        if not zone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Zone not found.",
            )
        await self.repository.soft_delete(zone)
