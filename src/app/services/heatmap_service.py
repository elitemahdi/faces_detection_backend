import asyncio
from collections import Counter
from datetime import datetime, timezone
import cv2
from fastapi import HTTPException, status
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.frame_cache import frame_cache
from app.models.detection_event import DetectionEvent
from app.repositories.camera_repository import CameraRepository
from app.repositories.detection_event_repository import DetectionEventRepository
from app.repositories.zone_repository import ZoneRepository
from app.schemas.heatmap import (
    HeatmapColormap,
    HeatmapJsonResponse,
    HeatmapPoint,
    HeatmapStatsResponse,
    UserRankingItem,
    ZoneRankingItem,
)
from app.services.zone_service import zone_engine



def _capture_fallback_frame(source: str, w: int, h: int) -> np.ndarray | None:
    """Safe fallback worker for capturing a single background frame.
    Avoids opening local webcams (indices like '0') if already held by an active stream to prevent MSMF driver crashes.
    """
    if str(source).strip().isdigit():
        # Local webcam: Avoid opening concurrent handle
        return None
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return cv2.resize(frame, (w, h))
    except Exception:
        pass
    return None


class HeatmapService:

    COLORMAP_MAP = {
        HeatmapColormap.JET: cv2.COLORMAP_JET,
        HeatmapColormap.HOT: cv2.COLORMAP_HOT,
        HeatmapColormap.INFERNO: cv2.COLORMAP_INFERNO,
        HeatmapColormap.VIRIDIS: cv2.COLORMAP_VIRIDIS,
        HeatmapColormap.TURBO: cv2.COLORMAP_TURBO,
        HeatmapColormap.OCEAN: cv2.COLORMAP_OCEAN,
    }

    def __init__(self, session: AsyncSession):
        self.session = session
        self.camera_repository = CameraRepository(session)
        self.event_repository = DetectionEventRepository(session)
        self.zone_repository = ZoneRepository(session)

    async def get_heatmap_data(
        self,
        camera_id: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        start_time_of_day: str | None = None,
        end_time_of_day: str | None = None,
        user_id: int | None = None,
        user_name: str | None = None,
        is_recognized: bool | None = None,
        zone_id: int | None = None,
        grid_size: int = 32,
    ) -> HeatmapJsonResponse:
        camera = await self.camera_repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera with ID {camera_id} not found.",
            )

        events = await self.event_repository.query_events(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            start_time_of_day=start_time_of_day,
            end_time_of_day=end_time_of_day,
            user_id=user_id,
            user_name=user_name,
            is_recognized=is_recognized,
            zone_id=zone_id,
            limit=10000,
        )

        zones = await self.zone_repository.list_by_camera(camera_id, active_only=False)
        zones_by_id = {z.id: z for z in zones}

        points = [
            HeatmapPoint(
                x=e.norm_x,
                y=e.norm_y,
                weight=e.confidence,
                user_id=e.user_id,
                user_name=e.user_name,
                timestamp=e.timestamp,
            )
            for e in events
        ]

        # Compute 2D Density Grid Matrix
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        for p in points:
            gx = min(grid_size - 1, max(0, int(p.x * grid_size)))
            gy = min(grid_size - 1, max(0, int(p.y * grid_size)))
            grid[gy, gx] += p.weight

        max_val = np.max(grid)
        if max_val > 0:
            grid = grid / max_val

        # Compute Detailed Analytics & Summary Stats
        stats = self._compute_stats(
            camera_id=camera_id,
            events=events,
            zones_by_id=zones_by_id,
            start_time=start_time,
            end_time=end_time,
            start_time_of_day=start_time_of_day,
            end_time_of_day=end_time_of_day,
        )

        return HeatmapJsonResponse(
            camera_id=camera_id,
            total_points=len(points),
            stats=stats,
            points=points,
            grid_matrix=grid.tolist(),
        )

    def _compute_stats(
        self,
        camera_id: int,
        events: list[DetectionEvent],
        zones_by_id: dict,
        start_time: datetime | None,
        end_time: datetime | None,
        start_time_of_day: str | None = None,
        end_time_of_day: str | None = None,
    ) -> HeatmapStatsResponse:
        total = len(events)
        rec_count = sum(1 for e in events if e.is_recognized)
        unknown_count = total - rec_count
        rec_rate = (rec_count / total * 100.0) if total > 0 else 0.0

        # Unique & Ranked Users
        user_counter = Counter(
            e.user_name or f"User #{e.user_id}" for e in events if e.is_recognized and (e.user_name or e.user_id)
        )
        unique_users = len(user_counter)
        top_user = user_counter.most_common(1)[0][0] if user_counter else None

        user_rankings = [
            UserRankingItem(
                user_id=None,
                user_name=uname,
                count=cnt,
                percentage=round((cnt / total * 100.0), 1) if total > 0 else 0.0,
            )
            for uname, cnt in user_counter.most_common(10)
        ]

        # 24-Hour Distribution (0 .. 23)
        hourly_dist = {h: 0 for h in range(24)}
        for e in events:
            if e.timestamp:
                hourly_dist[e.timestamp.hour] = hourly_dist.get(e.timestamp.hour, 0) + 1

        peak_hour = max(hourly_dist, key=hourly_dist.get) if total > 0 else None

        # Zone Distribution & Ranking
        zone_counter = Counter(e.zone_id for e in events if e.zone_id is not None)
        zone_rankings = []
        for zid, cnt in zone_counter.most_common(10):
            zone_obj = zones_by_id.get(zid)
            zname = zone_obj.name if zone_obj else f"Zone #{zid}"
            ztype = zone_obj.zone_type.value if zone_obj and hasattr(zone_obj.zone_type, "value") else "registered"
            zone_rankings.append(
                ZoneRankingItem(
                    zone_id=zid,
                    zone_name=zname,
                    zone_type=ztype,
                    count=cnt,
                    percentage=round((cnt / total * 100.0), 1) if total > 0 else 0.0,
                )
            )

        return HeatmapStatsResponse(
            camera_id=camera_id,
            total_detections=total,
            recognized_count=rec_count,
            unknown_count=unknown_count,
            recognition_rate=round(rec_rate, 1),
            unique_users_count=unique_users,
            top_user_name=top_user,
            peak_hour=peak_hour,
            start_time=start_time,
            end_time=end_time,
            start_time_of_day=start_time_of_day,
            end_time_of_day=end_time_of_day,
            hourly_distribution=hourly_dist,
            top_users_ranking=user_rankings,
            zone_traffic_ranking=zone_rankings,
        )

    async def generate_heatmap_image(
        self,
        camera_id: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        start_time_of_day: str | None = None,
        end_time_of_day: str | None = None,
        user_id: int | None = None,
        user_name: str | None = None,
        is_recognized: bool | None = None,
        zone_id: int | None = None,
        colormap: HeatmapColormap = HeatmapColormap.JET,
        radius: int = 35,
        alpha: float = 0.55,
        width: int = 960,
        height: int = 540,
    ) -> bytes:
        """Generates a high-resolution JPEG heatmap image overlay with time-of-day filtering."""
        camera = await self.camera_repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )

        events = await self.event_repository.query_events(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            start_time_of_day=start_time_of_day,
            end_time_of_day=end_time_of_day,
            user_id=user_id,
            user_name=user_name,
            is_recognized=is_recognized,
            zone_id=zone_id,
            limit=10000,
        )

        # 1. Base Frame: Check decoupled frame_cache first, fallback non-blockingly via asyncio.to_thread
        base_frame = None
        cached_frame = frame_cache.get(camera_id)
        if cached_frame is not None:
            base_frame = cv2.resize(cached_frame, (width, height))
        else:
            base_frame = await asyncio.to_thread(_capture_fallback_frame, camera.source, width, height)

        if base_frame is None:
            base_frame = np.full((height, width, 3), 20, dtype=np.uint8)
            for x in range(0, width, 60):
                cv2.line(base_frame, (x, 0), (x, height), (35, 35, 35), 1)
            for y in range(0, height, 60):
                cv2.line(base_frame, (0, y), (width, y), (35, 35, 35), 1)

        # 2. Accumulate spatial heat density additively
        density_map = np.zeros((height, width), dtype=np.float32)

        if events:
            for e in events:
                px = int(np.clip(e.norm_x * width, 0, width - 1))
                py = int(np.clip(e.norm_y * height, 0, height - 1))
                weight = float(np.clip(e.confidence, 0.2, 1.5))
                density_map[py, px] += weight

            ksize = int(radius * 4 + 1)
            if ksize % 2 == 0:
                ksize += 1
            sigma = float(radius / 2.0)
            density_map = cv2.GaussianBlur(density_map, (ksize, ksize), sigma)

            max_val = np.max(density_map)
            if max_val > 0:
                normalized_density = np.uint8(np.clip((density_map / max_val) * 255.0, 0, 255))
            else:
                normalized_density = np.zeros((height, width), dtype=np.uint8)

            cv_colormap = self.COLORMAP_MAP.get(colormap, cv2.COLORMAP_JET)
            color_heatmap = cv2.applyColorMap(normalized_density, cv_colormap)

            mask = normalized_density > 10
            overlay = base_frame.copy()
            overlay[mask] = color_heatmap[mask]

            cv2.addWeighted(overlay, alpha, base_frame, 1.0 - alpha, 0, base_frame)

        # 3. Render active security zones as subtle outlines
        active_zones = await self.zone_repository.list_by_camera(camera_id, active_only=True)
        if active_zones:
            zones_with_contours = [
                (z, zone_engine.denormalize_points(z.coordinates, width, height))
                for z in active_zones
            ]
            base_frame = zone_engine.render_zones_on_frame(base_frame, zones_with_contours, set())

        # 4. Render HUD Information Overlay
        total_pts = len(events)
        cv2.rectangle(base_frame, (10, 10), (340, 84), (15, 15, 15), -1)
        cv2.rectangle(base_frame, (10, 10), (340, 84), (70, 70, 70), 1)

        time_window = f" | Time: {start_time_of_day}-{end_time_of_day}" if (start_time_of_day and end_time_of_day) else ""
        title_text = f"HEATMAP: {camera.name.upper()}"
        stats_text = f"Detections: {total_pts}{time_window}"
        filter_text = f"Filter: {user_name or (f'User #{user_id}' if user_id else 'All Traffic')}"

        cv2.putText(base_frame, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(base_frame, stats_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(base_frame, filter_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        # 5. Render Colormap Gradient Bar Legend
        bar_x, bar_y, bar_w, bar_h = width - 180, 16, 160, 14
        grad = np.tile(np.linspace(0, 255, bar_w, dtype=np.uint8), (bar_h, 1))
        cv_colormap = self.COLORMAP_MAP.get(colormap, cv2.COLORMAP_JET)
        color_grad = cv2.applyColorMap(grad, cv_colormap)
        base_frame[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = color_grad
        cv2.rectangle(base_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        cv2.putText(base_frame, "Low", (bar_x - 30, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(base_frame, "High", (bar_x + bar_w + 6, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        ret, buffer = cv2.imencode(".jpg", base_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buffer.tobytes()
