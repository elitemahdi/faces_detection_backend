import threading
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import aiofiles
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user_repository import UserRepository
from app.repositories.zone_repository import ZoneRepository
from app.schemas.video import VideoProcessResponse
from app.services.face_recognition_service import face_recognition_service
from app.services.tracking_service import FaceTrackerManager
from app.services.zone_service import zone_engine

MATCH_THRESHOLD = 0.38
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage" / "videos"
METADATA_FILE = STORAGE_DIR / "metadata.json"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class VideoProcessingService:
    _meta_lock = threading.Lock()

    def __init__(self, session: AsyncSession):
        self.session = session
        self.user_repository = UserRepository(session)
        self.zone_repository = ZoneRepository(session)
        self.recognition_service = face_recognition_service

    def _read_metadata(self) -> dict[str, dict]:
        with self._meta_lock:
            if not METADATA_FILE.exists():
                return {}
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

    def _save_metadata(self, data: dict[str, dict]) -> None:
        with self._meta_lock:
            try:
                with open(METADATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception:
                pass

    def list_processed_videos(self) -> list[VideoProcessResponse]:
        meta = self._read_metadata()
        results = []
        for vid, item in meta.items():
            try:
                results.append(VideoProcessResponse.model_validate(item))
            except Exception:
                pass
        return sorted(results, key=lambda x: x.created_at, reverse=True)

    def get_video_file_path(self, video_id: str) -> Path:
        meta = self._read_metadata()
        if video_id not in meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video with ID '{video_id}' not found.",
            )
        filename = meta[video_id]["annotated_filename"]
        path = STORAGE_DIR / filename
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Annotated video file is missing on storage.",
            )
        return path

    def delete_processed_video(self, video_id: str) -> None:
        meta = self._read_metadata()
        if video_id not in meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video with ID '{video_id}' not found.",
            )
        item = meta.pop(video_id)
        self._save_metadata(meta)
        path = STORAGE_DIR / item["annotated_filename"]
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    async def annotate_video_file(
            self,
            video_file: UploadFile,
            sample_rate: int = 1,
            match_threshold: float = MATCH_THRESHOLD,
            camera_id: int | None = None,
            zones_data: list[dict] | None = None,
    ) -> dict:
        """Non-blocking video annotation with AdaFace and Zone detection policies."""
        video_id = str(uuid.uuid4())
        original_filename = video_file.filename or f"{video_id}.mp4"
        ext = os.path.splitext(original_filename)[1] or ".mp4"
        input_path = STORAGE_DIR / f"{video_id}_input{ext}"
        output_filename = f"{video_id}_annotated.mp4"
        output_path = STORAGE_DIR / output_filename

        # 1. Asynchronously stream uploaded video to disk
        async with aiofiles.open(input_path, "wb") as out_file:
            while content := await video_file.read(1024 * 1024):
                await out_file.write(content)

        # 2. Fetch enrolled users and zones asynchronously
        users = await self.user_repository.list_all(skip=0, limit=1000)
        known_users = [
            {
                "id": u.id,
                "name": u.full_name,
                "role": u.role.value if hasattr(u.role, "value") else str(u.role),
                "embedding": np.array(u.embedding, dtype=np.float32),
            }
            for u in users
            if u.embedding is not None
        ]

        active_zones = []
        if zones_data:
            active_zones = zones_data
        elif camera_id:
            db_zones = await self.zone_repository.list_by_camera(camera_id, active_only=True)
            active_zones = [
                {
                    "id": z.id,
                    "name": z.name,
                    "zone_type": z.zone_type.value if hasattr(z.zone_type, "value") else str(z.zone_type),
                    "coordinates": z.coordinates,
                }
                for z in db_zones
            ]

        # 3. Offload CPU/GPU heavy processing to background worker
        loop = asyncio.get_running_loop()
        try:
            total_frames, fps, total_detections, recognized_users, duration = (
                await loop.run_in_executor(
                    None,
                    partial(
                        self._process_video_sync,
                        str(input_path),
                        str(output_path),
                        known_users,
                        match_threshold,
                        sample_rate,
                        active_zones,
                    ),
                )
            )
        except ValueError as val_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(val_err),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Video processing failed: {str(exc)}",
            )
        finally:
            if input_path.exists():
                try:
                    input_path.unlink()
                except Exception:
                    pass

        file_size = output_path.stat().st_size if output_path.exists() else 0
        now = datetime.now(timezone.utc)

        response_obj = VideoProcessResponse(
            video_id=video_id,
            original_filename=original_filename,
            annotated_filename=output_filename,
            download_url=f"/api/v1/videos/{video_id}/download",
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            total_detections=total_detections,
            recognized_users=sorted(list(recognized_users)),
            file_size_bytes=file_size,
            created_at=now,
        )

        meta = self._read_metadata()
        meta[video_id] = response_obj.model_dump()
        self._save_metadata(meta)

        return {
            "metadata": response_obj,
            "output_path": output_path,
        }

    def _process_video_sync(
            self,
            input_path: str,
            output_path: str,
            known_users: list[dict],
            match_threshold: float,
            sample_rate: int = 1,
            zones_data: list[dict] | None = None,
    ) -> tuple[int, float, int, set[str], float]:
        """Synchronous, thread-isolated worker function executing in worker pool."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open uploaded video file '{input_path}'.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = (frame_count_est / fps) if fps > 0 else 0.0

        if known_users:
            enrolled_matrix = np.array(
                [u["embedding"] for u in known_users], dtype=np.float32
            )
            norms = np.linalg.norm(enrolled_matrix, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            enrolled_matrix = enrolled_matrix / norms
        else:
            enrolled_matrix = None

        zones_with_contours = []
        if zones_data:
            for z in zones_data:
                coords = z.get("coordinates", [])
                if len(coords) >= 3:
                    contour = zone_engine.denormalize_points(coords, width, height)
                    zones_with_contours.append((z, contour))

        detector = self.recognition_service.get_detector((width, height))
        tracker_manager = FaceTrackerManager(
            fps=int(fps),
            recheck_interval_frames=max(15, int(fps * 2)),
        )

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to initialize video writer with avc1 or mp4v codec.")

        total_frames = 0
        unique_track_ids = set()
        recognized_users = set()
        last_annotations = []
        last_breached_zones = set()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1
                fh, fw, _ = frame.shape

                if total_frames % sample_rate == 0:
                    detector.setInputSize((fw, fh))
                    _, raw_faces = detector.detect(frame)
                else:
                    raw_faces = None  # Advances Kalman filter without running detector

                last_annotations = []
                last_breached_zones = set()

                tracked_faces = tracker_manager.update(
                    frame=frame,
                    raw_faces=raw_faces,
                    recognition_service=self.recognition_service,
                    enrolled_matrix=enrolled_matrix,
                    known_users=known_users,
                    match_threshold=match_threshold,
                )

                for tf in tracked_faces:
                    unique_track_ids.add(tf.track_id)
                    if tf.is_recognized:
                        recognized_users.add(tf.name)

                    bx, by, bw, bh = tf.bbox_xywh
                    face_center = (bx + bw // 2, by + bh // 2)

                    label = f"ID:{tf.track_id} | {tf.name} ({tf.confidence:.2f})"
                    color = tf.color_bgr

                    for zone_obj, contour in zones_with_contours:
                        if zone_engine.is_point_in_zone(face_center, contour):
                            z_type = zone_obj.get("zone_type", "registered")
                            z_name = zone_obj.get("name", "Zone")
                            z_check = zone_engine.evaluate_zone_rule(
                                zone_type=z_type,
                                matched_user=tf.user_data,
                                is_recognized=tf.is_recognized,
                                face_detected=True,
                                zone_name=z_name,
                            )
                            color = z_check.color_bgr
                            label = f"ID:{tf.track_id} | {z_check.message}"
                            if z_check.is_violation:
                                z_id = zone_obj.get("id")
                                if z_id is not None:
                                    last_breached_zones.add(z_id)
                            break

                    last_annotations.append((tf.bbox_xywh, label, color))

                # Render zones
                if zones_with_contours:
                    frame = zone_engine.render_zones_on_frame(
                        frame, zones_with_contours, last_breached_zones
                    )

                # Render annotations
                for bbox, label, color in last_annotations:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(
                        frame, (bx, by), (bx + bw, by + bh), color, 2
                    )
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
                    )
                    cv2.rectangle(
                        frame,
                        (bx, max(0, by - 22)),
                        (bx + tw + 8, max(22, by)),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (bx + 4, max(16, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                writer.write(frame)

        finally:
            cap.release()
            writer.release()

        actual_duration = (total_frames / fps) if fps > 0 else duration
        total_unique_visitors = len(unique_track_ids)
        return total_frames, fps, total_unique_visitors, recognized_users, actual_duration
