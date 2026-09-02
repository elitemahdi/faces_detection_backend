import asyncio
import threading
import time
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.frame_cache import frame_cache
from app.repositories.camera_repository import CameraRepository
from app.repositories.detection_event_repository import DetectionEventRepository
from app.repositories.user_repository import UserRepository
from app.repositories.zone_repository import ZoneRepository
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate
from app.services.face_recognition_service import face_recognition_service
from app.services.tracking_service import FaceTrackerManager
from app.services.zone_service import zone_engine

MATCH_THRESHOLD = 0.38


class ThreadedCameraReader:
    """Non-blocking background camera reader.
    Prevents OpenCV frame buffer backlog and ensures real-time zero-latency frame delivery at 30+ FPS.
    """

    def __init__(self, source: int | str):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.running = True
        self.latest_frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                with self.lock:
                    self.latest_frame = frame
                    self.ret = ret
                time.sleep(0.005)  # Pace reads to avoid MSMF buffer overload
            except Exception:
                time.sleep(0.01)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get_dimensions(self) -> tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        return w, h

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if not self.ret or self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.3)
        self.cap.release()


async def _persist_detection_events(events: list[dict]) -> None:
    """Asynchronously logs live spatial detection events to the database without blocking the video stream."""
    if not events:
        return
    try:
        async with AsyncSessionLocal() as session:
            repo = DetectionEventRepository(session)
            await repo.bulk_log_events(events)
    except Exception as exc:
        print(f"[WARN] Failed to persist live detection events: {exc}")


class CameraService:
    _latest_raw_frames: dict[int, np.ndarray] = {}

    @classmethod
    def get_cached_frame(cls, camera_id: int) -> np.ndarray | None:
        return cls._latest_raw_frames.get(camera_id)

    @classmethod
    def set_cached_frame(cls, camera_id: int, frame: np.ndarray) -> None:
        cls._latest_raw_frames[camera_id] = frame

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = CameraRepository(session)
        self.user_repository = UserRepository(session)
        self.zone_repository = ZoneRepository(session)
        self.event_repository = DetectionEventRepository(session)
        self.recognition_service = face_recognition_service

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

    async def stream_camera_feed(
        self,
        camera_id: int,
        annotate_faces: bool = True,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> AsyncGenerator[bytes, None]:
        """Streams live camera frames via MJPEG with real-time AI recognition, motion gating, and 30+ FPS throughput."""
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )

        # 1. Preload enrolled users
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

        if known_users:
            enrolled_matrix = np.array(
                [u["embedding"] for u in known_users], dtype=np.float32
            )
            norms = np.linalg.norm(enrolled_matrix, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            enrolled_matrix = enrolled_matrix / norms
        else:
            enrolled_matrix = None

        active_zones = await self.zone_repository.list_by_camera(camera_id, active_only=True)

        device_src = int(camera.source) if camera.source.isdigit() else camera.source
        reader = ThreadedCameraReader(device_src)

        try:
            if not reader.isOpened():
                err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    err_frame,
                    f"Cannot open camera: {camera.name}",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                ret, buf = cv2.imencode(".jpg", err_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
                return

            w, h = reader.get_dimensions()
            detector = self.recognition_service.get_detector((w, h))
            tracker_manager = FaceTrackerManager(
                fps=30,
                recheck_interval_frames=25,
            )

            frame_count = 0
            fail_count = 0
            last_annotations = []
            last_breached_zones = set()
            cached_resolution = None
            zones_with_contours = []
            background_tasks: set[asyncio.Task] = set()

            while reader.isOpened():
                ret, frame = await asyncio.to_thread(reader.read)
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count > 60:
                        break
                    await asyncio.sleep(0.01)
                    continue

                fail_count = 0
                frame_cache.set(camera_id, frame)
                frame_count += 1
                fh, fw, _ = frame.shape

                # Cache denormalized zone contours
                if cached_resolution != (fw, fh):
                    zones_with_contours = [
                        (z, zone_engine.denormalize_points(z.coordinates, fw, fh))
                        for z in active_zones
                    ]
                    cached_resolution = (fw, fh)

                if annotate_faces:
                    # Run deep detector every 2 frames; on intermediate frames, ByteTrack tracks smoothly
                    run_detection = (frame_count % 2 == 1) or (frame_count < 5)
                    if run_detection:
                        detector.setInputSize((fw, fh))
                        _, raw_faces = detector.detect(frame)
                    else:
                        raw_faces = None

                    tracked_faces = tracker_manager.update(
                        frame=frame,
                        raw_faces=raw_faces,
                        recognition_service=self.recognition_service,
                        enrolled_matrix=enrolled_matrix,
                        known_users=known_users,
                        match_threshold=match_threshold,
                    )

                    last_annotations = []
                    last_breached_zones = set()
                    events_to_log = []

                    for tf in tracked_faces:
                        bx, by, bw, bh = tf.bbox_xywh
                        face_center = (bx + bw // 2, by + bh // 2)

                        in_zone = False
                        matched_zone_id = None
                        label = f"{tf.name} ({tf.confidence:.2f})" if tf.is_recognized else tf.name
                        color = tf.color_bgr  # Green (0, 255, 0) when recognized, Red/Orange when fast/unknown

                        for zone_obj, contour in zones_with_contours:
                            if zone_engine.is_point_in_zone(face_center, contour):
                                in_zone = True
                                matched_zone_id = zone_obj.id
                                z_check = zone_engine.evaluate_zone_rule(
                                    zone_type=zone_obj.zone_type,
                                    matched_user=tf.user_data,
                                    is_recognized=tf.is_recognized,
                                    face_detected=True,
                                    zone_name=zone_obj.name,
                                )
                                color = z_check.color_bgr
                                label = f"{tf.name} | {z_check.message}"
                                if z_check.is_violation:
                                    last_breached_zones.add(zone_obj.id)
                                break

                        last_annotations.append((tf.bbox_xywh, label, color, tf.is_recognized))

                        events_to_log.append({
                            "camera_id": camera_id,
                            "user_id": tf.user_data["id"] if (tf.is_recognized and tf.user_data) else None,
                            "user_name": tf.name if tf.is_recognized else None,
                            "norm_x": round(float(face_center[0]) / fw, 4),
                            "norm_y": round(float(face_center[1]) / fh, 4),
                            "confidence": round(float(tf.confidence), 2),
                            "is_recognized": tf.is_recognized,
                            "zone_id": matched_zone_id,
                        })

                    # Log events every 10 frames (~3x per sec)
                    if frame_count % 10 == 0 and events_to_log:
                        persist_task = asyncio.create_task(_persist_detection_events(events_to_log))
                        background_tasks.add(persist_task)
                        persist_task.add_done_callback(background_tasks.discard)

                    # 1. Render zone polygons
                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, last_breached_zones
                        )

                    # 2. Render bounding boxes and HUD tags
                    for bbox, label, color, is_rec in last_annotations:
                        bx, by, bw, bh = bbox
                        thickness = 3 if is_rec else 2
                        cv2.rectangle(
                            frame, (bx, by), (bx + bw, by + bh), color, thickness
                        )
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
                        )
                        cv2.rectangle(
                            frame,
                            (bx, max(0, by - 24)),
                            (bx + tw + 8, max(24, by)),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (bx + 4, max(18, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (255, 255, 255) if color != (0, 255, 0) else (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, set()
                        )

                # Fast JPEG compression
                ret, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

                # Non-blocking yield to event loop
                await asyncio.sleep(0.001)

        finally:
            frame_cache.remove(camera_id)
            reader.release()

    async def stream_camera_websocket(
        self,
        websocket: WebSocket,
        camera_id: int,
        annotate_faces: bool = True,
        match_threshold: float = MATCH_THRESHOLD,
    ):
        """Streams live camera frames via WebSocket with real-time AI recognition, motion gating, and 30+ FPS throughput."""
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Camera not found.",
            )
            return

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

        if known_users:
            enrolled_matrix = np.array(
                [u["embedding"] for u in known_users], dtype=np.float32
            )
            norms = np.linalg.norm(enrolled_matrix, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            enrolled_matrix = enrolled_matrix / norms
        else:
            enrolled_matrix = None

        active_zones = await self.zone_repository.list_by_camera(camera_id, active_only=True)

        device_src = int(camera.source) if camera.source.isdigit() else camera.source
        reader = ThreadedCameraReader(device_src)

        if not reader.isOpened():
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                err_frame,
                f"Cannot open camera: {camera.name}",
                (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            ret, buf = cv2.imencode(".jpg", err_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                await websocket.send_bytes(buf.tobytes())
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="Unable to open camera source.",
            )
            return

        w, h = reader.get_dimensions()
        detector = self.recognition_service.get_detector((w, h))

        current_annotate = annotate_faces
        current_threshold = match_threshold

        async def receive_client_controls():
            nonlocal current_annotate, current_threshold
            try:
                while True:
                    msg = await websocket.receive_json()
                    action = msg.get("action")
                    if action == "set_annotate":
                        current_annotate = bool(msg.get("value", True))
                    elif action == "set_threshold":
                        current_threshold = float(msg.get("value", MATCH_THRESHOLD))
                    elif action == "ping":
                        await websocket.send_json({"type": "pong"})
            except Exception:
                pass

        recv_task = asyncio.create_task(receive_client_controls())

        try:
            tracker_manager = FaceTrackerManager(
                fps=30,
                recheck_interval_frames=25,
            )
            frame_count = 0
            fail_count = 0
            last_annotations = []
            last_breached_zones = set()
            cached_resolution = None
            zones_with_contours = []
            background_tasks: set[asyncio.Task] = set()

            while reader.isOpened():
                ret, frame = await asyncio.to_thread(reader.read)
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count > 60:
                        break
                    await asyncio.sleep(0.01)
                    continue

                fail_count = 0
                frame_cache.set(camera_id, frame)
                frame_count += 1
                fh, fw, _ = frame.shape

                if cached_resolution != (fw, fh):
                    zones_with_contours = [
                        (z, zone_engine.denormalize_points(z.coordinates, fw, fh))
                        for z in active_zones
                    ]
                    cached_resolution = (fw, fh)

                if current_annotate:
                    run_detection = (frame_count % 2 == 1) or (frame_count < 5)
                    if run_detection:
                        detector.setInputSize((fw, fh))
                        _, raw_faces = detector.detect(frame)
                    else:
                        raw_faces = None

                    tracked_faces = tracker_manager.update(
                        frame=frame,
                        raw_faces=raw_faces,
                        recognition_service=self.recognition_service,
                        enrolled_matrix=enrolled_matrix,
                        known_users=known_users,
                        match_threshold=current_threshold,
                    )

                    last_annotations = []
                    last_breached_zones = set()
                    events_to_log = []

                    for tf in tracked_faces:
                        bx, by, bw, bh = tf.bbox_xywh
                        face_center = (bx + bw // 2, by + bh // 2)

                        in_zone = False
                        matched_zone_id = None
                        label = f"{tf.name} ({tf.confidence:.2f})" if tf.is_recognized else tf.name
                        color = tf.color_bgr

                        for zone_obj, contour in zones_with_contours:
                            if zone_engine.is_point_in_zone(face_center, contour):
                                in_zone = True
                                matched_zone_id = zone_obj.id
                                z_check = zone_engine.evaluate_zone_rule(
                                    zone_type=zone_obj.zone_type,
                                    matched_user=tf.user_data,
                                    is_recognized=tf.is_recognized,
                                    face_detected=True,
                                    zone_name=zone_obj.name,
                                )
                                color = z_check.color_bgr
                                label = f"{tf.name} | {z_check.message}"
                                if z_check.is_violation:
                                    last_breached_zones.add(zone_obj.id)
                                break

                        last_annotations.append((tf.bbox_xywh, label, color, tf.is_recognized))

                        events_to_log.append({
                            "camera_id": camera_id,
                            "user_id": tf.user_data["id"] if (tf.is_recognized and tf.user_data) else None,
                            "user_name": tf.name if tf.is_recognized else None,
                            "norm_x": round(float(face_center[0]) / fw, 4),
                            "norm_y": round(float(face_center[1]) / fh, 4),
                            "confidence": round(float(tf.confidence), 2),
                            "is_recognized": tf.is_recognized,
                            "zone_id": matched_zone_id,
                        })

                    if frame_count % 10 == 0 and events_to_log:
                        persist_task = asyncio.create_task(_persist_detection_events(events_to_log))
                        background_tasks.add(persist_task)
                        persist_task.add_done_callback(background_tasks.discard)

                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, last_breached_zones
                        )

                    for bbox, label, color, is_rec in last_annotations:
                        bx, by, bw, bh = bbox
                        thickness = 3 if is_rec else 2
                        cv2.rectangle(
                            frame, (bx, by), (bx + bw, by + bh), color, thickness
                        )
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
                        )
                        cv2.rectangle(
                            frame,
                            (bx, max(0, by - 24)),
                            (bx + tw + 8, max(24, by)),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (bx + 4, max(18, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (255, 255, 255) if color != (0, 255, 0) else (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, set()
                        )

                ret, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not ret:
                    continue

                try:
                    await websocket.send_bytes(buffer.tobytes())
                except WebSocketDisconnect:
                    break
                except Exception:
                    break

                await asyncio.sleep(0.001)

        finally:
            recv_task.cancel()
            frame_cache.remove(camera_id)
            reader.release()

    async def get_snapshot(
        self,
        camera_id: int,
        annotate_faces: bool = True,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> bytes:
        """Captures a single high-resolution JPEG frame from the camera with optional face annotations and zone overlays."""
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )

        frame = frame_cache.get(camera_id)
        if frame is None:
            device_src = int(camera.source) if camera.source.isdigit() else camera.source
            cap = cv2.VideoCapture(device_src)
            if not cap.isOpened():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Could not open camera stream '{camera.name}'.",
                )
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to capture snapshot frame from camera.",
                )

        frame = frame.copy()
        fh, fw, _ = frame.shape

        if annotate_faces:
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
            if known_users:
                enrolled_matrix = np.array(
                    [u["embedding"] for u in known_users], dtype=np.float32
                )
                norms = np.linalg.norm(enrolled_matrix, axis=1, keepdims=True)
                norms[norms < 1e-6] = 1.0
                enrolled_matrix = enrolled_matrix / norms
            else:
                enrolled_matrix = None

            active_zones = await self.zone_repository.list_by_camera(camera_id, active_only=True)
            zones_with_contours = [
                (z, zone_engine.denormalize_points(z.coordinates, fw, fh))
                for z in active_zones
            ]

            detector = self.recognition_service.get_detector((fw, fh))
            _, raw_faces = detector.detect(frame)

            tracker_manager = FaceTrackerManager(fps=30)
            tracked_faces = tracker_manager.update(
                frame=frame,
                raw_faces=raw_faces,
                recognition_service=self.recognition_service,
                enrolled_matrix=enrolled_matrix,
                known_users=known_users,
                match_threshold=match_threshold,
            )

            breached_zones = set()
            for tf in tracked_faces:
                bx, by, bw, bh = tf.bbox_xywh
                face_center = (bx + bw // 2, by + bh // 2)
                label = f"{tf.name} ({tf.confidence:.2f})" if tf.is_recognized else tf.name
                color = tf.color_bgr

                for zone_obj, contour in zones_with_contours:
                    if zone_engine.is_point_in_zone(face_center, contour):
                        z_check = zone_engine.evaluate_zone_rule(
                            zone_type=zone_obj.zone_type,
                            matched_user=tf.user_data,
                            is_recognized=tf.is_recognized,
                            face_detected=True,
                            zone_name=zone_obj.name,
                        )
                        color = z_check.color_bgr
                        label = f"{tf.name} | {z_check.message}"
                        if z_check.is_violation:
                            breached_zones.add(zone_obj.id)
                        break

                thickness = 3 if tf.is_recognized else 2
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, thickness)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (bx, max(0, by - 24)), (bx + tw + 8, max(24, by)), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (bx + 4, max(18, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255) if color != (0, 255, 0) else (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            if zones_with_contours:
                frame = zone_engine.render_zones_on_frame(frame, zones_with_contours, breached_zones)

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode snapshot image.",
            )
        return buffer.tobytes()
