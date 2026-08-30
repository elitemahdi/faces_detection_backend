import asyncio
from typing import AsyncGenerator
import cv2
import numpy as np
from fastapi import HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.camera_repository import CameraRepository
from app.repositories.user_repository import UserRepository
from app.repositories.zone_repository import ZoneRepository
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate
from app.services.face_recognition_service import face_recognition_service
from app.services.zone_service import zone_engine

MATCH_THRESHOLD = 0.38


class CameraService:

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = CameraRepository(session)
        self.user_repository = UserRepository(session)
        self.zone_repository = ZoneRepository(session)
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
        """Streams live camera frames via MJPEG multipart stream with real-time AI face recognition & zone rules."""
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )

        # 1. Preload enrolled users matrix for vectorized matching
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

        # 2. Preload active security zones for this camera
        active_zones = await self.zone_repository.list_by_camera(camera_id, active_only=True)

        device_src = int(camera.source) if camera.source.isdigit() else camera.source
        cap = cv2.VideoCapture(device_src)

        try:
            if not cap.isOpened():
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
                cv2.putText(
                    err_frame,
                    f"Source: {camera.source}",
                    (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )
                ret, buf = cv2.imencode(".jpg", err_frame)
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
                return

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            detector = self.recognition_service.get_detector((w, h))

            frame_count = 0
            last_annotations = []
            last_breached_zones = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.04)
                    continue

                frame_count += 1
                fh, fw, _ = frame.shape

                # Prepare denormalized zone contours for current resolution
                zones_with_contours = [
                    (z, zone_engine.denormalize_points(z.coordinates, fw, fh))
                    for z in active_zones
                ]

                if annotate_faces:
                    last_annotations = []
                    last_breached_zones = set()
                    detector.setInputSize((fw, fh))
                    _, faces = detector.detect(frame)

                    if faces is not None and len(faces) > 0:
                        bboxes = [f[:4].astype(int) for f in faces]
                        crops = [
                            self.recognition_service.align_and_crop(frame, f)
                            for f in faces
                        ]
                        embeddings = (
                            self.recognition_service.extract_batch_embeddings(
                                crops
                            )
                        )

                        for bbox, emb in zip(bboxes, embeddings):
                            bx, by, bw, bh = bbox
                            face_center = (bx + bw // 2, by + bh // 2)

                            best_match_user = None
                            highest_score = -1.0

                            if enrolled_matrix is not None and len(known_users) > 0:
                                sims = (
                                    self.recognition_service.batch_cosine_similarity(
                                        emb, enrolled_matrix
                                    )
                                )
                                best_idx = int(np.argmax(sims))
                                highest_score = float(sims[best_idx])
                                best_match_user = known_users[best_idx]

                            is_rec = (
                                highest_score >= match_threshold
                                and best_match_user is not None
                            )

                            # --- ZONE EVALUATION ---
                            in_zone = False
                            label = (
                                f"{best_match_user['name']} ({highest_score:.2f})"
                                if is_rec
                                else f"Unknown ({highest_score:.2f})"
                            )
                            color = (0, 255, 0) if is_rec else (0, 0, 255)

                            for zone_obj, contour in zones_with_contours:
                                if zone_engine.is_point_in_zone(face_center, contour):
                                    in_zone = True
                                    z_check = zone_engine.evaluate_zone_rule(
                                        zone_type=zone_obj.zone_type,
                                        matched_user=best_match_user,
                                        is_recognized=is_rec,
                                        face_detected=True,
                                        zone_name=zone_obj.name,
                                    )
                                    color = z_check.color_bgr
                                    label = z_check.message
                                    if z_check.is_violation:
                                        last_breached_zones.add(zone_obj.id)
                                    break

                            last_annotations.append((bbox, label, color))

                    # 1. Render semi-transparent zone polygons onto frame
                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, last_breached_zones
                        )

                    # 2. Render face bounding boxes and violation tags
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
                else:
                    # Still render zone polygons if annotations are off but zones exist
                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, set()
                        )

                ret, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

                await asyncio.sleep(0.033)

        finally:
            cap.release()

    async def stream_camera_websocket(
        self,
        websocket: WebSocket,
        camera_id: int,
        annotate_faces: bool = True,
        match_threshold: float = MATCH_THRESHOLD,
    ):
        """Streams live camera frames via WebSocket with real-time AI face recognition, zone security policies, and bidirectional controls."""
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
        cap = cv2.VideoCapture(device_src)

        if not cap.isOpened():
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
            ret, buf = cv2.imencode(".jpg", err_frame)
            if ret:
                await websocket.send_bytes(buf.tobytes())
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="Unable to open camera source.",
            )
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
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
            frame_count = 0
            last_annotations = []
            last_breached_zones = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.04)
                    continue

                frame_count += 1
                fh, fw, _ = frame.shape

                zones_with_contours = [
                    (z, zone_engine.denormalize_points(z.coordinates, fw, fh))
                    for z in active_zones
                ]

                if current_annotate:
                    last_annotations = []
                    last_breached_zones = set()
                    detector.setInputSize((fw, fh))
                    _, faces = detector.detect(frame)

                    if faces is not None and len(faces) > 0:
                        bboxes = [f[:4].astype(int) for f in faces]
                        crops = [
                            self.recognition_service.align_and_crop(frame, f)
                            for f in faces
                        ]
                        embeddings = (
                            self.recognition_service.extract_batch_embeddings(
                                crops
                            )
                        )

                        for bbox, emb in zip(bboxes, embeddings):
                            bx, by, bw, bh = bbox
                            face_center = (bx + bw // 2, by + bh // 2)

                            best_match_user = None
                            highest_score = -1.0

                            if enrolled_matrix is not None and len(known_users) > 0:
                                sims = (
                                    self.recognition_service.batch_cosine_similarity(
                                        emb, enrolled_matrix
                                    )
                                )
                                best_idx = int(np.argmax(sims))
                                highest_score = float(sims[best_idx])
                                best_match_user = known_users[best_idx]

                            is_rec = (
                                highest_score >= current_threshold
                                and best_match_user is not None
                            )

                            label = (
                                f"{best_match_user['name']} ({highest_score:.2f})"
                                if is_rec
                                else f"Unknown ({highest_score:.2f})"
                            )
                            color = (0, 255, 0) if is_rec else (0, 0, 255)

                            for zone_obj, contour in zones_with_contours:
                                if zone_engine.is_point_in_zone(face_center, contour):
                                    z_check = zone_engine.evaluate_zone_rule(
                                        zone_type=zone_obj.zone_type,
                                        matched_user=best_match_user,
                                        is_recognized=is_rec,
                                        face_detected=True,
                                        zone_name=zone_obj.name,
                                    )
                                    color = z_check.color_bgr
                                    label = z_check.message
                                    if z_check.is_violation:
                                        last_breached_zones.add(zone_obj.id)
                                    break

                            last_annotations.append((bbox, label, color))

                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, last_breached_zones
                        )

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
                else:
                    if zones_with_contours:
                        frame = zone_engine.render_zones_on_frame(
                            frame, zones_with_contours, set()
                        )

                ret, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if ret:
                    await websocket.send_bytes(buffer.tobytes())

                await asyncio.sleep(0.033)

        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            recv_task.cancel()
            cap.release()

    async def get_snapshot(
        self,
        camera_id: int,
        annotate_faces: bool = True,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> bytes:
        """Captures a single annotated JPEG frame from the camera with zone policies."""
        camera = await self.repository.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found.",
            )

        device_src = int(camera.source) if camera.source.isdigit() else camera.source
        cap = cv2.VideoCapture(device_src)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to connect to camera source '{camera.source}'.",
            )

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to capture frame from camera.",
            )

        fh, fw, _ = frame.shape
        active_zones = await self.zone_repository.list_by_camera(camera_id, active_only=True)
        zones_with_contours = [
            (z, zone_engine.denormalize_points(z.coordinates, fw, fh))
            for z in active_zones
        ]

        if annotate_faces:
            detector = self.recognition_service.get_detector((fw, fh))
            _, faces = detector.detect(frame)
            breached_zones = set()

            if faces is not None and len(faces) > 0:
                users = await self.user_repository.list_all(skip=0, limit=1000)
                known_users = [
                    {
                        "id": u.id,
                        "name": u.full_name,
                        "role": u.role.value if hasattr(u.role, "value") else str(u.role),
                        "embedding": np.array(u.embedding, dtype=np.float32),
                    }
                    for u in users if u.embedding is not None
                ]
                if known_users:
                    enrolled_matrix = np.array([u["embedding"] for u in known_users], dtype=np.float32)
                    norms = np.linalg.norm(enrolled_matrix, axis=1, keepdims=True)
                    norms[norms < 1e-6] = 1.0
                    enrolled_matrix = enrolled_matrix / norms
                else:
                    enrolled_matrix = None

                bboxes = [f[:4].astype(int) for f in faces]
                crops = [self.recognition_service.align_and_crop(frame, f) for f in faces]
                embeddings = self.recognition_service.extract_batch_embeddings(crops)

                for bbox, emb in zip(bboxes, embeddings):
                    bx, by, bw, bh = bbox
                    face_center = (bx + bw // 2, by + bh // 2)

                    best_match_user = None
                    highest_score = -1.0
                    if enrolled_matrix is not None and len(known_users) > 0:
                        sims = self.recognition_service.batch_cosine_similarity(emb, enrolled_matrix)
                        best_idx = int(np.argmax(sims))
                        highest_score = float(sims[best_idx])
                        best_match_user = known_users[best_idx]

                    is_rec = highest_score >= match_threshold and best_match_user is not None
                    label = f"{best_match_user['name']} ({highest_score:.2f})" if is_rec else f"Unknown ({highest_score:.2f})"
                    color = (0, 255, 0) if is_rec else (0, 0, 255)

                    for zone_obj, contour in zones_with_contours:
                        if zone_engine.is_point_in_zone(face_center, contour):
                            z_check = zone_engine.evaluate_zone_rule(
                                zone_type=zone_obj.zone_type,
                                matched_user=best_match_user,
                                is_recognized=is_rec,
                                face_detected=True,
                                zone_name=zone_obj.name,
                            )
                            color = z_check.color_bgr
                            label = z_check.message
                            if z_check.is_violation:
                                breached_zones.add(zone_obj.id)
                            break

                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(frame, (bx, max(0, by - 22)), (bx + tw + 8, max(22, by)), color, -1)
                    cv2.putText(frame, label, (bx + 4, max(16, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            if zones_with_contours:
                frame = zone_engine.render_zones_on_frame(frame, zones_with_contours, breached_zones)
        else:
            if zones_with_contours:
                frame = zone_engine.render_zones_on_frame(frame, zones_with_contours, set())

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
