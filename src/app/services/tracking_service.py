from dataclasses import dataclass
import cv2
import numpy as np
import supervision as sv

from app.core.config import settings


@dataclass
class TrackedFace:
    track_id: int
    bbox_xywh: tuple[int, int, int, int]
    name: str
    confidence: float
    is_recognized: bool
    color_bgr: tuple[int, int, int]
    user_data: dict | None = None
    landmarks: np.ndarray | None = None
    raw_detection_score: float = 1.0
    is_fast_motion: bool = False
    velocity: float = 0.0


class FaceTrackerManager:
    """Robust Multi-Object Face Tracker (MOT) with Persistent Identity Lock & Long-Distance Recognition.

    - Motion Gating: Strictly uses spatial velocity (px/frame) to detect fast movement without false blur rejections.
    - Long-Distance Recognition: Enables recognition on distant faces down to 16px.
    - Persistent Identity Lock: Once a user is verified, their identity and vibrant GREEN BORDER (0, 255, 0)
      remain locked even while moving around the room.
    """

    def __init__(
        self,
        fps: int = 30,
        track_activation_threshold: float = 0.20,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.30,
        recheck_interval_frames: int = 30,
        min_face_size_for_recognition: int = 16,  # Recognizes distant faces down to 16x16px
        max_velocity_for_recognition: float = 25.0,  # px per frame
    ):
        self.fps = max(1, fps)
        self.recheck_interval_frames = max(1, recheck_interval_frames)
        self.lost_track_buffer = lost_track_buffer
        self.min_face_size_for_recognition = min_face_size_for_recognition
        self.max_velocity_for_recognition = max_velocity_for_recognition

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=self.fps,
        )

        self.identity_cache: dict[int, dict] = {}
        self.frame_index: int = 0
        self.last_valid_raw_faces = np.empty((0, 15), dtype=np.float32)

    @staticmethod
    def _compute_iou(boxA: np.ndarray | tuple | list, boxB: np.ndarray | tuple | list) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        union_area = float(boxA_area + boxB_area - inter_area)
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)

    def _match_box_to_raw_face(
        self, tracked_xyxy: np.ndarray, raw_faces: np.ndarray
    ) -> np.ndarray | None:
        best_iou = 0.0
        best_face = None

        for face in raw_faces:
            x, y, w, h = face[:4]
            raw_xyxy = np.array([x, y, x + w, y + h])
            iou = self._compute_iou(tracked_xyxy, raw_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_face = face

        if best_iou >= 0.10:
            return best_face
        return None

    def update(
        self,
        frame: np.ndarray,
        raw_faces: np.ndarray | None,
        recognition_service,
        enrolled_matrix: np.ndarray | None,
        known_users: list[dict],
        match_threshold: float = 0.38,
    ) -> list[TrackedFace]:
        self.frame_index += 1
        fh, fw = frame.shape[:2]

        # 1. Filter valid detections
        valid_faces = []
        if raw_faces is not None and len(raw_faces) > 0:
            for f in raw_faces:
                det_score = float(f[14]) if len(f) > 14 else 1.0
                if det_score >= 0.20:
                    valid_faces.append(f)
            self.last_valid_raw_faces = np.array(valid_faces)
        elif raw_faces is not None:
            self.last_valid_raw_faces = np.empty((0, 15), dtype=np.float32)

        # 2. Build supervision Detections
        if len(self.last_valid_raw_faces) > 0:
            xyxy_boxes = []
            confidences = []
            for f in self.last_valid_raw_faces:
                x, y, w, h = f[:4]
                x1 = float(np.clip(x, 0, fw - 1))
                y1 = float(np.clip(y, 0, fh - 1))
                x2 = float(np.clip(x + w, x1 + 1, fw))
                y2 = float(np.clip(y + h, y1 + 1, fh))
                score = float(f[14]) if len(f) > 14 else 1.0
                xyxy_boxes.append([x1, y1, x2, y2])
                confidences.append(score)

            detections = sv.Detections(
                xyxy=np.array(xyxy_boxes, dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
            )
        else:
            detections = sv.Detections.empty()

        # 3. Update ByteTrack Kalman Filters
        tracked_detections = self.tracker.update_with_detections(detections)

        if len(tracked_detections) == 0 or tracked_detections.tracker_id is None:
            return []

        active_track_ids = set(tracked_detections.tracker_id.astype(int))

        # 4. Clean up stale tracks from identity cache
        self.identity_cache = {
            tid: data
            for tid, data in self.identity_cache.items()
            if tid in active_track_ids
            or (self.frame_index - data.get("last_evaluated_frame", self.frame_index)) <= self.lost_track_buffer * 2
        }

        # 5. Process tracked faces
        tracked_faces: list[TrackedFace] = []
        raw_faces_array = self.last_valid_raw_faces

        for idx in range(len(tracked_detections)):
            track_id = int(tracked_detections.tracker_id[idx])
            t_xyxy = tracked_detections.xyxy[idx]
            det_conf = (
                float(tracked_detections.confidence[idx])
                if tracked_detections.confidence is not None
                else 1.0
            )

            x1, y1, x2, y2 = t_xyxy
            bx = int(max(0, x1))
            by = int(max(0, y1))
            bw = int(max(1, min(fw - bx, x2 - x1)))
            bh = int(max(1, min(fh - by, y2 - y1)))
            bbox_xywh = (bx, by, bw, bh)
            center = (bx + bw / 2.0, by + bh / 2.0)

            # Retrieve track history
            cached_info = self.identity_cache.get(track_id, {})
            prev_center = cached_info.get("prev_center")
            prev_frame = cached_info.get("prev_frame", self.frame_index - 1)
            dt = max(1, self.frame_index - prev_frame)

            # Calculate motion velocity (pixels/frame)
            if prev_center is not None:
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                velocity = float(np.sqrt(dx * dx + dy * dy) / dt)
            else:
                velocity = 0.0

            # Fast motion test strictly based on spatial velocity
            is_fast_motion = (
                (velocity > self.max_velocity_for_recognition and dt <= 2)
                or (bw > 0 and velocity > bw * 0.45 and dt <= 2)
            )

            is_too_small = min(bw, bh) < self.min_face_size_for_recognition
            already_recognized = cached_info.get("is_rec", False)

            needs_eval = (
                not is_fast_motion
                and not is_too_small
                and (
                    not cached_info.get("evaluated", False)
                    or (self.frame_index - cached_info.get("last_evaluated_frame", 0)) >= self.recheck_interval_frames
                )
            )

            matched_raw_face = None
            if len(raw_faces_array) > 0:
                matched_raw_face = self._match_box_to_raw_face(t_xyxy, raw_faces_array)

            if needs_eval and matched_raw_face is not None:
                best_match_user = None
                highest_score = 0.0
                is_rec = False

                crop = recognition_service.align_and_crop(frame, matched_raw_face)
                embeddings = recognition_service.extract_batch_embeddings([crop])

                if embeddings and enrolled_matrix is not None and len(known_users) > 0:
                    emb = embeddings[0]
                    sims = recognition_service.batch_cosine_similarity(emb, enrolled_matrix)
                    if len(sims) > 0:
                        best_idx = int(np.argmax(sims))
                        highest_score = float(sims[best_idx])
                        best_match_user = known_users[best_idx]
                        is_rec = highest_score >= match_threshold and best_match_user is not None

                if not is_rec and already_recognized:
                    # Keep previous high-confidence identity during momentary fluctuations
                    name = cached_info.get("name", "Unknown")
                    highest_score = cached_info.get("score", 0.80)
                    is_rec = True
                    best_match_user = cached_info.get("user_data")
                else:
                    name = best_match_user["name"] if (is_rec and best_match_user) else "Unknown"

                self.identity_cache[track_id] = {
                    "name": name,
                    "score": highest_score,
                    "is_rec": is_rec,
                    "user_data": best_match_user,
                    "last_evaluated_frame": self.frame_index,
                    "evaluated": True,
                    "prev_center": center,
                    "prev_frame": self.frame_index,
                    "velocity": velocity,
                }
                cached_info = self.identity_cache[track_id]
            else:
                cached_info["prev_center"] = center
                cached_info["prev_frame"] = self.frame_index
                cached_info["velocity"] = velocity
                self.identity_cache[track_id] = cached_info

            # PERSISTENT IDENTITY LOCK:
            # Verified identity remains locked with GREEN BORDER (0, 255, 0)
            if already_recognized or cached_info.get("is_rec", False):
                name = cached_info.get("name", "Recognized")
                is_rec = True
                score = cached_info.get("score", 0.80)
                color = (0, 255, 0)  # VIBRANT SOLID GREEN BORDER LOCKED
                user_data = cached_info.get("user_data")
            else:
                name = cached_info.get("name", "Unknown")
                is_rec = False
                score = cached_info.get("score", 0.0)
                user_data = cached_info.get("user_data")
                if is_too_small:
                    color = (255, 200, 0)
                    name = "Tracking..."
                else:
                    color = (0, 0, 255)  # Red border for unverified / unknown

            tracked_faces.append(
                TrackedFace(
                    track_id=track_id,
                    bbox_xywh=bbox_xywh,
                    name=name,
                    confidence=score,
                    is_recognized=is_rec,
                    color_bgr=color,
                    user_data=user_data,
                    landmarks=matched_raw_face,
                    raw_detection_score=det_conf,
                    is_fast_motion=is_fast_motion,
                    velocity=velocity,
                )
            )

        return tracked_faces
