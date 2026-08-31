from dataclasses import dataclass

import numpy as np
import supervision as sv


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


class FaceTrackerManager:
    """Multi-Object Face Tracker (MOT) utilizing ByteTrack and persistent identity caching.

    Prevents running expensive deep facial embedding extraction (AdaFace) on every single frame.
    Extracts identity once per track (or periodically every N frames) and maintains Kalman
    filtered state across occlusions.
    """

    def __init__(
            self,
            fps: int = 30,
            track_activation_threshold: float = 0.40,
            lost_track_buffer: int = 30,
            minimum_matching_threshold: float = 0.70,
            recheck_interval_frames: int = 30,
    ):
        self.fps = max(1, fps)
        self.recheck_interval_frames = max(1, recheck_interval_frames)
        self.lost_track_buffer = lost_track_buffer

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=self.fps,
        )

        # Maps track_id -> {"name": str, "score": float, "is_rec": bool, "user_data": dict | None, "last_evaluated_frame": int}
        self.identity_cache: dict[int, dict] = {}
        self.frame_index: int = 0

    @staticmethod
    def _compute_iou(boxA: np.ndarray | tuple | list, boxB: np.ndarray | tuple | list) -> float:
        """Computes Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
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
        """Finds the corresponding raw YuNet face (with 5-point landmarks) matching the tracked box via IoU."""
        best_iou = 0.0
        best_face = None

        for face in raw_faces:
            x, y, w, h = face[:4]
            raw_xyxy = np.array([x, y, x + w, y + h])
            iou = self._compute_iou(tracked_xyxy, raw_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_face = face

        # Require a minimum IoU overlap of 0.20 to consider it a valid landmark source
        if best_iou >= 0.20:
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
        """Updates ByteTrack with detections and resolves persistent face identities."""
        self.frame_index += 1
        fh, fw = frame.shape[:2]

        # 1. Filter valid detections by YuNet score
        valid_faces = []
        if raw_faces is not None and len(raw_faces) > 0:
            for f in raw_faces:
                det_score = float(f[14]) if len(f) > 14 else 1.0
                if det_score >= 0.40:  # Minimum detection threshold
                    valid_faces.append(f)

        # 2. Build supervision Detections in XYXY format with boundary clipping
        if valid_faces:
            xyxy_boxes = []
            confidences = []
            for f in valid_faces:
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

        # 4. Clean up stale tracks from identity cache (not seen in lost_track_buffer frames)
        self.identity_cache = {
            tid: data
            for tid, data in self.identity_cache.items()
            if tid in active_track_ids
               or (self.frame_index - data.get("last_evaluated_frame", self.frame_index)) <= self.lost_track_buffer * 2
        }

        # 5. Process tracked faces and evaluate deep embeddings if needed
        tracked_faces: list[TrackedFace] = []
        raw_faces_array = np.array(valid_faces) if valid_faces else np.empty((0, 15))

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

            # Check if this track needs identity evaluation
            cached_info = self.identity_cache.get(track_id)
            needs_eval = (
                    cached_info is None
                    or (self.frame_index - cached_info["last_evaluated_frame"]) >= self.recheck_interval_frames
            )

            matched_raw_face = None
            if len(raw_faces_array) > 0:
                matched_raw_face = self._match_box_to_raw_face(t_xyxy, raw_faces_array)

            if needs_eval:
                best_match_user = None
                highest_score = 0.0
                is_rec = False

                if matched_raw_face is not None:
                    # Align and crop using 5-point facial landmarks
                    crop = recognition_service.align_and_crop(frame, matched_raw_face)
                    embeddings = recognition_service.extract_batch_embeddings([crop])

                    if embeddings and enrolled_matrix is not None and len(known_users) > 0:
                        emb = embeddings[0]
                        sims = recognition_service.batch_cosine_similarity(emb, enrolled_matrix)
                        best_idx = int(np.argmax(sims))
                        highest_score = float(sims[best_idx])
                        best_match_user = known_users[best_idx]
                        is_rec = highest_score >= match_threshold and best_match_user is not None

                name = best_match_user["name"] if (is_rec and best_match_user) else "Unknown"

                # If face couldn't be matched in this frame (e.g. occlusion) but was previously recognized, retain old identity
                if matched_raw_face is None and cached_info is not None:
                    name = cached_info["name"]
                    highest_score = cached_info["score"]
                    is_rec = cached_info["is_rec"]
                    best_match_user = cached_info["user_data"]

                self.identity_cache[track_id] = {
                    "name": name,
                    "score": highest_score,
                    "is_rec": is_rec,
                    "user_data": best_match_user,
                    "last_evaluated_frame": self.frame_index,
                }
                cached_info = self.identity_cache[track_id]

            name = cached_info["name"]
            score = cached_info["score"]
            is_rec = cached_info["is_rec"]
            user_data = cached_info["user_data"]
            color = (0, 255, 0) if is_rec else (0, 0, 255)

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
                )
            )

        return tracked_faces
