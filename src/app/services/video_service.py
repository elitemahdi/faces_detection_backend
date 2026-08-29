from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile
import uuid
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user_repository import UserRepository
from app.services.face_recognition_service import FaceRecognitionService

MATCH_THRESHOLD = 0.363
PROCESSED_DIR = Path("uploads/processed_videos")


class VideoProcessingService:

    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)
        self.recognition_service = FaceRecognitionService()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    async def annotate_video_file(
        self,
        video_file: UploadFile,
        sample_rate: int = 1,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> dict:
        """Processes an uploaded video, draws bounding boxes and labels for recognized
        and unknown faces on each frame, writes the output to an MP4 file, and returns
        structured summary metrics.
        """
        # 1. Fetch registered users with embeddings from DB
        users = await self.repository.list_all(skip=0, limit=1000)
        known_users = [
            {
                "id": u.id,
                "name": u.full_name,
                "embedding": np.array(u.embedding, dtype=np.float32),
            }
            for u in users
            if u.embedding is not None
        ]

        # 2. Save uploaded video to a temporary input file
        original_filename = video_file.filename or "input.mp4"
        suffix = Path(original_filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_in:
            content = await video_file.read()
            temp_in.write(content)
            temp_input_path = temp_in.name

        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            Path(temp_input_path).unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to open or decode video file.",
            )

        # 3. Read video metadata
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 4. Initialize VideoWriter with H.264 / mp4v codec
        video_id = uuid.uuid4().hex[:12]
        output_filename = f"annotated_{video_id}{suffix}"
        output_path = PROCESSED_DIR / output_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        frame_idx = 0
        total_detections = 0
        recognized_users_set = set()
        last_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run face detection and recognition on sampled frames
            if frame_idx % sample_rate == 0:
                last_detections = []
                detections = self.recognition_service.process_and_embed(frame)

                for bbox, embedding in detections:
                    total_detections += 1
                    query_vec = np.array(embedding, dtype=np.float32)
                    best_match_user = None
                    highest_score = -1.0

                    for user in known_users:
                        score = self.recognition_service.cosine_similarity(
                            query_vec, user["embedding"]
                        )
                        if score > highest_score:
                            highest_score = score
                            best_match_user = user

                    is_recognized = (
                        highest_score >= match_threshold
                        and best_match_user is not None
                    )
                    if is_recognized:
                        recognized_users_set.add(best_match_user["name"])

                    label = (
                        f"{best_match_user['name']} ({highest_score:.2f})"
                        if is_recognized
                        else f"Unknown ({highest_score:.2f})"
                    )
                    color = (0, 255, 0) if is_recognized else (0, 0, 255)

                    last_detections.append((bbox, label, color))

            # Draw annotations on the frame
            for bbox, label, color in last_detections:
                x, y, w, h = bbox
                # Bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Label background tag
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame, (x, max(0, y - 20)), (x + tw + 6, max(20, y)), color, -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x + 3, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Write annotated frame
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        Path(temp_input_path).unlink(missing_ok=True)

        file_size = output_path.stat().st_size if output_path.exists() else 0
        duration = frame_idx / fps if fps > 0 else 0.0
        now = datetime.now(timezone.utc)

        # Save metadata sidecar json
        metadata = {
            "video_id": video_id,
            "original_filename": original_filename,
            "annotated_filename": output_filename,
            "download_url": f"/api/v1/videos/{video_id}/download",
            "total_frames": frame_idx,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "total_detections": total_detections,
            "recognized_users": sorted(list(recognized_users_set)),
            "file_size_bytes": file_size,
            "created_at": now.isoformat(),
        }

        meta_path = PROCESSED_DIR / f"{video_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "output_path": output_path,
            "metadata": metadata,
        }

    def list_processed_videos(self) -> list[dict]:
        """Lists all processed videos stored in the processed directory."""
        items = []
        for meta_file in sorted(PROCESSED_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items.append(data)
            except Exception:
                pass
        return items

    def get_video_file_path(self, video_id: str) -> Path:
        """Finds the output video file path by video_id."""
        meta_path = PROCESSED_DIR / f"{video_id}.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                fn = data.get("annotated_filename")
                if fn:
                    p = PROCESSED_DIR / fn
                    if p.exists():
                        return p

        # Fallback: search for annotated_{video_id}.*
        matches = list(PROCESSED_DIR.glob(f"annotated_{video_id}*"))
        if matches and matches[0].exists():
            return matches[0]

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processed video #{video_id} not found.",
        )

    def delete_processed_video(self, video_id: str) -> bool:
        """Deletes a processed video and its metadata sidecar."""
        meta_path = PROCESSED_DIR / f"{video_id}.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    fn = data.get("annotated_filename")
                    if fn:
                        (PROCESSED_DIR / fn).unlink(missing_ok=True)
            except Exception:
                pass
            meta_path.unlink(missing_ok=True)

        for p in PROCESSED_DIR.glob(f"annotated_{video_id}*"):
            p.unlink(missing_ok=True)

        return True
