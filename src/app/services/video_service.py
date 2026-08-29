import asyncio
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import tempfile
import uuid
import aiofiles
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user_repository import UserRepository
from app.services.face_recognition_service import face_recognition_service

MATCH_THRESHOLD = 0.38
PROCESSED_DIR = Path("uploads/processed_videos")


class VideoProcessingService:

    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)
        self.recognition_service = face_recognition_service
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    async def annotate_video_file(
        self,
        video_file: UploadFile,
        sample_rate: int = 1,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> dict:
        """Non-blocking async video pipeline:
        - Fetches registered user embeddings asynchronously.
        - Streams uploaded video chunks to temp file.
        - Dispatches CPU-bound worker with dedicated thread-safe detector.
        - Properly translates worker exceptions to route HTTPExceptions.
        - Always guarantees temp file cleanup.
        """
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

        original_filename = video_file.filename or "input.mp4"
        suffix = Path(original_filename).suffix or ".mp4"

        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_input_path = temp_in.name
        temp_in.close()

        try:
            async with aiofiles.open(temp_input_path, "wb") as f_out:
                while chunk := await video_file.read(1024 * 1024):
                    await f_out.write(chunk)

            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(
                    None,
                    partial(
                        self._process_video_sync,
                        temp_input_path=temp_input_path,
                        original_filename=original_filename,
                        suffix=suffix,
                        known_users=known_users,
                        sample_rate=sample_rate,
                        match_threshold=match_threshold,
                    ),
                )
                return result
            except ValueError as ve:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Video processing failed: {e}",
                )
        finally:
            Path(temp_input_path).unlink(missing_ok=True)

    def _process_video_sync(
        self,
        temp_input_path: str,
        original_filename: str,
        suffix: str,
        known_users: list[dict],
        sample_rate: int,
        match_threshold: float,
    ) -> dict:
        """Synchronous CPU/GPU-bound video processing executed in a dedicated worker thread."""
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise ValueError("Unable to open or decode video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Dedicated per-thread FaceDetectorYN instance to prevent race conditions / SIGSEGV
        detector = self.recognition_service.get_detector((width, height))

        video_id = uuid.uuid4().hex[:12]
        output_filename = f"annotated_{video_id}{suffix}"
        output_path = PROCESSED_DIR / output_filename

        # Browser-compatible H.264 (avc1) encoding with mp4v fallback
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Precompute L2-normalized matrix of enrolled user embeddings
        if known_users:
            enrolled_matrix = np.array(
                [u["embedding"] for u in known_users], dtype=np.float32
            )
            norms = np.linalg.norm(enrolled_matrix, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            enrolled_matrix = enrolled_matrix / norms
        else:
            enrolled_matrix = None

        frame_idx = 0
        total_detections = 0
        recognized_users_set = set()
        last_detections = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    last_detections = []
                    _, faces = detector.detect(frame)

                    if faces is not None and len(faces) > 0:
                        bboxes = []
                        crops = []
                        for face in faces:
                            bboxes.append(face[:4].astype(int))
                            crops.append(self.recognition_service.align_and_crop(frame, face))

                        # Single batched ONNX evaluation for all faces in the frame
                        embeddings = self.recognition_service.extract_batch_embeddings(crops)

                        for bbox, emb in zip(bboxes, embeddings):
                            total_detections += 1
                            best_match_user = None
                            highest_score = -1.0

                            if enrolled_matrix is not None and len(known_users) > 0:
                                sims = self.recognition_service.batch_cosine_similarity(
                                    emb, enrolled_matrix
                                )
                                best_idx = int(np.argmax(sims))
                                highest_score = float(sims[best_idx])
                                best_match_user = known_users[best_idx]

                            is_rec = (
                                highest_score >= match_threshold
                                and best_match_user is not None
                            )
                            if is_rec:
                                recognized_users_set.add(best_match_user["name"])

                            label = (
                                f"{best_match_user['name']} ({highest_score:.2f})"
                                if is_rec
                                else f"Unknown ({highest_score:.2f})"
                            )
                            color = (0, 255, 0) if is_rec else (0, 0, 255)
                            last_detections.append((bbox, label, color))

                # Draw annotations on the frame
                for bbox, label, color in last_detections:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

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

                writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        file_size = output_path.stat().st_size if output_path.exists() else 0
        duration = frame_idx / fps if fps > 0 else 0.0

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
            "created_at": datetime.now(timezone.utc).isoformat(),
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
