from pathlib import Path
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
    ) -> Path:
        """Processes an uploaded video, draws bounding boxes and labels for recognized

        and unknown faces on each frame, and writes the output to an MP4 file.
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
        suffix = Path(video_file.filename or "input.mp4").suffix or ".mp4"
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 4. Initialize VideoWriter with H.264 / mp4v codec
        output_filename = f"annotated_{uuid.uuid4().hex}.mp4"
        output_path = PROCESSED_DIR / output_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        frame_idx = 0
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
                    query_vec = np.array(embedding, dtype=np.float32)
                    best_match_user = None
                    highest_score = -1.0

                    for user in known_users:
                        score = self.recognition_service.recognizer.match(
                            query_vec,
                            user["embedding"],
                            cv2.FaceRecognizerSF_FR_COSINE,
                        )
                        if score > highest_score:
                            highest_score = score
                            best_match_user = user

                    is_recognized = (
                        highest_score >= MATCH_THRESHOLD
                        and best_match_user is not None
                    )
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

        return output_path