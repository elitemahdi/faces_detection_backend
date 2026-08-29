from pathlib import Path
import uuid
import aiofiles
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.services.face_recognition_service import face_recognition_service

UPLOAD_DIR = Path("uploads/avatars")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


class UserService:

    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)
        self.recognition_service = face_recognition_service
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    async def _process_photo(
        self, photo: UploadFile
    ) -> tuple[str, list[float]]:
        """Validates the uploaded photo, verifies face presence and quality gate,
        extracts the 512-dim embedding, and saves the file to disk.
        """
        file_ext = Path(photo.filename or "").suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        content = await photo.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds the 5MB limit.",
            )

        # Decode image in memory for OpenCV processing
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Corrupt or unreadable image file.",
            )

        # Production Quality Gate: Confidence >= 0.90, Face >= 100x100px, exactly 1 face
        try:
            _, embedding_512d, score = self.recognition_service.enroll_face(
                img, min_confidence=0.90, min_face_size=100
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        # Save photo file to disk
        file_name = f"{uuid.uuid4()}{file_ext}"
        destination = UPLOAD_DIR / file_name

        async with aiofiles.open(destination, "wb") as f:
            await f.write(content)

        return str(destination), embedding_512d

    async def register_user(
        self, user_in: UserCreate, photo: UploadFile
    ) -> UserResponse:
        """Checks uniqueness, processes image & embedding, and creates the user."""
        existing = await self.repository.get_by_email_or_phone(
            email=user_in.email, phone=user_in.phone
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email or phone number already exists.",
            )

        photo_path, embedding = await self._process_photo(photo)
        db_user = await self.repository.create(
            user_in=user_in,
            photo_path=photo_path,
            embedding=embedding,
        )
        return UserResponse.model_validate(db_user)

    async def get_user(self, user_id: int) -> UserResponse:
        """Retrieves a single user by ID."""
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )
        return UserResponse.model_validate(user)

    async def list_users(
        self, skip: int = 0, limit: int = 20
    ) -> list[UserResponse]:
        """Lists active users with pagination."""
        users = await self.repository.list_all(skip=skip, limit=limit)
        return [UserResponse.model_validate(user) for user in users]

    async def update_user(
        self,
        user_id: int,
        user_in: UserUpdate,
        photo: UploadFile | None = None,
    ) -> UserResponse:
        """Updates user details and regenerates embedding if a new photo is provided."""
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )

        # Check for conflicts if email or phone is updated
        if user_in.email or user_in.phone:
            conflict = await self.repository.get_by_email_or_phone(
                email=user_in.email or user.email,
                phone=user_in.phone or user.phone,
                exclude_user_id=user_id,
            )
            if conflict:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Another user already has this email or phone number.",
                )

        photo_path = None
        if photo is not None:
            photo_path, new_embedding = await self._process_photo(photo)
            user.embedding = new_embedding

        updated_user = await self.repository.update(
            user=user, user_in=user_in, photo_path=photo_path
        )
        return UserResponse.model_validate(updated_user)

    async def delete_user(self, user_id: int) -> None:
        """Soft-deletes a user."""
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )
        await self.repository.soft_delete(user)
