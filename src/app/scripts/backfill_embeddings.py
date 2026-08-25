import asyncio
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
from sqlalchemy import select

from app.core.database import get_db
from app.models.user import User
from app.services.face_recognition_service import FaceRecognitionService


async def backfill():
    print("🚀 [1/4] Initializing Face Recognition Model...", flush=True)
    recognition_service = FaceRecognitionService()

    print("🔌 [2/4] Connecting to Database via get_db()...", flush=True)
    async for session in get_db():
        query = select(User).where(User.deleted_at.is_(None))
        result = await session.execute(query)
        users = result.scalars().all()

        print(
            f"📊 [3/4] Found {len(users)} total user(s) in database.",
            flush=True,
        )

        updated_count = 0
        skipped_count = 0

        for user in users:
            if user.embedding is not None and len(user.embedding) > 0:
                print(
                    f"  ⏭️ User #{user.id} ({user.full_name}): Embedding exists. Skipping.",
                    flush=True,
                )
                skipped_count += 1
                continue

            photo_file = Path(user.photo_path)
            if not photo_file.exists():
                print(
                    f"  ⚠️ User #{user.id} ({user.full_name}): Photo '{user.photo_path}' not found. Skipping.",
                    flush=True,
                )
                skipped_count += 1
                continue

            img = cv2.imread(str(photo_file))
            if img is None:
                print(
                    f"  ❌ User #{user.id} ({user.full_name}): Could not decode image. Skipping.",
                    flush=True,
                )
                skipped_count += 1
                continue

            detections = recognition_service.process_and_embed(img)
            if not detections:
                print(
                    f"  ⚠️ User #{user.id} ({user.full_name}): No face detected. Skipping.",
                    flush=True,
                )
                skipped_count += 1
                continue

            _, embedding_512d = detections[0]
            user.embedding = embedding_512d
            updated_count += 1
            print(
                f"  ✅ User #{user.id} ({user.full_name}): Embedding generated!",
                flush=True,
            )

        if updated_count > 0:
            print("💾 [4/4] Saving updates to PostgreSQL...", flush=True)
            await session.commit()
            print(
                f"✨ Successfully updated {updated_count} user(s)!", flush=True
            )
        else:
            print("✨ [4/4] No users needed updating.", flush=True)

        break


if __name__ == "__main__":
    try:
        asyncio.run(backfill())
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)




# for run the upper script, change the database.py with this code.
# from sqlalchemy.ext.asyncio import (
#     AsyncSession,
#     async_sessionmaker,
#     create_async_engine,
# )
# from sqlalchemy.orm import declarative_base
#
# # Adjust database URL if you load it from config/environment variables
# DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/facedb"
#
# engine = create_async_engine(
#     DATABASE_URL,
#     echo=False,
#     future=True,
# )
#
# # Export async_session_maker here
# async_session_maker = async_sessionmaker(
#     bind=engine,
#     class_=AsyncSession,
#     expire_on_commit=False,
#     autocommit=False,
#     autoflush=False,
# )
#
# Base = declarative_base()
#
#
# async def get_db():
#     async with async_session_maker() as session:
#         try:
#             yield session
#         finally:
#             await session.close()