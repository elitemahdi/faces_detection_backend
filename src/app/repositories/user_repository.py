from collections.abc import Sequence
from datetime import datetime, timezone
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class UserRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: int) -> User | None:
        query = select(User).where(
            User.id == user_id, User.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_email_or_phone(
        self, email: str, phone: str, exclude_user_id: int | None = None
    ) -> User | None:
        query = select(User).where(
            or_(User.email == email, User.phone == phone),
            User.deleted_at.is_(None),
        )
        if exclude_user_id is not None:
            query = query.where(User.id != exclude_user_id)

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def create(
            self,
            user_in: UserCreate,
            photo_path: str,
            embedding: list[float] | None = None,
    ) -> User:
        db_user = User(
            full_name=user_in.full_name,
            email=user_in.email,
            phone=user_in.phone,
            role=user_in.role,
            photo_path=photo_path,
            embedding=embedding,
        )
        self.session.add(db_user)
        await self.session.commit()
        await self.session.refresh(db_user)
        return db_user

    async def update(
        self,
        user: User,
        user_in: UserUpdate,
        photo_path: str | None = None,
    ) -> User:
        """Applies non-null fields to the existing user model."""
        update_data = user_in.model_dump(exclude_unset=True)

        if photo_path is not None:
            update_data["photo_path"] = photo_path

        for field, value in update_data.items():
            setattr(user, field, value)

        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def soft_delete(self, user: User) -> None:
        """Sets deleted_at timestamp instead of permanently removing row."""
        user.deleted_at = datetime.now(timezone.utc)
        await self.session.commit()

    async def list_all(self, skip: int = 0, limit: int = 20) -> Sequence[User]:
        query = (
            select(User)
            .where(User.deleted_at.is_(None))
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return result.scalars().all()