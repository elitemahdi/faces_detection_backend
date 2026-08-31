from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# Explicitly type as dict[str, Any] so it accepts bools, ints, etc.
engine_kwargs: dict[str, Any] = {
    "echo": False,
    "pool_pre_ping": True,
}

if "sqlite" not in settings.database_url.lower():
    engine_kwargs.update(
        {
            "pool_size": 20,
            "max_overflow": 10,
            "pool_recycle": 3600,
        }
    )

engine = create_async_engine(settings.database_url, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing an async database session per request."""
    async with AsyncSessionLocal() as session:
        yield session


async def close_db_connection() -> None:
    """Disposes all pooled connections on application shutdown."""
    await engine.dispose()
