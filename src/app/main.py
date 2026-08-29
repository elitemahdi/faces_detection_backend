from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import api_router
from app.core.database import Base, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(title="Face Recognition Microservice", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "Face Recognition Microservice"}