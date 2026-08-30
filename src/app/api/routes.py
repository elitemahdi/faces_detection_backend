from fastapi import APIRouter

from app.api.v1.camera_routes import router as camera_router
from app.api.v1.user_routes import router as user_router
from app.api.v1.video_routes import router as video_router
from app.api.v1.zone_routes import router as zone_router

api_router = APIRouter()
api_router.include_router(user_router)
api_router.include_router(video_router)
api_router.include_router(camera_router)
api_router.include_router(zone_router)
