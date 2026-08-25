from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.user import UserCreate, UserResponse, UserUpdate, UserRole
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    full_name: str = Form(..., min_length=2, max_length=100),
    email: EmailStr = Form(...),
    phone: str = Form(..., min_length=7, max_length=20),
    role: UserRole = Form(UserRole.USER),
    photo: UploadFile = File(...),
    service: UserService = Depends(get_user_service),
):
    user_in = UserCreate(
        full_name=full_name, email=email, phone=phone, role=role
    )
    return await service.register_user(user_in=user_in, photo=photo)

@router.get(
    "/",
    response_model=list[UserResponse],
    status_code=status.HTTP_200_OK,
)
async def get_all_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    service: UserService = Depends(get_user_service),
):
    return await service.list_users(skip=skip, limit=limit)


@router.get(
    "/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK
)
async def get_user_by_id(
    user_id: int,
    service: UserService = Depends(get_user_service),
):
    return await service.get_user(user_id=user_id)


@router.put(
    "/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK
)
async def update_user(
    user_id: int,
    full_name: str | None = Form(None, min_length=2, max_length=100),
    email: EmailStr | None = Form(None),
    phone: str | None = Form(None, min_length=7, max_length=20),
    photo: UploadFile | None = File(None),
    service: UserService = Depends(get_user_service),
):
    user_in = UserUpdate(full_name=full_name, email=email, phone=phone)
    return await service.update_user(
        user_id=user_id, user_in=user_in, photo=photo
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
):
    await service.delete_user(user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)