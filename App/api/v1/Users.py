from fastapi import APIRouter, Body, Depends, Response, status, Query
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from App.core.settings import settings
from App.api.dependencies.sqlite_connector import get_db
from App.api.dependencies.auth import (
    get_current_user,
    get_admin_user,
    bearer_scheme,
    cookie_scheme,
    refresh_cookie_scheme,
)
from App.services.user_service import UserService
from App.models.userModels import (
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    LogoutRequest,
    SignupRequest,
    SignupResponse,
    UserData,
    UserUpdateRequest,
    ChangePasswordRequest,
    ResetPasswordRequest,
    MessageResponse,
    UserListResponse,
)

router = APIRouter(prefix="/users", tags=["Users"])


def _set_auth_cookies(response: Response, access_token: str, refresh_token: Optional[str], expires_in: int) -> None:
    """Attach HttpOnly SameSite=Strict CSO & refresh_token cookies."""
    response.set_cookie(
        key="CSO",
        value=access_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="strict",
        max_age=expires_in,
        path="/",
    )
    if refresh_token:
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            secure=settings.COOKIE_SECURE,
            samesite="strict",
            max_age=settings.REFRESH_TOKEN_TTL_SECONDS,
            path="/app/v1/users/refresh",
        )


# ==================== AUTH ENDPOINTS ====================

@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: SignupRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user."""
    service = UserService(db)
    created_user = await service.register_user(user_data)
    return SignupResponse(message="User created successfully", user=created_user)


@router.post("/login", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    response: Response,
    cookie_login: bool = Query(
        default=False,
        description="If true, also sets HttpOnly CSO (access) & refresh_token cookies",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Authenticate user with optional cookie_login mode + Redis refresh token family."""
    service = UserService(db)
    result = await service.login_user(login_data)
    if cookie_login:
        _set_auth_cookies(
            response,
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            expires_in=result["expires_in"],
        )
    return LoginResponse(**result)


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    response: Response,
    payload: Optional[RefreshTokenRequest] = Body(default=None),
    cookie_login: bool = Query(default=False, description="Also update HttpOnly cookies"),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    cookie_at: Optional[str] = Depends(cookie_scheme),
    cookie_rt: Optional[str] = Depends(refresh_cookie_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Rotate refresh token & issue a fresh access token (supports JSON body or HttpOnly cookie)."""
    rt_val = (payload.refresh_token if payload and payload.refresh_token else None) or cookie_rt
    at_val = (bearer.credentials if bearer else None) or cookie_at
    service = UserService(db)
    result = await service.refresh_session(refresh_token_value=rt_val, access_token_value=at_val)
    if cookie_login or cookie_rt:
        _set_auth_cookies(
            response,
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            expires_in=result["expires_in"],
        )
    return LoginResponse(**result)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    payload: Optional[LogoutRequest] = Body(default=None),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    cookie_at: Optional[str] = Depends(cookie_scheme),
    cookie_rt: Optional[str] = Depends(refresh_cookie_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Revoke access token (Redis jti blocklist) & refresh family, and clear auth cookies."""
    at_val = (bearer.credentials if bearer else None) or cookie_at
    rt_val = (payload.refresh_token if payload and payload.refresh_token else None) or cookie_rt
    revoke_all = bool(payload and payload.logout_everywhere)

    service = UserService(db)
    result = await service.logout_user(
        access_token_value=at_val,
        refresh_token_value=rt_val,
        logout_everywhere=revoke_all,
    )
    response.delete_cookie(key="CSO", path="/")
    response.delete_cookie(key="refresh_token", path="/app/v1/users/refresh")
    if result.get("already_logged_out"):
        return MessageResponse(message="You are already logged out")
    if result.get("logout_everywhere"):
        return MessageResponse(message="Logged out from all devices successfully")
    return MessageResponse(message="Logged out successfully")


# ==================== PROFILE & ACCOUNT ENDPOINTS ====================

@router.get("/me", response_model=UserData)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current authenticated user's profile."""
    service = UserService(db)
    return await service.get_user_profile(current_user["id"])


@router.get("/", response_model=UserListResponse)
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all active users (admin only)."""
    service = UserService(db)
    return await service.list_users(skip=skip, limit=limit)


@router.get("/deleted/list", response_model=UserListResponse)
async def get_deleted_users(
    current_user: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all soft-deleted users (admin only)."""
    service = UserService(db)
    return await service.list_deleted_users()


@router.get("/{user_id}", response_model=UserData)
async def get_user_by_id(
    user_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get user by ID - normal users get own profile, admins can access any profile."""
    service = UserService(db)
    target_id = current_user["id"] if current_user.get("role") != "admin" else user_id
    return await service.get_user_profile(target_id)


@router.put("/{user_id}", response_model=UserData)
async def update_user(
    user_id: int,
    user_update: UserUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user profile."""
    service = UserService(db)
    target_id = current_user["id"] if current_user.get("role") != "admin" else user_id
    return await service.update_user_profile(target_id, user_update, current_user)


@router.delete("/{user_id}", response_model=MessageResponse)
async def delete_user(
    user_id: int,
    hard_delete: bool = Query(False, description="Hard delete if true (admin only)"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete user - soft delete by default, hard delete if admin specifies."""
    service = UserService(db)
    target_id = current_user["id"] if current_user.get("role") != "admin" else user_id
    success = await service.remove_user(target_id, current_user, hard_delete=hard_delete)
    return MessageResponse(message="User deleted successfully" if success else "User not found")


@router.post("/{user_id}/restore", response_model=UserData)
async def restore_user(
    user_id: int,
    current_user: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Restore a soft-deleted user (admin only)."""
    service = UserService(db)
    return await service.restore_user(user_id)


@router.post("/{user_id}/deactivate", response_model=MessageResponse)
async def deactivate_user(
    user_id: int,
    current_user: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a user account (admin only)."""
    service = UserService(db)
    await service.deactivate_user(user_id)
    return MessageResponse(message="User deactivated successfully")


@router.post("/{user_id}/activate", response_model=MessageResponse)
async def activate_user(
    user_id: int,
    current_user: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Activate a user account (admin only)."""
    service = UserService(db)
    await service.activate_user(user_id)
    return MessageResponse(message="User activated successfully")


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change password for current user."""
    service = UserService(db)
    await service.change_password(current_user["id"], password_data)
    return MessageResponse(message="Password changed successfully")


@router.post("/{user_id}/reset-password", response_model=MessageResponse)
async def reset_password(
    user_id: int,
    password_data: ResetPasswordRequest,
    current_user: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Reset a user's password (admin only)."""
    service = UserService(db)
    await service.reset_password(user_id, password_data)
    return MessageResponse(message="Password reset successfully")
