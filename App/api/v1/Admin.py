# App/api/v1/Admin.py
"""
Unique Admin & Granular RBAC Permission endpoints (zero overlap with Users.py).

Routes:
    POST   /admin/account/temp_token/{user_id} -- issue short-lived temporary token (SLT)
    GET    /admin/get_all_permissions          -- list granular RBAC permission catalog
    POST   /admin/set_permissions              -- assign permissions to a user
    POST   /admin/set_permissions_bulk         -- bulk assign permissions to multiple users
    GET    /admin/users/permissions            -- inspect a user's permissions
    DELETE /admin/remove_permissions           -- revoke specific or all user permissions
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from App.api.dependencies.sqlite_connector import get_db
from App.api.dependencies.auth import get_admin_user
from App.models.PermissionModel import (
    PermissionModel,
    BulkPermissionModel,
    RemovePermissionsModel,
)
from App.services.admin_service import AdminService

admin_router = APIRouter(prefix="/admin", tags=["Admin"])


@admin_router.post(
    "/account/temp_token/{user_id}",
    status_code=status.HTTP_200_OK,
    summary="Issue short-lived temporary token (SLT)",
)
async def temp_token_maker(
    user_id: int,
    expires_minutes: Optional[int] = Query(default=None, ge=1, le=60, description="Validity in minutes (default 10)"),
    current_admin: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a short-lived temporary token for account recovery or password updates (admin only)."""
    service = AdminService(db)
    return await service.create_temp_token(user_id=user_id, expires_minutes=expires_minutes)


@admin_router.get(
    "/get_all_permissions",
    status_code=status.HTTP_200_OK,
    summary="List all available RBAC permissions",
)
async def get_all_permissions(
    current_admin: dict = Depends(get_admin_user),
):
    """List the full catalog of supported granular permissions (admin only)."""
    return AdminService.get_all_permissions()


@admin_router.post(
    "/set_permissions",
    status_code=status.HTTP_200_OK,
    summary="Set permissions for a single user",
)
async def set_permissions(
    payload: PermissionModel,
    current_admin: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Assign or update granular permissions for a user (admin only)."""
    service = AdminService(db)
    return await service.set_user_permissions(payload)


@admin_router.post(
    "/set_permissions_bulk",
    status_code=status.HTTP_200_OK,
    summary="Bulk set permissions for multiple users",
)
async def set_permissions_bulk(
    payload: BulkPermissionModel,
    current_admin: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Bulk assign granular permissions for multiple users (admin only)."""
    service = AdminService(db)
    return await service.set_bulk_permissions(payload)


@admin_router.get(
    "/users/permissions",
    status_code=status.HTTP_200_OK,
    summary="Get permissions assigned to a user",
)
async def get_users_permissions(
    user_id: int = Query(..., gt=0, description="Target user ID"),
    current_admin: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Inspect granular permissions currently assigned to a user (admin only)."""
    service = AdminService(db)
    return await service.get_user_permissions(user_id=user_id)


@admin_router.delete(
    "/remove_permissions",
    status_code=status.HTTP_200_OK,
    summary="Remove permissions from a user",
)
async def remove_permissions(
    payload: RemovePermissionsModel,
    current_admin: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove specific permission keys or clear all permissions for a user (admin only)."""
    service = AdminService(db)
    return await service.remove_user_permissions(payload)
