# App/services/admin_service.py
from datetime import timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from App.core.LoggingInit import get_core_logger
from App.core.settings import settings
from App.core.exceptions import DomainError, UserNotFoundError
from App.repository.userRepository import UserRepository
from App.api.dependencies.auth import create_access_token
from App.models.Permissions import Permission
from App.models.PermissionModel import (
    PermissionModel,
    BulkPermissionModel,
    RemovePermissionsModel,
)

logger = get_core_logger(__name__)


class AdminService:
    """
    Service Layer for unique Admin operations:
      - Short-lived temporary tokens (SLT)
      - Granular RBAC permission catalog & assignment
    No HTTPExceptions inside service — raises domain exceptions for main.py handlers.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UserRepository(db)

    async def create_temp_token(
        self,
        user_id: int,
        expires_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Issue a short-lived temporary token (SLT) for password reset or account recovery."""
        user = await self.repo.get_by_id_include_deleted(user_id)
        if not user:
            raise UserNotFoundError(f"User ID {user_id} not found")

        mins = max(1, min(int(expires_minutes or settings.SLT_TOKEN_EXPIRE_MINUTES), 60))
        token = create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "email": user.email,
                "role": user.user_role,
            },
            expires_delta=timedelta(minutes=mins),
            is_slt=True,
        )
        logger.info(f"Issued temporary SLT token for user ID {user_id} ({mins} min)")
        return {
            "temp_token": token,
            "token_type": "bearer",
            "expires_in_minutes": mins,
            "user_id": user.id,
            "username": user.username,
        }

    @staticmethod
    def get_all_permissions() -> Dict[str, Any]:
        """Return full catalog of available granular RBAC permissions."""
        perms = [{"name": p.name, "value": p.value} for p in Permission]
        return {"total": len(perms), "permissions": perms}

    async def set_user_permissions(self, payload: PermissionModel) -> Dict[str, Any]:
        """Merge/set granular permissions for a single user."""
        user = await self.repo.get_by_id(payload.user_id)
        if not user:
            raise UserNotFoundError(f"User ID {payload.user_id} not found")

        merged = dict(user.permissions or {})
        merged.update(payload.permissions)
        updated = await self.repo.update_permissions(payload.user_id, merged)
        return {
            "user_id": updated.id,
            "user_email": updated.email,
            "user_name": updated.username,
            "permissions": updated.permissions or {},
        }

    async def set_bulk_permissions(self, payload: BulkPermissionModel) -> Dict[str, Any]:
        """Bulk assign granular permissions for multiple users."""
        results: List[Dict[str, Any]] = []
        not_found: List[int] = []
        for item in payload.all_user_permissions:
            user = await self.repo.get_by_id(item.user_id)
            if not user:
                not_found.append(item.user_id)
                continue
            merged = dict(user.permissions or {})
            merged.update(item.permissions)
            updated = await self.repo.update_permissions(item.user_id, merged)
            results.append({
                "user_id": updated.id,
                "user_email": updated.email,
                "user_name": updated.username,
                "permissions": updated.permissions or {},
            })
        return {"updated": results, "not_found_user_ids": not_found}

    async def get_user_permissions(self, user_id: int) -> Dict[str, Any]:
        """Retrieve assigned permissions for a specific user."""
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User ID {user_id} not found")
        return {
            "user_id": user.id,
            "user_email": user.email,
            "user_name": user.username,
            "permissions": user.permissions or {},
            "role": user.user_role,
        }

    async def remove_user_permissions(self, payload: RemovePermissionsModel) -> Dict[str, Any]:
        """Remove specific permission keys or clear all permissions for a user."""
        user = await self.repo.get_by_id(payload.user_id)
        if not user:
            raise UserNotFoundError(f"User ID {payload.user_id} not found")

        current_perms = dict(user.permissions or {})
        if payload.remove_all:
            new_perms: Dict[str, bool] = {}
        else:
            if not payload.permission_keys:
                raise DomainError("Provide permission_keys or set remove_all=true")
            new_perms = {k: v for k, v in current_perms.items() if k not in set(payload.permission_keys)}

        updated = await self.repo.update_permissions(payload.user_id, new_perms)
        return {
            "user_id": updated.id,
            "user_email": updated.email,
            "user_name": updated.username,
            "permissions": updated.permissions or {},
        }
