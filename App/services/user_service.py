# App/services/user_service.py
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from App.core.LoggingInit import get_core_logger
from App.core.settings import settings
from App.core.exceptions import (
    DomainError,
    UserNotFoundError,
    DuplicateEmailError,
    DuplicateNameError,
    AccountAlreadyDisabledError,
    AccountNotDisabledError,
    IncorrectPasswordError,
    PermissionDeniedError,
)
from App.repository.userRepository import UserRepository
from App.store import token_store
from App.api.dependencies.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_jwt,
    decode_jwt_ignore_expiry,
)
from App.models.userModels import (
    SignupRequest,
    LoginRequest,
    UserUpdateRequest,
    ChangePasswordRequest,
    ResetPasswordRequest,
    UserData,
)

logger = get_core_logger(__name__)


class UserService:
    """
    Service Layer: Owns all application business logic for users & auth sessions.
    Delegates database access to UserRepository and session revocation to Redis token_store.
    Raises domain exceptions cleanly for main.py handlers.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UserRepository(db)

    async def register_user(self, user_data: SignupRequest) -> UserData:
        """Register a new user after verifying uniqueness."""
        if await self.repo.get_by_username(user_data.username):
            raise DuplicateNameError("Username already registered")

        if await self.repo.get_by_email(user_data.email):
            raise DuplicateEmailError("Email already registered")

        hashed_password = get_password_hash(user_data.password)
        db_user_data = {
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "password_hash": hashed_password,
            "user_role": "user",
            "is_active": True,
            "disabled": False,
            "is_admin": False,
            "permissions": {},
        }

        user = await self.repo.create_user(db_user_data)
        logger.info(f"User registered successfully: {user.username} (ID: {user.id})")
        return UserData.model_validate(user)

    async def login_user(self, login_data: LoginRequest) -> Dict[str, Any]:
        """Authenticate user, issue access + rotated refresh token family, and track in Redis."""
        user = await self.repo.get_by_username(login_data.username)
        if not user:
            user = await self.repo.get_by_email(login_data.username)

        if not user or not verify_password(login_data.password, user.password_hash):
            raise IncorrectPasswordError("Incorrect username or password")

        if not user.is_active or user.disabled:
            raise AccountAlreadyDisabledError("Account is disabled")

        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "email": user.email,
            "role": user.user_role,
        }
        access_token = create_access_token(data=token_data)
        refresh_token, rt_jti, family_id, rt_ttl = create_refresh_token(data=token_data)
        expires_in = int(settings.ACCESS_TOKEN_EXPIRE_MINUTES) * 60

        try:
            await token_store.store_refresh(rt_jti, user.id, family_id, rt_ttl)
            await token_store.track_family_for_user(user.id, family_id, rt_ttl)
        except Exception as exc:
            logger.warning(f"Redis token_store unavailable during login (fail-open): {exc}")

        logger.info(f"User logged in successfully: {user.username} (ID: {user.id})")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            "token_type": "bearer",
            "user": UserData.model_validate(user),
        }

    async def refresh_session(
        self,
        refresh_token_value: Optional[str],
        access_token_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate & rotate refresh token with stolen-token family reuse protection."""
        if not refresh_token_value:
            raise IncorrectPasswordError("Refresh token is required")

        payload = decode_jwt(refresh_token_value)
        if not payload or payload.get("type") != "refresh":
            raise IncorrectPasswordError("Invalid or expired refresh token")

        if payload.get("types") == "slts":
            raise DomainError("Short-lived temporary tokens cannot be refreshed")

        user_id = payload.get("user_id")
        rt_jti = payload.get("jti")
        family_id = payload.get("family_id")
        if not user_id or not rt_jti:
            raise IncorrectPasswordError("Malformed refresh token payload")

        # Validate rotation / reuse detection in Redis (when Redis is online)
        try:
            if await token_store.is_refresh_revoked(rt_jti):
                meta = await token_store.get_revoked_refresh_meta(rt_jti)
                fam = (meta or {}).get("family_id") or family_id
                if fam:
                    await token_store.revoke_family(fam, int(settings.REFRESH_TOKEN_TTL_SECONDS))
                    await token_store.forget_family_for_user(int(user_id), fam)
                await token_store.set_user_revocation_cutoff(int(user_id))
                raise IncorrectPasswordError("Refresh token reuse detected. Session family revoked.")

            stored = await token_store.consume_refresh(rt_jti, revoke_ttl=int(settings.REFRESH_TOKEN_TTL_SECONDS))
            if stored is None:
                raise IncorrectPasswordError("Refresh token is no longer active or already used")
            family_id = stored.get("family_id") or family_id
        except IncorrectPasswordError:
            raise
        except Exception as exc:
            logger.warning(f"Redis token_store unavailable during refresh (fail-open): {exc}")

        user = await self.repo.get_by_id(int(user_id))
        if not user:
            raise UserNotFoundError("User not found")
        if not user.is_active or user.disabled:
            raise AccountAlreadyDisabledError("Account is disabled")

        # Blocklist old access token jti if provided
        if access_token_value:
            old_at = decode_jwt_ignore_expiry(access_token_value)
            if old_at and old_at.get("jti") and old_at.get("exp"):
                rem = int(old_at["exp"] - datetime.now(timezone.utc).timestamp())
                if rem > 0:
                    try:
                        await token_store.blocklist_access(old_at["jti"], rem)
                    except Exception:
                        pass

        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "email": user.email,
            "role": user.user_role,
        }
        new_access = create_access_token(data=token_data)
        new_refresh, new_rt_jti, new_fam, rt_ttl = create_refresh_token(data=token_data, family_id=family_id)
        expires_in = int(settings.ACCESS_TOKEN_EXPIRE_MINUTES) * 60

        try:
            await token_store.store_refresh(new_rt_jti, user.id, new_fam, rt_ttl)
            await token_store.track_family_for_user(user.id, new_fam, rt_ttl)
        except Exception as exc:
            logger.warning(f"Redis store failed after refresh rotation (fail-open): {exc}")

        return {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "expires_in": expires_in,
            "token_type": "bearer",
            "user": UserData.model_validate(user),
        }

    async def logout_user(
        self,
        access_token_value: Optional[str] = None,
        refresh_token_value: Optional[str] = None,
        logout_everywhere: bool = False,
    ) -> Dict[str, Any]:
        """Idempotently revoke access token (jti blocklist) & refresh family (or all families)."""
        user_id: Optional[int] = None
        at_payload = decode_jwt_ignore_expiry(access_token_value) if access_token_value else None
        rt_payload = decode_jwt_ignore_expiry(refresh_token_value) if refresh_token_value else None

        # Nothing valid to revoke -> session already ended
        if not at_payload and not rt_payload:
            return {"already_logged_out": True, "logout_everywhere": False}

        # Access token already blocklisted and no refresh token supplied -> already logged out
        if at_payload and not rt_payload and not logout_everywhere and at_payload.get("jti"):
            try:
                if await token_store.is_access_blocked(at_payload["jti"]):
                    return {"already_logged_out": True, "logout_everywhere": False}
            except Exception:
                pass

        if at_payload:
            if True:
                user_id = at_payload.get("user_id")
                at_jti = at_payload.get("jti")
                at_exp = at_payload.get("exp")
                if at_jti and at_exp:
                    rem = int(at_exp - datetime.now(timezone.utc).timestamp())
                    if rem > 0:
                        try:
                            await token_store.blocklist_access(at_jti, rem)
                        except Exception as exc:
                            logger.warning(f"Redis blocklist_access skipped (fail-open): {exc}")

        if rt_payload:
            if True:
                user_id = user_id or rt_payload.get("user_id")
                rt_jti = rt_payload.get("jti")
                fam_id = rt_payload.get("family_id")
                try:
                    if fam_id:
                        await token_store.revoke_family(fam_id, int(settings.REFRESH_TOKEN_TTL_SECONDS))
                        if user_id:
                            await token_store.forget_family_for_user(int(user_id), fam_id)
                    elif rt_jti:
                        await token_store.revoke_refresh(rt_jti, int(settings.REFRESH_TOKEN_TTL_SECONDS))
                except Exception as exc:
                    logger.warning(f"Redis refresh revoke skipped (fail-open): {exc}")

        if logout_everywhere and user_id:
            try:
                families = await token_store.get_user_families(int(user_id))
                for fam in families:
                    await token_store.revoke_family(fam, int(settings.REFRESH_TOKEN_TTL_SECONDS))
                await token_store.clear_user_families(int(user_id))
                await token_store.set_user_revocation_cutoff(int(user_id))
            except Exception as exc:
                logger.warning(f"Redis logout_everywhere skipped (fail-open): {exc}")

        return {"already_logged_out": False, "logout_everywhere": bool(logout_everywhere and user_id)}

    async def get_user_profile(self, user_id: int) -> UserData:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise UserNotFoundError("User not found")
        return UserData.model_validate(user)

    async def list_users(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        users = await self.repo.get_users(skip=skip, limit=limit)
        total = await self.repo.count_active_users()
        user_list = [UserData.model_validate(u) for u in users]
        return {"users": user_list, "total": total, "skip": skip, "limit": limit}

    async def update_user_profile(
        self,
        target_user_id: int,
        user_update: UserUpdateRequest,
        acting_user: dict,
    ) -> UserData:
        user = await self.repo.get_by_id(target_user_id)
        if not user:
            raise UserNotFoundError("User not found")

        update_dict: Dict[str, Any] = {}
        if user_update.user_role is not None:
            if acting_user.get("role") != "admin":
                raise PermissionDeniedError("Only admins can update user role")
            update_dict["user_role"] = user_update.user_role.value

        if user_update.email is not None:
            existing_email = await self.repo.get_by_email(user_update.email)
            if existing_email and existing_email.id != target_user_id:
                raise DuplicateEmailError("Email already in use")
            update_dict["email"] = user_update.email

        if user_update.full_name is not None:
            update_dict["full_name"] = user_update.full_name

        if update_dict:
            user = await self.repo.update_user(target_user_id, update_dict)

        logger.info(f"User profile updated: ID {target_user_id} by {acting_user.get('id')}")
        return UserData.model_validate(user)

    async def remove_user(
        self,
        target_user_id: int,
        acting_user: dict,
        hard_delete: bool = False,
    ) -> bool:
        existing_user = await self.repo.get_by_id_include_deleted(target_user_id)
        if not existing_user:
            raise UserNotFoundError("User not found")

        is_admin = acting_user.get("role") == "admin"
        effective_soft_delete = not (hard_delete and is_admin)
        success = await self.repo.delete_user(target_user_id, soft_delete=effective_soft_delete)
        logger.warning(f"User deleted: ID {target_user_id} by {acting_user.get('id')} (hard_delete={not effective_soft_delete})")
        return success

    async def restore_user(self, user_id: int) -> UserData:
        user = await self.repo.restore_user(user_id)
        if not user:
            raise UserNotFoundError("Deleted user not found")
        logger.info(f"User restored: ID {user_id}")
        return UserData.model_validate(user)

    async def deactivate_user(self, user_id: int) -> bool:
        success = await self.repo.deactivate_user(user_id)
        if not success:
            raise UserNotFoundError("User not found")
        logger.info(f"User deactivated: ID {user_id}")
        return True

    async def activate_user(self, user_id: int) -> bool:
        success = await self.repo.activate_user(user_id)
        if not success:
            raise UserNotFoundError("User not found")
        logger.info(f"User activated: ID {user_id}")
        return True

    async def list_deleted_users(self) -> Dict[str, Any]:
        users = await self.repo.get_deleted_users()
        user_list = [UserData.model_validate(u) for u in users]
        return {"users": user_list, "total": len(user_list), "skip": 0, "limit": len(user_list)}

    async def change_password(self, user_id: int, password_data: ChangePasswordRequest) -> bool:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise UserNotFoundError("User not found")
        if not verify_password(password_data.old_password, user.password_hash):
            raise IncorrectPasswordError("Incorrect current password")
        new_hash = get_password_hash(password_data.new_password)
        await self.repo.change_password(user_id, new_hash)
        logger.info(f"Password changed for user ID: {user_id}")
        return True

    async def reset_password(self, target_user_id: int, new_password_data: ResetPasswordRequest) -> bool:
        user = await self.repo.get_by_id(target_user_id)
        if not user:
            raise UserNotFoundError("User not found")
        new_hash = get_password_hash(new_password_data.new_password)
        await self.repo.reset_password(target_user_id, new_hash)
        logger.info(f"Password reset for user ID: {target_user_id}")
        return True
