# App/services/user_service.py
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from App.core.LoggingInit import get_core_logger
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
from App.api.dependencies.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
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
    Service Layer: Owns all application business logic for users.
    Delegates database access to UserRepository.
    Raises domain exceptions on business constraint violations so main.py handles them cleanly.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UserRepository(db)

    async def register_user(self, user_data: SignupRequest) -> UserData:
        """Register a new user after verifying uniqueness"""
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
        }

        user = await self.repo.create_user(db_user_data)
        logger.info(f"User registered successfully: {user.username} (ID: {user.id})")
        return UserData.from_orm(user)

    async def login_user(self, login_data: LoginRequest) -> Dict[str, Any]:
        """Authenticate user and issue access token"""
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
        logger.info(f"User logged in successfully: {user.username} (ID: {user.id})")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserData.from_orm(user),
        }

    async def get_user_profile(self, user_id: int) -> UserData:
        """Fetch user profile by ID"""
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise UserNotFoundError("User not found")
        return UserData.from_orm(user)

    async def list_users(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """List active users with pagination"""
        users = await self.repo.get_users(skip=skip, limit=limit)
        total = await self.repo.count_active_users()
        user_list = [UserData.from_orm(u) for u in users]
        return {
            "users": user_list,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    async def update_user_profile(
        self,
        target_user_id: int,
        user_update: UserUpdateRequest,
        acting_user: dict,
    ) -> UserData:
        """Update user information with RBAC validation"""
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
        return UserData.from_orm(user)

    async def remove_user(
        self,
        target_user_id: int,
        acting_user: dict,
        hard_delete: bool = False,
    ) -> bool:
        """Remove or soft delete a user"""
        existing_user = await self.repo.get_by_id_include_deleted(target_user_id)
        if not existing_user:
            raise UserNotFoundError("User not found")

        is_admin = acting_user.get("role") == "admin"
        effective_soft_delete = not (hard_delete and is_admin)

        success = await self.repo.delete_user(target_user_id, soft_delete=effective_soft_delete)
        logger.warning(f"User deleted: ID {target_user_id} by {acting_user.get('id')} (hard_delete={not effective_soft_delete})")
        return success

    async def restore_user(self, user_id: int) -> UserData:
        """Restore soft-deleted user"""
        user = await self.repo.restore_user(user_id)
        if not user:
            raise UserNotFoundError("Deleted user not found")
        logger.info(f"User restored: ID {user_id}")
        return UserData.from_orm(user)

    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user"""
        success = await self.repo.deactivate_user(user_id)
        if not success:
            raise UserNotFoundError("User not found")
        logger.info(f"User deactivated: ID {user_id}")
        return True

    async def activate_user(self, user_id: int) -> bool:
        """Activate user"""
        success = await self.repo.activate_user(user_id)
        if not success:
            raise UserNotFoundError("User not found")
        logger.info(f"User activated: ID {user_id}")
        return True

    async def list_deleted_users(self) -> Dict[str, Any]:
        """List soft-deleted users"""
        users = await self.repo.get_deleted_users()
        user_list = [UserData.from_orm(u) for u in users]
        return {
            "users": user_list,
            "total": len(user_list),
            "skip": 0,
            "limit": len(user_list),
        }

    async def change_password(self, user_id: int, password_data: ChangePasswordRequest) -> bool:
        """Change user password after validating old password"""
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
        """Admin resets user password"""
        user = await self.repo.get_by_id(target_user_id)
        if not user:
            raise UserNotFoundError("User not found")

        new_hash = get_password_hash(new_password_data.new_password)
        await self.repo.reset_password(target_user_id, new_hash)
        logger.info(f"Password reset for user ID: {target_user_id}")
        return True
