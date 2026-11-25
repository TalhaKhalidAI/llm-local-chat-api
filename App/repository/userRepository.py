# App/repository/userRepository.py
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update, delete
from App.api.databases.Users import User


class UserRepository:
    """
    Data Access Layer (Repository).
    Pure database queries with no HTTP/FastAPI exceptions or business logic.
    All methods are asynchronous and operate directly on the database session.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        stmt = select(User).where(and_(User.id == user_id, User.is_deleted == False))
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_id_include_deleted(self, user_id: int) -> Optional[User]:
        """Get user by ID including deleted"""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        stmt = select(User).where(and_(User.username == username, User.is_deleted == False))
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        stmt = select(User).where(and_(User.email == email, User.is_deleted == False))
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get active users with pagination"""
        stmt = select(User).where(User.is_deleted == False).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def count_active_users(self) -> int:
        """Get total count of active users"""
        stmt = select(func.count(User.id)).where(User.is_deleted == False)
        result = await self.db.execute(stmt)
        return result.scalar() or 0

    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create and persist a new user"""
        db_user = User(**user_data)
        self.db.add(db_user)
        await self.db.flush()
        await self.db.refresh(db_user)
        return db_user

    async def update_user(self, user_id: int, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user columns"""
        db_user = await self.get_by_id(user_id)
        if db_user:
            for key, value in update_data.items():
                if hasattr(db_user, key) and key not in ['id', 'password_hash']:
                    setattr(db_user, key, value)
            await self.db.flush()
            await self.db.refresh(db_user)
        return db_user

    async def delete_user(self, user_id: int, soft_delete: bool = True) -> bool:
        """Soft or hard delete user"""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        db_user = result.scalars().first()
        if not db_user:
            return False

        if soft_delete:
            db_user.is_deleted = True
            db_user.deleted_at = datetime.now(timezone.utc)
            db_user.is_active = False
            await self.db.flush()
        else:
            await self.db.delete(db_user)
            await self.db.flush()
        return True

    async def restore_user(self, user_id: int) -> Optional[User]:
        """Restore a soft-deleted user"""
        stmt = select(User).where(and_(User.id == user_id, User.is_deleted == True))
        result = await self.db.execute(stmt)
        db_user = result.scalars().first()
        if db_user:
            db_user.is_deleted = False
            db_user.deleted_at = None
            db_user.is_active = True
            await self.db.flush()
            await self.db.refresh(db_user)
        return db_user

    async def change_password(self, user_id: int, new_password_hash: str) -> bool:
        """Update user password hash"""
        db_user = await self.get_by_id(user_id)
        if not db_user:
            return False
        db_user.password_hash = new_password_hash
        await self.db.flush()
        return True

    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user"""
        db_user = await self.get_by_id(user_id)
        if not db_user:
            return False
        db_user.disabled = True
        db_user.is_active = False
        await self.db.flush()
        return True

    async def activate_user(self, user_id: int) -> bool:
        """Activate user"""
        db_user = await self.get_by_id(user_id)
        if not db_user:
            return False
        db_user.disabled = False
        db_user.is_active = True
        await self.db.flush()
        return True

    async def get_deleted_users(self) -> List[User]:
        """Get all soft-deleted users"""
        stmt = select(User).where(User.is_deleted == True)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def reset_password(self, user_id: int, new_password_hash: str) -> bool:
        """Reset password hash"""
        return await self.change_password(user_id, new_password_hash)
