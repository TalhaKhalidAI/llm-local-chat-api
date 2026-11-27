from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, text
from datetime import datetime
from typing import List, Optional
from App.api.dependencies.sqlite_connector import Base, engine


# User Model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    password_hash = Column(String)
    user_role = Column(String, default="user")  # user, admin
    is_active = Column(Boolean, default=True)
    disabled = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)

    # Granular RBAC Permissions & MFA columns
    permissions = Column(JSON, default=dict, nullable=False)
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String, nullable=True)
    mfa_method = Column(String, default="totp", nullable=True)

    # Soft Delete attributes
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)


async def init_db():
    """Asynchronously initialize database schema and safely add new columns to existing SQLite tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        res = await conn.execute(text("PRAGMA table_info(users)"))
        existing_cols = {row[1] for row in res.fetchall()}
        migrations = [
            ("permissions", "ALTER TABLE users ADD COLUMN permissions JSON DEFAULT '{}'"),
            ("mfa_enabled", "ALTER TABLE users ADD COLUMN mfa_enabled BOOLEAN DEFAULT 0"),
            ("mfa_secret", "ALTER TABLE users ADD COLUMN mfa_secret VARCHAR"),
            ("mfa_method", "ALTER TABLE users ADD COLUMN mfa_method VARCHAR DEFAULT 'totp'"),
        ]
        for col_name, ddl in migrations:
            if col_name not in existing_cols:
                await conn.execute(text(ddl))
