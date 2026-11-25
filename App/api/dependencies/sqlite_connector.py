# App/api/dependencies/sqlite_connector.py
import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import event, text
from App.core.settings import settings

# Determine async sqlite URL
raw_url = getattr(settings, "SQLITE_DATABASE_URL", "sqlite:///./app.db")
if raw_url.startswith("sqlite:///") and not raw_url.startswith("sqlite+aiosqlite:///"):
    async_db_url = raw_url.replace("sqlite:///", "sqlite+aiosqlite:///")
elif not raw_url.startswith("sqlite+aiosqlite:///"):
    async_db_url = f"sqlite+aiosqlite:///{raw_url}"
else:
    async_db_url = raw_url

# Password / passphrase for SQLite encryption (e.g. SQLCipher PRAGMA key)
db_password = getattr(settings, "SQLITE_DATABASE_PASSWORD", None) or os.getenv("SQLITE_DATABASE_PASSWORD", "")

engine = create_async_engine(
    async_db_url,
    poolclass=NullPool,
    connect_args={"check_same_thread": False},
    echo=False,
)

# Apply PRAGMA key on connection if password is configured
if db_password:
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_key(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA key = '{db_password}';")
        except Exception:
            pass
        finally:
            cursor.close()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# Backwards compatibility alias
SessionLocal = AsyncSessionLocal
Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
