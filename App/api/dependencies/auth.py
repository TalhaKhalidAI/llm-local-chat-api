# App/api/dependencies/auth.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import uuid

from argon2 import PasswordHasher, exceptions as argon2_exceptions
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyCookie
from sqlalchemy.ext.asyncio import AsyncSession

from App.core.settings import settings
from App.api.dependencies.sqlite_connector import get_db
from App.repository.userRepository import UserRepository
from App.core.LoggingInit import get_core_logger
from App.store import token_store

logger = get_core_logger(__name__)

pwd_context = PasswordHasher(
    memory_cost=settings.MEMORY_COST,
    parallelism=settings.PARALLELISM,
    hash_len=settings.HASH_LENGTH,
    salt_len=settings.SALT_LENGTH,
)

oauth2_scheme = HTTPBearer(auto_error=False)
bearer_scheme = oauth2_scheme
cookie_scheme = APIKeyCookie(name="CSO", auto_error=False)
refresh_cookie_scheme = APIKeyCookie(name="refresh_token", auto_error=False)


# ========== PASSWORD FUNCTIONS ==========

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using Argon2."""
    try:
        return pwd_context.verify(hash=hashed_password, password=plain_password)
    except argon2_exceptions.VerifyMismatchError:
        return False


def get_password_hash(password: str) -> str:
    """Hash password using Argon2."""
    return pwd_context.hash(password)


def validate_password_strength(password: str) -> bool:
    if len(password) < 8:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?`~" for c in password)
    return has_upper and has_lower and has_digit and has_special


# ========== JWT TOKEN FUNCTIONS ==========

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    is_slt: bool = False,
) -> str:
    """Create JWT access token with unique jti and iat claims."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        mins = settings.SLT_TOKEN_EXPIRE_MINUTES if is_slt else settings.ACCESS_TOKEN_EXPIRE_MINUTES
        expire = now + timedelta(minutes=mins)

    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": str(uuid.uuid4()),
        "type": "access",
    })
    if is_slt:
        to_encode["types"] = "slts"

    return jwt.encode(
        to_encode,
        settings.secret_key_str,
        algorithm=settings.ALGORITHM,
    )


def create_refresh_token(
    data: Dict[str, Any],
    family_id: Optional[str] = None,
) -> Tuple[str, str, str, int]:
    """Create JWT refresh token with jti & token family_id. Returns (token, jti, family_id, ttl_seconds)."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    ttl = int(getattr(settings, "REFRESH_TOKEN_TTL_SECONDS", 7 * 24 * 3600))
    expire = now + timedelta(seconds=ttl)
    jti = str(uuid.uuid4())
    fam = family_id or str(uuid.uuid4())

    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": jti,
        "family_id": fam,
        "type": "refresh",
    })
    encoded = jwt.encode(
        to_encode,
        settings.secret_key_str,
        algorithm=settings.ALGORITHM,
    )
    return encoded, jti, fam, ttl


def decode_jwt(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify JWT token."""
    try:
        return jwt.decode(
            token,
            settings.secret_key_str,
            algorithms=[settings.ALGORITHM],
        )
    except JWTError as e:
        logger.debug(f"JWT decode failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected token decode error: {e}")
        return None


def decode_jwt_ignore_expiry(token: str) -> Optional[Dict[str, Any]]:
    """Decode JWT signature without failing on exp (used by logout)."""
    try:
        return jwt.decode(
            token,
            settings.secret_key_str,
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": False},
        )
    except JWTError:
        return None
    except Exception:
        return None


async def _is_token_revoked_in_redis(jti: Optional[str], user_id: int, iat: Optional[int]) -> bool:
    """Check server-side Redis blocklist & user cutoff timestamp (fail-open if Redis offline)."""
    try:
        if jti and await token_store.is_access_blocked(jti):
            return True
        cutoff = await token_store.get_user_revocation_cutoff(user_id)
        if cutoff is not None and iat is not None and float(iat) < float(cutoff):
            return True
    except Exception as exc:
        logger.warning(f"Redis revocation check skipped (fail-open): {exc}")
    return False


# ========== DEPENDENCY INJECTIONS ==========

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    cookie_token: Optional[str] = Depends(cookie_scheme),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Authenticate user from Bearer Authorization header OR HttpOnly CSO cookie."""
    token = credentials.credentials if credentials else cookie_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_jwt(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") == "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh tokens cannot be used for authentication. Use an access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if await _is_token_revoked_in_redis(payload.get("jti"), int(user_id), payload.get("iat")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    repo = UserRepository(db)
    user = await repo.get_by_id(int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.disabled or not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "name": user.full_name or user.username,
        "role": user.user_role,
        "is_active": user.is_active,
        "disabled": user.disabled,
        "is_admin": user.is_admin or user.user_role == "admin",
        "permissions": user.permissions or {},
        "mfa_enabled": bool(getattr(user, "mfa_enabled", False)),
        "token_jti": payload.get("jti"),
        "is_slt": payload.get("types") == "slts",
    }


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if current_user.get("disabled", False) or not current_user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive or disabled account")
    return current_user


async def get_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
) -> Dict[str, Any]:
    if current_user.get("role") != "admin" and not current_user.get("is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


async def authenticate_user(username: str, password: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
    repo = UserRepository(db)
    user = await repo.get_by_username(username) or await repo.get_by_email(username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    if user.disabled or not user.is_active:
        return None
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "name": user.full_name or user.username,
        "role": user.user_role,
        "is_active": user.is_active,
        "disabled": user.disabled,
    }


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    cookie_token: Optional[str] = Depends(cookie_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[Dict[str, Any]]:
    if not credentials and not cookie_token:
        return None
    try:
        return await get_current_user(credentials=credentials, cookie_token=cookie_token, db=db)
    except HTTPException:
        return None
