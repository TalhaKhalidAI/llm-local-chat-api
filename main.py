from contextlib import asynccontextmanager
import asyncio
import os
from fastapi import FastAPI, HTTPException, status, Request, applications
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from App.api.v1 import v1_router

from App.core.RedisConnector import redis_client
from App.middleware.rate_limit_middleware import GlobalRateLimitMiddleware
from App.middleware.kill_switch_middleware import KillSwitchMiddleware
from App.middleware.body_size_middleware import BodySizeLimitMiddleware
from App.middleware.request_id_middleware import RequestIDMiddleware
from App.middleware.gzip_request_middleware import GZipRequestMiddleware
from App.middleware.gzip_response_middleware import GZipResponseMiddleware
from App.middleware.idempotency_middleware import IdempotencyMiddleware

from App.core.settings import settings
from App.core.LoggingInit import get_core_logger
from App.core.exceptions import AppError
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import fastapi_offline_swagger_ui
from pathlib import Path
# Initialize Logger
logger = get_core_logger(__name__)

# Initialize Limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT] if settings.RATE_LIMIT_DEFAULT else ["100/minute"]
)

async def create_default_admin():
    """Create default admin user on startup if not exists asynchronously"""
    from App.api.dependencies.sqlite_connector import AsyncSessionLocal
    from App.api.databases.Users import init_db
    from App.repository.userRepository import UserRepository
    from App.api.dependencies.auth import get_password_hash
    
    # Ensure tables are created asynchronously
    try:
        await init_db()
    except Exception as e:
        logger.error(f"Database schema initialization error: {e}")

    async with AsyncSessionLocal() as session:
        try:
            repo = UserRepository(session)
            admin_username = os.getenv("ADMIN_USERNAME", "admin")
            admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
            admin_password = os.getenv("ADMIN_PASSWORD", "Admin@123")
            
            # Check if admin exists
            existing_admin = await repo.get_by_username(admin_username)
            if not existing_admin:
                admin_data = {
                    "username": admin_username,
                    "email": admin_email,
                    "full_name": "System Administrator",
                    "password_hash": get_password_hash(admin_password),
                    "user_role": "admin",
                    "is_active": True,
                    "disabled": False,
                    "is_admin": True
                }
                await repo.create_user(admin_data)
                await session.commit()
                logger.info(f"Default admin user created: {admin_username}")
            else:
                logger.info(f"Admin user already exists: {admin_username}")
        except Exception as e:
            await session.rollback()
            logger.error(f"Error creating default admin: {e}")

@asynccontextmanager
async def lifespan(app):
    logger.info("App started")
    
    # Connect Redis
    try:
        await redis_client.connect()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed during startup (fail-open mode active): {e}")
    
    # Create default admin user
    await create_default_admin()
    
    yield
    
    # Disconnect Redis
    try:
        await redis_client.disconnect()
        logger.info("Disconnected from Redis")
    except Exception:
        pass
    logger.info("app end")

app = FastAPI(title="Langchain API", version="0.0.1", lifespan=lifespan)

# State and Exception Handlers
app.state.limiter = limiter
app.state.auto_kill_enabled = False  # Global flag for automatic protection

assets_path = fastapi_offline_swagger_ui.__path__[0]
if os.path.exists(assets_path + "/swagger-ui.css") and os.path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")
    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )
    applications.get_swagger_ui_html = swagger_monkey_patch


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Robust rate limit handler that avoids AttributeError if exc is not as expected"""
    detail = getattr(exc, "detail", str(exc))
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": f"Rate limit exceeded: {detail}"}
    )



# ============================================================================
# DOMAIN & INFRASTRUCTURE EXCEPTION HANDLERS
# ============================================================================
from App.core.exceptions import (
    DomainError,
    InfrastructureError,
    UserNotFoundError,
    DuplicateEmailError,
    DuplicateNameError,
    IncorrectPasswordError,
    AccountAlreadyDisabledError,
    AccountNotDisabledError,
    PermissionDeniedError,
    RateLimitError,
    MFARequiredError,
    InvalidMFACodeError,
)

@app.exception_handler(InfrastructureError)
async def infrastructure_error_handler(request: Request, exc: InfrastructureError):
    req_id = getattr(request.state, "request_id", "-")
    logger.error(f"[RequestID: {req_id}] Infrastructure Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=jsonable_encoder({
            "detail": str(exc) or "Service temporarily unavailable",
            "error_code": "INFRASTRUCTURE_UNAVAILABLE",
            "request_id": req_id,
        })
    )

@app.exception_handler(UserNotFoundError)
async def user_not_found_handler(request: Request, exc: UserNotFoundError):
    req_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=jsonable_encoder({
            "detail": str(exc) or "User not found",
            "error_code": "USER_NOT_FOUND",
            "request_id": req_id,
        })
    )

@app.exception_handler(DuplicateEmailError)
@app.exception_handler(DuplicateNameError)
async def duplicate_entity_handler(request: Request, exc: DomainError):
    req_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({
            "detail": str(exc),
            "error_code": "DUPLICATE_RESOURCE",
            "request_id": req_id,
        })
    )

@app.exception_handler(IncorrectPasswordError)
async def incorrect_password_handler(request: Request, exc: IncorrectPasswordError):
    req_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={"WWW-Authenticate": "Bearer"},
        content=jsonable_encoder({
            "detail": str(exc) or "Incorrect username or password",
            "error_code": "INCORRECT_CREDENTIALS",
            "request_id": req_id,
        })
    )

@app.exception_handler(AccountAlreadyDisabledError)
async def account_disabled_handler(request: Request, exc: AccountAlreadyDisabledError):
    req_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=jsonable_encoder({
            "detail": str(exc) or "Account is disabled",
            "error_code": "ACCOUNT_DISABLED",
            "request_id": req_id,
        })
    )

@app.exception_handler(PermissionDeniedError)
async def permission_denied_handler(request: Request, exc: PermissionDeniedError):
    req_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=jsonable_encoder({
            "detail": str(exc) or "Permission denied",
            "error_code": "PERMISSION_DENIED",
            "request_id": req_id,
        })
    )

@app.exception_handler(DomainError)
async def generic_domain_error_handler(request: Request, exc: DomainError):
    req_id = getattr(request.state, "request_id", "-")
    logger.warning(f"[RequestID: {req_id}] Domain Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({
            "detail": str(exc),
            "error_code": "DOMAIN_ERROR",
            "request_id": req_id,
        })
    )

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """Centralized handler for all custom AppError exceptions (ModelNotLoadedError, etc.)."""
    logger.warning(f"[{exc.error_code}] {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "detail": exc.detail},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Centralized handler for FastAPI HTTPException."""
    logger.warning(f"[HTTP_{exc.status_code}] {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": f"HTTP_{exc.status_code}", "detail": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Centralized handler for request validation errors (safely encodes non-serializable ctx exceptions)."""
    safe_errors = jsonable_encoder(
        exc.errors(),
        custom_encoder={Exception: lambda e: str(e)},
    )
    logger.warning(f"[VALIDATION_ERROR] {request.url.path}: {safe_errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"error": "REQUEST_VALIDATION_ERROR", "detail": safe_errors},
    )


@app.exception_handler(asyncio.QueueFull)
async def queue_full_handler(request: Request, exc: asyncio.QueueFull):
    logger.warning(f"[QUEUE_FULL] {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "QUEUE_FULL", "detail": str(exc) or "Request queue is full"},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    logger.warning(f"[FILE_NOT_FOUND] {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "FILE_NOT_FOUND", "detail": str(exc)},
    )


@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, exc: MemoryError):
    logger.error(f"[INSUFFICIENT_MEMORY] {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
        content={"error": "INSUFFICIENT_MEMORY", "detail": str(exc)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"[VALUE_ERROR] {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "VALIDATION_ERROR", "detail": str(exc)},
    )

@app.middleware("http")
async def kill_switch_middleware(request: Request, call_next):
    """Global middleware for emergency kill switch (Maintenance Mode)"""
    # Whitelist System and Documentation endpoints (including offline Swagger /assets)
    path = request.url.path
    is_whitelisted = (
        path == "/health"
        or path == "/reset-kill-switch"
        or path.startswith("/docs")
        or path.startswith("/redoc")
        or path.startswith("/openapi.json")
        or path.startswith("/assets")
    )
    
    # Check both manual and automatic kill switches
    is_killed = settings.KILL_SWITCH_ENABLED or getattr(app.state, "auto_kill_enabled", False)
    
    if is_killed and not is_whitelisted:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Service is temporarily unavailable due to maintenance.",
                "type": "auto_kill" if getattr(app.state, "auto_kill_enabled", False) else "manual_kill"
            }
        )
    
    try:
        response = await call_next(request)
        return response
    except (AppError, HTTPException, RequestValidationError, asyncio.QueueFull, ValueError, FileNotFoundError, MemoryError):
        raise
    except Exception as e:
        logger.error(f"CRITICAL: Unhandled exception detected. Triggering AUTO-KILL. Error: {e}")
        app.state.auto_kill_enabled = True
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred. System has entered safety mode."}
        )

# Add Middlewares (Order: Outermost -> Innermost)
# Middleware Stack -- order matters (last added = outermost)
# ============================================================================
app.add_middleware(GZipResponseMiddleware)
app.add_middleware(GZipRequestMiddleware)
app.add_middleware(BodySizeLimitMiddleware, max_size=settings.MAX_BODY_SIZE)
app.add_middleware(RequestIDMiddleware, header_name="X-Request-ID")   # innermost
app.add_middleware(
    GlobalRateLimitMiddleware,
    default_limit=settings.RATE_LIMIT_DEFAULT or "100/minute",
)
app.add_middleware(KillSwitchMiddleware, recovery_seconds=60)
app.add_middleware(IdempotencyMiddleware)

# Outermost CORS
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)



@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """Simple health check endpoint for monitoring"""
    return {"status": "healthy", "version": "0.0.1", "auto_kill_enabled": getattr(app.state, "auto_kill_enabled", False)}


@app.post("/reset-kill-switch", status_code=status.HTTP_200_OK, tags=["System"])
async def reset_kill_switch():
    """Reset the automatic safety kill switch without restarting the server."""
    app.state.auto_kill_enabled = False
    return {"status": "ok", "auto_kill_enabled": False, "message": "Kill switch reset; service is active."}

app.include_router(v1_router, prefix="/app/v1")
