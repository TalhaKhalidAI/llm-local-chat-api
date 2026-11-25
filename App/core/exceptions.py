"""
App/core/exceptions.py — Custom Exception Classes
=================================================
All custom exceptions inherit from `AppError` and are caught by `@app.exception_handler` in `main.py`.
"""

from __future__ import annotations

from fastapi import status


class AppError(Exception):
    """Base application exception handled by `@app.exception_handler(AppError)` in main.py."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "APP_ERROR"

    def __init__(self, detail: str, status_code: int | None = None, error_code: str | None = None):
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code


class ModelNotLoadedError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "MODEL_NOT_LOADED"


class VisionModelNotLoadedError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "VISION_MODEL_NOT_LOADED"


class VisionModelNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "VISION_MODEL_NOT_FOUND"


class VisionModelLoadError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "VISION_MODEL_LOAD_ERROR"


class RemoteVisionApiError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "REMOTE_VISION_API_ERROR"


class ModelNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "MODEL_NOT_FOUND"


class ModelLoadError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "MODEL_LOAD_ERROR"


class QueueFullError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "QUEUE_FULL"


class InvalidImageError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "INVALID_IMAGE"


class InvalidFileError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "INVALID_FILE"


class ImageTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    error_code = "IMAGE_TOO_LARGE"


class FileTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    error_code = "FILE_TOO_LARGE"


class InvalidCollectionError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "INVALID_COLLECTION"


class MissingRequestFieldError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_code = "MISSING_REQUEST_FIELD"


class InsufficientMemoryError(AppError):
    status_code = status.HTTP_507_INSUFFICIENT_STORAGE
    error_code = "INSUFFICIENT_MEMORY"


class InfrastructureError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "INFRASTRUCTURE_UNAVAILABLE"
    default_message = "Service temporarily unavailable"

    def __init__(self, message: str | None = None):
        super().__init__(detail=message or self.default_message, status_code=self.status_code, error_code=self.error_code)


# =====================================================================
# DOMAIN EXCEPTIONS FROM BOILERPLATE
# =====================================================================
"""
Domain exceptions for the repository layer.

These are raised by repositories and translated to HTTP responses
by the route layer. They are deliberately NOT HTTPException subclasses --
the repository should not know about HTTP.
"""


# =====================================================================
# BASE
# =====================================================================

class DomainError(Exception):
    """Base class for all business-logic errors."""
    default_message = "A domain error occurred"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.default_message)


class InfrastructureError(DomainError):
    """Raised when a dependent service (Redis, external API) is unreachable.

    Subclass of DomainError so it's raised from the same layer, but routes
    must catch it *before* DomainError to map it to 503 instead of 401/400.
    """
    default_message = "Service temporarily unavailable"


# =====================================================================
# USER / AUTH
# =====================================================================

class UserNotFoundError(DomainError):
    default_message = "User not found"


class DuplicateEmailError(DomainError):
    default_message = "Email already registered"


class DuplicateNameError(DomainError):
    default_message = "Name already taken"


class AccountAlreadyDisabledError(DomainError):
    default_message = "Account is already disabled"


class AccountNotDisabledError(DomainError):
    default_message = "Account is not disabled"


class AccountAlreadyDeletedError(DomainError):
    default_message = "Account is already deleted"


class AccountNotDeletedError(DomainError):
    default_message = "Account is not deleted"


class AdminCreationError(DomainError):
    default_message = "Failed to create admin"


class PasswordRequiredError(DomainError):
    default_message = "Password required for this action"


class IncorrectPasswordError(DomainError):
    default_message = "Incorrect password"


class RateLimitError(DomainError):
    """Raised when a caller exceeds a per-action rate limit (e.g. login)."""
    default_message = "Too many attempts. Please try again later."


# =====================================================================
# CHANNEL
# =====================================================================

class ChannelNotFoundError(DomainError):
    default_message = "Channel not found"


class ChannelAlreadyExistsError(DomainError):
    default_message = "Channel already exists for this user"


class DuplicateHandleError(DomainError):
    default_message = "Handle already taken"


class ChannelAlreadyDeletedError(DomainError):
    default_message = "Channel is already deleted"


class ChannelMemberNotFoundError(DomainError):
    default_message = "Channel member not found"


class DuplicateChannelMemberError(DomainError):
    default_message = "User is already a member of this channel"


class CannotAddOwnerAsMemberError(DomainError):
    """Owner is implicit; they must not appear in channel_members."""
    default_message = "Channel owner cannot be added as a member"


class InvalidMemberRoleError(DomainError):
    default_message = "Invalid member role"


# =====================================================================
# CHANNEL GRANTS
# =====================================================================

class ChannelGrantNotFoundError(DomainError):
    default_message = "Channel grant not found"


class DuplicateChannelGrantError(DomainError):
    default_message = "Grant already exists for this grantee"


class InvalidGranteeError(DomainError):
    """Raised when the grantee is invalid (self-grant, missing user, etc.)."""
    default_message = "Invalid grantee"


class GrantExpiredError(DomainError):
    default_message = "Grant has expired"


class GrantRevokedError(DomainError):
    default_message = "Grant has been revoked"


class GrantUsageExhaustedError(DomainError):
    default_message = "Grant has reached its maximum number of uses"


class DuplicateVideoGrantError(DomainError):
    default_message = "Grant already exists for this grantee"

class CommentLikeNotFoundError(DomainError):
    default_message = "Comment like not found"

class PermissionDeniedError(DomainError):
    """Raised when an authenticated user lacks the required permission."""
    default_message = "You don't have permission to perform this action"

class ValidationError(DomainError):
    default_message = "Invalid input"

class IntegrityError(DomainError):
    default_message = "Integrity error"
class MFARequiredError(DomainError):
    """User has MFA enabled but no code was provided in the login request."""
    default_message = "MFA code required"

    def __init__(self, message: str | None = None, method: str = "totp"):
        super().__init__(message or self.default_message)
        self.method = method
class InvalidMFACodeError(DomainError):
    """The provided MFA code did not match the user's TOTP."""
    default_message = "Invalid MFA code"
    