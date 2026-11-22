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
