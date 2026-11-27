# App/middleware/body_size_middleware.py
import json
from App.core.LoggingInit import get_core_logger
from App.core.size_parser import format_size, parse_size

logger = get_core_logger(__name__)


async def _send_json_error(send, status_code: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status_code,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
        "more_body": False,
    })


class BodySizeLimitMiddleware:
    """
    Pure ASGI body size limit middleware (avoids BaseHTTPMiddleware stream wrapping).
    Supports both Content-Length and Transfer-Encoding: chunked uploads while strictly
    enforcing max_size in bytes.
    """

    def __init__(self, app, max_size: int = 1 * 1024 * 1024):
        self.app = app
        self.max_size = parse_size(max_size) if isinstance(max_size, (str, int)) else int(max_size)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        max_size = int(self.max_size)

        content_length_raw = headers.get(b"content-length")
        if content_length_raw is not None:
            try:
                length_value = int(content_length_raw.decode("latin-1").strip())
            except (TypeError, ValueError):
                return await _send_json_error(send, 400, {"error": "Invalid Content-Length header"})
            if length_value > max_size:
                return await _send_json_error(
                    send,
                    413,
                    {
                        "error": "Request body too large",
                        "max_bytes": max_size,
                        "max_human": format_size(max_size),
                    },
                )
            return await self.app(scope, receive, send)

        # Streaming / chunked body: count bytes incrementally without rejecting chunked uploads
        bytes_seen = 0
        response_started = False

        async def checked_receive():
            nonlocal bytes_seen
            message = await receive()
            if message["type"] == "http.request":
                bytes_seen += len(message.get("body", b""))
                if bytes_seen > max_size:
                    raise _BodyTooLargeError()
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            return await self.app(scope, checked_receive, tracked_send)
        except _BodyTooLargeError:
            if not response_started:
                return await _send_json_error(
                    send,
                    413,
                    {
                        "error": "Request body too large",
                        "max_bytes": max_size,
                        "max_human": format_size(max_size),
                    },
                )


class _BodyTooLargeError(Exception):
    pass