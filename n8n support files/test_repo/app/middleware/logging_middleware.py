"""
Structured request/response logging middleware for FastAPI.

Logs every request with:
  - Method, path, status code
  - Processing duration
  - Request ID (for distributed tracing)
  - Client IP

Mimics a production observability stack (without the actual tracing backend).
"""
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured access log with request ID injection."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:12]
        start = time.monotonic()

        # Inject request ID into state for downstream use
        request.state.request_id = request_id

        logger.info(
            f"[Access] → {request.method} {request.url.path} "
            f"req_id={request_id} "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        try:
            response = await call_next(request)
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                f"[Access] ✗ {request.method} {request.url.path} "
                f"req_id={request_id} error={type(e).__name__}: {e} "
                f"elapsed={elapsed_ms:.1f}ms"
            )
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            f"[Access] ← {request.method} {request.url.path} "
            f"status={response.status_code} "
            f"req_id={request_id} "
            f"elapsed={elapsed_ms:.1f}ms",
        )

        response.headers["X-Request-ID"] = request_id
        return response
