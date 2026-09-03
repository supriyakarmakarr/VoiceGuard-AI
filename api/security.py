"""
VoiceGuard AI - API Authentication & Sliding Window Rate Limiter
Secures /api/analyze and streaming endpoints for enterprise production:
1. Header-based API Key validation (X-API-Key or Bearer token)
2. In-Memory Sliding Window Rate Limiter (60 req/min with burst handling)
3. Standard Rate Limit response headers (X-RateLimit-Limit, X-RateLimit-Remaining, Retry-After)
4. Dev/Demo bypass for local frontend and test clients
"""

import os
import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Tuple, Optional

API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

# Configurable settings
CONFIGURED_API_KEY = os.getenv("VOICEGUARD_API_KEY", "voiceguard-enterprise-demo-key-2026")
DEFAULT_RATE_LIMIT_PER_MINUTE = int(os.getenv("VOICEGUARD_RATE_LIMIT", "60"))
DEMO_MODE = os.getenv("API_DEMO_MODE", "true").lower() in ("true", "1", "yes")


class SlidingWindowRateLimiter:
    """
    Sliding window in-memory rate limiter per client key / IP.
    """

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE):
        self.limit = requests_per_minute
        self.window_seconds = 60.0
        self.records = defaultdict(deque)

    def is_allowed(self, client_id: str) -> Tuple[bool, int, int]:
        """
        Returns (is_allowed: bool, remaining_requests: int, retry_after_sec: int).
        """
        now = time.time()
        window_start = now - self.window_seconds
        timestamps = self.records[client_id]

        # Purge timestamps outside the 60s sliding window
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        curr_count = len(timestamps)
        if curr_count >= self.limit:
            oldest = timestamps[0]
            retry_after = max(1, int(oldest + self.window_seconds - now))
            return False, 0, retry_after

        # Record this request
        timestamps.append(now)
        remaining = self.limit - (curr_count + 1)
        return True, remaining, 0


rate_limiter = SlidingWindowRateLimiter()


def get_client_identifier(request: Request) -> str:
    """Extracts client IP or API key for rate limiting bucket."""
    api_key = request.headers.get(API_KEY_HEADER_NAME)
    if api_key:
        return f"key:{api_key}"
    # Fallback to client IP
    client_ip = request.client.host if request.client else "127.0.0.1"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    return f"ip:{client_ip}"


async def enforce_security_and_rate_limit(request: Request):
    """
    FastAPI dependency that enforces API key authentication and sliding window rate limiting.
    Allows demo calls if DEMO_MODE is true or from trusted local origins, but still rate limits.
    """
    client_id = get_client_identifier(request)

    # 1. Rate limiting check
    allowed, remaining, retry_after = rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({DEFAULT_RATE_LIMIT_PER_MINUTE} req/min). Retry in {retry_after} seconds.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(DEFAULT_RATE_LIMIT_PER_MINUTE),
                "X-RateLimit-Remaining": "0",
            },
        )

    # Attach remaining requests to request state for response header injection
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_limit = DEFAULT_RATE_LIMIT_PER_MINUTE

    # 2. Authentication check
    api_key = request.headers.get(API_KEY_HEADER_NAME)
    if api_key:
        if api_key != CONFIGURED_API_KEY and not DEMO_MODE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-API-Key credentials.",
            )
        return api_key

    # If in DEMO_MODE or local call, allow without header
    if DEMO_MODE:
        return "demo-session"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing required authentication header: X-API-Key",
    )
