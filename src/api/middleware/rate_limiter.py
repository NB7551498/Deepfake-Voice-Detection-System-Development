"""
Sliding-window IP Rate Limiter for FastAPI.
Protects inference pipeline from denial-of-service and high compute abuse.
"""

import time
from collections import defaultdict
from fastapi import Request, HTTPException

# 40 requests per minute per IP address
DEFAULT_RATE_LIMIT = 40
WINDOW_SECONDS = 60


class SlidingWindowRateLimiter:
    def __init__(self, limit: int = DEFAULT_RATE_LIMIT, window_s: int = WINDOW_SECONDS):
        self.limit = limit
        self.window_s = window_s
        self.requests = defaultdict(list)

    def check(self, request: Request):
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        timestamps = self.requests[client_ip]

        # Purge timestamps older than window
        self.requests[client_ip] = [t for t in timestamps if now - t < self.window_s]

        if len(self.requests[client_ip]) >= self.limit:
            retry_after = int(self.window_s - (now - self.requests[client_ip][0]))
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.limit} requests per minute. Retry in {max(retry_after, 1)}s.",
                headers={"Retry-After": str(max(retry_after, 1))}
            )

        self.requests[client_ip].append(now)


rate_limiter = SlidingWindowRateLimiter()
