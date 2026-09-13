"""Local-first authentication and bounded per-IP rate limiting."""
import hmac
import os
import time
import threading
from collections import OrderedDict, deque
from fastapi import Request, HTTPException

CONFIGURED_API_KEY = os.getenv('VOICEGUARD_API_KEY', '')
DEFAULT_RATE_LIMIT_PER_MINUTE = int(os.getenv('VOICEGUARD_RATE_LIMIT', '120'))
DEMO_MODE = os.getenv('API_DEMO_MODE', 'false').lower() == 'true'

class SlidingWindowRateLimiter:
    def __init__(self, requests_per_minute=DEFAULT_RATE_LIMIT_PER_MINUTE):
        self.limit = requests_per_minute
        self.records = OrderedDict()
        self.lock = threading.Lock()

    def is_allowed(self, client_id):
        with self.lock:
            now = time.monotonic()
            if client_id not in self.records and len(self.records) >= 4096:
                self.records.popitem(last=False)
            queue = self.records.setdefault(client_id, deque())
            self.records.move_to_end(client_id)
            while queue and queue[0] < now - 60:
                queue.popleft()
            if len(queue) >= self.limit:
                return False, 0, max(1, int(queue[0] + 61 - now))
            queue.append(now)
            return True, self.limit - len(queue), 0

rate_limiter = SlidingWindowRateLimiter()

def get_client_identifier(request):
    # Never trust a client-supplied forwarding header or varying API key as a rate bucket.
    return request.client.host if request.client else 'unknown'

async def enforce_security_and_rate_limit(request: Request):
    client = get_client_identifier(request)
    allowed, remaining, retry = rate_limiter.is_allowed(client)
    if not allowed:
        raise HTTPException(429, 'Too many requests. Retry shortly.', headers={'Retry-After': str(retry)})
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_limit = rate_limiter.limit
    supplied = request.headers.get('x-api-key', '')
    if not supplied and request.headers.get('authorization', '').startswith('Bearer '):
        supplied = request.headers['authorization'][7:]
    if CONFIGURED_API_KEY:
        if not hmac.compare_digest(supplied, CONFIGURED_API_KEY):
            raise HTTPException(401, 'Enter a valid API key in connection settings.')
        return 'authenticated'
    if client in ('127.0.0.1', '::1', 'testclient') or DEMO_MODE:
        return 'local' if not DEMO_MODE else 'public-demo'
    raise HTTPException(401, 'Remote access requires VOICEGUARD_API_KEY or explicit public demo mode.')
