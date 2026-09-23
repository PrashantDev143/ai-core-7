"""Client-side throttling for an API whose real limits aren't published.

Google removed the free-tier rate limit tables from its docs; the numbers in
.env are a conservative guess, not a spec. So the limiter treats the configured
RPM as a ceiling it may never exceed, and as something it will drop below on
its own if the API starts returning 429s: multiplicative decrease on rejection,
slow additive recovery on sustained success. Same shape as TCP congestion
control, for the same reason — the true limit is only observable by hitting it.
"""

import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I)


def parse_retry_delay(error: BaseException) -> float | None:
    """Pull Google's suggested retry delay out of an error, if it sent one."""
    match = _RETRY_DELAY_RE.search(str(error))
    return float(match.group(1)) if match else None


def is_rate_limited(error: BaseException) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code in (429, 503):
        return True
    text = str(error).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text


@dataclass
class DailyCounter:
    """In-process request count for the day.

    A process restart resets this, which means a crash loop could still burn
    the daily allowance. Phase 3 moves it into Redis once that's available.
    """

    limit: int
    _count: int = 0
    _day: int = field(default_factory=lambda: int(time.time() // 86400))

    def _roll(self) -> None:
        today = int(time.time() // 86400)
        if today != self._day:
            self._day, self._count = today, 0

    def check_and_increment(self) -> None:
        self._roll()
        if self._count >= self.limit:
            raise DailyQuotaExceeded(
                f"local daily cap of {self.limit} requests reached; "
                "raise GEMINI_MAX_RPD or wait for UTC midnight"
            )
        self._count += 1

    @property
    def used(self) -> int:
        self._roll()
        return self._count


class DailyQuotaExceeded(RuntimeError):
    pass


class AdaptiveRateLimiter:
    def __init__(
        self,
        max_rpm: int,
        max_rpd: int,
        *,
        min_rpm: int = 2,
        recovery_after_successes: int = 20,
    ):
        self._ceiling_rpm = max_rpm
        self._effective_rpm = float(max_rpm)
        self._min_rpm = min(min_rpm, max_rpm)
        self._recovery_after = recovery_after_successes

        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._consecutive_ok = 0
        self.daily = DailyCounter(limit=max_rpd)

        self.observed_429s = 0
        self.total_wait_seconds = 0.0

    @property
    def effective_rpm(self) -> float:
        return self._effective_rpm

    async def acquire(self) -> None:
        """Block until sending another request is within the current budget."""
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()

                if len(self._calls) < self._effective_rpm:
                    self.daily.check_and_increment()
                    self._calls.append(now)
                    return

                wait = 60.0 - (now - self._calls[0]) + 0.05

            self.total_wait_seconds += wait
            await asyncio.sleep(wait)

    def record_success(self) -> None:
        self._consecutive_ok += 1
        # Additive increase, and only after a decent run of clean calls, so one
        # lucky response doesn't undo a backoff we just learned was necessary.
        if self._consecutive_ok >= self._recovery_after:
            self._consecutive_ok = 0
            self._effective_rpm = min(self._ceiling_rpm, self._effective_rpm + 1.0)

    def record_rate_limited(self) -> None:
        self.observed_429s += 1
        self._consecutive_ok = 0
        self._effective_rpm = max(self._min_rpm, self._effective_rpm * 0.5)

    def backoff_delay(self, attempt: int, base: float, suggested: float | None) -> float:
        """Full-jitter exponential backoff, floored by the server's own hint.

        Jitter matters even single-client: without it, several coroutines that
        were throttled together wake together and immediately re-collide.
        """
        capped = min(base * (2**attempt), 60.0)
        delay = random.uniform(0, capped)
        if suggested is not None:
            delay = max(delay, suggested)
        return delay

    def stats(self) -> dict[str, float | int]:
        return {
            "effective_rpm": round(self._effective_rpm, 2),
            "ceiling_rpm": self._ceiling_rpm,
            "requests_today": self.daily.used,
            "daily_limit": self.daily.limit,
            "observed_429s": self.observed_429s,
            "total_wait_seconds": round(self.total_wait_seconds, 2),
        }
