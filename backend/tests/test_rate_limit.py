import pytest

from app.llm.rate_limit import (
    AdaptiveRateLimiter,
    DailyQuotaExceeded,
    is_rate_limited,
    parse_retry_delay,
)


class FakeApiError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def test_parses_retry_delay_from_error_body():
    err = FakeApiError('{"error": {"details": [{"retryDelay": "17s"}]}}')
    assert parse_retry_delay(err) == 17.0


def test_missing_retry_delay_is_none():
    assert parse_retry_delay(FakeApiError("boom")) is None


@pytest.mark.parametrize(
    "error,expected",
    [
        (FakeApiError("x", code=429), True),
        (FakeApiError("x", code=503), True),
        (FakeApiError("RESOURCE_EXHAUSTED: quota"), True),
        (FakeApiError("400 invalid argument", code=400), False),
    ],
)
def test_rate_limit_detection(error, expected):
    assert is_rate_limited(error) is expected


def test_429_halves_effective_rate():
    limiter = AdaptiveRateLimiter(max_rpm=10, max_rpd=100)
    limiter.record_rate_limited()
    assert limiter.effective_rpm == 5.0
    limiter.record_rate_limited()
    assert limiter.effective_rpm == 2.5


def test_rate_never_drops_below_floor():
    limiter = AdaptiveRateLimiter(max_rpm=10, max_rpd=100, min_rpm=2)
    for _ in range(20):
        limiter.record_rate_limited()
    assert limiter.effective_rpm == 2.0


def test_recovery_requires_sustained_success():
    limiter = AdaptiveRateLimiter(max_rpm=10, max_rpd=100, recovery_after_successes=5)
    limiter.record_rate_limited()
    assert limiter.effective_rpm == 5.0

    for _ in range(4):
        limiter.record_success()
    assert limiter.effective_rpm == 5.0  # not yet

    limiter.record_success()
    assert limiter.effective_rpm == 6.0


def test_recovery_cannot_exceed_configured_ceiling():
    limiter = AdaptiveRateLimiter(max_rpm=3, max_rpd=100, recovery_after_successes=1)
    for _ in range(50):
        limiter.record_success()
    assert limiter.effective_rpm == 3.0


def test_backoff_respects_server_hint_over_jitter():
    limiter = AdaptiveRateLimiter(max_rpm=10, max_rpd=100)
    # Jittered exponential could return anything in [0, 2]; the server said 30.
    assert limiter.backoff_delay(0, base=2.0, suggested=30.0) >= 30.0


def test_backoff_is_bounded_without_a_hint():
    limiter = AdaptiveRateLimiter(max_rpm=10, max_rpd=100)
    assert all(0 <= limiter.backoff_delay(10, base=2.0, suggested=None) <= 60.0 for _ in range(50))


def test_daily_cap_raises_once_exhausted():
    limiter = AdaptiveRateLimiter(max_rpm=100, max_rpd=3)
    for _ in range(3):
        limiter.daily.check_and_increment()
    with pytest.raises(DailyQuotaExceeded):
        limiter.daily.check_and_increment()


async def test_acquire_permits_up_to_effective_rate():
    limiter = AdaptiveRateLimiter(max_rpm=5, max_rpd=100)
    for _ in range(5):
        await limiter.acquire()
    assert limiter.stats()["requests_today"] == 5
