"""Shared bounded and testable HTTP reliability layer."""

from __future__ import annotations

import email.utils
import logging
import math
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


def parse_datetime(value: datetime | str | None) -> datetime | None:
    """Parse persisted wall-clock values into timezone-aware UTC datetimes."""

    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise TypeError(f"unsupported datetime value: {type(value).__name__}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_retry_after_seconds(
    value: int | float | str | None,
    *,
    now: datetime | str | None = None,
) -> float | None:
    """Return a non-negative retry delay from seconds or an absolute time."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("retry_after must not be boolean")
    if isinstance(value, (int, float)):
        delay = float(value)
        return max(0.0, delay) if math.isfinite(delay) else None
    text = value.strip()
    if not text:
        return None
    try:
        delay = float(text)
    except ValueError:
        current = parse_datetime(now) or datetime.now(UTC)
        try:
            retry_at = parse_datetime(text)
        except ValueError:
            try:
                retry_at = email.utils.parsedate_to_datetime(text)
                retry_at = parse_datetime(retry_at)
            except (TypeError, ValueError):
                return None
        return max(0.0, (retry_at - current).total_seconds()) if retry_at else None
    return max(0.0, delay) if math.isfinite(delay) else None


class ResponseTooLarge(ValueError):
    pass


class ResilientHttpClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        user_agent: str = "LiteratureRadar/1.0 (contact: local-application)",
        timeout: float = 20.0,
        max_retries: int = 2,
        max_response_bytes: int = 2_000_000,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
        rate_limit_per_second: float = 5.0,
        circuit_failures: int = 4,
        max_backoff_seconds: float = 60.0,
        max_inline_sleep_seconds: float = 5.0,
    ) -> None:
        self.client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self.max_retries = int(max_retries)
        self.max_response_bytes = int(max_response_bytes)
        self.sleeper = sleeper
        self.clock = clock
        self.jitter = jitter
        rate = float(rate_limit_per_second)
        self.rate_interval = 1 / rate if rate > 0 else 0.0
        self.circuit_failures = int(circuit_failures)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.max_inline_sleep_seconds = float(max_inline_sleep_seconds)
        self._last_request: dict[str, float] = {}
        self._failures: dict[str, int] = {}

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        host = (urlsplit(url).hostname or "").lower()
        if self._failures.get(host, 0) >= self.circuit_failures:
            raise httpx.TransportError(f"circuit open for {host}")
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            try:
                response = self.client.get(url, **kwargs)
                if (response.status_code == 429 or 500 <= response.status_code < 600) and (
                    attempt < self.max_retries
                ):
                    delay = self._retry_delay(response, attempt)
                    if delay > self.max_inline_sleep_seconds:
                        response.raise_for_status()
                    self.sleeper(delay)
                    continue
                response.raise_for_status()
                if len(response.content) > self.max_response_bytes:
                    raise ResponseTooLarge("response exceeded configured byte limit")
                self._failures[host] = 0
                return response
            except (httpx.TimeoutException, httpx.TransportError):
                self._failures[host] = self._failures.get(host, 0) + 1
                if attempt >= self.max_retries:
                    raise
                self.sleeper((2**attempt) + self.jitter())
        raise AssertionError("unreachable")

    def get_json(self, url: str, **kwargs: object) -> dict:
        response = self.get(url, **kwargs)
        content_type = response.headers.get("content-type", "").lower()
        if "json" not in content_type:
            raise ValueError("expected JSON content type")
        try:
            value = response.json()
        except ValueError as exc:
            raise ValueError("invalid JSON response") from exc
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value

    def _throttle(self, host: str) -> None:
        now = float(self.clock())
        wait = self._last_request.get(host, now - self.rate_interval) + self.rate_interval - now
        if wait > 0:
            self.sleeper(wait)
        self._last_request[host] = float(self.clock())

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        value = response.headers.get("retry-after")
        if value:
            delay = parse_retry_after_seconds(value)
            if delay is not None:
                return delay
        return min(self.max_backoff_seconds, (2**attempt) + float(self.jitter()))

    def close(self) -> None:
        self.client.close()
