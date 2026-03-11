"""Thread-safe global rate limiters for DART API and LLM API."""

import logging
import threading
import time

import requests

logger = logging.getLogger(__name__)


class DartRateLimiter:
    """Thread-safe global rate limiter for DART API.

    Enforces minimum interval between requests (Lock-based serialization).
    Also provides a Semaphore to cap concurrent in-flight HTTP requests
    (used by dart_get_with_retry).
    """

    def __init__(
        self, min_interval: float = 0.5, max_concurrent: int = 2,
    ) -> None:
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._last_call_time: float = 0
        self._min_interval = min_interval

    def wait(self) -> None:
        """Enforce minimum interval between calls. For OpenDartReader lib calls."""
        with self._lock:
            elapsed = time.time() - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()

    @property
    def semaphore(self) -> threading.Semaphore:
        return self._semaphore


class LlmRateLimiter:
    """Thread-safe global rate limiter for LLM API calls.

    Enforces a minimum interval between LLM API calls across all workers
    to prevent rate limiting from the LLM provider.
    """

    def __init__(self, min_interval: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._last_call_time: float = 0
        self._min_interval = min_interval

    def wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()


# Module-level singletons
_dart_limiter = DartRateLimiter(min_interval=1.0, max_concurrent=1)
_llm_limiter = LlmRateLimiter(min_interval=1.0)


def get_dart_limiter() -> DartRateLimiter:
    return _dart_limiter


def get_llm_limiter() -> LlmRateLimiter:
    return _llm_limiter


def dart_call_with_retry(
    fn,
    *args,
    max_retries: int = 3,
    **kwargs,
):
    """Call an OpenDartReader method with rate limiting, semaphore, and retry.

    Wraps any callable (e.g. dart.sub_docs, dart.list) with the full
    DART rate-limiting protection: interval wait + semaphore + exponential backoff.

    Returns the result of fn(*args, **kwargs), or raises on final failure.
    """
    limiter = get_dart_limiter()

    for attempt in range(max_retries + 1):
        limiter.wait()
        limiter.semaphore.acquire()
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception:
            if attempt < max_retries:
                wait_time = min(3 * (2 ** attempt), 60)
                logger.warning(
                    "DART call %s failed, retrying in %ds (attempt %d/%d)",
                    fn.__name__ if hasattr(fn, '__name__') else str(fn),
                    wait_time, attempt + 1, max_retries,
                )
                time.sleep(wait_time)
            else:
                raise
        finally:
            limiter.semaphore.release()


def dart_get_with_retry(
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> requests.Response:
    """Make an HTTP GET with DART rate limiting, concurrency cap, and 429 retry.

    Combines interval enforcement, semaphore-based concurrency cap,
    and exponential backoff on HTTP 429.

    Args:
        url: Request URL.
        max_retries: Maximum number of retries on HTTP 429.
        **kwargs: Additional kwargs passed to requests.get().

    Returns:
        requests.Response object.

    Raises:
        requests.RequestException: On non-retryable errors or retries exhausted.
    """
    kwargs.setdefault("timeout", 30)
    limiter = get_dart_limiter()

    for attempt in range(max_retries + 1):
        limiter.wait()
        limiter.semaphore.acquire()
        try:
            r = requests.get(url, **kwargs)
        except Exception:
            if attempt < max_retries:
                wait_time = min(2 ** attempt, 30)
                logger.warning(
                    "DART request error, retrying in %ds (attempt %d/%d): %s",
                    wait_time, attempt + 1, max_retries, url,
                )
                time.sleep(wait_time)
                continue
            raise
        finally:
            limiter.semaphore.release()

        if r.status_code == 429:
            wait_time = min(2 ** attempt, 30)
            logger.warning(
                "DART API 429 Too Many Requests, retrying in %ds (attempt %d/%d)",
                wait_time, attempt + 1, max_retries,
            )
            time.sleep(wait_time)
            continue

        return r

    raise requests.RequestException(
        f"DART API request failed after {max_retries} retries: {url}",
    )
