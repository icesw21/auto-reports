"""
Retry utility with exponential backoff for handling transient network failures.

This module provides a retry mechanism for HTTP requests and other operations
that may fail due to temporary issues like network timeouts or server errors.
"""

import time
import requests
from typing import Callable, Any


def retry_request(func: Callable, max_retries: int = 3, base_delay: float = 1.0,
                  logger=None, *args, **kwargs) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Retries the given function up to max_retries times with exponential backoff
    delays between attempts. Catches transient network-related exceptions.

    Args:
        func: The callable function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        logger: Optional logger instance for logging retry attempts
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        The return value of the successful function call

    Raises:
        The last exception raised if all retry attempts fail

    Example:
        >>> import requests
        >>> from core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>>
        >>> # Retry a GET request with custom parameters
        >>> response = retry_request(
        ...     requests.get,
        ...     max_retries=3,
        ...     base_delay=1,
        ...     logger=logger,
        ...     url='https://api.example.com/data',
        ...     headers={'User-Agent': 'MyApp/1.0'}
        ... )
        >>>
        >>> # Retry any callable function
        >>> def fetch_data(endpoint, timeout=10):
        ...     return requests.get(f'https://api.example.com/{endpoint}', timeout=timeout)
        >>>
        >>> result = retry_request(fetch_data, logger=logger, endpoint='users', timeout=30)
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Execute the function with provided arguments
            result = func(*args, **kwargs)
            return result

        except (requests.RequestException,
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError) as e:
            last_exception = e

            # If this was the last attempt, don't retry
            if attempt == max_retries - 1:
                if logger:
                    logger.error(f"Final retry attempt failed after {max_retries} attempts: {e}")
                raise

            # Calculate exponential backoff delay: 1s, 2s, 4s, ...
            delay = base_delay * (2 ** attempt)

            if logger:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay}s..."
                )

            time.sleep(delay)

    # This should never be reached due to the raise in the exception handler,
    # but included for completeness
    if last_exception:
        raise last_exception
