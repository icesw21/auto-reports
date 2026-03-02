"""Unified logging: loguru primary with stdlib intercept for finance-parser modules."""

import logging
import os
import sys
import threading

from loguru import logger

DEFAULT_LOG_DIR = "stocks"

_logging_initialized = False
_logging_lock = threading.Lock()

# Set default extra so {extra[name]} never raises KeyError
logger.configure(extra={"name": "auto-reports"})


class InterceptHandler(logging.Handler):
    """Route stdlib logging through loguru so all modules share one log output."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_dir: str = DEFAULT_LOG_DIR, log_level: str = "INFO") -> None:
    """Initialize unified loguru logging with console + file sinks and stdlib intercept."""
    global _logging_initialized

    with _logging_lock:
        if _logging_initialized:
            return
        _setup_logging_unlocked(log_dir, log_level)
        _logging_initialized = True


def _setup_logging_unlocked(log_dir: str, log_level: str) -> None:
    """Internal: actual logging setup (called under lock)."""

    os.makedirs(log_dir, exist_ok=True)

    # Install intercept handler for stdlib logging (finance-parser modules)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Remove default loguru handler
    logger.remove()

    # Console sink (colorized)
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[name]}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )

    # File sink (rotating 10MB, keep 5 backups)
    logger.add(
        os.path.join(log_dir, "auto-reports.log"),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]} - {message}",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )


def get_logger(name: str = None):
    """Get a loguru logger bound to the given module name."""
    if not _logging_initialized:
        setup_logging()
    return logger.bind(name=name or "auto-reports")
