"""
Base collector abstract class for DART research report collector.

This module defines the interface that all collectors must implement.
"""

import os
from abc import ABC, abstractmethod

from auto_reports.utils.logging import get_logger


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.

    Defines the common interface and shared functionality for collectors
    that gather financial documents from various sources.

    Attributes:
        logger: Logger instance for the collector

    Example:
        >>> class MyCollector(BaseCollector):
        ...     def collect(self, stock_code, company_name):
        ...         # Implementation here
        ...         return downloaded_count
    """

    def __init__(self, output_dir: str = None):
        """Initialize the collector with a logger and output directory."""
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = output_dir or "./stocks"

    @abstractmethod
    def collect(self, stock_code: str, company_name: str, **kwargs) -> int:
        """
        Collect documents for a given stock.

        Args:
            stock_code: 6-digit stock code (e.g., '005930')
            company_name: Company name for folder organization
            **kwargs: Additional collector-specific parameters

        Returns:
            int: Number of files successfully downloaded

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement collect()")

    def _ensure_directory(self, directory: str) -> None:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory: Path to the directory
        """
        os.makedirs(directory, exist_ok=True)
