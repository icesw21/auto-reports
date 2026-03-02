"""
Download utility functions for DART research report collector.

This module provides download-related helper functions including
download completion detection and popup handling.
"""

import logging
import os
import time
import glob
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

logger = logging.getLogger(__name__)


def wait_for_download(download_dir: str, timeout: int = 60) -> bool:
    """
    Wait for download completion by monitoring .crdownload files.

    Chrome creates .crdownload files during active downloads. This function
    monitors the download directory until all such files disappear or timeout.

    Args:
        download_dir: Directory where files are being downloaded
        timeout: Maximum seconds to wait (default: 60)

    Returns:
        bool: True if download completed, False if timeout occurred

    Example:
        >>> if wait_for_download('/path/to/downloads'):
        ...     print("Download complete")
        ... else:
        ...     print("Download timed out")
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        crdownload_files = glob.glob(os.path.join(download_dir, "*.crdownload"))
        if not crdownload_files:
            time.sleep(0.5)
            # Double-check for filesystem delay
            crdownload_files = glob.glob(os.path.join(download_dir, "*.crdownload"))
            if not crdownload_files:
                return True
        time.sleep(1)
    return False


def handle_html_popup(driver) -> None:
    """
    Handle HTML dialog popups by clicking the confirm button.

    Waits for and clicks a confirm button in HTML-based dialog popups,
    commonly used for duplicate login confirmations on Korean websites.

    Args:
        driver: Selenium WebDriver instance

    Example:
        >>> from selenium import webdriver
        >>> driver = webdriver.Chrome()
        >>> handle_html_popup(driver)  # Handles any popup if present
    """
    try:
        # Wait briefly for popup confirm button
        confirm_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '\ud655\uc778')]"))
        )
        confirm_button.click()
        logger.info("Duplicate login popup confirmed.")
        time.sleep(3)
    except TimeoutException:
        pass
