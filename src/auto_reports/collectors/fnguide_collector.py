"""
FnGuide research report collector module.

This module provides the FnGuideCollector class for collecting research reports
from FnGuide using Selenium browser automation.
"""

import json
import os
import time
from typing import Dict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from auto_reports.collectors.base import BaseCollector
from auto_reports.utils.file_utils import sanitize_filename
from auto_reports.utils.download_utils import wait_for_download, handle_html_popup


class FnGuideCollector(BaseCollector):
    """
    Collector for FnGuide research reports.

    Uses Selenium WebDriver to automate browser interactions with FnGuide,
    including login and PDF downloads from the research report viewer.

    Attributes:
        user_id: FnGuide login username
        user_pw: FnGuide login password

    Example:
        >>> collector = FnGuideCollector()
        >>> count = collector.collect(
        ...     stock_code='005930',
        ...     company_name='Samsung Electronics'
        ... )
    """

    def __init__(self, user_id: str = None, user_pw: str = None, output_dir: str = None):
        """
        Initialize the FnGuide collector.

        Args:
            user_id: FnGuide username (loads from env if not provided)
            user_pw: FnGuide password (loads from env if not provided)
            output_dir: Output directory for collected data
        """
        super().__init__(output_dir=output_dir)
        if user_id and user_pw:
            self.user_id = user_id
            self.user_pw = user_pw
        else:
            from auto_reports.config import Settings
            settings = Settings()
            self.user_id = settings.fnguide_id
            self.user_pw = settings.fnguide_pw

    def _setup_driver(self, download_dir: str) -> webdriver.Chrome:
        """
        Configure and create a Chrome WebDriver instance.

        Args:
            download_dir: Directory for downloaded files

        Returns:
            webdriver.Chrome: Configured Chrome WebDriver
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True
        }
        chrome_options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(options=chrome_options)

        # Headless Chrome requires explicit CDP command to enable downloads
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": download_dir,
        })

        return driver

    def collect(self, stock_code: str, company_name: str, **kwargs) -> int:
        """
        Collect FnGuide research reports for a company.

        Args:
            stock_code: 6-digit stock code (not used, search is by name)
            company_name: Company name for search and folder organization

        Returns:
            int: Number of reports successfully downloaded
        """
        self.logger.info(f"FnGuide collection started: {company_name}")

        # Setup download directory (reports subdirectory)
        download_dir = os.path.abspath(os.path.join(self.output_dir, company_name, 'reports'))
        self._ensure_directory(download_dir)

        # Load download manifest for URL-based duplicate detection
        manifest_path = os.path.join(download_dir, '_downloads.json')
        manifest = self._load_manifest(manifest_path)

        # Setup browser
        downloaded_count = 0
        driver = None

        try:
            driver = self._setup_driver(download_dir)
            wait = WebDriverWait(driver, 15)
            # 1. Login
            self.logger.info("FnGuide login attempt")
            driver.get("https://www.fnguide.com/Users/Login")

            id_input = wait.until(EC.element_to_be_clickable((By.ID, "userId")))
            id_input.clear()
            id_input.send_keys(self.user_id)

            pw_input = driver.find_element(By.ID, "userPw")
            pw_input.clear()
            pw_input.send_keys(self.user_pw)
            pw_input.send_keys(Keys.ENTER)

            time.sleep(1)
            handle_html_popup(driver)
            self.logger.info("FnGuide login complete")

            # 2. Navigate to research page and search
            driver.get("https://www.fnguide.com/Research/SearchReport")
            handle_html_popup(driver)

            search_box = wait.until(EC.presence_of_element_located((By.ID, "srchKeyword")))
            search_box.clear()
            search_box.send_keys(company_name)
            search_box.send_keys(Keys.ENTER)

            time.sleep(3)

            # Check for empty results
            try:
                empty_msg = driver.find_elements(By.CSS_SELECTOR, ".empty-case p.caption")
                if empty_msg and ("No search results" in empty_msg[0].text or "검색결과가 없습니다" in empty_msg[0].text):
                    self.logger.warning(f"FnGuide no search results: {company_name}")
                    self.logger.warning(f"No FnGuide research reports found: {company_name}")
                    return 0
            except Exception:
                pass

            # 3. Collect reports
            try:
                report_links = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.report-title")))

                download_urls = []
                for link in report_links:
                    url = link.get_attribute("href")
                    raw_title = link.text.strip()
                    clean_title = sanitize_filename(raw_title)
                    broker = self._extract_broker_from_row(link, raw_title, company_name)
                    download_urls.append((url, broker, clean_title))

                self.logger.info(f"FnGuide found {len(download_urls)} reports: {company_name}")
                self.logger.info(f"FnGuide found {len(download_urls)} report candidates: {company_name}")

                for i, (url, broker, title) in enumerate(download_urls):
                    try:
                        # Build filename: {broker}_{company}_{title}.pdf
                        if broker:
                            file_name = f"{broker}_{company_name}_{title}.pdf"
                        else:
                            file_name = f"{company_name}_{title}.pdf"
                        file_path = os.path.join(download_dir, file_name)

                        # Duplicate check 1: target file already exists
                        if os.path.exists(file_path):
                            self.logger.info(f"Duplicate file skipped: {file_name}")
                            self.logger.info(f"Duplicate file skipped: {file_name}")
                            continue

                        # Duplicate check 2: URL already downloaded (manifest)
                        if url in manifest:
                            self.logger.info(f"Duplicate URL skipped: {title}")
                            self.logger.info(f"Duplicate URL skipped (manifest): {title}")
                            continue

                        self.logger.info(f"Accessing viewer for: {title}")

                        # Snapshot PDFs before download to detect new file by set difference
                        before_pdfs = set(f for f in os.listdir(download_dir) if f.endswith('.pdf'))

                        driver.get(url)
                        time.sleep(5)

                        # Click download button
                        download_btn = wait.until(EC.element_to_be_clickable((By.ID, "pdfViewer_download")))
                        driver.execute_script("arguments[0].click();", download_btn)
                        self.logger.info("Download button clicked")

                        if wait_for_download(download_dir):
                            # Find newly downloaded PDF by set difference
                            after_pdfs = set(f for f in os.listdir(download_dir) if f.endswith('.pdf'))
                            new_files = after_pdfs - before_pdfs
                            if new_files:
                                downloaded_file = new_files.pop()
                                downloaded_path = os.path.join(download_dir, downloaded_file)
                                # Rename only if it's not already our target name
                                if downloaded_file != file_name:
                                    os.replace(downloaded_path, file_path)
                                    self.logger.info(f"FnGuide report saved: {downloaded_file} -> {file_name}")
                                else:
                                    self.logger.info(f"FnGuide report saved: {file_name}")

                                # Record in manifest and increment count
                                manifest[url] = file_name
                                downloaded_count += 1
                                self.logger.info(f"Download complete: {file_name}")
                            else:
                                self.logger.warning(f"FnGuide download completed but no new PDF detected: {title}")
                                self.logger.warning("Download completed but no new PDF detected")
                        else:
                            self.logger.warning(f"FnGuide download timeout: {title}")
                            self.logger.warning("Download timeout (60s exceeded)")

                        driver.back()
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.report-title")))

                    except Exception as e:
                        self.logger.exception(f"FnGuide download error: {title}")
                        self.logger.error(f"Download error: {e}")
                        driver.get("https://www.fnguide.com/Research/SearchReport")

                self.logger.info(f"FnGuide collection complete: {company_name} - {downloaded_count} downloaded")

            except TimeoutException:
                self.logger.error(f"FnGuide report list loading failed: {company_name}")
                self.logger.error(f"FnGuide report list loading failed: {company_name}")
            except Exception as e:
                self.logger.exception(f"FnGuide list processing error: {company_name}")
                self.logger.error(f"FnGuide list processing error: {e}")

        finally:
            self._save_manifest(manifest_path, manifest)
            self.logger.info(f"FnGuide process ended: {company_name}")
            self.logger.info(f"FnGuide process ended: {company_name}")
            if driver:
                time.sleep(2)
                driver.quit()

        return downloaded_count

    def _extract_broker_from_row(self, link_element, raw_title: str, company_name: str = '') -> str:
        """
        Extract broker/securities firm name (제공사) from the report's parent row.

        First attempts to locate the '제공사' column by inspecting table headers,
        then falls back to heuristic cell scanning if headers aren't found.

        Args:
            link_element: Selenium WebElement for the report title link
            raw_title: Raw title text for filtering
            company_name: Company name to skip (prevents returning stock name as broker)

        Returns:
            str: Sanitized broker name, or empty string if not found
        """
        try:
            row = link_element.find_element(By.XPATH, './ancestor::tr[1]')
            cells = row.find_elements(By.TAG_NAME, 'td')

            # Primary: find 제공사 column index from table headers
            try:
                table = link_element.find_element(By.XPATH, './ancestor::table[1]')
                headers = table.find_elements(By.CSS_SELECTOR, 'thead th')
                if not headers:
                    headers = table.find_elements(By.TAG_NAME, 'th')
                for i, header in enumerate(headers):
                    if '제공사' in header.text:
                        if i < len(cells):
                            text = cells[i].text.strip()
                            if text:
                                return sanitize_filename(text)
                        break
            except Exception:
                pass

            # Fallback: heuristic scan (skip date, title, company name, author-like cells)
            for cell in cells:
                if cell.find_elements(By.CSS_SELECTOR, 'a.report-title'):
                    continue
                text = cell.text.strip()
                if not text or text[0].isdigit():
                    continue
                if raw_title and raw_title in text:
                    continue
                if company_name and text == company_name:
                    continue
                # Skip author-like cells (contain comma-separated names)
                if ',' in text and not text.endswith('증권') and not text.endswith('투자') and not text.endswith('자산운용'):
                    continue
                return sanitize_filename(text)
        except Exception:
            pass
        return ''

    def _load_manifest(self, manifest_path: str) -> Dict[str, str]:
        """Load download manifest (URL -> filename mapping)."""
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                self.logger.warning(f"Corrupt manifest, starting fresh: {manifest_path}")
        return {}

    def _save_manifest(self, manifest_path: str, manifest: Dict[str, str]) -> None:
        """Save download manifest (URL -> filename mapping)."""
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except IOError as e:
            self.logger.warning(f"Failed to save manifest: {e}")
