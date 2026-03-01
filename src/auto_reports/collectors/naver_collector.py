"""
Naver Finance research report collector module.

This module provides the NaverCollector class for collecting research reports
from Naver Finance using HTTP requests and BeautifulSoup parsing.
"""

import os
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from auto_reports.collectors.base import BaseCollector
from auto_reports.utils.file_utils import sanitize_filename
from auto_reports.utils.retry import retry_request


class NaverCollector(BaseCollector):
    """
    Collector for Naver Finance research reports.

    Uses HTTP requests to fetch the research report list page and downloads
    PDF files directly. Supports date filtering to collect only recent reports.

    Attributes:
        headers: HTTP headers for requests

    Example:
        >>> collector = NaverCollector()
        >>> count = collector.collect(
        ...     stock_code='005930',
        ...     company_name='Samsung Electronics',
        ...     start_date='20240101'
        ... )
    """

    def __init__(self, output_dir: str = None):
        """Initialize the Naver collector."""
        super().__init__(output_dir=output_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def collect(self, stock_code: str, company_name: str,
                start_date: str = None, **kwargs) -> int:
        """
        Collect Naver Finance research reports for a company.

        Args:
            stock_code: 6-digit stock code for Naver Finance API
            company_name: Company name for folder organization
            start_date: Start date in YYYYMMDD format (only reports after this date)

        Returns:
            int: Number of reports successfully downloaded
        """
        self.logger.info(f"Naver research collection started: {company_name} (after {start_date})")
        print(f"Searching Naver research reports for [{company_name}] (after {start_date})...")

        # Setup save directory
        save_dir = os.path.join(self.output_dir, company_name)
        self._ensure_directory(save_dir)

        # Parse start date for comparison
        start_dt_obj = datetime.strptime(start_date, '%Y%m%d') if start_date else None

        url = f"https://finance.naver.com/research/company_list.naver?searchType=itemCode&itemCode={stock_code}"

        try:
            res = retry_request(requests.get, max_retries=3, base_delay=1, logger=self.logger,
                               url=url, headers=self.headers, timeout=30)
            res.encoding = 'euc-kr'
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table.type_1 tr')

            downloaded_count = 0
            for row in rows:
                pdf_td = row.select_one('td.file')
                if pdf_td and pdf_td.find('a'):
                    # Extract and parse date (YY.MM.DD -> YYYYMMDD)
                    date_raw = row.select_one('td:nth-child(5)').text.strip()
                    report_date_str = "20" + date_raw.replace('.', '')
                    report_dt_obj = datetime.strptime(report_date_str, '%Y%m%d')

                    # Skip reports before start date (list is sorted descending)
                    if start_dt_obj and report_dt_obj < start_dt_obj:
                        break

                    title = row.select_one('td:nth-child(2) a').text.strip()
                    broker = row.select_one('td:nth-child(3)').text.strip()
                    pdf_url = pdf_td.find('a')['href']

                    # Generate filename
                    date_dash = "20" + date_raw.replace('.', '-')
                    clean_title = sanitize_filename(title)
                    clean_broker = sanitize_filename(broker)
                    file_name = f"{date_dash}_[Research]_{clean_broker}_{clean_title}.pdf"
                    file_path = os.path.join(save_dir, file_name)

                    if not os.path.exists(file_path):
                        print(f"  - Downloading report: {clean_broker} - {clean_title}")
                        pdf_res = retry_request(requests.get, max_retries=3, base_delay=1, logger=self.logger,
                                               url=pdf_url, headers=self.headers, timeout=30)
                        if pdf_res.status_code != 200:
                            self.logger.warning(f"PDF download failed with HTTP {pdf_res.status_code}: {clean_title}")
                            continue
                        with open(file_path, 'wb') as f:
                            f.write(pdf_res.content)
                        downloaded_count += 1
                        self.logger.info(f"Naver report downloaded: {clean_broker} - {clean_title}")
                        self.logger.info("Waiting: 0.5 seconds")
                        time.sleep(0.5)
                    else:
                        self.logger.info(f"Duplicate file skipped: {file_name}")
                        print(f"  - Already exists: {file_name}")

            self.logger.info(f"Naver collection complete: {company_name} - {downloaded_count} downloaded")

        except Exception as e:
            self.logger.exception(f"Naver research collection error: {company_name}")
            print(f"Error during research collection: {e}")
            return 0

        return downloaded_count
