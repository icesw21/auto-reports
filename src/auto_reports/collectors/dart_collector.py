"""
DART (Data Analysis, Retrieval and Transfer System) collector module.

This module provides the DartCollector class for collecting regulatory filings
from the Korean Financial Supervisory Service's DART system.
"""

import json
import os
import re
import time
from datetime import datetime

import OpenDartReader
import requests

from auto_reports.collectors.base import BaseCollector
from auto_reports.utils.file_utils import sanitize_filename
from auto_reports.utils.retry import retry_request


class DartCollector(BaseCollector):
    """
    Collector for DART regulatory filings.

    Uses the OpenDartReader library to access the DART API and download
    regulatory filings as PDF files. Falls back to URL lists for filings
    that cannot be downloaded directly (e.g., exchange disclosures).
    """

    def __init__(self, api_key: str = None, output_dir: str = None):
        """Initialize the DART collector with API credentials."""
        super().__init__(output_dir=output_dir)
        if not api_key:
            from auto_reports.config import Settings
            api_key = Settings().dart_api_key
        self.dart = OpenDartReader(api_key)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://dart.fss.or.kr/'
        }

    def collect(self, stock_code: str, company_name: str,
                start_date: str = None, end_date: str = None,
                keywords: list[str] = None, **kwargs) -> int:
        """
        Collect DART regulatory filings for a company.

        Args:
            stock_code: 6-digit stock code (can be empty string for name-based search)
            company_name: Company name for folder organization and fallback search
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            keywords: List of keywords to filter filings

        Returns:
            int: Number of PDF files successfully downloaded
        """
        self.logger.info(f"DART collection started: {company_name} ({start_date} ~ {end_date})")
        print(f"Searching DART filings for [{company_name}] ({start_date} ~ {end_date})...")

        # Query DART API
        if stock_code == '':
            df = self.dart.list(company_name, start=start_date, end=end_date)
        else:
            df = self.dart.list(stock_code, start=start_date, end=end_date)

        if df is None or df.empty:
            self.logger.warning(f"No DART results for: {company_name}")
            print(f"[{company_name}] No filings found.")
            return 0

        # Filter by keywords
        if keywords:
            safe_keywords = [re.escape(k) for k in keywords]
            pattern = '|'.join(safe_keywords)
            filtered_df = df[df['report_nm'].str.contains(pattern, case=False, na=False, regex=True)].copy()
        else:
            filtered_df = df.copy()

        self.logger.info(f"DART filtering complete: {company_name} - {len(filtered_df)} filings found")

        # Prepare save directory
        save_dir = os.path.join(self.output_dir, company_name)
        self._ensure_directory(save_dir)

        # Error URL lists for failed downloads
        error_url_list = []
        error_url_list_full = []
        downloaded_count = 0

        for _, row in filtered_df.iterrows():
            rcept_no = row['rcept_no']
            report_nm = sanitize_filename(row['report_nm'])
            file_name = f"{row['rcept_dt']}_[Filing]_{rcept_no}_{report_nm}.pdf"
            file_path = os.path.join(save_dir, file_name)

            if os.path.exists(file_path):
                self.logger.info(f"Duplicate file skipped: {file_name}")
                continue

            # Step 1: Retrieve sub-document list (may fail for exchange disclosures)
            try:
                sub_df = self.dart.sub_docs(rcept_no)
            except Exception:
                # OpenDartReader raises NameError for filings without sub-documents
                self.logger.info(f"No sub_docs available (exchange disclosure): {report_nm}")
                print(f"  - Exchange disclosure: {report_nm}")
                view_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
                error_url_list.append(view_url)
                error_url_list_full.append(f"[{row['rcept_dt']}] | {row['report_nm']} | {view_url}")
                continue

            # Step 2: Extract PDF download URL from sub-document
            try:
                if sub_df is None or sub_df.empty or 'url' not in sub_df.columns:
                    raise ValueError("No downloadable sub_docs")

                first_url = sub_df.iloc[0]['url']
                match = re.search(r'dcmNo=(\d+)', first_url)
                if not match:
                    raise ValueError("Cannot parse dcmNo")

                dcm_no = match.group(1)
                download_url = f"https://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcept_no}&dcm_no={dcm_no}"

                print(f"  - Downloading filing: {report_nm}")
                res = retry_request(requests.get, max_retries=3, base_delay=1, logger=self.logger,
                                   url=download_url, headers=self.headers, timeout=30)
                if res.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(res.content)
                    downloaded_count += 1
                    self.logger.info(f"DART filing downloaded: {report_nm}")
                    self.logger.info("Waiting: 0.5 seconds")
                    time.sleep(0.5)
                else:
                    raise ValueError(f"HTTP {res.status_code}")

            except Exception as e:
                # PDF download failed - save to URL fallback list
                self.logger.warning(f"DART PDF download failed: {report_nm} - {e}")
                print(f"  - Exchange disclosure: {report_nm}")
                view_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
                error_url_list.append(view_url)
                error_url_list_full.append(f"[{row['rcept_dt']}] | {row['report_nm']} | {view_url}")

        # Save error URL lists in multiple formats
        if error_url_list:
            # 1. JSON format (structured data like news_collector)
            disclosure_json = []
            for i, full_str in enumerate(error_url_list_full):
                # Parse: [YYYYMMDD] | Title | URL
                parts = full_str.split(' | ')
                if len(parts) == 3:
                    date_str = parts[0].strip('[] ')
                    title = parts[1].strip()
                    url = parts[2].strip()
                    # Format date as YYYY-MM-DD
                    try:
                        date_obj = datetime.strptime(date_str, '%Y%m%d')
                        date_formatted = date_obj.strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        date_formatted = date_str

                    disclosure_json.append({
                        'date': date_formatted,
                        'title': title,
                        'url': url,
                        'source': 'dart'
                    })

            # Save JSON file
            if disclosure_json:
                json_path = os.path.join(save_dir, f"{company_name}_exchange_disclosure.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(disclosure_json, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Exchange disclosure JSON saved: {json_path} ({len(disclosure_json)} items)")
                print(f"Exchange disclosure JSON saved: {json_path}")

            # 2. Copy-friendly version (URLs only)
            txt_dir = os.path.join(save_dir, 'txt')
            self._ensure_directory(txt_dir)
            txt_path = os.path.join(txt_dir, f"{company_name}_exchange_disclosure_list(copy).txt")
            price_link = f"\nhttps://stock.naver.com/domestic/stock/{stock_code}/price"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(error_url_list))
                f.write(price_link)

            self.logger.info(f"Exchange disclosure list created: {txt_path} ({len(error_url_list)} items)")
            print(f"Exchange disclosure list saved: {txt_path}")

        self.logger.info(f"DART collection complete: {company_name} - PDF {downloaded_count}, Exchange {len(error_url_list)}")

        return downloaded_count
