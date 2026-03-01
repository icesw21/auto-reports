"""DART report business section fetcher.

Fetches 'II. 사업의 내용' subsections from annual/quarterly reports
via OpenDartReader sub_docs navigation.
"""

from __future__ import annotations

import logging
import re
import time

import OpenDartReader
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[\s\xa0]+")

# Sections we want from "II. 사업의 내용"
_TARGET_SECTIONS = {
    "사업개요": "1. 사업의 개요",
    "주요제품": "2. 주요 제품 및 서비스",
    "원재료": "3. 원재료 및 생산설비",
    "매출수주": "4. 매출 및 수주상황",
}


class BusinessSections:
    """Container for extracted business section text."""

    def __init__(self) -> None:
        self.사업개요: str = ""
        self.주요제품: str = ""
        self.원재료: str = ""
        self.매출수주: str = ""
        self.report_title: str = ""  # e.g. "사업보고서 (2024.12)"
        self.rcept_no: str = ""


class DartBusinessFetcher:
    """Fetches business content sections from DART periodic reports."""

    def __init__(self, api_key: str, delay: float = 0.5) -> None:
        self.dart = OpenDartReader(api_key)
        self.delay = delay

    def _find_latest_report_rcept(self, corp: str) -> tuple[str, str] | None:
        """Find the most recent periodic report receipt number.

        Returns (rcept_no, report_name) or None.

        Prefers the most recently filed report by date, regardless of type.
        This ensures the latest quarterly report is used over an older
        annual report (e.g. Q3 2025 report filed Nov 2025 vs annual 2024
        filed Mar 2025).
        """
        from datetime import date

        today = date.today()
        start = f"{today.year - 2}0101"
        end = today.strftime("%Y%m%d")

        try:
            df = self.dart.list(corp, start=start, end=end, kind="A")
        except Exception:
            logger.exception("dart.list failed for %s", corp)
            return None
        time.sleep(self.delay)

        if df is None or df.empty:
            return None

        # Sort by receipt date descending to get most recent first
        if "rcept_dt" in df.columns:
            df = df.sort_values("rcept_dt", ascending=False)

        # Filter to only periodic reports (사업/반기/분기보고서)
        report_keywords = "사업보고서|반기보고서|분기보고서"
        mask = df["report_nm"].str.contains(report_keywords, na=False, regex=True)
        filtered = df[mask]

        if not filtered.empty:
            # Take the most recent by filing date (already sorted descending)
            row = filtered.iloc[0]
            return str(row["rcept_no"]), str(row["report_nm"])

        return None

    def _fetch_section_text(self, url: str) -> str:
        """Fetch a section URL and return clean text content."""
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            logger.exception("Failed to fetch section: %s", url)
            return ""

    def fetch_business_sections(self, corp: str) -> BusinessSections | None:
        """Fetch all business content sections for a company.

        Args:
            corp: Ticker or corp_code.

        Returns:
            BusinessSections with text content, or None if no report found.
        """
        result = self._find_latest_report_rcept(corp)
        if not result:
            logger.warning("No periodic report found for %s", corp)
            return None

        rcept_no, report_name = result
        logger.info("Fetching business sections from: %s (%s)", report_name, rcept_no)

        try:
            docs = self.dart.sub_docs(rcept_no)
        except Exception:
            logger.exception("sub_docs failed for %s", rcept_no)
            return None
        time.sleep(self.delay)

        if docs is None or docs.empty:
            return None

        sections = BusinessSections()
        sections.report_title = report_name
        sections.rcept_no = rcept_no

        # Build title -> url map
        title_url_map: dict[str, str] = {}
        for _, row in docs.iterrows():
            title = str(row.get("title", "")).strip()
            url = str(row.get("url", ""))
            title_norm = _WHITESPACE_RE.sub("", title)
            title_url_map[title_norm] = url

        # Fetch each target section
        for field, section_title in _TARGET_SECTIONS.items():
            section_norm = _WHITESPACE_RE.sub("", section_title)
            # Find matching title (substring match for flexibility)
            matched_url = None
            for title_key, url in title_url_map.items():
                if section_norm in title_key:
                    matched_url = url
                    break

            if matched_url:
                text = self._fetch_section_text(matched_url)
                setattr(sections, field, text)
                logger.info(
                    "Fetched %s: %d chars", section_title, len(text),
                )
                time.sleep(self.delay)
            else:
                logger.warning("Section not found: %s", section_title)

        return sections
