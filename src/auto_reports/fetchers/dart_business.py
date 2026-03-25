"""DART report business section fetcher.

Fetches 'II. 사업의 내용' subsections from annual/quarterly reports
via OpenDartReader sub_docs navigation.
"""

from __future__ import annotations

import logging
import re
import OpenDartReader
from bs4 import BeautifulSoup

from auto_reports.fetchers.rate_limiter import dart_call_with_retry, dart_get_with_retry

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[\s\xa0]+")
_SEGMENT_PREFIX_RE = re.compile(r"^\d+\.(?:\([^)]*\))?")


def _core_section_name(normalized: str) -> str:
    """Strip number prefix and optional segment prefix.

    Examples (after whitespace normalization):
        '1.사업의개요'           → '사업의개요'
        '1.(제조서비스업)사업의개요' → '사업의개요'
        '4.(금융업)매출및수주상황'  → '매출및수주상황'
    """
    return _SEGMENT_PREFIX_RE.sub("", normalized)

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

    def _find_periodic_reports(
        self, corp: str, max_reports: int = 8,
    ) -> list[tuple[str, str]]:
        """Find the most recent periodic reports.

        Returns list of (rcept_no, report_name) sorted by filing date descending,
        up to max_reports entries.
        """
        from datetime import date

        today = date.today()
        start = f"{today.year - 3}0101"
        end = today.strftime("%Y%m%d")

        try:
            df = dart_call_with_retry(self.dart.list, corp, start=start, end=end, kind="A")
        except Exception:
            logger.exception("dart.list failed for %s", corp)
            return []

        if df is None or df.empty:
            return []

        if "rcept_dt" in df.columns:
            df = df.sort_values("rcept_dt", ascending=False)

        report_keywords = "사업보고서|반기보고서|분기보고서"
        mask = df["report_nm"].str.contains(report_keywords, na=False, regex=True)
        filtered = df[mask]

        if filtered.empty:
            return []

        results: list[tuple[str, str]] = []
        for _, row in filtered.head(max_reports).iterrows():
            results.append((str(row["rcept_no"]), str(row["report_nm"])))

        return results

    def fetch_order_backlog_history(
        self, corp: str, max_reports: int = 8,
    ) -> list[tuple[str, str, str]]:
        """Fetch order backlog (수주잔고) text from multiple periodic reports.

        Returns list of (period_label, report_name, raw_text) sorted newest-first.
        period_label is YYYYMMDD format (e.g. "20250930").
        """
        from auto_reports.fetchers.opendart import _parse_reference_date

        reports = self._find_periodic_reports(corp, max_reports)
        if not reports:
            logger.warning("No periodic reports found for order backlog: %s", corp)
            return []

        section_norm = _WHITESPACE_RE.sub("", "4. 매출 및 수주상황")
        results: list[tuple[str, str, str]] = []

        for rcept_no, report_name in reports:
            ref_date = _parse_reference_date(report_name)
            if not ref_date:
                continue

            try:
                docs = dart_call_with_retry(self.dart.sub_docs, rcept_no)
            except Exception:
                logger.debug("sub_docs failed for %s", rcept_no)
                continue

            if docs is None or docs.empty:
                continue

            # Find the "매출 및 수주상황" section URL
            matched_url = None
            section_core = _core_section_name(section_norm)
            for _, row in docs.iterrows():
                title = str(row.get("title", "")).strip()
                title_key = _WHITESPACE_RE.sub("", title)
                if section_core in _core_section_name(title_key):
                    matched_url = str(row.get("url", ""))
                    break

            if not matched_url:
                continue

            text = self._fetch_section_text(matched_url)
            if text:
                results.append((ref_date, report_name, text))
                logger.info(
                    "Backlog history: %s from %s (%d chars)",
                    ref_date, report_name, len(text),
                )

        return results

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
            df = dart_call_with_retry(self.dart.list, corp, start=start, end=end, kind="A")
        except Exception:
            logger.exception("dart.list failed for %s", corp)
            return None

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
            r = dart_get_with_retry(url, timeout=15)
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
            docs = dart_call_with_retry(self.dart.sub_docs, rcept_no)
        except Exception:
            logger.exception("sub_docs failed for %s", rcept_no)
            return None

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
            # Find matching title (core name match, ignores segment prefix)
            matched_url = None
            section_core = _core_section_name(section_norm)
            for title_key, url in title_url_map.items():
                if section_core in _core_section_name(title_key):
                    matched_url = url
                    break

            if matched_url:
                text = self._fetch_section_text(matched_url)
                setattr(sections, field, text)
                logger.info(
                    "Fetched %s: %d chars", section_title, len(text),
                )
            else:
                logger.warning("Section not found: %s", section_title)

        return sections
