"""DART supplementary section fetcher for shareholder, dividend, and subsidiary data.

Fetches specific sections from 반기보고서/사업보고서 (NOT 분기보고서) via
OpenDartReader sub_docs navigation, then returns raw HTML text for LLM extraction.
"""

from __future__ import annotations

import logging
import re
import OpenDartReader
from bs4 import BeautifulSoup

from auto_reports.fetchers.rate_limiter import dart_call_with_retry, dart_get_with_retry

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[\s\xa0]+")


class DartSupplementaryFetcher:
    """Fetches supplementary report sections (shareholder, dividend, subsidiary)."""

    def __init__(self, api_key: str, delay: float = 0.5) -> None:
        self.dart = OpenDartReader(api_key)
        self.delay = delay

    # ------------------------------------------------------------------
    # Report discovery (반기/사업보고서 only)
    # ------------------------------------------------------------------

    def _find_half_or_annual_rcept(self, corp: str) -> tuple[str, str] | None:
        """Find the most recent 반기보고서 or 사업보고서 receipt number.

        Excludes 분기보고서 because 배당/종속회사 data is only in 반기/사업보고서.
        Returns (rcept_no, report_name) or None.
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

        # Sort by receipt date descending
        if "rcept_dt" in df.columns:
            df = df.sort_values("rcept_dt", ascending=False)

        # Filter to 반기보고서 or 사업보고서 only (exclude 분기보고서)
        mask = df["report_nm"].str.contains(
            r"사업보고서|반기보고서", na=False, regex=True,
        )
        # Exclude 분기보고서 that might match
        exclude = df["report_nm"].str.contains("분기보고서", na=False)
        filtered = df[mask & ~exclude]

        if not filtered.empty:
            row = filtered.iloc[0]
            return str(row["rcept_no"]), str(row["report_nm"])

        return None

    # ------------------------------------------------------------------
    # Section fetching
    # ------------------------------------------------------------------

    def _fetch_sub_docs(self, rcept_no: str) -> dict[str, str]:
        """Fetch sub_docs for a report and build title->url map."""
        try:
            docs = dart_call_with_retry(self.dart.sub_docs, rcept_no)
        except Exception:
            logger.exception("sub_docs failed for %s", rcept_no)
            return {}

        if docs is None or docs.empty:
            return {}

        title_url_map: dict[str, str] = {}
        for _, row in docs.iterrows():
            title = str(row.get("title", "")).strip()
            url = str(row.get("url", ""))
            title_norm = _WHITESPACE_RE.sub("", title)
            title_url_map[title_norm] = url

        return title_url_map

    def _fetch_section_text(self, url: str) -> str:
        """Fetch section HTML and return clean text."""
        try:
            r = dart_get_with_retry(url, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            logger.exception("Failed to fetch section: %s", url)
            return ""

    def _find_section_url(
        self, title_url_map: dict[str, str], keywords: list[str],
    ) -> str | None:
        """Find a section URL by matching keywords in normalized titles.

        Keywords are tried in order (most specific first), so the first
        keyword that matches any title wins.
        """
        for kw in keywords:
            kw_norm = _WHITESPACE_RE.sub("", kw)
            for title_key, url in title_url_map.items():
                if kw_norm in title_key:
                    return url
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_supplementary_sections(
        self, corp: str,
    ) -> dict[str, str]:
        """Fetch all supplementary sections (shareholder, dividend, subsidiary).

        Returns dict with keys: 'shareholder', 'dividend', 'subsidiary',
        'report_name'. Values are raw text from DART HTML sections.
        """
        result = {"shareholder": "", "dividend": "", "subsidiary": "", "report_name": ""}

        rcept_info = self._find_half_or_annual_rcept(corp)
        if not rcept_info:
            logger.warning("No 반기/사업보고서 found for %s", corp)
            return result

        rcept_no, report_name = rcept_info
        result["report_name"] = report_name
        logger.info("Fetching supplementary sections from: %s (%s)", report_name, rcept_no)

        title_url_map = self._fetch_sub_docs(rcept_no)
        if not title_url_map:
            return result

        # 1. Shareholder: VII. 주주에 관한 사항
        shareholder_url = self._find_section_url(title_url_map, [
            "최대주주및특수관계인의주식소유현황",
            "최대주주및그특수관계인의주식소유현황",
            "주주에관한사항",
        ])
        if shareholder_url:
            result["shareholder"] = self._fetch_section_text(shareholder_url)
            logger.info("Fetched shareholder section: %d chars", len(result["shareholder"]))

        # 2. Dividend: III. 재무에 관한 사항 - 6. 배당에 관한 사항
        dividend_url = self._find_section_url(title_url_map, [
            "배당에관한사항",
        ])
        if dividend_url:
            result["dividend"] = self._fetch_section_text(dividend_url)
            logger.info("Fetched dividend section: %d chars", len(result["dividend"]))

        # 3. Subsidiary: XII. 상세표 - 1. 연결대상 종속회사 현황
        subsidiary_url = self._find_section_url(title_url_map, [
            "연결대상종속회사현황(상세)",
            "연결대상종속회사현황",
        ])
        if subsidiary_url:
            result["subsidiary"] = self._fetch_section_text(subsidiary_url)
            logger.info("Fetched subsidiary section: %d chars", len(result["subsidiary"]))

        return result
