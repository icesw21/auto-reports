"""Fetch and prepare DART disclosure HTML pages for parsing."""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DART_BASE = "https://dart.fss.or.kr"


class DartHtmlFetcher:
    """Fetch DART disclosure HTML pages."""

    def __init__(self, delay: float = 1.0, timeout: int = 30) -> None:
        self.delay = delay
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_DEFAULT_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_disclosure(self, url: str) -> tuple[str, BeautifulSoup]:
        """Fetch a DART disclosure page.

        Handles the frame-based structure by detecting the inner document URL
        and fetching it.  Returns (title, soup) where soup is the BeautifulSoup
        of the actual document content.
        """
        logger.info("Fetching disclosure: %s", url)
        # Fetch raw HTML so regex frame detection can work on original source
        raw_html, main_soup = self._get_page_with_raw(url)

        title = self._extract_title(main_soup)

        doc_url = self._extract_document_url(main_soup, url, raw_html=raw_html)
        if doc_url and doc_url != url:
            logger.info("Fetching document content from: %s", doc_url)
            time.sleep(self.delay)
            doc_soup = self._get_page(doc_url)
            # Prefer title from document page if available
            doc_title = self._extract_title(doc_soup)
            if doc_title:
                title = doc_title
            return title, doc_soup

        # Fallback: return the main page soup as-is
        return title, main_soup

    def fetch_disclosure_list(
        self, urls: list[str]
    ) -> list[tuple[str, str, BeautifulSoup]]:
        """Fetch multiple disclosure pages with delay between requests.

        Returns list of (url, title, soup).
        """
        results: list[tuple[str, str, BeautifulSoup]] = []
        for i, url in enumerate(urls):
            if i > 0:
                time.sleep(self.delay)
            try:
                title, soup = self.fetch_disclosure(url)
                results.append((url, title, soup))
            except Exception:
                logger.exception("Failed to fetch disclosure: %s", url)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_document_url(
        self, main_soup: BeautifulSoup, base_url: str,
        raw_html: str = "",
    ) -> str:
        """Find the actual document URL from the main page frame structure.

        DART's main disclosure page (dsaf001/main.do) embeds the report inside
        an iframe.  This method tries several strategies in order:

        1. Look for <iframe> whose src points to the document viewer.
        2. Look for <frame> elements.
        3. Regex fallback on raw HTML (html.parser may miss <frame> in framesets).
        4. Extract viewDoc() parameters from JavaScript (dynamic iframe loading).
        5. Construct the viewer URL from the rcpNo query parameter (least reliable).
        """
        # Strategy 1 & 2: scan frame / iframe elements
        for tag_name in ("iframe", "frame"):
            for tag in main_soup.find_all(tag_name):
                src = tag.get("src", "")
                if not src:
                    continue
                if "viewer.do" in src or "report" in src.lower():
                    return urljoin(base_url, src)

        # Strategy 3: regex fallback on raw HTML
        # html.parser may not properly parse <frameset>/<frame> tags
        search_html = raw_html or str(main_soup)
        for match in re.finditer(
            r'<frame[^>]+src=["\']([^"\']*viewer\.do[^"\']*)["\']',
            search_html,
            re.IGNORECASE,
        ):
            src = match.group(1)
            if "left.do" not in src:  # Skip the menu frame
                return urljoin(base_url, src)

        # Strategy 4: extract viewDoc() parameters from JavaScript
        # DART dynamically loads the iframe via:
        #   viewDoc(rcpNo, dcmNo, eleId, offset, length, dtd[, tocNo])
        _Q = r"""['"]"""  # single or double quote
        _V = r"""[^'"]*"""  # value inside quotes
        _SEP = r"""\s*,\s*"""  # comma separator
        _ARG = _Q + "(" + _V + ")" + _Q  # quoted capture group
        vd_match = re.search(
            r"viewDoc\s*\(\s*"
            + _ARG + _SEP + _ARG + _SEP + _ARG + _SEP
            + _ARG + _SEP + _ARG + _SEP + _ARG
            + r"(?:" + _SEP + _ARG + r")?",
            search_html,
            re.IGNORECASE,
        )
        if vd_match:
            viewer_url = (
                f"{DART_BASE}/report/viewer.do"
                f"?rcpNo={vd_match.group(1)}"
                f"&dcmNo={vd_match.group(2)}"
                f"&eleId={vd_match.group(3)}"
                f"&offset={vd_match.group(4)}"
                f"&length={vd_match.group(5)}"
                f"&dtd={vd_match.group(6)}"
            )
            toc_no = vd_match.group(7)
            if toc_no:
                viewer_url += f"&tocNo={toc_no}"
            logger.debug("Extracted viewer URL from JavaScript: %s", viewer_url)
            return viewer_url

        # Strategy 5: build viewer URL from rcpNo (least reliable)
        rcp_no = self._extract_rcp_no(base_url)
        if rcp_no:
            viewer_url = (
                f"{DART_BASE}/report/viewer.do?rcpNo={rcp_no}"
            )
            logger.debug("Using constructed viewer URL: %s", viewer_url)
            return viewer_url

        return base_url

    def _get_page(self, url: str) -> BeautifulSoup:
        """Low-level HTTP GET with encoding handling (EUC-KR fallback)."""
        _, soup = self._get_page_with_raw(url)
        return soup

    def _get_page_with_raw(self, url: str) -> tuple[str, BeautifulSoup]:
        """HTTP GET returning both raw HTML text and parsed soup.

        Returns (raw_html, soup).  The raw text is needed for regex-based
        frame detection when the HTML parser misses <frame> tags.
        """
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("HTTP request failed for URL: %s", url)
            raise

        # DART pages are sometimes served as EUC-KR but declared incorrectly.
        content_type = response.headers.get("Content-Type", "")
        if "euc-kr" in content_type.lower():
            response.encoding = "euc-kr"
        elif "utf-8" not in content_type.lower():
            # Try to detect from the HTML meta charset
            raw = response.content
            if b"euc-kr" in raw[:2048].lower():
                response.encoding = "euc-kr"
            else:
                response.encoding = response.apparent_encoding or "utf-8"

        raw_html = response.text
        return raw_html, BeautifulSoup(raw_html, "html.parser")

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """Extract page title from <title> tag or <h1>."""
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    @staticmethod
    def _extract_rcp_no(url: str) -> str:
        """Parse rcpNo from a DART URL query string."""
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        values = params.get("rcpNo", [])
        return values[0] if values else ""
