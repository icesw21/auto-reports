"""DART Report HTML parser for financial statement extraction.

Fallback for when the structured finstate_all API lacks data.
Fetches report HTML via sub_docs URLs and parses financial tables.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse

import OpenDartReader
import requests
from bs4 import BeautifulSoup, Tag

from auto_reports.models.financial import BalanceSheet, IncomeStatementItem

logger = logging.getLogger(__name__)

# ── Amount pattern: standard Korean won format ──
# Matches: "1,234,567", "(1,234,567)", "0", but NOT "4,5,6,32" (note refs)
_AMOUNT_RE = re.compile(r"^\(?\d{1,3}(,\d{3})*\)?$")

# Roman numeral prefixes (Unicode fullwidth + ASCII)
_ROMAN_PREFIX_RE = re.compile(
    r"^[IVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫivx]+[.\s]+",
)
_ARABIC_PREFIX_RE = re.compile(r"^\d+[.\s]+")
_WHITESPACE_RE = re.compile(r"[\s\xa0]+")


# ── Account name normalization ──


def _normalize_account_name(name: str) -> str:
    """Normalize a Korean account name for matching.

    Strips Roman/Arabic prefixes, whitespace, and nbsp.
    Prefix stripping runs before whitespace collapse so that
    space-separated prefixes like "1 자산총계" are handled correctly.
    """
    name = name.strip()
    name = _ROMAN_PREFIX_RE.sub("", name)
    name = _ARABIC_PREFIX_RE.sub("", name)
    name = _WHITESPACE_RE.sub("", name)
    return name


# ── Field matching maps ──

_BS_FIELDS: dict[str, list[str]] = {
    "total_assets": ["자산총계"],
    "cash_and_equivalents": ["현금및현금성자산"],
    "short_term_investments": ["단기금융상품"],
    "total_liabilities": ["부채총계"],
    "short_term_borrowings": ["단기차입금"],
    "current_long_term_debt": ["유동성장기부채", "유동성장기차입금"],
    "bonds": ["사채"],
    "total_equity": ["자본총계"],
}

_IS_FIELDS: dict[str, list[str]] = {
    "revenue": ["매출액", "수익(매출액)", "영업수익"],
    "operating_income": ["영업이익", "영업이익(손실)", "영업손익", "영업손실"],
    "net_income": [
        "당기순이익",
        "당기순이익(손실)",
        "당기순손실(이익)",
        "당기순손실",
        "반기순이익",
        "반기순이익(손실)",
        "반기순손실(이익)",
        "반기순손실",
        "분기순이익",
        "분기순이익(손실)",
        "분기순손실(이익)",
        "분기순손실",
    ],
}


# ── Amount parsing ──


def _parse_report_amount(text: str) -> int | None:
    """Parse a won amount from report HTML cell text."""
    text = text.strip().replace("\xa0", "").replace(" ", "")
    if not text or text == "-":
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace(",", "")
    try:
        value = int(float(text))
        return -value if negative else value
    except (ValueError, TypeError):
        return None


def _is_amount_cell(text: str) -> bool:
    """Check if text looks like a formatted financial amount.

    Distinguishes real amounts (1,234,567) from note references (4,5,6,32).
    """
    text = text.strip().replace("\xa0", "").replace(" ", "")
    if not text or text == "-" or text == "0":
        return True
    return bool(_AMOUNT_RE.match(text))


# ── Table identification ──


def _identify_table_type(header_table: Tag) -> str | None:
    """Identify a header table's financial statement type."""
    text = _WHITESPACE_RE.sub("", header_table.get_text())
    if "재무상태표" in text:
        return "BS"
    if "손익계산서" in text or "포괄손익계산서" in text:
        return "IS"
    if "자본변동표" in text:
        return "SCE"
    if "현금흐름표" in text:
        return "CF"
    return None


def _identify_by_content(data_table: Tag) -> str | None:
    """Identify table type from data content (fallback for older reports).

    Older DART reports omit the statement type name from the header table,
    so we infer the type from the first few rows of the data table.
    """
    rows = data_table.find_all("tr")
    # Collect text from the first ~8 rows for identification
    sample = ""
    for row in rows[:8]:
        sample += _WHITESPACE_RE.sub("", row.get_text())

    # Check order matters: BS first (most distinctive), then IS, SCE, CF
    if "자산총계" in sample or ("자산" in sample and "유동자산" in sample):
        return "BS"
    if "매출액" in sample or "영업수익" in sample:
        return "IS"
    if "자본변동" in sample or ("자본금" in sample and "자본잉여금" in sample):
        return "SCE"
    if "영업활동" in sample:
        return "CF"
    return None


_SUBSTRING_EXCLUSIONS: dict[str, list[str]] = {
    "사채": ["유동성", "전환", "할인", "할증", "상환", "교환"],
    "자본총계": ["부채"],
    "부채총계": ["자본"],
}


def _match_field(
    normalized_name: str, field_map: dict[str, list[str]],
) -> str | None:
    """Match a normalized account name to a model field."""
    # Exact match
    for field, patterns in field_map.items():
        for pattern in patterns:
            if normalized_name == pattern:
                return field
    # Substring match (with exclusions for ambiguous names)
    for field, patterns in field_map.items():
        for pattern in patterns:
            if pattern in normalized_name:
                exclusions = _SUBSTRING_EXCLUSIONS.get(pattern, [])
                if any(excl in normalized_name for excl in exclusions):
                    continue
                return field
    return None


# ── Data table parsing ──


def _parse_data_table(
    data_table: Tag,
    amount_col: int = 0,
) -> list[tuple[str, int | None]]:
    """Parse a financial data table into (account_name, current_period_amount) pairs.

    Handles both column layouts:
    - With notes column: [과목, 주석, 당기, (전기...)]
    - Without notes: [과목, 당기, (전기...)]
    - Quarterly IS with 3개월/누적 sub-cols (first amount = 3-month current)

    Args:
        data_table: The HTML table element to parse.
        amount_col: Which valid-amount column to extract (0-based).
            For quarterly IS: 0=3개월당기, 1=3개월전기, 2=누적당기, 3=누적전기.
            For annual IS/BS: 0=당기, 1=전기.
    """
    rows = data_table.find_all("tr")
    if not rows:
        return []

    results = []
    for row in rows[1:]:  # Skip header row(s)
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if not cells or not any(cells):
            continue

        # Check for sub-header rows like "3개월", "누적" — skip
        first_clean = _WHITESPACE_RE.sub("", cells[0])
        if first_clean in ("3개월", "누적", ""):
            continue

        # Cell 0 = account name
        account_name = cells[0].replace("\xa0", " ").strip()
        if not account_name:
            continue

        # Find the Nth valid amount cell (skip empty & note cells)
        amount = None
        found_count = 0
        for cell in cells[1:]:
            cleaned = cell.strip().replace("\xa0", "").replace(" ", "")
            if not cleaned:
                continue  # Skip genuinely empty cells
            # Short integers without commas are likely note references (e.g. "15")
            if len(cleaned) <= 4 and "," not in cleaned and cleaned.isdigit():
                continue
            if _is_amount_cell(cell):
                if found_count == amount_col:
                    amount = _parse_report_amount(cell)
                    break
                found_count += 1

        normalized = _normalize_account_name(account_name)
        if normalized:
            results.append((normalized, amount))

    return results


# ── Main fetcher class ──


class DartReportFetcher:
    """Fetches and parses DART report HTML for financial data."""

    def __init__(
        self, api_key: str, delay: float = 0.5, fs_pref: str = "연결",
    ) -> None:
        self.dart = OpenDartReader(api_key)
        self.delay = delay
        self.fs_pref = fs_pref

    # ------------------------------------------------------------------
    # Report discovery
    # ------------------------------------------------------------------

    def find_rcept_no(
        self, corp: str, year: int, quarter: int = 0,
    ) -> str | None:
        """Find the receipt number for a report filing.

        Args:
            corp: Ticker or corp_code.
            year: Business year.
            quarter: 0=annual, 1=Q1, 2=H1, 3=Q3.
        """
        # Wide search window to catch late filings and corrections
        # (e.g. 기재정정 filings can appear years after the original)
        from datetime import date as _date

        today_str = _date.today().strftime("%Y%m%d")

        if quarter == 0:
            start = f"{year}0101"
            end = today_str
            keyword = "사업보고서"
            period_hint = f"({year}.12)"
        elif quarter == 1:
            start = f"{year}0101"
            end = today_str
            keyword = "분기보고서"
            period_hint = f"({year}.03)"
        elif quarter == 2:
            start = f"{year}0101"
            end = today_str
            keyword = "반기보고서"
            period_hint = f"({year}.06)"
        elif quarter == 3:
            start = f"{year}0101"
            end = today_str
            keyword = "분기보고서"
            period_hint = f"({year}.09)"
        else:
            return None

        try:
            df = self.dart.list(corp, start=start, end=end, kind="A")
        except Exception:
            logger.exception("dart.list failed for %s", corp)
            return None
        time.sleep(self.delay)

        if df is None or df.empty:
            return None

        # Filter by report type keyword
        mask = df["report_nm"].str.contains(keyword, na=False)
        filtered = df[mask]

        # Further filter by period hint (e.g. "(2023.12)")
        period_mask = filtered["report_nm"].str.contains(
            re.escape(period_hint), na=False, regex=True,
        )
        if period_mask.any():
            filtered = filtered[period_mask]
        else:
            # No report matches this specific year/period — don't return wrong year
            return None

        if filtered.empty:
            return None

        # Most recent filing first (dart.list returns descending by date)
        return str(filtered.iloc[0]["rcept_no"])

    # ------------------------------------------------------------------
    # HTML section fetching
    # ------------------------------------------------------------------

    def _fetch_section_html(
        self, rcept_no: str,
    ) -> tuple[str, str] | None:
        """Fetch the financial statement section HTML.

        Returns (html_text, statement_type_label) or None.
        Prefers consolidated (연결) or individual (별도) based on fs_pref.
        """
        try:
            docs = self.dart.sub_docs(rcept_no)
        except Exception:
            logger.exception("sub_docs failed for %s", rcept_no)
            return None
        time.sleep(self.delay)

        if docs is None or docs.empty:
            return None

        consolidated_url = None
        individual_url = None
        con_size = 0
        ind_size = 0

        for _, row in docs.iterrows():
            title = str(row.get("title", ""))
            url = str(row.get("url", ""))

            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            length = int(parsed.get("length", ["0"])[0])

            title_norm = _WHITESPACE_RE.sub("", title)
            if "연결재무제표" in title_norm and "주석" not in title_norm:
                consolidated_url = url
                con_size = length
            elif (
                title_norm.endswith("재무제표")
                and "주석" not in title_norm
                and "연결" not in title_norm
            ):
                individual_url = url
                ind_size = length

        # Choose section: prefer fs_pref, fall back to the other
        chosen_url = None
        stmt_type = ""
        min_size = 500  # Sections < 500 bytes are just "해당사항 없음"

        if self.fs_pref == "연결" and consolidated_url and con_size > min_size:
            chosen_url = consolidated_url
            stmt_type = "연결"
        elif individual_url and ind_size > min_size:
            chosen_url = individual_url
            stmt_type = "별도"
        elif consolidated_url and con_size > min_size:
            chosen_url = consolidated_url
            stmt_type = "연결"

        if not chosen_url:
            logger.warning("No financial section found: rcept_no=%s", rcept_no)
            return None

        try:
            r = requests.get(chosen_url, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text, stmt_type
        except Exception:
            logger.exception("Failed to fetch report HTML: %s", chosen_url)
            return None

    # ------------------------------------------------------------------
    # Table parsing
    # ------------------------------------------------------------------

    def _parse_tables(
        self, html: str, is_amount_col: int = 0,
    ) -> dict[str, list[tuple[str, int | None]]]:
        """Parse HTML into identified financial statement data.

        Returns e.g. {'BS': [(name, amount), ...], 'IS': [...]}.
        Tables come in header+data pairs.  Two identification strategies:
        1. Header-based: statement name in even-indexed header table.
        2. Content-based fallback: infer type from data table rows
           (older reports omit the statement name from headers).

        Args:
            is_amount_col: Which amount column to use for IS tables (0=first).
                BS tables always use column 0.
        """
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")

        result: dict[str, list[tuple[str, int | None]]] = {}
        i = 0
        while i < len(tables) - 1:
            # Strategy 1: identify by header keywords
            table_type = _identify_table_type(tables[i])
            if table_type and table_type not in result:
                col = is_amount_col if table_type == "IS" else 0
                data = _parse_data_table(tables[i + 1], amount_col=col)
                if data:
                    result[table_type] = data
                i += 2
                continue

            # Strategy 2: identify by data table content
            if not table_type:
                table_type = _identify_by_content(tables[i + 1])
                if table_type and table_type not in result:
                    col = is_amount_col if table_type == "IS" else 0
                    data = _parse_data_table(tables[i + 1], amount_col=col)
                    if data:
                        result[table_type] = data
                    i += 2
                    continue

            i += 1

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_balance_sheet(
        self, corp: str, year: int, quarter: int = 0,
    ) -> BalanceSheet:
        """Fetch and parse balance sheet from report HTML."""
        period_label = str(year) if quarter == 0 else f"{year}.Q{quarter}"
        logger.info(
            "Report HTML: fetching BS for %s %d Q%d", corp, year, quarter,
        )

        rcept_no = self.find_rcept_no(corp, year, quarter)
        if not rcept_no:
            logger.warning("No report found: %s %d Q%d", corp, year, quarter)
            return BalanceSheet(period=period_label)

        result = self._fetch_section_html(rcept_no)
        if not result:
            return BalanceSheet(period=period_label)

        html, stmt_type = result
        tables = self._parse_tables(html)
        bs_data = tables.get("BS", [])

        if not bs_data:
            return BalanceSheet(period=period_label)

        bs = BalanceSheet(period=period_label, statement_type=stmt_type)
        for name, amount in bs_data:
            field = _match_field(name, _BS_FIELDS)
            if field and amount is not None:
                setattr(bs, field, amount)

        return bs

    def get_income_statement(
        self, corp: str, year: int, quarter: int = 0,
        cumulative: bool = False,
    ) -> IncomeStatementItem:
        """Fetch and parse income statement from report HTML.

        Args:
            cumulative: When True and quarter > 1, extract the cumulative
                (누적) column instead of the 3-month standalone column.
                For Q1 or annual reports, this has no effect (Q1 cum = standalone).
        """
        period_label = str(year) if quarter == 0 else f"{year}.Q{quarter}"
        logger.info(
            "Report HTML: fetching IS for %s %d Q%d cum=%s",
            corp, year, quarter, cumulative,
        )

        rcept_no = self.find_rcept_no(corp, year, quarter)
        if not rcept_no:
            return IncomeStatementItem(period=period_label)

        result = self._fetch_section_html(rcept_no)
        if not result:
            return IncomeStatementItem(period=period_label)

        html, stmt_type = result

        # For cumulative quarterly (H1, Q3), use amount column 2 (누적당기)
        # Q1 cumulative = standalone, so no change needed
        is_col = 2 if (cumulative and quarter > 1) else 0
        tables = self._parse_tables(html, is_amount_col=is_col)
        is_data = tables.get("IS", [])

        # Fallback: if cumulative column yielded no values, use first column
        if cumulative and quarter > 1 and is_col > 0:
            has_values = any(amount is not None for _, amount in is_data)
            if not has_values:
                logger.debug(
                    "Cumulative column empty for %d Q%d, falling back to col 0",
                    year, quarter,
                )
                tables = self._parse_tables(html, is_amount_col=0)
                is_data = tables.get("IS", [])

        if not is_data:
            return IncomeStatementItem(period=period_label)

        item = IncomeStatementItem(period=period_label)
        for name, amount in is_data:
            field = _match_field(name, _IS_FIELDS)
            if field and amount is not None:
                setattr(item, field, amount)

        return item
