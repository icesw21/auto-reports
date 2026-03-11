"""Parser for 매출액또는손익구조 변동 disclosures."""

from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup, Tag

from auto_reports.models.disclosure import Performance
from auto_reports.parsers.base import clean_text, table_to_dict, table_to_rows

logger = logging.getLogger(__name__)

# Keys for the income changes section
_INCOME_KEYS = ("- 매출액", "- 영업이익", "- 당기순이익")
# Keys for the financial status section
_FINANCIAL_KEYS = ("- 자산총계", "- 부채총계", "- 자본총계", "- 자본금")
# IncomeItem field aliases
_INCOME_ITEM_ALIASES = ("당해사업연도", "직전사업연도", "증감금액", "증감비율(%)", "흑자적자전환여부")
# FinancialStatus field aliases
_FINANCIAL_STATUS_ALIASES = ("당해사업연도", "직전사업연도")


def _is_income_table(header_text: str) -> bool:
    return "당해사업연도" in header_text and "증감금액" in header_text


def _is_financial_table(header_text: str) -> bool:
    return "당해사업연도" in header_text and ("자산" in header_text or "재무현황" in header_text)


def _build_income_changes(rows: list[dict[str, str]]) -> dict:
    """Build the income_changes dict from a multi-row income table.

    The table is expected to have a first-column label (e.g. "- 매출액") and
    columns matching IncomeItem aliases.
    """
    result: dict = {}
    for row in rows:
        # First non-empty value in the row is the label
        label = ""
        data: dict = {}
        cells = list(row.items())
        if not cells:
            continue
        label = clean_text(cells[0][1]) if cells else ""
        if not label:
            continue
        for alias in _INCOME_ITEM_ALIASES:
            if alias in row:
                data[alias] = row[alias]
        if data:
            result[label] = data
    return result


def _build_financial_status(rows: list[dict[str, str]]) -> dict:
    """Build the financial_status dict from a multi-row financial table."""
    result: dict = {}
    for row in rows:
        cells = list(row.items())
        if not cells:
            continue
        label = clean_text(cells[0][1]) if cells else ""
        if not label:
            continue
        data: dict = {}
        for alias in _FINANCIAL_STATUS_ALIASES:
            if alias in row:
                data[alias] = row[alias]
        if data:
            result[label] = data
    return result


def _parse_merged_table(
    table: "Tag",
) -> tuple[Optional[dict], Optional[dict], dict[str, str]]:
    """Parse income/financial sections from a single merged table.

    Some DART disclosures embed all sections (metadata, income, financial)
    in one table instead of separate tables.  This function scans row-by-row
    for section headers and extracts each section accordingly.

    Returns (income_changes, financial_status, kv_pairs).
    """
    rows = table.find_all("tr")
    income_changes: Optional[dict] = None
    financial_status: Optional[dict] = None
    kv_pairs: dict[str, str] = {}

    i = 0
    while i < len(rows):
        cells = rows[i].find_all(["td", "th"])
        if not cells:
            i += 1
            continue

        cell_texts = [clean_text(c.get_text()) for c in cells]
        row_text = " ".join(cell_texts)

        if _is_income_table(row_text):
            # This row is the income section header
            headers = cell_texts
            data_rows: list[dict[str, str]] = []
            i += 1
            while i < len(rows):
                data_cells = rows[i].find_all(["td", "th"])
                data_texts = [clean_text(c.get_text()) for c in data_cells]
                joined = " ".join(data_texts)
                # Stop at the next numbered section header
                if _is_financial_table(joined) or re.match(r"^\d+\.", data_texts[0] if data_texts else ""):
                    break
                row_dict: dict[str, str] = {}
                for j, text in enumerate(data_texts):
                    key = headers[j] if j < len(headers) else f"col_{j}"
                    row_dict[key] = text
                data_rows.append(row_dict)
                i += 1
            income_changes = _build_income_changes(data_rows)
            continue

        elif _is_financial_table(row_text):
            headers = cell_texts
            data_rows2: list[dict[str, str]] = []
            i += 1
            while i < len(rows):
                data_cells = rows[i].find_all(["td", "th"])
                data_texts = [clean_text(c.get_text()) for c in data_cells]
                joined = " ".join(data_texts)
                if re.match(r"^\d+\.", data_texts[0] if data_texts else ""):
                    break
                row_dict2: dict[str, str] = {}
                for j, text in enumerate(data_texts):
                    key = headers[j] if j < len(headers) else f"col_{j}"
                    row_dict2[key] = text
                data_rows2.append(row_dict2)
                i += 1
            financial_status = _build_financial_status(data_rows2)
            continue

        else:
            # Key-value row
            if len(cells) >= 2:
                key = clean_text(cells[0].get_text())
                value = clean_text(cells[1].get_text())
                if key:
                    kv_pairs[key] = value
            i += 1

    return income_changes, financial_status, kv_pairs


def parse_performance(soup: BeautifulSoup) -> Performance:
    """Parse a 매출액또는손익구조 변동 disclosure into a Performance model.

    Args:
        soup: BeautifulSoup object for the disclosure document.

    Returns:
        Performance populated with whatever data could be extracted.
    """
    tables = soup.find_all("table")
    all_kv: dict[str, str] = {}
    income_changes: Optional[dict] = None
    financial_status: Optional[dict] = None
    period_dict: Optional[dict] = None

    consumed: set[int] = set()  # track tables consumed by dedicated parsers
    for idx, table in enumerate(tables):
        rows_raw = table.find_all("tr")
        if not rows_raw:
            continue

        header_text = " ".join(
            clean_text(cell.get_text())
            for cell in rows_raw[0].find_all(["th", "td"])
        )

        if _is_income_table(header_text):
            rows = table_to_rows(table)
            income_changes = _build_income_changes(rows)
            consumed.add(idx)
            continue

        if _is_financial_table(header_text):
            rows = table_to_rows(table)
            financial_status = _build_financial_status(rows)
            consumed.add(idx)
            continue

        kv = table_to_dict(table)
        all_kv.update(kv)

    # Fallback: scan for merged tables where all sections are in one table
    if not income_changes or not financial_status:
        for idx, table in enumerate(tables):
            if idx in consumed:
                continue
            ic, fs, extra_kv = _parse_merged_table(table)
            if ic and not income_changes:
                income_changes = ic
            if fs and not financial_status:
                financial_status = fs
            all_kv.update(extra_kv)

    # Statement type
    statement_type = (
        all_kv.get("1. 재무제표의 종류")
        or all_kv.get("재무제표의 종류")
        or all_kv.get("재무제표 종류")
        or None
    )

    # Period
    start = (
        all_kv.get("- 시작일")
        or all_kv.get("시작일")
        or None
    )
    end = (
        all_kv.get("- 종료일")
        or all_kv.get("종료일")
        or None
    )
    if start or end:
        period_dict = {}
        if start:
            period_dict["- 시작일"] = start
        if end:
            period_dict["- 종료일"] = end

    # If income changes weren't found in a dedicated table, try to build from kv
    if not income_changes:
        income_changes_raw: dict = {}
        for label in _INCOME_KEYS:
            item_data: dict = {}
            for alias in _INCOME_ITEM_ALIASES:
                # Try composite keys like "- 매출액_당해사업연도"
                candidate = all_kv.get(f"{label}_{alias}") or all_kv.get(f"{label} {alias}")
                if candidate:
                    item_data[alias] = candidate
            if item_data:
                income_changes_raw[label] = item_data
        if income_changes_raw:
            income_changes = income_changes_raw

    # If financial status wasn't found in a dedicated table, try kv
    if not financial_status:
        financial_raw: dict = {}
        for label in _FINANCIAL_KEYS:
            item_data2: dict = {}
            for alias in _FINANCIAL_STATUS_ALIASES:
                candidate = all_kv.get(f"{label}_{alias}") or all_kv.get(f"{label} {alias}")
                if candidate:
                    item_data2[alias] = candidate
            if item_data2:
                financial_raw[label] = item_data2
        if financial_raw:
            financial_status = financial_raw

    change_reason = (
        all_kv.get("5. 매출액 또는 손익구조 변동 주요원인")
        or all_kv.get("변동 주요원인")
        or None
    )

    # Detect unit from document text (e.g., "단위:천원", "단위 : 백만원")
    # Target the income changes section header to avoid picking up a
    # different unit from the financial status section.
    unit = "원"
    full_text = soup.get_text()
    unit_m = re.search(
        r'손익구조변동.*?단위\s*[:：]\s*(천원|백만원|억원|원)', full_text, re.DOTALL,
    )
    if not unit_m:
        unit_m = re.search(r'단위\s*[:：]\s*(천원|백만원|억원|원)', full_text)
    if unit_m:
        unit = unit_m.group(1)

    try:
        return Performance(
            unit=unit,
            **{
                "1. 재무제표의 종류": statement_type,
                "2. 결산기간": period_dict,
                "3. 매출액 또는 손익구조변동내용(단위: 원)": income_changes,
                "4. 재무현황(단위 : 원)": financial_status,
                "5. 매출액 또는 손익구조 변동 주요원인": change_reason,
            }
        )
    except Exception as exc:
        logger.error("Failed to build Performance: %s", exc)
        return Performance()
