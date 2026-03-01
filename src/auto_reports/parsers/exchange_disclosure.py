"""Parser for KRX exchange disclosure HTML files (kind.krx.co.kr).

Handles three disclosure categories:
- Backlog (수주계약): 단일판매ㆍ공급계약체결/해지
- Overhang exercise (행사): 전환/신주인수권/교환청구권 행사
- Overhang price adjustment (조정): 전환가액/행사가액 조정
- Sales (실적): 매출액또는손익구조 변동 (delegated to parse_performance)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup, Tag

from auto_reports.models.disclosure import parse_korean_float, parse_korean_number
from auto_reports.parsers.base import clean_text, table_to_dict
from auto_reports.parsers.performance import parse_performance

logger = logging.getLogger(__name__)

__all__ = [
    "parse_exchange_disclosure",
    "parse_exchange_disclosure_from_soup",
    "parse_exchange_backlog",
    "parse_exchange_overhang_exercise",
    "parse_exchange_overhang_price_adj",
]


# ---------------------------------------------------------------------------
# Encoding / soup helpers
# ---------------------------------------------------------------------------

def _decode_html(html_bytes: bytes) -> str:
    return html_bytes.decode("euc-kr", errors="replace")


def _parse_soup(html_bytes: bytes) -> BeautifulSoup:
    return BeautifulSoup(_decode_html(html_bytes), "html.parser")


def _row_texts(row: Tag) -> list[str]:
    return [clean_text(c.get_text()) for c in row.find_all(["td", "th"])]


# ---------------------------------------------------------------------------
# Substring-based key matching helpers
# ---------------------------------------------------------------------------

def _kv_get(kv: dict[str, str], fragment: str) -> Optional[str]:
    """Find a key containing fragment and return its value."""
    for k, v in kv.items():
        if fragment in k:
            return v.strip() or None
    return None


def _kv_has(kv: dict[str, str], fragment: str) -> bool:
    """Check if any key contains the given fragment."""
    return any(fragment in k for k in kv)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# These strings appear in the EUC-KR -> replace-decoded title (garbled)
# We match on decoded Korean text from the bold heading span instead.

def _get_heading(soup: BeautifulSoup) -> str:
    """Return the primary document heading (decoded Korean)."""
    for span in soup.find_all("span"):
        style = span.get("style", "")
        if "font-weight:bold" in style or "font-weight: bold" in style:
            txt = clean_text(span.get_text())
            if txt:
                return txt
    title = soup.find("title")
    if title:
        return clean_text(title.get_text())
    return ""


def _classify(soup: BeautifulSoup) -> str:
    """Return one of: backlog, exercise, price_adj, sales, unknown."""
    heading = _get_heading(soup)

    # Price adjustment keywords (check before exercise - both contain 행사)
    if any(k in heading for k in ("가액의조정", "가액의 조정", "조정(안내공문)")):
        return "price_adj"

    # Exercise keywords
    if any(k in heading for k in ("행사", "청구권행사", "인수권행사")):
        return "exercise"

    # Check for combined price adj with 조정 in heading
    if "조정" in heading and "행사" not in heading:
        return "price_adj"

    if any(k in heading for k in ("계약체결", "계약해지", "수주계약")):
        return "backlog"

    if any(k in heading for k in ("매출액또는손익", "손익구조변동", "손익구조")):
        return "sales"

    # Fallback: inspect first table's first few rows
    first_table = soup.find("table")
    if first_table:
        rows = first_table.find_all("tr")
        for row in rows[:5]:
            texts = _row_texts(row)
            row_text = " ".join(texts)
            if "해지금액" in row_text or "해지일자" in row_text:
                return "backlog"
            if "계약금액" in row_text:
                return "backlog"
            if "조정전" in row_text and "조정후" in row_text:
                return "price_adj"
            if "청구일자" in row_text or "행사일자" in row_text:
                return "exercise"
            if "매출액" in row_text and "손익" in row_text:
                return "sales"

    return "unknown"


# ---------------------------------------------------------------------------
# Backlog parser
# ---------------------------------------------------------------------------

def parse_exchange_backlog(soup: BeautifulSoup) -> dict:
    """Parse 단일판매ㆍ공급계약체결/해지 disclosure.

    Uses substring-based key matching via ``_kv_get`` to handle
    key name variations between different KRX disclosure formats.
    """
    tables = soup.find_all("table")
    kv: dict[str, str] = {}
    for t in tables:
        kv.update(table_to_dict(t))

    is_cancel = _kv_has(kv, "해지금액") or _kv_has(kv, "해지일자") or _kv_has(kv, "해지 내용")
    doc_type = "단일판매ㆍ공급계약해지" if is_cancel else "단일판매ㆍ공급계약체결"

    result: dict = {"type": doc_type}

    if is_cancel:
        result["description"] = (
            _kv_get(kv, "해지 구분") or _kv_get(kv, "해지 내용")
            or _kv_get(kv, "공급계약 내용") or _kv_get(kv, "공급계약 구분")
        )
        result["detail"] = (
            _kv_get(kv, "해지계약명") or _kv_get(kv, "계약내용")
            or _kv_get(kv, "해지내용상세")
        )
        result["cancel_amount"] = parse_korean_number(_kv_get(kv, "해지금액"))
        result["recent_revenue"] = parse_korean_number(_kv_get(kv, "매출액(원)"))
        result["revenue_ratio_pct"] = parse_korean_float(_kv_get(kv, "매출액대비") or _kv_get(kv, "매출액 대비"))
        result["is_large_corp"] = _parse_large_corp(_kv_get(kv, "법인여부"))
        result["counterparty"] = _kv_get(kv, "계약상대")
        result["counterparty_relationship"] = _kv_get(kv, "회사와의 관계")
        result["period"] = _parse_period(kv)
        result["conditions"] = (
            _kv_get(kv, "주요사유") or _kv_get(kv, "주요내용")
            or _kv_get(kv, "주요 내용")
        )
        result["cancel_date"] = _kv_get(kv, "해지일자")
    else:
        # Description: key varies — "공급계약 내용", "공급계약 구분", etc.
        result["description"] = (
            _kv_get(kv, "공급계약 내용") or _kv_get(kv, "공급계약 구분")
        )
        result["detail"] = _kv_get(kv, "체결내용") or _kv_get(kv, "체결계약명")
        result["contract_amount"] = parse_korean_number(_kv_get(kv, "계약금액(원)") or _kv_get(kv, "계약금액"))
        result["recent_revenue"] = parse_korean_number(_kv_get(kv, "매출액(원)"))
        result["revenue_ratio_pct"] = parse_korean_float(_kv_get(kv, "매출액대비") or _kv_get(kv, "매출액 대비"))
        result["is_large_corp"] = _parse_large_corp(_kv_get(kv, "법인여부"))
        result["counterparty"] = _kv_get(kv, "계약상대")
        result["counterparty_relationship"] = _kv_get(kv, "회사와의 관계")
        result["region"] = _kv_get(kv, "지역") or _kv_get(kv, "공급지역")
        result["period"] = _parse_period(kv)
        result["conditions"] = _parse_conditions(kv)
        result["contract_date"] = _kv_get(kv, "수주)일자") or _kv_get(kv, "계약일자")

    result["notes"] = _find_notes(kv)
    result["history"] = _find_history(kv)
    return result


def _parse_large_corp(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    val = val.strip()
    if "비해당" in val or "해당없음" in val:
        return False
    if "해당" in val:
        return True
    return None


def _parse_period(kv: dict[str, str]) -> dict:
    start = _kv_get(kv, "시작일")
    end = _kv_get(kv, "종료일")
    return {"start": start, "end": end}


def _parse_conditions(kv: dict[str, str]) -> dict:
    conditions: dict[str, str] = {}
    for k, v in kv.items():
        if k in ("계약금ㆍ선급금 유무", "선급금 지급 시", "대금결제조건"):
            conditions[k] = v
    return conditions if conditions else {}


def _find_notes(kv: dict[str, str]) -> Optional[str]:
    for key in kv:
        if "기타 투자판단에 참고할 사항" in key or "기타 투자판단" in key:
            return kv[key].strip() or None
    return None


def _find_history(kv: dict[str, str]) -> Optional[str]:
    for key in kv:
        if "공시이력" in key or "관련공시" in key:
            return kv[key].strip() or None
    return None


# ---------------------------------------------------------------------------
# Overhang – Exercise parser
# ---------------------------------------------------------------------------

def parse_exchange_overhang_exercise(soup: BeautifulSoup) -> dict:
    """Parse 전환/신주인수권/교환청구권 행사 disclosure."""
    heading = _get_heading(soup)
    doc_type = _classify_exercise_type(heading)

    tables = soup.find_all("table")
    if not tables:
        return {"type": doc_type}

    kv = table_to_dict(tables[0])

    cumulative_shares = _find_cumulative_shares(kv)
    total_shares = _find_total_shares(kv)
    ratio_pct = _find_ratio(kv)

    # Find data tables dynamically: skip KV (table 0) and label-only tables (< 2 rows).
    # Some formats wrap section labels in separate <table> (5 tables total),
    # others use <span> labels (3 tables total).
    data_tables = [t for t in tables[1:] if len(t.find_all("tr")) >= 2]

    daily_claims: list[dict] = []
    if len(data_tables) >= 1:
        daily_claims = _parse_daily_claims(data_tables[0])

    cb_balance: list[dict] = []
    if len(data_tables) >= 2:
        cb_balance = _parse_cb_balance(data_tables[1])

    notes = _find_notes(kv)
    history = _find_history(kv)

    return {
        "type": doc_type,
        "cumulative_shares": cumulative_shares,
        "total_shares": total_shares,
        "ratio_pct": ratio_pct,
        "daily_claims": daily_claims,
        "cb_balance": cb_balance,
        "notes": notes,
        "history": history,
    }


def _classify_exercise_type(heading: str) -> str:
    # Combined type contains all three: 전환, 신주인수권, 교환
    if "신주인수권" in heading and "전환" in heading and "교환" in heading:
        return "전환청구권ㆍ신주인수권ㆍ교환청구권행사"
    if "신주인수권" in heading:
        return "신주인수권행사"
    # "전환주식의 전환청구권 행사" — distinct from combined type
    if "전환주식" in heading and "전환청구권" in heading:
        return "전환주식의전환청구권행사"
    if "전환청구권" in heading:
        return "전환청구권행사"
    return "행사"


def _find_cumulative_shares(kv: dict[str, str]) -> Optional[int]:
    for k, v in kv.items():
        if "누계" in k and ("행사" in k or "청구" in k):
            return parse_korean_number(v)
    # Variant B: "2. 행사주식수 누계(주)"
    for k, v in kv.items():
        if "누계" in k:
            return parse_korean_number(v)
    return None


def _find_total_shares(kv: dict[str, str]) -> Optional[int]:
    for k, v in kv.items():
        if "발행주식총수" in k and "대비" not in k:
            return parse_korean_number(v)
    return None


def _find_ratio(kv: dict[str, str]) -> Optional[float]:
    for k, v in kv.items():
        if "발행주식총수 대비" in k or "발행주식총수대비" in k or (
            "발행주식" in k and "대비" in k
        ):
            return parse_korean_float(v)
    return None


def _parse_daily_claims(table: Tag) -> list[dict]:
    """Parse daily claims table (Table index 2).

    Columns vary by sub-type but always start with date, then series/type, amount,
    price, shares, optional listing_date.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    # Skip header rows (rows without a date-like first cell)
    claims: list[dict] = []
    for row in rows:
        texts = _row_texts(row)
        if not texts:
            continue
        # A data row has a date in first cell (YYYY-MM-DD pattern)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", texts[0]):
            continue

        claim: dict = {"date": texts[0]}
        if len(texts) >= 2:
            # Second column may be a date (전환주식 format) or series number
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", texts[1]):
                claim["issue_date"] = texts[1]
            else:
                claim["series"] = parse_korean_number(texts[1])
        if len(texts) >= 3:
            claim["bond_type"] = texts[2]
        if len(texts) >= 4:
            claim["amount"] = parse_korean_number(texts[3])
        if len(texts) >= 5:
            claim["conversion_price"] = parse_korean_number(texts[4])
        if len(texts) >= 6:
            claim["shares"] = parse_korean_number(texts[5])
        if len(texts) >= 7:
            claim["listing_date"] = texts[6]
        claims.append(claim)

    return claims


def _parse_cb_balance(table: Tag) -> list[dict]:
    """Parse bond balance table (Table index 4).

    Columns: 회차, 권면총액, 통화단위(KRW text), 미잔액, 통화단위(KRW text), 전환가액, 전환가능주식수
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    balances: list[dict] = []
    for row in rows:
        texts = _row_texts(row)
        if not texts:
            continue

        # Date-based format (전환주식): issue_date, bond_type, total, converted, price, remaining
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", texts[0]):
            bal: dict = {"issue_date": texts[0]}
            if len(texts) >= 2:
                bal["bond_type"] = texts[1]
            if len(texts) >= 3:
                bal["total_shares"] = parse_korean_number(texts[2])
            if len(texts) >= 4:
                bal["converted_shares"] = parse_korean_number(texts[3])
            if len(texts) >= 5:
                bal["conversion_price"] = parse_korean_number(texts[4])
            if len(texts) >= 6:
                bal["remaining_shares"] = parse_korean_number(texts[5])
            balances.append(bal)
            continue

        # Series-based format (CB/BW): first cell is a series number
        series = parse_korean_number(texts[0])
        if series is None:
            continue

        bal = {"series": series}
        if len(texts) >= 2:
            bal["face_value"] = parse_korean_number(texts[1])

        # Find all KRW/USD positions
        currency_indices = [j for j, t in enumerate(texts) if re.match(r"[A-Z]{3}", t)]

        if currency_indices:
            ccy1_idx = currency_indices[0]
            bal["currency"] = _extract_currency(texts[ccy1_idx])

            # Remaining balance: the numeric value right before first currency
            if ccy1_idx >= 3:
                bal["remaining"] = parse_korean_number(texts[ccy1_idx - 1])
            # The numeric values after the last KRW block are conversion_price and shares
            last_ccy_idx = currency_indices[-1]
            after = [texts[j] for j in range(last_ccy_idx + 1, len(texts)) if texts[j]]
            if len(after) >= 2:
                bal["conversion_price"] = parse_korean_number(after[0])
                bal["convertible_shares"] = parse_korean_number(after[1])
            elif len(after) == 1:
                bal["conversion_price"] = parse_korean_number(after[0])
        balances.append(bal)

    return balances


def _extract_currency(text: str) -> str:
    m = re.match(r"([A-Z]{3})", text.strip())
    return m.group(1) if m else text.strip()


# ---------------------------------------------------------------------------
# Overhang – Price Adjustment parser
# ---------------------------------------------------------------------------

def parse_exchange_overhang_price_adj(soup: BeautifulSoup) -> dict:
    """Parse 전환가액/신주인수권행사가액/교환가액의조정 disclosure.

    All data is in a single table. The table contains multi-row sub-sections
    for adjustments and share changes, identified by header rows.
    """
    heading = _get_heading(soup)
    doc_type = _classify_price_adj_type(heading)

    tables = soup.find_all("table")
    if not tables:
        return {"type": doc_type}

    table = tables[0]
    rows = table.find_all("tr")

    adjustments: list[dict] = []
    share_changes: list[dict] = []

    # We scan rows and identify sections by their header patterns.
    # Section 1 / 3 (depending on file): header row for adjustments has columns
    #   [section_label, 회차, 상장여부, 조정전가액(원), 조정후가액(원)]
    # Section 2 / 4: header row for share changes has columns
    #   [section_label, 회차, 권면총액, 통화단위, 조정전주식수, 조정후주식수]
    # Other rows are simple KV pairs.

    i = 0
    # Collect all rows with their text lists
    row_data: list[list[str]] = [_row_texts(r) for r in rows]

    kv: dict[str, str] = {}

    while i < len(row_data):
        texts = row_data[i]
        if not texts:
            i += 1
            continue

        if _is_adj_section_header(texts):
            joined_hdr = " ".join(texts)
            has_listed = "상장여부" in joined_hdr
            has_series = "회차" in joined_hdr
            i += 1
            while i < len(row_data):
                dtexts = row_data[i]
                if not dtexts:
                    i += 1
                    continue
                if _is_any_section_header(dtexts):
                    break
                adj = _parse_adj_data_row(
                    dtexts, has_listed_col=has_listed, has_series_col=has_series,
                )
                if adj:
                    adjustments.append(adj)
                i += 1
            continue

        if _is_share_change_section_header(texts):
            joined_hdr = " ".join(texts)
            has_currency = "통화단위" in joined_hdr
            has_series = "회차" in joined_hdr
            i += 1
            while i < len(row_data):
                dtexts = row_data[i]
                if not dtexts:
                    i += 1
                    continue
                if _is_any_section_header(dtexts):
                    break
                sc = _parse_share_change_data_row(
                    dtexts, has_currency_col=has_currency, has_series_col=has_series,
                )
                if sc:
                    share_changes.append(sc)
                i += 1
            continue

        # KV row
        if len(texts) >= 2 and texts[0]:
            kv[texts[0]] = texts[1]
        elif len(texts) == 1 and texts[0]:
            kv[texts[0]] = ""
        i += 1

    reason = _kv_get(kv, "조정사유")
    detail = _kv_get(kv, "조정근거 및 내용") or _kv_get(kv, "조정근거")
    effective_date = _kv_get(kv, "조정가액 적용일") or _kv_get(kv, "가액 적용일")
    notes = _find_notes(kv)
    history = _find_history(kv)

    return {
        "type": doc_type,
        "adjustments": adjustments,
        "share_changes": share_changes,
        "reason": reason,
        "detail": detail,
        "effective_date": effective_date,
        "notes": notes,
        "history": history,
    }


def _classify_price_adj_type(heading: str) -> str:
    if "교환가액" in heading and "전환" in heading and "신주인수권" in heading:
        return "전환가액ㆍ신주인수권행사가액ㆍ교환가액의조정"
    if "신주인수권행사가액" in heading or ("신주인수권" in heading and "조정" in heading):
        return "신주인수권행사가액의조정"
    # "전환주식의 전환가액 조정" — distinct from standard 전환가액의조정
    if "전환주식" in heading and "전환가액" in heading:
        return "전환주식의전환가액조정"
    if "전환가액" in heading:
        return "전환가액의조정"
    return "가액조정"


def _is_adj_section_header(texts: list[str]) -> bool:
    """True if row is a header for the price adjustment sub-table.

    The header row for adjustments always contains both:
    - a section label with 가액 조정 or similar
    - column headers like 회차, 조정전, 조정후
    Specifically: len >= 3 and contains both 조정전 and 조정후.
    """
    joined = " ".join(texts)
    return (
        len(texts) >= 3
        and "조정전" in joined
        and "조정후" in joined
        and ("가액" in joined or "가액의 조정" in joined)
        and "주식수" not in joined
    )


def _is_share_change_section_header(texts: list[str]) -> bool:
    """True if row is a header for the share change sub-table."""
    joined = " ".join(texts)
    return (
        len(texts) >= 3
        and "조정전" in joined
        and "조정후" in joined
        and "주식수" in joined
    )


def _is_any_section_header(texts: list[str]) -> bool:
    """True if this row is a numbered section header (e.g. '3. 조정사유')."""
    if not texts:
        return False
    # Numbered section: first cell starts with digit+dot
    if re.match(r"^\d+\.", texts[0]):
        return True
    # Or it's another sub-table header
    joined = " ".join(texts)
    return (
        (len(texts) >= 3 and "조정전" in joined and "조정후" in joined)
    )


def _parse_adj_data_row(
    texts: list[str], has_listed_col: bool = True, has_series_col: bool = True,
) -> Optional[dict]:
    """Parse a data row from the price adjustment table.

    Standard (4 cols): 회차, 상장여부, 조정전가액(원), 조정후가액(원)
    No-listed (3 cols): 회차, 조정전가액(원), 조정후가액(원)
    No-series (3 cols): 상장여부, 조정전가액(원), 조정후가액(원)  (전환주식 format)
    """
    if not texts or texts[0] in ("회차", ""):
        return None

    if not has_series_col:
        # 전환주식 format: [상장여부, 조정전, 조정후] — no series number
        if texts[0] in ("상장여부",):
            return None
        listed = texts[0] if not parse_korean_number(texts[0]) else None
        price_before = parse_korean_number(texts[1]) if len(texts) > 1 else None
        price_after = parse_korean_number(texts[2]) if len(texts) > 2 else None
        return {
            "listed": listed,
            "price_before": price_before,
            "price_after": price_after,
        }

    series = parse_korean_number(texts[0])
    if series is None:
        return None
    if has_listed_col and len(texts) >= 4:
        listed = texts[1]
        price_before = parse_korean_number(texts[2])
        price_after = parse_korean_number(texts[3])
    else:
        # No 상장여부 column
        listed = None
        price_before = parse_korean_number(texts[1]) if len(texts) > 1 else None
        price_after = parse_korean_number(texts[2]) if len(texts) > 2 else None
    return {
        "series": series,
        "listed": listed,
        "price_before": price_before,
        "price_after": price_after,
    }


def _parse_share_change_data_row(
    texts: list[str], has_currency_col: bool = True, has_series_col: bool = True,
) -> Optional[dict]:
    """Parse a data row from the share change table.

    Standard (5 cols): 회차, 권면총액, 통화단위(KRW...), 조정전주식수, 조정후주식수
    No-currency (4 cols): 회차, 권면총액, 조정전주식수, 조정후주식수
    No-series (3 cols): 현재주식수, 조정전주식수, 조정후주식수  (전환주식 format)
    """
    if not texts or texts[0] in ("회차", ""):
        return None

    if not has_series_col:
        # 전환주식 format: [현재주식수, 조정전, 조정후]
        current = parse_korean_number(texts[0])
        if current is None:
            return None
        shares_before = parse_korean_number(texts[1]) if len(texts) > 1 else None
        shares_after = parse_korean_number(texts[2]) if len(texts) > 2 else None
        return {
            "current_shares": current,
            "shares_before": shares_before,
            "shares_after": shares_after,
        }

    series = parse_korean_number(texts[0])
    if series is None:
        return None

    unconverted = parse_korean_number(texts[1]) if len(texts) > 1 else None

    currency = None
    shares_before = None
    shares_after = None

    if has_currency_col:
        # Find currency field by scanning for KRW/USD/EUR pattern
        currency_idx = None
        for j, t in enumerate(texts):
            if re.match(r"[A-Z]{3}", t):
                currency = _extract_currency(t)
                currency_idx = j
                break

        if currency_idx is not None:
            after = [texts[j] for j in range(currency_idx + 1, len(texts)) if texts[j]]
            if len(after) >= 2:
                shares_before = parse_korean_number(after[0])
                shares_after = parse_korean_number(after[1])
            elif len(after) == 1:
                shares_after = parse_korean_number(after[0])
        else:
            # Fallback positional
            shares_before = parse_korean_number(texts[3]) if len(texts) > 3 else None
            shares_after = parse_korean_number(texts[4]) if len(texts) > 4 else None
    else:
        # No currency column: [회차, 권면총액, 조정전주식수, 조정후주식수]
        shares_before = parse_korean_number(texts[2]) if len(texts) > 2 else None
        shares_after = parse_korean_number(texts[3]) if len(texts) > 3 else None

    return {
        "series": series,
        "unconverted": unconverted,
        "currency": currency,
        "shares_before": shares_before,
        "shares_after": shares_after,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_exchange_disclosure_from_soup(soup: BeautifulSoup) -> tuple[str, dict]:
    """Parse a KRX exchange disclosure from an already-decoded BeautifulSoup.

    Use this when the HTML has already been fetched and decoded (e.g. by
    DartHtmlFetcher which handles EUC-KR encoding automatically).

    Args:
        soup: Parsed BeautifulSoup of the disclosure HTML.

    Returns:
        Tuple of (category, parsed_data) where category is one of:
        'backlog', 'exercise', 'price_adj', 'sales', 'unknown'.
    """
    try:
        category = _classify(soup)

        if category == "backlog":
            return category, parse_exchange_backlog(soup)
        elif category == "exercise":
            return category, parse_exchange_overhang_exercise(soup)
        elif category == "price_adj":
            return category, parse_exchange_overhang_price_adj(soup)
        elif category == "sales":
            perf = parse_performance(soup)
            return category, perf.model_dump(exclude_none=True)
        else:
            logger.warning("Unknown exchange disclosure category; returning raw kv.")
            tables = soup.find_all("table")
            kv: dict = {}
            for t in tables:
                kv.update(table_to_dict(t))
            return "unknown", {"type": "unknown", "raw": kv}
    except Exception as exc:
        logger.error("Failed to parse exchange disclosure: %s", exc)
        return "unknown", {"type": "unknown", "error": str(exc)}


def parse_exchange_disclosure(html_bytes: bytes) -> dict:
    """Parse a KRX exchange disclosure HTML file.

    Classifies the document type and routes to the appropriate sub-parser.

    Args:
        html_bytes: Raw bytes of the HTML file (EUC-KR encoded).

    Returns:
        Dict with parsed disclosure data.
    """
    try:
        soup = _parse_soup(html_bytes)
        _, result = parse_exchange_disclosure_from_soup(soup)
        return result
    except Exception as exc:
        logger.error("Failed to parse exchange disclosure: %s", exc)
        return {"type": "unknown", "error": str(exc)}
