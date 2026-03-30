"""Parser for KRX exchange disclosure HTML files (kind.krx.co.kr).

Handles disclosure categories:
- Backlog (수주계약): 단일판매ㆍ공급계약체결/해지
- Overhang exercise (행사): 전환/신주인수권/교환청구권 행사
- Overhang price adjustment (조정): 전환가액/행사가액 조정
- Sales (실적): 매출액또는손익구조 변동 (delegated to parse_performance)
- Preliminary results (잠정실적): 영업(잠정)실적
- Forecast (실적전망): 영업실적등에대한전망(공정공시)
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
    "parse_exchange_preliminary_results",
    "parse_exchange_forecast",
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
    """Find a key containing fragment and return its value.

    When multiple keys match, prefer the shortest key to avoid
    merged summary-table headers that concatenate column names.
    """
    best_key = None
    best_len = float('inf')
    for k in kv:
        if fragment in k and len(k) < best_len:
            best_key = k
            best_len = len(k)
    if best_key is not None:
        v = kv[best_key].strip()
        return v or None
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
    """Return the primary document heading (decoded Korean).

    For correction disclosures (기재정정), the first bold span is
    '정정신고(보고)' which is not the actual disclosure type.  Skip it
    and return the next meaningful bold span instead.
    """
    for span in soup.find_all("span"):
        style = span.get("style", "")
        if "font-weight:bold" in style or "font-weight: bold" in style:
            txt = clean_text(span.get_text())
            if txt and "정정신고" not in txt:
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

    # Exercise keywords (includes balance change — same parser handles cb_balance)
    if any(k in heading for k in ("행사", "청구권행사", "인수권행사", "잔액변경")):
        return "exercise"

    # Check for combined price adj with 조정 in heading
    if "조정" in heading and "행사" not in heading:
        return "price_adj"

    if any(k in heading for k in ("계약체결", "계약해지", "수주계약")):
        return "backlog"

    # Preliminary results (잠정실적)
    if "잠정" in heading and ("실적" in heading or "영업" in heading):
        return "preliminary"

    # Forecast (실적전망 / 공정공시)
    if "전망" in heading:
        return "forecast"

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
            if "미잔액" in row_text or "전환가능주식수" in row_text:
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
        result["contract_amount"] = parse_korean_number(
            _kv_get(kv, "계약금액 총액") or _kv_get(kv, "계약금액 합계")
            or _kv_get(kv, "계약금액(원)") or _kv_get(kv, "계약금액")
        )
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
    """Parse 전환/신주인수권/교환청구권/주식매수선택권 행사 disclosure."""
    heading = _get_heading(soup)
    doc_type = _classify_exercise_type(heading)

    tables = soup.find_all("table")
    if not tables:
        return {"type": doc_type}

    # Stock option exercise has a different table structure
    if doc_type == "주식매수선택권행사":
        return _parse_so_exercise(tables, doc_type)

    # Identify tables by content rather than position, to handle correction
    # disclosures (기재정정) which prepend metadata tables.
    kv_table = None
    daily_table = None
    balance_table = None

    for t in tables:
        rows = t.find_all("tr")
        if not rows:
            continue
        first_row_text = " ".join(_row_texts(rows[0]))
        full_text = t.get_text()

        if "청구일자" in first_row_text or "행사일자" in first_row_text:
            daily_table = t
        elif "회차" in first_row_text and ("권면" in first_row_text or "미잔액" in first_row_text or "미전환" in first_row_text or "전환가능" in first_row_text):
            balance_table = t
        elif "행사주식수 누계" in full_text and "발행주식총수" in full_text:
            kv_table = t

    if kv_table is None:
        kv_table = tables[0]

    kv = table_to_dict(kv_table)

    cumulative_shares = _find_cumulative_shares(kv)
    total_shares = _find_total_shares(kv)
    ratio_pct = _find_ratio(kv)

    # Fallback: if content-based detection missed tables, use positional
    if daily_table is None or balance_table is None:
        kv_idx = tables.index(kv_table) if kv_table in tables else 0
        data_tables = [t for t in tables[kv_idx + 1:] if len(t.find_all("tr")) >= 2]
        if daily_table is None and len(data_tables) >= 1:
            daily_table = data_tables[0]
        if balance_table is None and len(data_tables) >= 2:
            balance_table = data_tables[1]

    daily_claims: list[dict] = []
    if daily_table is not None:
        daily_claims = _parse_daily_claims(daily_table)

    cb_balance: list[dict] = []
    if balance_table is not None:
        cb_balance = _parse_cb_balance(balance_table)

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
    # Stock option exercise
    if "주식매수선택권" in heading:
        return "주식매수선택권행사"
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


def _parse_so_exercise(tables: list[Tag], doc_type: str) -> dict:
    """Parse 주식매수선택권 행사 disclosure (stock option exercise).

    Table layout:
      Table 0: KV-like summary (총발행주식수, 행사주식수 breakdown, 비율, 상장예정일, 비고)
      Table 1: 일별 행사 내역 (date, relation, name, exercise_price, par_value, shares...)
      Table 2: 주식매수선택권 잔여현황 (total, 신주, 자기주식, 차액보상, 기타)
    """
    kv = table_to_dict(tables[0])

    # Extract fields from KV table — parse raw rows for numeric data
    exercise_shares: Optional[int] = None
    total_shares: Optional[int] = None
    ratio_pct: Optional[float] = None
    listing_date: Optional[str] = None

    for row in tables[0].find_all("tr"):
        texts = _row_texts(row)
        if not texts:
            continue
        # Data row: first cell is total_shares (numeric), second is exercise shares
        first_num = parse_korean_number(texts[0])
        if first_num is not None and first_num > 0:
            total_shares = first_num
            if len(texts) >= 2:
                exercise_shares = parse_korean_number(texts[1])
            # Ratio is typically the last cell with %
            for t in reversed(texts):
                r = parse_korean_float(t)
                if r is not None:
                    ratio_pct = r
                    break

    # Listing date from KV
    for k, v in kv.items():
        if "상장예정일" in k:
            m = re.search(r"\d{4}-\d{2}-\d{2}", v)
            if m:
                listing_date = m.group(0)
            break

    notes = _find_notes(kv)

    # Data tables (skip KV table 0 and label-only tables)
    data_tables = [t for t in tables[1:] if len(t.find_all("tr")) >= 2]

    daily_exercises: list[dict] = []
    if data_tables:
        daily_exercises = _parse_so_daily_exercises(data_tables[0])

    so_remaining: Optional[dict] = None
    if len(data_tables) >= 2:
        so_remaining = _parse_so_remaining(data_tables[1])

    result: dict = {
        "type": doc_type,
        "exercise_shares": exercise_shares,
        "total_shares": total_shares,
        "ratio_pct": ratio_pct,
        "listing_date": listing_date,
        "daily_exercises": daily_exercises,
        "notes": notes,
    }
    if so_remaining:
        result["so_remaining"] = so_remaining
    return result


def _parse_so_daily_exercises(table: Tag) -> list[dict]:
    """Parse stock option daily exercise table.

    Columns: 행사일 | 회사와의관계 | 성명 | 행사가격 | 1주당 액면가 |
             신주 | 자기주식 | 차액보상 | 주식수 합계
    """
    rows = table.find_all("tr")
    exercises: list[dict] = []
    for row in rows:
        texts = _row_texts(row)
        if not texts:
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", texts[0]):
            continue
        entry: dict = {"date": texts[0]}
        if len(texts) >= 2:
            entry["relation"] = texts[1]
        if len(texts) >= 3:
            entry["name"] = texts[2]
        if len(texts) >= 4:
            entry["exercise_price"] = parse_korean_number(texts[3])
        if len(texts) >= 5:
            entry["par_value"] = parse_korean_number(texts[4])
        if len(texts) >= 6:
            entry["shares_new"] = parse_korean_number(texts[5]) or 0
        if len(texts) >= 7:
            entry["shares_treasury"] = parse_korean_number(texts[6]) or 0
        if len(texts) >= 8:
            entry["shares_cash"] = parse_korean_number(texts[7]) or 0
        if len(texts) >= 9:
            entry["shares_total"] = parse_korean_number(texts[8]) or 0
        exercises.append(entry)
    return exercises


def _parse_so_remaining(table: Tag) -> Optional[dict]:
    """Parse 주식매수선택권 잔여현황 table.

    Columns: 잔여주식수(총) | 신주(주) | 자기주식(주) | 차액보상(주) | 기타(주)
    Returns dict with total, new_shares, treasury_shares, cash_settlement, other.
    """
    rows = table.find_all("tr")
    # Find the data row (skip header rows)
    for row in rows:
        texts = _row_texts(row)
        if not texts:
            continue
        # Data row: first cell is a number (total remaining shares)
        total = parse_korean_number(texts[0])
        if total is None:
            continue
        return {
            "total": total,
            "new_shares": parse_korean_number(texts[1]) or 0 if len(texts) >= 2 else 0,
            "treasury_shares": parse_korean_number(texts[2]) or 0 if len(texts) >= 3 else 0,
            "cash_settlement": parse_korean_number(texts[3]) or 0 if len(texts) >= 4 else 0,
            "other": parse_korean_number(texts[4]) or 0 if len(texts) >= 5 else 0,
        }
    return None


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

            # Remaining balance (미잔액):
            # When two currency markers exist (e.g. | 권면총액 | KRW | 미잔액 | KRW |),
            # remaining sits between the two markers.
            if len(currency_indices) >= 2:
                between_start = currency_indices[0] + 1
                between_end = currency_indices[1]
                found_remaining = False
                for idx in range(between_start, between_end):
                    val = parse_korean_number(texts[idx])
                    if val is not None:
                        bal["remaining"] = val
                        found_remaining = True
                        break
                # "-" in the remaining cell means 0 (fully converted)
                if not found_remaining:
                    raw_between = [texts[idx].strip() for idx in range(between_start, between_end)]
                    if any(t == "-" for t in raw_between):
                        bal["remaining"] = 0
            elif ccy1_idx >= 3:
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
# Preliminary Results (잠정실적) parser
# ---------------------------------------------------------------------------

# Financial item keywords that appear in 잠정실적 tables
_PRELIMINARY_ITEM_KEYS = ("매출액", "영업이익", "법인세비용", "당기순이익", "지배", "소유주지분")


def parse_exchange_preliminary_results(soup: BeautifulSoup) -> dict:
    """Parse 연결재무제표기준영업(잠정)실적 disclosure.

    Returns dict with type, statement_type, unit, periods,
    quarterly comparisons, cumulative data, company info, and notes.
    """
    heading = _get_heading(soup)
    statement_type = "연결" if "연결" in heading else "별도"

    tables = soup.find_all("table")
    result: dict = {
        "type": "잠정실적",
        "statement_type": statement_type,
    }

    if len(tables) < 2:
        return result

    # Table 1: Period dates
    result["periods"] = _parse_preliminary_periods(tables[0])

    # Table 2: Financial data + metadata
    unit, quarterly, cumulative = _parse_preliminary_financials(tables[1])
    result["unit"] = unit
    result["quarterly"] = quarterly
    result["cumulative"] = cumulative

    # Company info and notes by scanning rows directly
    # (table_to_dict doesn't handle complex multi-column tables well)
    result["company"] = _extract_company_info(tables[1])
    result["audited"] = False
    kv = table_to_dict(tables[1])
    result["notes"] = _find_notes(kv) or _find_notes_from_rows(tables[1])

    return result


def _parse_preliminary_periods(table: Tag) -> dict:
    """Extract period date ranges from the period info table."""
    rows = table.find_all("tr")
    periods: dict = {}

    # Period label mapping (Korean fragment -> output key)
    # Actual EUC-KR decoded labels: 당기실적, 전기실적, 전년동기실적,
    # 당기누계실적, 전년동기누적실적
    label_map = {
        "당기실적": "당해실적",
        "전기실적": "전분기실적",
        "전년동기실적": "전년동기실적",
        "당기누계": "누적당해실적",
        "전년동기누적": "누적전년동기실적",
    }

    for row in rows:
        texts = _row_texts(row)
        if len(texts) < 4:
            continue

        label_text = texts[0]
        # Match against known period labels
        matched_key = None
        for fragment, key in label_map.items():
            if fragment in label_text:
                matched_key = key
                break

        if not matched_key:
            continue

        # Extract start/end dates (texts[1] = start, texts[3] = end)
        start_date = texts[1].strip()
        end_date = texts[3].strip() if len(texts) > 3 else texts[2].strip()

        period_data: dict = {"start": start_date, "end": end_date}

        # Derive quarter label from end date (e.g. "2025-12-31" -> "2025.4Q")
        if matched_key in ("당해실적", "전분기실적", "전년동기실적"):
            label = _derive_quarter_label(end_date)
            if label:
                period_data["label"] = label

        periods[matched_key] = period_data

    return periods


def _derive_quarter_label(end_date: str) -> Optional[str]:
    """Derive quarter label from period end date (e.g. '2025-12-31' -> '2025.4Q')."""
    m = re.match(r"(\d{4})-(\d{2})-\d{2}", end_date)
    if not m:
        return None
    year, month = m.group(1), int(m.group(2))
    quarter_map = {3: 1, 6: 2, 9: 3, 12: 4}
    q = quarter_map.get(month)
    return f"{year}.{q}Q" if q else None


def _parse_preliminary_financials(table: Tag) -> tuple[str, dict, dict]:
    """Parse the financial data table from a 잠정실적 disclosure.

    Returns (unit, quarterly_data, cumulative_data).
    """
    rows = table.find_all("tr")
    quarterly: dict = {}
    cumulative: dict = {}
    unit = "백만원"

    # Detect unit from header row
    for row in rows[:5]:
        row_text = " ".join(_row_texts(row))
        if "단위" in row_text:
            if "억원" in row_text:
                unit = "억원"
            elif "백만원" in row_text:
                unit = "백만원"
            elif "천원" in row_text:
                unit = "천원"
            break

    current_item: Optional[str] = None

    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        texts = _row_texts(row)
        if not texts:
            continue

        first_cell = cells[0]
        rowspan = int(first_cell.get("rowspan", 1))

        # Row with rowspan >= 2: new financial item starts
        if rowspan >= 2:
            item_name = texts[0].strip()
            # Check if this is a financial item
            if not any(k in item_name for k in _PRELIMINARY_ITEM_KEYS):
                continue

            # Normalise item name
            item_name = _normalise_item_name(item_name)
            current_item = item_name

            # Extract quarterly data from this row
            # Layout: [item_name, period_type, q_val, prev_q_val, qoq_pct,
            #          qoq_exchange, yoy_val, yoy_pct, yoy_exchange]
            if len(texts) >= 3:
                period_type = texts[1].strip()
                if "당해" in period_type:
                    quarterly[current_item] = _extract_quarterly_row(texts[2:])

        elif current_item and len(texts) >= 2:
            # Continuation row (누적실적) — fewer cells due to rowspan
            period_type = texts[0].strip()
            if "누계" in period_type or "누적" in period_type:
                cumulative[current_item] = _extract_cumulative_row(texts[1:])
            current_item = None

    return unit, quarterly, cumulative


def _extract_company_info(table: Tag) -> dict:
    """Extract company name and business from table rows directly."""
    name = None
    business = None
    for row in table.find_all("tr"):
        texts = _row_texts(row)
        row_text = " ".join(texts)
        if "회사명" in row_text or "해당회사명" in row_text:
            # Company name is the last non-empty cell
            for t in reversed(texts):
                t = t.strip()
                if t and "회사명" not in t and "해당회사" not in t and not re.match(r"^\d+\.", t):
                    name = t
                    break
        if "사업내용" in row_text or "주요사업" in row_text:
            for t in reversed(texts):
                t = t.strip()
                if t and "사업내용" not in t and "주요사업" not in t:
                    business = t
                    break
    return {"name": name, "business": business}


def _find_notes_from_rows(table: Tag) -> Optional[str]:
    """Extract notes by scanning rows for '기타 투자판단' section.

    Handles two layouts:
    - Same row: header + content in one row (잠정실적)
    - Separate rows: header in one row, content in the next (실적전망)
    """
    found_header = False
    for row in table.find_all("tr"):
        texts = _row_texts(row)
        row_text = " ".join(texts)
        if "기타 투자판단" in row_text or "투자판단에 참고" in row_text:
            # Check if content is in same row
            for t in reversed(texts):
                t = t.strip()
                if t and "투자판단" not in t and "중요사항" not in t and not re.match(r"^\d+\.", t):
                    return t
            # Content might be in next row
            found_header = True
            continue
        if found_header:
            content = " ".join(t.strip() for t in texts if t.strip())
            if content:
                return content
            found_header = False
    return None


def _normalise_item_name(name: str) -> str:
    """Normalise financial item names for consistency."""
    name = name.strip()
    # Handle combined names like "법인세비용차감전계속사업이익(당기순이익)"
    if "법인세비용" in name or "계속사업이익" in name:
        return "법인세비용차감전계속사업이익"
    if "지배" in name or "소유주지분" in name:
        return "지배주주당기순이익"
    if "당기순이익" in name:
        return "당기순이익"
    if "영업이익" in name:
        return "영업이익"
    if "매출액" in name:
        return "매출액"
    return name


def _extract_quarterly_row(values: list[str]) -> dict:
    """Extract quarterly comparison values from a data row.

    Expected order: [당해, 전분기, QoQ%, QoQ환산, 전년동기, YoY%, YoY환산]
    """
    result: dict = {}
    if len(values) >= 1:
        result["당해실적"] = parse_korean_number(values[0])
    if len(values) >= 2:
        result["전분기실적"] = parse_korean_number(values[1])
    if len(values) >= 3:
        result["전분기대비증감율"] = parse_korean_float(values[2])
    # values[3] = 전분기대비증감환산율 (usually "-", skip)
    if len(values) >= 5:
        result["전년동기실적"] = parse_korean_number(values[4])
    if len(values) >= 6:
        result["전년동기대비증감율"] = parse_korean_float(values[5])
    # values[6] = 전년동기대비증감환산율 — check for 흑자/적자전환
    if len(values) >= 7:
        exchange = values[6].strip()
        if exchange and exchange != "-":
            result["흑자적자전환"] = exchange
    return result


def _extract_cumulative_row(values: list[str]) -> dict:
    """Extract cumulative comparison values from a data row.

    Expected order: [당해누적, (skip), (skip), (skip), 전년동기누적, YoY%, YoY환산]
    """
    result: dict = {}
    if len(values) >= 1:
        result["당해실적"] = parse_korean_number(values[0])
    # values[1..3] are typically "-" for cumulative rows
    if len(values) >= 5:
        result["전년동기실적"] = parse_korean_number(values[4])
    if len(values) >= 6:
        result["전년동기대비증감율"] = parse_korean_float(values[5])
    if len(values) >= 7:
        exchange = values[6].strip()
        if exchange and exchange != "-":
            result["흑자적자전환"] = exchange
    return result


# ---------------------------------------------------------------------------
# Forecast (실적전망 / 공정공시) parser
# ---------------------------------------------------------------------------

# Financial items in forecast disclosures
_FORECAST_ITEM_KEYS = ("매출액", "영업이익", "법인세비용", "당기순이익")


def parse_exchange_forecast(soup: BeautifulSoup) -> dict:
    """Parse 연결재무제표기준영업실적등에대한전망(공정공시) disclosure.

    Returns dict with type, statement_type, unit, forecasts,
    assumptions, prior_forecast_comparison, company info, and notes.
    """
    heading = _get_heading(soup)
    statement_type = "연결" if "연결" in heading else "별도"

    tables = soup.find_all("table")
    result: dict = {
        "type": "실적전망",
        "statement_type": statement_type,
    }

    if not tables:
        return result

    # Main forecast table (first table)
    unit, forecasts, kv = _parse_forecast_table(tables[0])
    result["unit"] = unit
    result["forecasts"] = forecasts

    # Assumptions
    result["assumptions"] = (
        _kv_get(kv, "근거") or _kv_get(kv, "구체적 근거")
        or _kv_get(kv, "전망 또는")
    )

    # Company info (scan rows directly for complex tables)
    result["company"] = _extract_company_info(tables[0])

    result["notes"] = _find_notes(kv) or _find_notes_from_rows(tables[0])

    # Prior forecast comparison table (second table, in a separate div)
    if len(tables) >= 2:
        comparison = _parse_forecast_comparison(tables[1])
        if comparison:
            result["prior_forecast_comparison"] = comparison

    return result


def _parse_forecast_table(table: Tag) -> tuple[str, list[dict], dict[str, str]]:
    """Parse the main forecast data table.

    Returns (unit, forecasts_list, kv_pairs).
    """
    rows = table.find_all("tr")
    unit = "억원"
    kv: dict[str, str] = {}

    # Detect unit and forecast years from header row
    years: list[int] = []
    for row in rows[:5]:
        texts = _row_texts(row)
        row_text = " ".join(texts)
        if "단위" in row_text:
            if "백만원" in row_text:
                unit = "백만원"
            elif "억원" in row_text:
                unit = "억원"
            elif "천원" in row_text:
                unit = "천원"
        # Extract year numbers (4-digit numbers in header)
        for t in texts:
            t = t.strip()
            if re.fullmatch(r"\d{4}", t):
                years.append(int(t))

    # Build forecast dicts keyed by year
    forecast_map: dict[int, dict] = {}
    for y in years:
        forecast_map[y] = {"year": y}

    # Parse period rows and item rows
    period_starts: dict[int, str] = {}
    period_ends: dict[int, str] = {}
    i = 0
    while i < len(rows):
        texts = _row_texts(rows[i])
        if not texts:
            i += 1
            continue

        row_text = " ".join(texts)

        # Period start/end rows
        if "시작일" in row_text or "종료일" in row_text:
            is_start = "시작일" in row_text
            # Extract dates from the row
            dates = [t.strip() for t in texts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t.strip())]
            for j, d in enumerate(dates):
                if j < len(years):
                    if is_start:
                        period_starts[years[j]] = d
                    else:
                        period_ends[years[j]] = d
            i += 1
            continue

        # Financial item rows (must have at least 2 cells to avoid matching notes text)
        item_name = texts[0].strip()
        if len(texts) >= 2 and any(k in item_name for k in _FORECAST_ITEM_KEYS):
            item_name = _normalise_item_name(item_name)
            # Values follow after the item name (one per year)
            # The exact column positions depend on colspan, extract numbers
            vals = [parse_korean_number(t) for t in texts[1:] if t.strip() not in ("", "년사업연도")]
            # Filter out "년사업연도" text columns
            for j, y in enumerate(years):
                if j < len(vals):
                    forecast_map[y][item_name] = vals[j]
                else:
                    forecast_map[y][item_name] = None
            i += 1
            continue

        # Section header or KV pair
        if len(texts) >= 2 and texts[0]:
            first = texts[0].strip()
            if re.match(r"^\d+\.", first):
                # Numbered section (kv pair: section header + value)
                value = " ".join(t for t in texts[1:] if t.strip())
                kv[first] = value
            elif first:
                kv[first] = " ".join(t for t in texts[1:] if t.strip())
        i += 1

    # Attach periods to forecast dicts
    forecasts: list[dict] = []
    for y in years:
        fd = forecast_map[y]
        period: dict = {}
        if y in period_starts:
            period["start"] = period_starts[y]
        if y in period_ends:
            period["end"] = period_ends[y]
        if period:
            fd["period"] = period
        forecasts.append(fd)

    return unit, forecasts, kv


def _parse_forecast_comparison(table: Tag) -> Optional[dict]:
    """Parse 최근 영업실적 등 전망 공시의 괴리 현황 table.

    Returns dict with period and per-item comparison (전망, 실적, 괴리율).
    """
    rows = table.find_all("tr")
    if len(rows) < 3:
        return None

    comparison: dict = {"items": {}}

    for row in rows:
        texts = _row_texts(row)
        if not texts:
            continue

        row_text = " ".join(texts)

        # Period row
        if "대상기간" in row_text or "기간" in texts[0]:
            dates = [t.strip() for t in texts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t.strip())]
            if len(dates) >= 2:
                comparison["period"] = {"start": dates[0], "end": dates[1]}
            elif len(dates) == 1:
                comparison["period"] = {"start": dates[0]}
            continue

        # Item rows
        item_name = texts[0].strip()
        if any(k in item_name for k in _FORECAST_ITEM_KEYS):
            item_name = _normalise_item_name(item_name)
            item_data: dict = {}
            if len(texts) >= 2:
                item_data["전망"] = parse_korean_number(texts[1])
            if len(texts) >= 3:
                item_data["실적"] = parse_korean_number(texts[2])
            if len(texts) >= 4:
                item_data["괴리율"] = parse_korean_float(texts[3])
            comparison["items"][item_name] = item_data

    return comparison if comparison.get("items") else None


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
        elif category == "preliminary":
            return category, parse_exchange_preliminary_results(soup)
        elif category == "forecast":
            return category, parse_exchange_forecast(soup)
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
