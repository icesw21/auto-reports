"""Parse overhang instruments from financial statement notes (재무제표 주석).

Extracts CB, BW, convertible preferred stock, and stock option data
from the notes section of periodic reports (분기/반기/사업보고서).
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

_WS = re.compile(r"[\s\xa0]+")

# Minimum conversion price threshold: par value (액면가) cannot be below 100원.
# This guards against parsing errors where par value is mistaken for conversion price.
_MIN_CONVERSION_PRICE = 100

# Keywords that indicate an instrument has been fully converted or redeemed
_INACTIVE_PATTERNS = [
    re.compile(r"모두\s*보통주로\s*전환"),
    re.compile(r"전액\s*상환"),
    re.compile(r"전액\s*전환"),
    re.compile(r"전부\s*전환"),
    re.compile(r"전환.*완료"),
    re.compile(r"상환.*완료"),
]

# Section header patterns for overhang-related notes
_SECTION_PATTERNS = {
    "CB": re.compile(r"^\d+\.\s*전환사채"),
    "BW": re.compile(r"^\d+\.\s*신주인수권부사채"),
    "RCPS": re.compile(r"\(\d+\)\s*(?:전환우선주|상환전환우선주|전환상환우선주)"),
    "SO": re.compile(r"^\d+\.\s*주식기준보상"),
}


def parse_notes_overhang(
    html: str,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> list[dict]:
    """Parse overhang instruments from financial statement notes HTML.

    If ``api_key`` is provided, uses LLM-based extraction first and falls
    back to regex-based parsing when the LLM call fails or returns nothing.
    Without ``api_key``, uses regex-based parsing only.

    Returns list of dicts with keys:
        category, series, kind, face_value, convertible_shares,
        conversion_price, exercise_start, exercise_end, active
    """
    # Try LLM-based extraction first when API key is available
    if api_key:
        try:
            from auto_reports.parsers.notes_overhang_llm import (
                parse_notes_overhang_llm,
            )

            results = parse_notes_overhang_llm(
                html, api_key=api_key, model=model or "gpt-4.1-mini",
                base_url=base_url,
            )
            if results:
                logger.info(
                    "LLM-based overhang extraction: %d instruments", len(results),
                )
                return results
            logger.info(
                "LLM returned no instruments, falling back to regex parsing",
            )
        except Exception as e:
            logger.warning(
                "LLM overhang extraction failed (%s), falling back to regex", e,
            )

    # Regex-based parsing (original logic)
    return _parse_notes_overhang_regex(html)


def _parse_notes_overhang_regex(html: str) -> list[dict]:
    """Regex/table-based overhang extraction (original logic)."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    # Find all <p> elements to locate section boundaries
    p_elements = [el for el in soup.find_all("p")]

    # Process each instrument type
    for cat_key, pattern in _SECTION_PATTERNS.items():
        section_start = _find_section_start(p_elements, pattern)
        if section_start is None:
            continue

        section_end = _find_section_end(p_elements, section_start)
        section_ps = p_elements[section_start:section_end]

        if cat_key in ("CB", "BW"):
            instruments = _parse_bond_section(soup, section_ps, cat_key)
        elif cat_key == "SO":
            instruments = _parse_stock_option_section(soup, section_ps)
        else:
            instruments = _parse_preferred_section(soup, section_ps)

        results.extend(instruments)

    # Secondary scan: find CB/BW embedded in other sections
    # e.g., "전환상환우선주부채 등" may contain a "전환사채" subsection
    if not any(r["category"] == "CB" for r in results):
        embedded = _find_embedded_cb(soup, p_elements)
        results.extend(embedded)

    return results


def _find_section_start(
    p_elements: list[Tag], pattern: re.Pattern,
) -> int | None:
    """Find the index of the <p> element matching the section header pattern."""
    for i, p in enumerate(p_elements):
        text = _WS.sub("", p.get_text(strip=True))
        if pattern.search(text):
            return i
    return None


def _find_section_end(p_elements: list[Tag], start_idx: int) -> int:
    """Find the end of a section (next numbered section header)."""
    for i in range(start_idx + 1, len(p_elements)):
        text = p_elements[i].get_text(strip=True)
        # New numbered section header (e.g., "14. 종업원급여")
        if re.match(r"^\d+\.\s+\S", text):
            return i
    return len(p_elements)


def _find_subsection_end(p_elements: list[Tag], start_idx: int) -> int:
    """Find the end of a (N) subsection.

    Stops at the next (N) sibling or the next top-level numbered section header.
    """
    for i in range(start_idx + 1, len(p_elements)):
        text = p_elements[i].get_text(strip=True)
        if re.match(r"\(\d+\)", text):
            return i
        if re.match(r"^\d+\.\s+\S", text):
            return i
    return len(p_elements)


def _parse_bond_section(
    soup: BeautifulSoup, section_ps: list[Tag], category: str,
) -> list[dict]:
    """Parse CB or BW section: find per-series subsections and extract data."""
    instruments: list[dict] = []

    # Find subsection headers: ① 제N회..., ② 제N회...
    subsection_indices: list[tuple[int, str, int]] = []
    for i, p in enumerate(section_ps):
        text = p.get_text(strip=True)
        m = re.match(r"[①②③④⑤⑥⑦⑧⑨⑩]\s*(제\s*(\d+)\s*회.+)", text)
        if m:
            kind_text = m.group(1)
            series = int(m.group(2))
            subsection_indices.append((i, kind_text, series))

    if not subsection_indices:
        # Fallback: some reports list CB details in a direct KV table
        # without ①②③ subsection headers (e.g., single-CB companies).
        return _parse_bond_direct(soup, section_ps, category)

    for idx, (sub_i, kind_text, series) in enumerate(subsection_indices):
        # Get the <p> element for this subsection
        sub_p = section_ps[sub_i]

        # Find the next subsection boundary
        if idx + 1 < len(subsection_indices):
            next_p = section_ps[subsection_indices[idx + 1][0]]
        else:
            next_p = None

        # Find the key-value table following this subsection header
        table = _find_next_table(sub_p)
        if table is None:
            continue

        kv = _parse_kv_table(table)
        if not kv:
            continue

        # Extract fields using smart extraction (handles multiple key formats)
        face_value = _extract_face_value_smart(kv)
        conv_shares = _extract_conv_shares_smart(kv, category)
        conv_price = _extract_conv_price_smart(kv, category)
        ex_start, ex_end = _extract_exercise_period_smart(kv, category)

        # Check if this instrument is inactive (fully converted/redeemed)
        # Look at text between this table and the next subsection
        footnote_text = _get_footnote_text(table, next_p)
        active = not _is_inactive(footnote_text)

        instruments.append({
            "category": category,
            "series": series,
            "kind": kind_text.strip(),
            "face_value": face_value or 0,
            "convertible_shares": conv_shares or 0,
            "conversion_price": conv_price or 0,
            "exercise_start": ex_start,
            "exercise_end": ex_end,
            "active": active,
        })

    return instruments


def _find_embedded_cb(
    soup: BeautifulSoup, p_elements: list[Tag],
) -> list[dict]:
    """Find CB data embedded in non-CB sections.

    Handles two cases:
    1. Multi-instrument: "(3) 전환사채" subsection under "차입금 및 전환사채",
       containing ①②③ per-series subsections → delegates to _parse_bond_section.
    2. Single-instrument: "(5) 전환사채 중 전환청구기간..." in a non-CB section
       → extracts from the KV table directly.
    """
    results: list[dict] = []

    for i, p in enumerate(p_elements):
        text = p.get_text(strip=True)
        # Match: "(N) 전환사채" or "(N) ... 전환사채..."
        if not re.search(r"\(\d+\)\s*.*전환사채", text):
            continue

        # Determine subsection scope: from this (N) to next (N+1) or top-level
        sub_end = _find_subsection_end(p_elements, i)
        section_ps = p_elements[i:sub_end]

        # Try multi-instrument parsing (①②③ subsections)
        instruments = _parse_bond_section(soup, section_ps, "CB")
        if instruments:
            results.extend(instruments)
            continue

        # Fallback: single KV table (e.g., embedded CB description)
        table = _find_next_table(p)
        if table is None:
            continue

        kv = _parse_kv_table(table)
        if not kv:
            continue

        # Extract series from "구분" value (e.g., "9회차 ... 전환사채")
        series = 1
        kind_text = ""
        for k, v in kv.items():
            if "구분" in k:
                m = re.search(r"(\d+)\s*회", v)
                if m:
                    series = int(m.group(1))
                kind_text = v.strip()
                break

        face_value = _extract_face_value_smart(kv)
        conv_price = _extract_conv_price_smart(kv, "CB")
        conv_shares = _extract_conv_shares_smart(kv, "CB")
        ex_start, ex_end = _extract_exercise_period_smart(kv, "CB")

        # Calculate shares from face_value / conv_price if not found directly
        if not conv_shares and face_value and conv_price and conv_price >= _MIN_CONVERSION_PRICE:
            conv_shares = face_value // conv_price

        footnote_text = _get_footnote_text(table, None)
        active = not _is_inactive(footnote_text)

        if face_value or conv_shares:
            results.append({
                "category": "CB",
                "series": series,
                "kind": kind_text,
                "face_value": face_value or 0,
                "convertible_shares": conv_shares or 0,
                "conversion_price": conv_price or 0,
                "exercise_start": ex_start,
                "exercise_end": ex_end,
                "active": active,
            })

    return results


def _parse_bond_direct(
    soup: BeautifulSoup, section_ps: list[Tag], category: str,
) -> list[dict]:
    """Parse CB/BW section without circled-number subsections.

    Some quarterly reports list CB details in a direct KV table without
    ① 제N회... headers. The series is identified from the table's first
    row (구분 → "제 N회 ...").

    Handles both single-instrument (2-column KV) and multi-instrument
    (N+1 column) table layouts. Iterates through all tables in the
    section to find the one with conversion-related data (e.g., 조건 table
    may follow a 내역 summary table).
    """
    instruments: list[dict] = []

    if not section_ps:
        return instruments

    # Collect all unique tables reachable from section <p> elements
    tables_seen: set[int] = set()
    tables: list[Tag] = []
    for p in section_ps:
        t = _find_next_table(p)
        if t and id(t) not in tables_seen:
            tables_seen.add(id(t))
            tables.append(t)

    if not tables:
        return instruments

    # Try each table for multi-column parsing (조건 table has series columns)
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue

        first_cells = rows[0].find_all(["td", "th"])
        if len(first_cells) < 2:
            continue

        first_label = _WS.sub("", first_cells[0].get_text(strip=True))

        # Determine which row has series headers and where data starts
        series_cells = None
        data_start_row = 1

        if first_label == "구분" and len(first_cells) > 2:
            # Row 0 directly has series columns: 구분 | 제1회 | 제2회 | ...
            series_cells = first_cells
            data_start_row = 1
        elif first_label == "구분" and len(rows) > 1:
            # Row 0 has merged header (구분 | 내역), check row 1 for series
            second_cells = rows[1].find_all(["td", "th"])
            if len(second_cells) > 2:
                series_cells = second_cells
                data_start_row = 2

        if series_cells and len(series_cells) > 2:
            found = []
            for col_idx in range(1, len(series_cells)):
                col_text = series_cells[col_idx].get_text(strip=True)
                m = re.search(r"(?:제\s*)?(\d+)\s*회", col_text)
                if m:
                    series = int(m.group(1))
                    inst = _parse_bond_column(
                        rows, col_idx, series, col_text, category, data_start_row,
                    )
                    if inst:
                        found.append(inst)
            if found:
                instruments.extend(found)
                return instruments

    # Fallback: single-column KV table (try first table)
    table = tables[0]
    kv = _parse_kv_table(table)
    rows = table.find_all("tr")
    first_cells = rows[0].find_all(["td", "th"]) if rows else []

    # Extract series from first row (may use <th> cells not captured by _parse_kv_table)
    series = 1
    kind_text = ""
    for cell in first_cells:
        text = cell.get_text(strip=True)
        m = re.search(r"(?:제\s*)?(\d+)\s*회", text)
        if m:
            series = int(m.group(1))
            kind_text = text.strip()
            break

    face_value = _extract_face_value_smart(kv)
    conv_price = _extract_conv_price_smart(kv, category)
    conv_shares = _extract_conv_shares_smart(kv, category)
    ex_start, ex_end = _extract_exercise_period_smart(kv, category)

    # Calculate shares from face_value / conv_price if not found directly
    if not conv_shares and face_value and conv_price and conv_price >= _MIN_CONVERSION_PRICE:
        conv_shares = face_value // conv_price

    footnote_text = _get_footnote_text(table, None)
    active = not _is_inactive(footnote_text)

    if face_value or conv_shares:
        instruments.append({
            "category": category,
            "series": series,
            "kind": kind_text,
            "face_value": face_value or 0,
            "convertible_shares": conv_shares or 0,
            "conversion_price": conv_price or 0,
            "exercise_start": ex_start,
            "exercise_end": ex_end,
            "active": active,
        })

    return instruments


def _parse_bond_column(
    rows: list[Tag], col_idx: int, series: int, kind_text: str, category: str,
    data_start_row: int = 1,
) -> dict | None:
    """Parse a single column from a multi-column bond table."""
    kv: dict[str, str] = {}
    for row in rows[data_start_row:]:
        cells = row.find_all(["td", "th"])
        if len(cells) <= col_idx:
            continue
        label = _WS.sub("", cells[0].get_text(strip=True))
        label = re.sub(r"\(\*\d+\)", "", label)
        value = cells[col_idx].get_text(separator="\n", strip=True)
        if label:
            kv[label] = value

    face_value = _extract_face_value_smart(kv)
    conv_price = _extract_conv_price_smart(kv, category)
    conv_shares = _extract_conv_shares_smart(kv, category)
    ex_start, ex_end = _extract_exercise_period_smart(kv, category)

    # Calculate shares from face_value / conv_price if not found directly
    if not conv_shares and face_value and conv_price and conv_price >= _MIN_CONVERSION_PRICE:
        conv_shares = face_value // conv_price

    if not face_value and not conv_shares:
        return None

    return {
        "category": category,
        "series": series,
        "kind": kind_text.strip(),
        "face_value": face_value or 0,
        "convertible_shares": conv_shares or 0,
        "conversion_price": conv_price or 0,
        "exercise_start": ex_start,
        "exercise_end": ex_end,
        "active": True,
    }


# ------------------------------------------------------------------
# Smart extraction helpers (handle multiple key name formats)
# ------------------------------------------------------------------

def _extract_face_value_smart(kv: dict[str, str]) -> int | None:
    """Extract face value from KV table, handling multiple key names and units.

    Handles:
    - 발행총액: 10,000,000,000원
    - 사채의액면(권면)총액(원): 25,000,000천원  (note: 천원 unit)
    - 권면총액: 10,000,000
    """
    # Try standard key first
    for key_fragment in ("발행총액", "권면총액", "총발행", "총액"):
        for k, v in kv.items():
            if key_fragment in k:
                return _extract_amount_with_unit(v)
    return None


def _extract_amount_with_unit(text: str | None) -> int | None:
    """Extract numeric amount, handling 천원 (×1,000) and 백만원 (×1,000,000) units."""
    if not text:
        return None
    text = re.sub(r"\(\*\d+\)", "", text).strip()

    # Try Korean text number first (e.g., "금 삼백억원", "오백억원")
    korean_val = _parse_korean_text_number(text)
    if korean_val is not None:
        return korean_val

    m = re.search(r"([\d,]+)\s*(천원|백만원|억원)?", text)
    if not m:
        return None
    try:
        value = int(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = m.group(2) or ""
    if unit == "천원":
        value *= 1_000
    elif unit == "백만원":
        value *= 1_000_000
    elif unit == "억원":
        value *= 1_0000_0000
    return value


# Korean digit/unit mappings for text number parsing
_KR_DIGITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_KR_UNITS = {"십": 10, "백": 100, "천": 1_000}
_KR_BIG_UNITS = {"만": 10_000, "억": 1_0000_0000, "조": 1_0000_0000_0000}


def _parse_korean_text_number(text: str) -> int | None:
    """Parse Korean text number like '금 삼백억원' → 30,000,000,000.

    Supports patterns:
    - 금 삼백억원, 오백억원, 이십억원
    - 금 일천오백억원 (150,000,000,000)
    """
    if not text:
        return None
    # Strip common prefixes/suffixes
    text = re.sub(r"^금\s*", "", text.strip())
    text = re.sub(r"원$", "", text.strip())

    # Must contain at least one Korean digit (일~구) — otherwise it's a
    # numeric string with unit suffix like "25,000,000천원", not a text number.
    if not any(c in text for c in _KR_DIGITS):
        return None

    # Parse: accumulate value with positional units
    # e.g., 삼백 → 3×100=300, then 억 → ×100,000,000
    total = 0
    current = 0  # current segment (before big unit)
    last_digit = 1  # implicit 1 for units without digit prefix (e.g., 백=100)

    for ch in text:
        if ch in _KR_DIGITS:
            last_digit = _KR_DIGITS[ch]
        elif ch in _KR_UNITS:
            current += last_digit * _KR_UNITS[ch]
            last_digit = 1
        elif ch in _KR_BIG_UNITS:
            current += last_digit if last_digit > 1 and current == 0 else 0
            if current == 0:
                current = 1
            total += current * _KR_BIG_UNITS[ch]
            current = 0
            last_digit = 1

    total += current
    return total if total > 0 else None


def _extract_conv_price_smart(kv: dict[str, str], category: str) -> int | None:
    """Extract conversion price from KV table, handling direct and text-block formats."""
    # Direct key: 전환가격(원/주), 전환가액(원/주), 행사가격(원/주), 행사가액(원/주)
    # Note: some reports use 가액 (value) instead of 가격 (price)
    price_keys = (
        ["행사가격", "행사가액", "전환가격", "전환가액", "발행가액"]
        if category == "BW"
        else ["전환가격", "전환가액", "행사가격", "행사가액", "발행가액"]
    )
    # Substrings that indicate per-share par value, NOT conversion price
    # Note: "액면" alone is NOT a marker — compound keys like "액면가액/발행가액(원)"
    # contain valid conversion prices.
    _PAR_VALUE_MARKERS = ("1주당", "주당")

    for pk in price_keys:
        for k, v in kv.items():
            if pk in k:
                # Guard: "발행가액" should not match per-share par value or total face value
                if pk == "발행가액" and ("총" in k or any(m in k for m in _PAR_VALUE_MARKERS)):
                    continue
                amt = _extract_amount(v)
                if amt and amt >= _MIN_CONVERSION_PRICE:
                    return amt

    # Fall back to text-block parsing: 전환에 관한 사항 / 신주인수권에 관한 사항
    conv_text = kv.get("전환에관한사항", "")
    if not conv_text:
        conv_text = kv.get("신주인수권에관한사항", "")
    price = _extract_conversion_price(conv_text, category)
    if price and price >= _MIN_CONVERSION_PRICE:
        return price
    return None



def _extract_conv_shares_smart(kv: dict[str, str], category: str) -> int | None:
    """Extract convertible shares from KV table, handling multiple key names."""
    # Direct keys (in order of specificity)
    share_keys = [
        "전환에따라발행할주식수",
        "전환시전환주식수",
        "행사시발행주식수",
    ]
    for sk in share_keys:
        for k, v in kv.items():
            if sk in k:
                amt = _extract_amount(v)
                if amt:
                    return amt
    return None


def _extract_exercise_period_smart(
    kv: dict[str, str], category: str,
) -> tuple[str, str]:
    """Extract exercise period from KV table, handling direct and text-block formats."""
    # Direct key: 전환청구기간, 행사기간, 신주인수권행사기간, 행사가능기간
    period_keys = ["전환청구기간", "행사기간", "신주인수권행사기간", "행사가능기간"]
    for pk in period_keys:
        for k, v in kv.items():
            if pk in k:
                # Try absolute dates first
                m = re.search(
                    r"(\d{4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})\s*일?"
                    r".*?"
                    r"(\d{4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})",
                    v, re.DOTALL,
                )
                if m:
                    start = f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"
                    end = f"{m.group(4)}.{int(m.group(5)):02d}.{int(m.group(6)):02d}"
                    return start, end
                # Try relative period (발행일 + N년 ~ 만기 - N개월)
                result = _resolve_relative_period(v, kv)
                if result[0]:
                    return result

    # Fall back to text-block parsing
    conv_text = kv.get("전환에관한사항", "")
    if not conv_text:
        conv_text = kv.get("신주인수권에관한사항", "")
    return _extract_exercise_period(conv_text, category)


def _resolve_relative_period(
    period_text: str, kv: dict[str, str],
) -> tuple[str, str]:
    """Resolve relative exercise period using 발행일/만기일 from KV table.

    Handles patterns like:
    - "발행일로부터 1년이 경과하는 날부터 만기 직전일까지"
    - "발행일로부터 1년이 경과하는 날부터 만기 1개월 전까지"
    """
    # Extract 발행일 and 만기일 from KV
    issue_date = _extract_date_from_kv(kv, "발행일")
    maturity_date = _extract_date_from_kv(kv, "만기일")
    if not issue_date or not maturity_date:
        return "", ""

    # Parse start offset: "N년이 경과" or "N년 후"
    start_date = None
    m = re.search(r"발행일.*?(\d+)\s*년", period_text)
    if m:
        years = int(m.group(1))
        start_date = issue_date.replace(year=issue_date.year + years)

    # Parse end: "만기 직전일" or "만기 N개월 전" or "만기일"
    end_date = None
    if "만기" in period_text:
        m_months = re.search(r"만기.*?(\d+)\s*개월\s*전", period_text)
        if m_months:
            months = int(m_months.group(1))
            # Subtract months from maturity date
            new_month = maturity_date.month - months
            new_year = maturity_date.year
            while new_month <= 0:
                new_month += 12
                new_year -= 1
            end_date = maturity_date.replace(year=new_year, month=new_month)
        elif "직전일" in period_text:
            end_date = maturity_date
        else:
            # Default: use maturity date
            end_date = maturity_date

    if start_date and end_date:
        return (
            start_date.strftime("%Y.%m.%d"),
            end_date.strftime("%Y.%m.%d"),
        )
    return "", ""


def _extract_date_from_kv(kv: dict[str, str], key_fragment: str):
    """Extract a date from KV table by key fragment (e.g. '발행일', '만기일')."""
    from datetime import date as _date

    for k, v in kv.items():
        if key_fragment in k:
            m = re.search(
                r"(\d{4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})",
                v,
            )
            if m:
                return _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _parse_preferred_section(
    soup: BeautifulSoup, section_ps: list[Tag],
) -> list[dict]:
    """Parse 전환우선주 section: find per-issuance subsections."""
    instruments: list[dict] = []

    # Find subsection headers: ① YYYY년 MM월 DD일 발행 전환우선주
    subsection_indices: list[tuple[int, str]] = []
    for i, p in enumerate(section_ps):
        text = p.get_text(strip=True)
        m = re.match(r"[①②③④⑤⑥⑦⑧⑨⑩]\s*(.+발행\s*전환우선주)", text)
        if m:
            subsection_indices.append((i, m.group(1).strip()))
        # Also handle: ① 전환우선주 (without date), ① 제N차 전환우선주
        elif re.match(r"[①②③④⑤⑥⑦⑧⑨⑩]\s*.*(?:전환우선주|상환전환)", text):
            kind = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", text).strip()
            subsection_indices.append((i, kind))

    if not subsection_indices:
        # Fallback: direct KV table without ①②③ subsections
        return _parse_preferred_direct(soup, section_ps)

    for idx, (sub_i, kind_text) in enumerate(subsection_indices):
        sub_p = section_ps[sub_i]

        if idx + 1 < len(subsection_indices):
            next_p = section_ps[subsection_indices[idx + 1][0]]
        else:
            next_p = None

        table = _find_next_table(sub_p)
        if table is None:
            continue

        kv = _parse_kv_table(table)
        if not kv:
            continue

        # Extract fields for preferred stock
        shares = _extract_amount(kv.get("발행주식수"))
        issue_price = _extract_amount(kv.get("1주당발행금액"))
        face_value = _extract_amount(kv.get("총발행가액"))

        # Conversion period
        conv_period_text = kv.get("전환기간", "")
        ex_start, ex_end = _extract_preferred_period(conv_period_text, kv)

        # Conversion ratio to determine actual convertible shares
        conv_condition = kv.get("전환조건", "")
        conv_ratio = _extract_conversion_ratio(conv_condition)
        convertible_shares = int(shares * conv_ratio) if shares else 0

        # Conversion price: issue_price adjusted by ratio
        conv_price = int(issue_price / conv_ratio) if issue_price and conv_ratio else issue_price or 0

        # Check inactive
        footnote_text = _get_footnote_text(table, next_p)
        active = not _is_inactive(footnote_text)

        instruments.append({
            "category": "전환우선주",
            "series": 0,
            "kind": kind_text.strip(),
            "face_value": face_value or 0,
            "convertible_shares": convertible_shares,
            "conversion_price": conv_price,
            "exercise_start": ex_start,
            "exercise_end": ex_end,
            "active": active,
        })

    return instruments


def _parse_preferred_direct(
    soup: BeautifulSoup, section_ps: list[Tag],
) -> list[dict]:
    """Parse preferred stock from direct KV table (no ①②③ subsections).

    Fallback for _parse_preferred_section when the section has a single
    preferred stock issuance described in a direct KV table.
    Iterates through all tables in the section to find one with
    conversion-related keys (조건 table may follow a 내역 summary table).
    """
    if not section_ps:
        return []

    # Collect all unique tables in the section
    tables_seen: set[int] = set()
    tables: list[Tag] = []
    for p in section_ps:
        t = _find_next_table(p)
        if t and id(t) not in tables_seen:
            tables_seen.add(id(t))
            tables.append(t)

    if not tables:
        return []

    # Merge KV data from all tables (later tables override earlier ones)
    kv: dict[str, str] = {}
    last_table = tables[0]
    for table in tables:
        table_kv = _parse_kv_table(table)
        if table_kv:
            kv.update(table_kv)
            last_table = table

    if not kv:
        return []

    # Extract kind from section header (remove "(N) " prefix)
    kind_text = section_ps[0].get_text(strip=True)
    kind_text = re.sub(r"^\(\d+\)\s*", "", kind_text).strip()

    # Face value (총발행가액, 발행총액)
    face_value = _extract_face_value_smart(kv)

    # Shares (발행주식수)
    shares = None
    for k, v in kv.items():
        if "발행주식수" in k or "발행수량" in k:
            shares = _extract_amount(v)
            if shares:
                break

    # Issue price per share (1주당 발행금액)
    issue_price = None
    for k, v in kv.items():
        if "1주당" in k and ("발행" in k or "금액" in k or "가액" in k):
            issue_price = _extract_amount(v)
            if issue_price:
                break

    # Conversion price (전환가격, 전환가액)
    conv_price = _extract_conv_price_smart(kv, "CB")

    # Exercise period: try direct date keys, then fall back to _extract_preferred_period
    ex_start, ex_end = "", ""
    for k, v in kv.items():
        if "전환청구기간" in k or "전환기간" in k or "행사가능기간" in k:
            # Try direct date range extraction
            m = re.search(
                r"(\d{4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})\s*일?"
                r".*?"
                r"(\d{4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})",
                v, re.DOTALL,
            )
            if m:
                ex_start = f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"
                ex_end = f"{m.group(4)}.{int(m.group(5)):02d}.{int(m.group(6)):02d}"
            else:
                # Try relative period ("N년이 경과한 날부터 ... N년이 되는 날")
                ex_start, ex_end = _extract_preferred_period(v, kv)
                if not ex_start:
                    ex_start, ex_end = _resolve_relative_period(v, kv)
            if ex_start:
                break

    # Conversion ratio
    conv_condition = ""
    for k, v in kv.items():
        if "전환조건" in k or "전환비율" in k:
            conv_condition = v
            break
    conv_ratio = _extract_conversion_ratio(conv_condition)

    # Calculate convertible shares and conversion price
    if shares:
        convertible_shares = int(shares * conv_ratio)
    elif face_value and (conv_price or issue_price):
        price = conv_price or issue_price
        convertible_shares = face_value // price
    else:
        convertible_shares = 0

    if not conv_price:
        if issue_price and conv_ratio:
            conv_price = int(issue_price / conv_ratio)
        else:
            conv_price = issue_price or 0

    # Check inactive
    footnote_text = _get_footnote_text(last_table, None)
    active = not _is_inactive(footnote_text)

    if face_value or convertible_shares:
        return [{
            "category": "전환우선주",
            "series": 0,
            "kind": kind_text,
            "face_value": face_value or 0,
            "convertible_shares": convertible_shares,
            "conversion_price": conv_price or 0,
            "exercise_start": ex_start,
            "exercise_end": ex_end,
            "active": active,
        }]

    return []


def _parse_stock_option_section(
    soup: BeautifulSoup, section_ps: list[Tag],
) -> list[dict]:
    """Parse 주식기준보상 section: extract stock option grants.

    Stock option notes typically contain a summary table with columns:
    구분(차수), 부여일, 행사가격, 부여수량, 행사/소멸수량, 잔여수량, 행사기간 등.
    """
    instruments: list[dict] = []

    # Find all tables in this section using section boundary detection
    tables: list[Tag] = []
    if section_ps:
        for p in section_ps:
            for sibling in p.find_all_next():
                if not isinstance(sibling, Tag):
                    continue
                if sibling.name == "table" and sibling not in tables:
                    tables.append(sibling)
                # Stop at next numbered section header
                if sibling.name == "p":
                    text = sibling.get_text(strip=True)
                    if re.match(r"^\d+\.\s+\S", text) and sibling not in section_ps:
                        break
            if tables:
                break  # Found tables from the first relevant <p>

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Detect header row to find column indices
        header_cells = rows[0].find_all(["td", "th"])
        header_texts = [_WS.sub("", c.get_text(strip=True)) for c in header_cells]

        col_map: dict[str, int] = {}
        for i, h in enumerate(header_texts):
            if "행사가격" in h or "행사가액" in h:
                col_map["exercise_price"] = i
            elif "잔여" in h or "미행사" in h:
                col_map["remaining"] = i
            elif "부여수량" in h or "부여주식" in h:
                col_map["granted"] = i
            elif "행사기간" in h:
                col_map["period"] = i
            elif "구분" in h or "차수" in h:
                col_map["series"] = i

        # Need at least exercise_price or remaining to be useful
        if not col_map.get("exercise_price") and not col_map.get("remaining"):
            continue

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if not texts or len(texts) <= max(col_map.values(), default=0):
                continue

            # Skip header-like or empty rows
            first = _WS.sub("", texts[0])
            if not first or first in ("합계", "계", "총계"):
                continue

            # Extract series number
            series = 0
            if "series" in col_map:
                series_text = texts[col_map["series"]]
                m = re.search(r"(\d+)", series_text)
                if m:
                    series = int(m.group(1))

            # Extract exercise price
            exercise_price = 0
            if "exercise_price" in col_map:
                exercise_price = _extract_amount(texts[col_map["exercise_price"]]) or 0

            # Extract remaining (exercisable) shares
            remaining_shares = 0
            if "remaining" in col_map:
                remaining_shares = _extract_amount(texts[col_map["remaining"]]) or 0
            elif "granted" in col_map:
                remaining_shares = _extract_amount(texts[col_map["granted"]]) or 0

            if remaining_shares <= 0:
                continue

            # Extract exercise period
            ex_start, ex_end = "", ""
            if "period" in col_map:
                period_text = texts[col_map["period"]]
                m = re.search(
                    r"(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2}).*?"
                    r"(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})",
                    period_text,
                )
                if m:
                    ex_start = f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"
                    ex_end = f"{m.group(4)}.{int(m.group(5)):02d}.{int(m.group(6)):02d}"

            instruments.append({
                "category": "SO",
                "series": series,
                "kind": f"주식매수선택권 {series}차" if series else "주식매수선택권",
                "face_value": 0,
                "convertible_shares": remaining_shares,
                "conversion_price": exercise_price,
                "exercise_start": ex_start,
                "exercise_end": ex_end,
                "active": True,
            })

    return instruments


# ------------------------------------------------------------------
# HTML navigation helpers
# ------------------------------------------------------------------

def _find_next_table(element: Tag) -> Tag | None:
    """Find the first <table> after a given element."""
    for sibling in element.find_all_next():
        if isinstance(sibling, Tag):
            if sibling.name == "table":
                return sibling
            # Stop if we hit another subsection header (circled number)
            if sibling.name == "p":
                text = sibling.get_text(strip=True)
                if re.match(r"[①②③④⑤⑥⑦⑧⑨⑩]", text):
                    return None
    return None


def _parse_kv_table(table: Tag) -> dict[str, str]:
    """Parse a 2-column key-value table into a dict.

    Handles rowspan: when a label cell spans multiple rows, subsequent
    rows have only 1 <td> (the value).  All values for the same label
    are combined with newline.

    Normalizes keys by removing whitespace and footnote markers (*N).
    """
    kv: dict[str, str] = {}
    current_label = ""
    remaining_rowspan = 0

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue

        if len(cells) >= 2:
            # Normal 2-cell row or start of a rowspanned label
            raw_label = cells[0].get_text(strip=True)
            value = cells[1].get_text(separator="\n", strip=True)

            # Check for rowspan on the label cell
            rowspan = int(cells[0].get("rowspan", 1))
            if rowspan > 1:
                remaining_rowspan = rowspan - 1

            label = _WS.sub("", raw_label)
            label = re.sub(r"\(\*\d+\)", "", label)
            if label:
                current_label = label
        elif len(cells) == 1 and remaining_rowspan > 0:
            # Continuation row under a rowspanned label
            value = cells[0].get_text(separator="\n", strip=True)
            label = current_label
            remaining_rowspan -= 1
        else:
            continue

        if not label:
            continue

        # Combine values for the same label
        if label in kv:
            kv[label] = kv[label] + "\n" + value
        else:
            kv[label] = value

    return kv


def _get_footnote_text(table: Tag, next_boundary: Tag | None) -> str:
    """Get text between a table and the next subsection boundary.

    This captures footnotes like (*1) ... (*2) ... that indicate
    whether an instrument has been converted/redeemed.
    """
    parts: list[str] = []
    for sibling in table.find_all_next():
        if not isinstance(sibling, Tag):
            continue
        if next_boundary and sibling == next_boundary:
            break
        # Stop at the next table or next subsection
        if sibling.name == "table":
            break
        if sibling.name == "p":
            text = sibling.get_text(strip=True)
            # Stop at next circled-number subsection
            if re.match(r"[①②③④⑤⑥⑦⑧⑨⑩]", text):
                break
            # Stop at next numbered section
            if re.match(r"^\d+\.\s+\S", text):
                break
            parts.append(text)
    return "\n".join(parts)


# ------------------------------------------------------------------
# Value extraction helpers
# ------------------------------------------------------------------

def _extract_amount(text: str | None) -> int | None:
    """Extract numeric amount from text like '10,000,000,000원' or '877,346주'."""
    if not text:
        return None
    # Remove footnote markers
    text = re.sub(r"\(\*\d+\)", "", text).strip()
    # Remove "N주당" prefix (e.g., "1주당 16,672원" → "16,672원")
    text = re.sub(r"^\d+주당\s*", "", text)
    # Find the numeric part
    m = re.search(r"([\d,]+)", text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_conversion_price(text: str, category: str) -> int | None:
    """Extract conversion/exercise price from 전환에 관한 사항 text."""
    if not text:
        return None

    if category == "CB":
        # Pattern: "전환가격: 11,398원 / 주" or "전환가격(*1): 14,646원 / 주"
        m = re.search(r"전환가격[^:：]*[:：]\s*([\d,]+)\s*원", text)
    elif category == "BW":
        # Pattern: "행사가격: 30,000원 / 주"
        m = re.search(r"행사가격[^:：]*[:：]\s*([\d,]+)\s*원", text)
    else:
        return None

    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _extract_exercise_period(
    text: str, category: str,
) -> tuple[str, str]:
    """Extract exercise period dates from 전환에 관한 사항 text.

    Returns (start_date, end_date) in 'YYYY.MM.DD' format.
    """
    start, end = "", ""
    if not text:
        return start, end

    if category == "CB":
        # Pattern: "전환청구가능기간: ... (2023년 09월 28일)로부터 ... (2027년 08월 28일)"
        m = re.search(
            r"전환청구가능기간[^:：]*[:：].*?"
            r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일.*?"
            r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
            text, re.DOTALL,
        )
    elif category == "BW":
        # Pattern: "행사기간: ... (2024년 03월 15일)로부터 ... (2028년 03월 14일)"
        m = re.search(
            r"(?:행사기간|신주인수권행사기간)[^:：]*[:：].*?"
            r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일.*?"
            r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
            text, re.DOTALL,
        )
    else:
        return start, end

    if m:
        start = f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"
        end = f"{m.group(4)}.{int(m.group(5)):02d}.{int(m.group(6)):02d}"

    return start, end


def _extract_preferred_period(
    period_text: str, kv: dict[str, str],
) -> tuple[str, str]:
    """Extract conversion period for preferred stock.

    Pattern: "최초발행일 후 1년이 경과한 날부터 ... 5년이 되는 날"
    Uses 발행일 from kv to compute actual dates.
    """
    start, end = "", ""
    if not period_text:
        return start, end

    issue_date_text = kv.get("발행일", "")
    # Extract issue date: "2024년 12월 14일" or "2024-12-14"
    m = re.search(r"(\d{4})\s*[년-]\s*(\d{1,2})\s*[월-]\s*(\d{1,2})", issue_date_text)
    if not m:
        return start, end

    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Extract N years for start: "N년이 경과한 날"
    start_years_m = re.search(r"(\d+)\s*년이\s*경과한\s*날", period_text)
    # Extract N years for end: "N년이 되는 날"
    end_years_m = re.search(r"(\d+)\s*년이\s*되는\s*날", period_text)

    if start_years_m:
        n = int(start_years_m.group(1))
        start = f"{year + n}.{month:02d}.{day:02d}"
    if end_years_m:
        n = int(end_years_m.group(1))
        end = f"{year + n}.{month:02d}.{day:02d}"

    return start, end


def _extract_conversion_ratio(text: str) -> float:
    """Extract conversion ratio from text like '전환우선주 1주당 보통주 1주'.

    Returns the ratio (e.g., 1.0, 0.891).
    """
    if not text:
        return 1.0
    m = re.search(r"(\d+)\s*주당\s*보통주\s*([\d.]+)\s*주", text)
    if m:
        try:
            return float(m.group(2)) / float(m.group(1))
        except (ValueError, ZeroDivisionError):
            return 1.0
    return 1.0


def _is_inactive(text: str) -> bool:
    """Check if footnote text indicates the instrument is inactive."""
    for pattern in _INACTIVE_PATTERNS:
        if pattern.search(text):
            return True
    return False
