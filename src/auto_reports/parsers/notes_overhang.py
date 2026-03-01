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


def parse_notes_overhang(html: str) -> list[dict]:
    """Parse overhang instruments from financial statement notes HTML.

    Searches for 전환사채, 신주인수권부사채, and 전환우선주 sections,
    then extracts instrument details from 2-column key-value tables.

    Returns list of dicts with keys:
        category, series, kind, face_value, convertible_shares,
        conversion_price, exercise_start, exercise_end, active
    """
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
        return instruments

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

        # Extract fields
        face_value = _extract_amount(kv.get("발행총액"))
        conv_shares = _extract_amount(kv.get("전환시전환주식수"))
        if conv_shares is None:
            # BW uses different field name
            conv_shares = _extract_amount(kv.get("행사시발행주식수"))

        # Conversion price and period from "전환에 관한 사항" text
        conv_text = kv.get("전환에관한사항", "")
        # BW uses "신주인수권에 관한 사항"
        if not conv_text:
            conv_text = kv.get("신주인수권에관한사항", "")

        conv_price = _extract_conversion_price(conv_text, category)
        ex_start, ex_end = _extract_exercise_period(conv_text, category)

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
        return instruments

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
