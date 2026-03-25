"""LLM-based overhang extraction from financial statement notes (재무제표 주석).

Searches for keywords (전환사채, 신주인수권부사채, 전환우선주, 주식매수선택권 등)
in the notes HTML, extracts surrounding context (text + tables), and uses an LLM
to extract structured instrument data.

This approach handles diverse table formats across different companies
where deterministic regex/table parsing fails.
"""

from __future__ import annotations

import json
import logging
import re

from bs4 import BeautifulSoup, Tag
import time

from openai import OpenAI, RateLimitError

from auto_reports.fetchers.rate_limiter import get_llm_limiter

logger = logging.getLogger(__name__)

OVERHANG_KEYWORDS = [
    "전환사채",
    "신주인수권부사채",
    "전환우선주",
    "전환상환우선주",
    "상환전환우선주",
    "주식매수선택권",
    "주식기준보상",
]

_WS = re.compile(r"[\s\xa0]+")

_SYSTEM_PROMPT = """\
You are a Korean financial document parser. You extract structured data about \
convertible/dilutive securities (overhang instruments) from Korean periodic \
report financial notes (재무제표 주석).

You MUST return a JSON object with key "instruments" containing an array. \
Each element represents one instrument with these fields:

{
  "category": "CB" | "BW" | "전환우선주" | "SO",
  "series": <int>,
  "kind": "<string>",
  "face_value": <int>,
  "remaining_balance": <int>,
  "convertible_shares": <int>,
  "conversion_price": <int>,
  "exercise_start": "YYYY.MM.DD",
  "exercise_end": "YYYY.MM.DD",
  "active": true | false
}

Field definitions:
- category: "CB" for 전환사채, "BW" for 신주인수권부사채, \
"전환우선주" for 전환우선주/전환상환우선주/상환전환우선주, \
"SO" for 주식매수선택권/주식기준보상
- series: 회차 number (제1회→1, 제2회→2, 1차→1). 전환우선주→0
- kind: Full instrument name (e.g. "제1회 무기명식 이권부 무보증 사모 전환사채", \
"주식매수선택권 3차")
- face_value: Total face value in KRW (원). \
Convert units: 천원→×1000, 백만원→×1000000, 억원→×100000000. \
Korean text numbers: 금 삼백억원 = 30000000000. \
For 주식매수선택권(SO), face_value is 0 (not applicable).
- remaining_balance: Unconverted/unredeemed remaining balance in KRW (원). \
Look for: 미전환사채 잔액, 미상환 잔액, 미행사 잔액, 신고일 현재 미전환사채 잔액. \
If "전액 전환", "전액 상환", "전부 전환", or 잔액 = 0, return 0. \
If not explicitly stated, use face_value as default. \
For SO, remaining_balance is 0.
- convertible_shares: Shares issuable on conversion/exercise. \
For CB/BW: if not stated, calculate: face_value ÷ conversion_price (floor division). \
For SO: use 잔여수량/미행사수량 (remaining exercisable shares). \
If only 부여수량 is available, use that.
- conversion_price: Price per share in KRW (원). \
For SO: use 행사가격/행사가액.
- exercise_start / exercise_end: Exercise period dates in "YYYY.MM.DD" format. \
"" if unknown. Resolve relative dates (e.g. "발행일로부터 1년이 경과한 날") \
using 발행일/만기일 from the document.
- active: false if fully converted/redeemed/exercised. \
Indicators: "모두 보통주로 전환", "전액 상환", "전액 전환", \
"전부 전환", "전환 완료", "상환 완료", "전량 행사", "행사 완료", \
"소멸", "취소"

Key extraction rules:
1. Face value keys: 발행총액, 권면(액면)총액, 사채의 액면총액, 총발행가액
2. Conversion price keys: 전환가격, 전환가액, 행사가격, 행사가액
3. Convertible shares keys: 전환에 따라 발행할 주식수, 전환시 전환주식수, \
행사시 발행주식수
4. Exercise period keys: 전환청구기간, 행사기간, 신주인수권행사기간, 행사가능기간
5. For 전환우선주 with conversion ratio (e.g. "우선주 1주당 보통주 1주"): \
convertible_shares = 발행주식수 × ratio, \
conversion_price = 1주당발행금액 ÷ ratio
6. Multiple instruments (제1회, 제2회, 1차, 2차): create separate entries for each.
7. Stock options (주식매수선택권/주식기준보상):
   - Table columns: 구분/차수, 부여일, 행사가격, 부여수량, 행사수량, 소멸수량, \
잔여수량/미행사수량, 행사기간
   - face_value = 0 (always)
   - convertible_shares = 잔여수량 or 미행사수량 (remaining unexercised shares). \
If only 부여수량 is given, use 부여수량 - 행사수량 - 소멸수량.
   - conversion_price = 행사가격/행사가액
   - Create one entry per 차수/grant with remaining shares > 0.
   - Skip rows where 잔여수량 = 0 (fully exercised or expired).

If no instruments found, return {"instruments": []}."""


def extract_overhang_context(html: str) -> str:
    """Extract relevant HTML context around overhang keywords from notes HTML.

    Searches for sections containing overhang keywords and extracts
    the section content including tables and surrounding text.
    Returns simplified HTML suitable for LLM processing.
    """
    soup = BeautifulSoup(html, "html.parser")

    p_elements = list(soup.find_all("p"))
    context_parts: list[str] = []
    covered_indices: set[int] = set()

    # Pass 1: Find section/subsection headers containing keywords
    for i, p in enumerate(p_elements):
        if i in covered_indices:
            continue
        text = _WS.sub("", p.get_text(strip=True))
        if not any(kw in text for kw in OVERHANG_KEYWORDS):
            continue

        # Determine section scope based on header pattern
        is_top_section = bool(re.match(r"^\d+\.", text))
        is_subsection = bool(re.match(r"\(\d+\)", text))

        if is_top_section:
            end_idx = _find_section_end_idx(p_elements, i)
        elif is_subsection:
            end_idx = _find_subsection_end_idx(p_elements, i)
        else:
            # Inline keyword mention: grab limited surrounding context
            end_idx = min(i + 15, len(p_elements))

        for j in range(i, end_idx):
            covered_indices.add(j)

        section_html = _collect_section_html(p_elements, i, end_idx)
        if section_html.strip():
            context_parts.append(section_html)

    # Pass 2: Find tables containing keywords not already covered
    collected_table_ids: set[int] = set()
    for table in soup.find_all("table"):
        if id(table) in collected_table_ids:
            continue
        table_text = table.get_text()
        if not any(kw in table_text for kw in OVERHANG_KEYWORDS):
            continue

        # Check if this table is already within a covered section
        table_p = table.find_previous("p")
        if table_p:
            try:
                p_idx = p_elements.index(table_p)
                if p_idx in covered_indices:
                    continue
            except ValueError:
                pass

        collected_table_ids.add(id(table))
        # Include preceding header paragraph for context
        header = ""
        if table_p:
            header = f"<p>{table_p.get_text(strip=True)}</p>\n"
        context_parts.append(header + str(table))

    if not context_parts:
        return ""

    combined = "\n\n<!-- section break -->\n\n".join(context_parts)

    # Strip style/class attributes to reduce size
    combined = re.sub(
        r'\s+(?:style|class|width|height|align|valign|bgcolor|border'
        r'|cellpadding|cellspacing|rowspan|colspan)="[^"]*"',
        "", combined,
    )

    # Truncate at section boundary to avoid splitting mid-HTML-tag
    # Use 30000 chars to accommodate companies with many CB series (e.g. 15+)
    if len(combined) > 30000:
        cutoff = combined.rfind("<!-- section break -->", 0, 30000)
        combined = combined[:cutoff] if cutoff > 0 else combined[:30000]

    return combined


def _find_section_end_idx(p_elements: list[Tag], start: int) -> int:
    """Find end of a numbered section (e.g. '13. 전환사채')."""
    for i in range(start + 1, len(p_elements)):
        text = p_elements[i].get_text(strip=True)
        if re.match(r"^\d+\.\s+\S", text):
            return i
    return len(p_elements)


def _find_subsection_end_idx(p_elements: list[Tag], start: int) -> int:
    """Find end of a (N) subsection."""
    for i in range(start + 1, len(p_elements)):
        text = p_elements[i].get_text(strip=True)
        if re.match(r"\(\d+\)", text):
            return i
        if re.match(r"^\d+\.\s+\S", text):
            return i
    return len(p_elements)


def _collect_section_html(
    p_elements: list[Tag], start_idx: int, end_idx: int,
) -> str:
    """Collect HTML content for a section: paragraphs and tables between them."""
    parts: list[str] = []
    seen_tables: set[int] = set()

    for i in range(start_idx, end_idx):
        p = p_elements[i]
        parts.append(str(p))

        # Find all tables between this <p> and the next one
        next_p = p_elements[i + 1] if i + 1 < end_idx else None
        for sibling in p.find_all_next():
            if not isinstance(sibling, Tag):
                continue
            if next_p and sibling == next_p:
                break
            if sibling.name == "table" and id(sibling) not in seen_tables:
                seen_tables.add(id(sibling))
                parts.append(str(sibling))

    return "\n".join(parts)


# ------------------------------------------------------------------
# LLM extraction
# ------------------------------------------------------------------


def parse_notes_overhang_llm(
    html: str,
    api_key: str,
    model: str = "gpt-4.1-mini",
    base_url: str = "",
) -> list[dict]:
    """Extract overhang instruments from notes HTML using LLM.

    Args:
        html: Raw HTML of the 재무제표 주석 section
        api_key: OpenAI API key
        model: Model to use (default: gpt-4.1-mini)

    Returns:
        List of instrument dicts with 9 standard fields, or empty list on failure.
    """
    context = extract_overhang_context(html)
    if not context.strip():
        logger.info("No overhang-related context found in notes HTML")
        return []

    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )

    user_prompt = (
        "다음은 한국 상장기업 정기보고서의 재무제표 주석에서 추출한 "
        "전환사채/신주인수권부사채/전환우선주/주식매수선택권 관련 HTML입니다.\n\n"
        f"{context}\n\n"
        "위 내용에서 모든 전환사채(CB), 신주인수권부사채(BW), "
        "전환우선주/전환상환우선주/상환전환우선주, "
        "주식매수선택권(SO) 정보를 추출하여 "
        "JSON으로 반환해주세요.\n"
        "각 항목에 대해 category, series, kind, face_value, "
        "convertible_shares, conversion_price, exercise_start, "
        "exercise_end, active를 추출합니다.\n"
        "금액은 반드시 원(KRW) 단위로 변환하세요.\n"
        "전환가능주식수가 직접 명시되지 않은 경우 "
        "액면총액 ÷ 전환가격으로 계산하세요.\n"
        "주식매수선택권은 face_value=0이며, "
        "잔여수량/미행사수량을 convertible_shares로, "
        "행사가격을 conversion_price로 사용하세요."
    )

    _llm_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    get_llm_limiter().wait()
    try:
        response = client.chat.completions.create(**_llm_kwargs)
        content = (response.choices[0].message.content or "").strip()
    except RateLimitError as e:
        logger.warning("LLM rate limited: %s — retrying with backoff", e)
        content = ""
        for attempt in range(3):
            time.sleep(2 ** (attempt + 1))
            get_llm_limiter().wait()
            try:
                response = client.chat.completions.create(**_llm_kwargs)
                content = (response.choices[0].message.content or "").strip()
                break
            except RateLimitError:
                continue
            except Exception:
                break
        if not content:
            logger.warning("LLM rate limit retries exhausted")
            return []
    except Exception as e:
        logger.warning("LLM overhang extraction failed: %s", e)
        return []

    return _parse_llm_response(content)


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

_VALID_CATEGORIES = {"CB", "BW", "전환우선주", "SO"}


def _parse_llm_response(content: str) -> list[dict]:
    """Parse LLM JSON response into list of instrument dicts."""
    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON array from response text
        m = re.search(r"\[.*]", content, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return []
        else:
            logger.warning("No JSON array found in LLM response")
            return []

    # Handle wrapper objects like {"instruments": [...]}
    if isinstance(data, dict):
        for key in ("instruments", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            return []

    if not isinstance(data, list):
        return []

    results: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        inst = _normalize_instrument(item)
        if inst:
            results.append(inst)

    return results


def _normalize_instrument(item: dict) -> dict | None:
    """Normalize a single instrument dict from LLM output."""
    category = str(item.get("category", ""))
    if category not in _VALID_CATEGORIES:
        # Map common variants
        if "전환사채" in category:
            category = "CB"
        elif "신주인수권" in category:
            category = "BW"
        elif any(kw in category for kw in ("전환우선주", "전환상환", "상환전환", "RCPS")):
            category = "전환우선주"
        elif any(kw in category for kw in ("주식매수선택권", "주식기준보상", "스톡옵션")):
            category = "SO"
        else:
            return None

    face_value = _safe_int(item.get("face_value", 0))
    conversion_price = _safe_int(item.get("conversion_price", 0))
    convertible_shares = _safe_int(item.get("convertible_shares", 0))

    # Validate face_value against conversion_price × convertible_shares.
    # LLMs sometimes misconvert 천원/백만원 units, producing a face_value
    # that is 1000x or 1,000,000x the expected amount.
    if face_value and conversion_price and convertible_shares:
        expected = conversion_price * convertible_shares
        if expected > 0:
            ratio = face_value / expected
            if 900 < ratio < 1100:
                face_value = expected
            elif 900_000 < ratio < 1_100_000:
                face_value = expected

    # Calculate convertible_shares if missing (CB/BW only; SO has no face_value)
    if not convertible_shares and face_value and conversion_price and category != "SO":
        convertible_shares = face_value // conversion_price

    # Skip instruments with no meaningful data
    # SO instruments have face_value=0, so only check convertible_shares
    if category == "SO":
        if not convertible_shares:
            return None
    elif not face_value and not convertible_shares:
        return None

    return {
        "category": category,
        "series": _safe_int(item.get("series", 0)),
        "kind": str(item.get("kind", "")).strip(),
        "face_value": face_value,
        "convertible_shares": convertible_shares,
        "conversion_price": conversion_price,
        "exercise_start": _normalize_date(str(item.get("exercise_start", ""))),
        "exercise_end": _normalize_date(str(item.get("exercise_end", ""))),
        "active": bool(item.get("active", True)),
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_int(value) -> int:
    """Convert value to int safely."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[,\s원주]", "", value)
        try:
            return int(cleaned)
        except (ValueError, TypeError):
            return 0
    return 0


def _normalize_date(date_str: str) -> str:
    """Normalize date to YYYY.MM.DD format."""
    if not date_str or date_str in ("", "None", "null"):
        return ""
    # YYYY.MM.DD or YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", date_str)
    if m:
        return f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"
    # YYYY년 MM월 DD일
    m = re.match(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})", date_str)
    if m:
        return f"{m.group(1)}.{int(m.group(2)):02d}.{int(m.group(3)):02d}"
    return ""
