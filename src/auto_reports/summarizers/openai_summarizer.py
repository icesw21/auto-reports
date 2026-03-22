"""OpenAI-based summarizer for DART business section content."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI, RateLimitError

from auto_reports.fetchers.rate_limiter import get_llm_limiter

logger = logging.getLogger(__name__)


def _detect_revenue_unit(text: str) -> str:
    """Detect monetary unit from DART revenue section text.

    Looks for patterns like '(단위: 천원)', '(단위 : 백만원)',
    '(단위: 천RMB)', '(단위: 백만CNY)' etc.
    Returns the matched unit string, or '백만원' as default.
    """
    # Match: 천원, 백만원, 억원, 천RMB, 백만CNY, 천USD, 억EUR, etc.
    m = re.search(
        r"단위\s*[:：]\s*((천|백만|억)(원|[A-Z]{2,3}|RMB))",
        text,
    )
    if m:
        return m.group(1)
    return "백만원"


def _build_conv_rule(source_unit: str, display_unit: str) -> str:
    """Build conversion rule string for LLM prompt.

    Maps source_unit (e.g. '천원', '천RMB', '백만CNY') to display_unit
    (e.g. '억원', '백만 RMB').
    """
    # Parse source unit: prefix (천/백만/억) + currency part (원/RMB/CNY/...)
    m = re.match(r"(천|백만|억)(.*)", source_unit)
    if not m:
        return f"원문의 금액 단위를 {display_unit}로 변환하세요."
    prefix = m.group(1)

    # If source and display are equivalent, no conversion needed
    if source_unit == display_unit or source_unit == display_unit.replace(" ", ""):
        return f"원문의 금액 단위: {source_unit} (변환 불필요, 그대로 사용)"

    # Determine divisor from source prefix to display unit
    # For KRW: target is 억 (1억 = 100,000천원 = 100백만원 = 1억원)
    # For foreign: target is 백만 (1백만 = 1,000천 = 1백만)
    if display_unit == "억원":
        if prefix == "천":
            return (
                f"원문의 금액 단위: {source_unit}\n"
                f"  → {display_unit} 변환법: 숫자 ÷ 100,000 후 반올림\n"
                f"  → 예시: 1,192,400 ({source_unit}) ÷ 100,000 = 12 ({display_unit})"
            )
        elif prefix == "백만":
            return (
                f"원문의 금액 단위: {source_unit}\n"
                f"  → {display_unit} 변환법: 숫자 ÷ 100 후 반올림\n"
                f"  → 예시: 1,192 ({source_unit}) ÷ 100 = 12 ({display_unit})"
            )
        else:  # 억
            return f"원문의 금액 단위: {source_unit} (변환 불필요, 그대로 사용)"
    else:
        # Foreign currency: target is 백만 {currency}
        if prefix == "천":
            return (
                f"원문의 금액 단위: {source_unit}\n"
                f"  → {display_unit} 변환법: 숫자 ÷ 1,000 후 반올림\n"
                f"  → 예시: 1,192,400 ({source_unit}) ÷ 1,000 = 1,192 ({display_unit})"
            )
        elif prefix == "백만":
            return f"원문의 금액 단위: {source_unit} (변환 불필요, 그대로 사용)"
        else:  # 억
            return (
                f"원문의 금액 단위: {source_unit}\n"
                f"  → {display_unit} 변환법: 숫자 × 100\n"
                f"  → 예시: 12 ({source_unit}) × 100 = 1,200 ({display_unit})"
            )


def _strip_revenue_summary(text: str) -> str:
    """Remove aggregate summary rows from DART revenue table.

    Revenue tables often end with a 합계/총계 section that re-aggregates
    data by segment (e.g., EV Components total across 제품+상품).
    This causes the LLM to use aggregated values instead of the detailed
    per-type breakdown. Stripping the summary forces the LLM to work
    with individual 제품/상품/기타매출 rows only.
    """
    lines = text.split('\n')

    # Find last 소계 (subtotal within individual sections)
    last_subtotal = -1
    for i, line in enumerate(lines):
        if line.strip() == '소계':
            last_subtotal = i

    if last_subtotal < 0:
        return text

    # Find first standalone "합계" after the last 소계 (summary section header)
    # Use prefix match to also catch "합계  88,197" on a single line.
    summary_start = None
    for i in range(last_subtotal + 1, len(lines)):
        if re.match(r'^합계\b', lines[i].strip()):
            summary_start = i
            break

    if summary_start is None:
        return text

    # Find next Korean section header (나., 다., 라., etc.) or end.
    # If no header follows, everything after summary_start is stripped.
    section_end = len(lines)
    for i in range(summary_start + 1, len(lines)):
        if re.match(r'^[가-힣]\.\s', lines[i].strip()):
            section_end = i
            break

    logger.debug(
        "Stripped revenue summary: lines %d-%d (of %d)",
        summary_start, section_end, len(lines),
    )
    return '\n'.join(lines[:summary_start] + lines[section_end:])


def _fix_revenue_breakdown_scale(
    table_text: str,
    known_revenue_won: int | None,
    display_unit: str = "억원",
) -> str:
    """Auto-correct revenue breakdown table if amounts are off by a power of 10.

    Parses the markdown table, sums the amounts from non-합계/총계 rows,
    compares against known_revenue_won, and rescales if the ratio is close
    to a power of 10 (e.g. 10x, 100x).
    """
    if not known_revenue_won or not table_text:
        return table_text

    # Convert known revenue to display unit
    if display_unit == "억원":
        known_display = known_revenue_won / 1_0000_0000
    else:
        known_display = known_revenue_won / 1_000_000

    if known_display <= 0:
        return table_text

    # Parse amounts from table rows (excluding 합계/총계 rows)
    amount_pattern = re.compile(r"^\|[^|]+\|[^|]+\|\s*([\d,]+)\s*\|")
    total_pattern = re.compile(r"합계|총계")
    amounts: list[int] = []
    for line in table_text.split("\n"):
        if total_pattern.search(line):
            continue
        m = amount_pattern.match(line.strip())
        if m:
            try:
                amounts.append(int(m.group(1).replace(",", "")))
            except ValueError:
                continue

    if not amounts:
        return table_text

    table_sum = sum(amounts)
    if table_sum <= 0:
        return table_text

    ratio = table_sum / known_display
    # Find nearest power of 10
    for factor in (10, 100, 1000):
        if 0.7 * factor <= ratio <= 1.5 * factor:
            logger.info(
                "Revenue breakdown scale fix: table_sum=%d, known=%d, factor=%dx",
                table_sum, int(known_display), factor,
            )
            # Rescale all numeric amounts in the table
            def _rescale(match: re.Match) -> str:
                raw = match.group(0)
                try:
                    val = int(raw.replace(",", ""))
                    corrected = round(val / factor)
                    return f"{corrected:,}" if corrected >= 1000 else str(corrected)
                except ValueError:
                    return raw

            # Replace amounts in the amount column (3rd column)
            fixed_lines = []
            for line in table_text.split("\n"):
                if line.startswith("|") and not line.startswith("|--") and not line.startswith("| --"):
                    parts = line.split("|")
                    if len(parts) >= 4:
                        # parts[0]='', parts[1]=col1, parts[2]=col2, parts[3]=amount, parts[4]=pct
                        amount_cell = parts[3]
                        parts[3] = re.sub(r"[\d,]+", _rescale, amount_cell, count=1)
                        line = "|".join(parts)
                fixed_lines.append(line)
            return "\n".join(fixed_lines)

    return table_text


def extract_order_backlog_timeseries(
    backlog_history: list[tuple[str, str, str]],
    api_key: str,
    model: str = "",
    base_url: str = "",
    currency: str = "KRW",
) -> str:
    """Extract structured order backlog data from multiple periodic reports
    and build a time-series markdown table.

    Args:
        backlog_history: List of (ref_date_YYYYMMDD, report_name, raw_text).
        api_key: LLM API key.
        model: LLM model name.
        base_url: LLM base URL.
        currency: Currency code for display unit.

    Returns:
        Markdown table string, or empty string on failure.
    """
    import json as _json

    if not backlog_history or not api_key:
        return ""

    if not model:
        model = "gpt-4.1-mini"

    display_unit = "억원" if currency == "KRW" else f"백만 {currency}"
    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )

    # Step 1: Extract structured backlog data from each report
    snapshots: list[dict] = []  # [{period, segments: {name: amount}}]

    for ref_date, report_name, raw_text in backlog_history:
        # Convert YYYYMMDD to period label (e.g. "2025.Q3")
        period_label = _ref_date_to_period(ref_date)
        if not period_label:
            continue

        # Extract the order backlog subsection ("다. 수주" or similar)
        # to avoid truncating it away when the full section is long.
        text = _extract_backlog_subsection(raw_text)

        # Detect unit from the subsection itself (not full text, which may
        # have a different unit for the revenue section)
        source_unit = _detect_revenue_unit(text)
        conv_rule = _build_conv_rule(source_unit, display_unit)

        prompt = (
            f"다음은 정기보고서의 수주 현황 섹션입니다.\n\n"
            f"{text}\n\n"
            f"위 내용에서 **수주잔고** (기말잔고/기말잔량) 금액 데이터만 추출해주세요.\n\n"
            f"### 규칙\n"
            f"- 금액만 추출 (수량은 무시)\n"
            f"- 사업부문/품목별로 구분하여 추출\n"
            f"- {conv_rule}\n"
            f"- 합계/총계 행은 제외 (개별 사업부문만)\n"
            f"- 수주잔고 데이터가 없으면 빈 JSON 반환\n\n"
            f"### 응답 형식 (JSON만, 다른 텍스트 없이)\n"
            f'{{"segments": {{"사업부문A": 금액, "사업부문B": 금액}}}}\n\n'
            f"예시: {json_example(display_unit)}"
        )

        get_llm_limiter().wait()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            content = (response.choices[0].message.content or "").strip()
            # Extract JSON from response (may have markdown fences)
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            data = _json.loads(content)
            segments = data.get("segments", {})
            if segments:
                # Ensure all values are numeric
                clean_segments = {}
                for k, v in segments.items():
                    try:
                        clean_segments[k] = round(float(v))
                    except (ValueError, TypeError):
                        continue
                if clean_segments:
                    snapshots.append({"period": period_label, "segments": clean_segments})
                    logger.info("Backlog extracted: %s -> %s", period_label, clean_segments)
        except Exception as e:
            logger.warning("Backlog extraction failed for %s: %s", period_label, e)
            continue

    if not snapshots:
        return ""

    # Step 2: Build the time-series table (fetch 3 years for YoY, display 2 years)
    return _build_backlog_table(snapshots, display_unit, display_quarters=8)


def _extract_backlog_subsection(text: str) -> str:
    """Extract the order backlog subsection from '매출 및 수주상황' text.

    Looks for '다. 수주' or '수주상황' subsection header and returns
    the text from that point onward (up to 4000 chars).
    Falls back to the last 4000 chars if no header found.
    """
    # Primary patterns — section headers that include unit declaration
    for pattern in [
        r"다\.\s*수주",          # "다. 수주상황" or "다. 수주 상황"
        r"다\)\s*수주",          # "다) 수주상황"
        r"[Cc]\.\s*수주",       # "C. 수주상황"
    ]:
        m = re.search(pattern, text)
        if m:
            return text[m.start():][:4000]

    # Fallback patterns — table column headers like "수주잔고".
    # Include 500 chars of preceding context to capture the unit declaration
    # (e.g., "(단위 : 억원)") which typically appears before the table.
    for pattern in [r"수주잔[고량]"]:
        m = re.search(pattern, text)
        if m:
            start = max(0, m.start() - 500)
            return text[start:][:4500]

    # Last resort: use the last 4000 chars
    if len(text) > 4000:
        return text[-4000:]
    return text


def json_example(unit: str) -> str:
    return f'{{"segments": {{"반도체장비": 150, "디스플레이": 80}}}} (단위: {unit})'


def _ref_date_to_period(ref_date: str) -> str:
    """Convert YYYYMMDD to period label like '2025.Q3' or '2025'."""
    if len(ref_date) != 8:
        return ""
    year = ref_date[:4]
    month = int(ref_date[4:6])
    quarter_map = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}
    # Find nearest quarter end
    for end_month, q_label in quarter_map.items():
        if month <= end_month:
            return f"{year}.{q_label}"
    return f"{year}.Q4"


def _normalize_segment_name(name: str) -> str:
    """Normalize segment name for fuzzy matching.

    Strips all whitespace so '방사선 관리 용역' == '방사선관리 용역'.
    """
    return re.sub(r"\s+", "", name)


def _unify_segment_names(snapshots: list[dict]) -> list[dict]:
    """Merge segment names that differ only by whitespace.

    Uses the first-seen form as the canonical name.
    Returns new snapshots with unified segment names.
    """
    # Build canonical name mapping: normalized -> first-seen display name
    canonical: dict[str, str] = {}
    for snap in snapshots:
        for name in snap["segments"]:
            norm = _normalize_segment_name(name)
            if norm not in canonical:
                canonical[norm] = name

    # Rewrite each snapshot's segments using canonical names
    unified: list[dict] = []
    for snap in snapshots:
        new_segments: dict[str, int] = {}
        for name, val in snap["segments"].items():
            norm = _normalize_segment_name(name)
            canon_name = canonical[norm]
            # Sum if same normalized name appears multiple times
            new_segments[canon_name] = new_segments.get(canon_name, 0) + val
        unified.append({"period": snap["period"], "segments": new_segments})
    return unified


def _build_backlog_table(
    snapshots: list[dict],
    display_unit: str,
    display_quarters: int = 8,
) -> str:
    """Build a markdown time-series table from backlog snapshots.

    Each snapshot: {period: "2025.Q3", segments: {"부문A": 100, "부문B": 200}}
    All snapshots are used for YoY calculation, but only the most recent
    `display_quarters` are shown in the table.
    """
    if not snapshots:
        return ""

    # Unify segment names (fuzzy match: strip whitespace)
    snapshots = _unify_segment_names(snapshots)

    # Sort snapshots by period descending
    snapshots.sort(key=lambda s: s["period"], reverse=True)

    # Build period -> total map for YoY lookup (ALL snapshots)
    period_totals: dict[str, int] = {}
    period_segments: dict[str, dict[str, int]] = {}
    for snap in snapshots:
        total = sum(snap["segments"].values())
        period_totals[snap["period"]] = total
        period_segments[snap["period"]] = snap["segments"]

    # Only display the most recent N quarters
    display_snapshots = snapshots[:display_quarters]

    # Build rows: 분기 | 합계 (YoY%) | 비고 (segment breakdown)
    headers = ["분기", f"합계({display_unit})", "비고"]
    separator = ["---", "---", "---"]

    rows: list[list[str]] = []
    for snap in display_snapshots:
        period = snap["period"]
        total = period_totals[period]

        # YoY for total
        yoy_period = _get_yoy_period(period)
        yoy_str = ""
        if yoy_period and yoy_period in period_totals:
            prev_total = period_totals[yoy_period]
            if prev_total > 0:
                pct = (total - prev_total) / prev_total * 100
                sign = "+" if pct >= 0 else ""
                yoy_str = f" ({sign}{pct:.0f}%)"

        total_cell = f"{total:,}{yoy_str}"

        # Notes: segment breakdown
        segments = snap["segments"]
        notes_parts = [f"{name} {val:,}" for name, val in segments.items()]
        notes_cell = " / ".join(notes_parts)

        rows.append([f"**{period}**", total_cell, notes_cell])

    # Render markdown
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _calc_yoy(
    period: str, seg_name: str, current_val: int,
    period_data: dict[str, dict[str, int]],
) -> str | None:
    """Calculate YoY % for a segment. Returns formatted string or None."""
    yoy_period = _get_yoy_period(period)
    if not yoy_period or yoy_period not in period_data:
        return None
    prev_val = period_data[yoy_period].get(seg_name)
    if prev_val is None or prev_val == 0:
        return None
    pct = (current_val - prev_val) / prev_val * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _calc_yoy_total(
    period: str, all_segments: list[str],
    period_data: dict[str, dict[str, int]],
) -> str | None:
    """Calculate YoY % for total across all segments."""
    yoy_period = _get_yoy_period(period)
    if not yoy_period or yoy_period not in period_data:
        return None
    current_total = sum(period_data[period].get(s, 0) for s in all_segments)
    prev_total = sum(period_data[yoy_period].get(s, 0) for s in all_segments)
    if prev_total == 0:
        return None
    pct = (current_total - prev_total) / prev_total * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _get_yoy_period(period: str) -> str:
    """Get same quarter of previous year. e.g. '2025.Q3' -> '2024.Q3'."""
    if ".Q" in period:
        parts = period.split(".Q")
        try:
            year = int(parts[0])
            return f"{year - 1}.Q{parts[1]}"
        except (ValueError, IndexError):
            return ""
    return ""


def _system_prompt(display_unit: str = "억원") -> str:
    """Generate system prompt with the correct display unit."""
    return f"""\
당신은 한국 상장기업의 사업보고서를 분석하는 금융 애널리스트입니다.
사업보고서의 특정 섹션 원문을 받아, 투자 분석에 필요한 핵심 내용만 간결하게 요약합니다.
- 불필요한 수식어 제거, 핵심 팩트 중심
- 테이블 데이터는 마크다운 테이블로 정리
- 금액은 {display_unit} 단위로 통일하여 표기
- 주요 고객사/매입처 이름이 비공개(A사 등)면 그대로 유지
- 모든 문장은 명사/명사구로 자연스럽게 종결 (예: 수주 확보, 실적 개선, 매출 성장 등). '~임', '~함' 같은 어색한 접미사 금지. 동사형/형용사형 어미(~한다, ~이다, ~하고 있다)도 사용 금지
"""


@dataclass
class BusinessSummary:
    """Summarized business sections for report."""

    business_model: str = ""
    major_suppliers: str = ""
    major_customers: str = ""
    revenue_breakdown: str = ""
    order_backlog: str = ""
    report_source: str = ""  # e.g. "사업보고서 (2024.12)"


def summarize_business_sections(
    사업개요: str,
    주요제품: str,
    원재료: str,
    매출수주: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
    currency: str = "KRW",
    known_revenue_won: int | None = None,
) -> BusinessSummary:
    """Summarize business sections using OpenAI.

    Args:
        사업개요: Raw text from "1. 사업의 개요"
        주요제품: Raw text from "2. 주요 제품 및 서비스"
        원재료: Raw text from "3. 원재료 및 생산설비"
        매출수주: Raw text from "4. 매출 및 수주상황"
        api_key: OpenAI API key
        model: Model to use (default: gpt-4.1-mini)

    Returns:
        BusinessSummary with summarized content.
    """
    if not model:
        model = "gpt-4.1-mini"
    client = OpenAI(
        api_key=api_key, max_retries=5,
        **({"base_url": base_url} if base_url else {}),
    )
    summary = BusinessSummary()

    # Determine display unit for amounts
    if currency == "KRW":
        _display_unit = "억원"
    else:
        _display_unit = f"백만 {currency}"
    _sys = _system_prompt(_display_unit)

    # 1. Business model (사업개요 + 주요제품)
    if 사업개요 or 주요제품:
        prompt = f"""다음은 사업보고서의 '사업의 개요'와 '주요 제품 및 서비스' 섹션입니다.

[사업의 개요]
{사업개요[:3000]}

[주요 제품 및 서비스]
{주요제품[:3000]}

위 내용을 바탕으로 핵심 사업, 주요 제품, 경쟁력을 `- ` 불릿 리스트로 3줄 이내로 요약해주세요.
형식: - (첫째 줄) - (둘째 줄) - (셋째 줄)
각 줄은 한 문장으로, 명사/명사구로 자연스럽게 종결 (예: 수주 확보, 매출 성장). '~임', '~함' 접미사 금지."""
        summary.business_model = _call_llm(client, model, prompt, _sys)
        time.sleep(2)

    if currency == "KRW":
        _supplier_conv = "금액은 억원 단위로 변환하여 표기하세요 (천원이면 100,000으로, 백만원이면 100으로 나누어 반올림)."
    else:
        _supplier_conv = f"금액은 백만 {currency} 단위로 변환하여 표기하세요 (천{currency}이면 1,000으로 나누어 반올림)."

    # 2. Major suppliers (원재료)
    if 원재료:
        prompt = f"""다음은 사업보고서의 '원재료 및 생산설비' 섹션입니다.

{원재료[:3000]}

위 내용에서 핵심 원재료와 주요 공급업체를 자회사/사업부문별로 `- ` 불릿 리스트로 작성해주세요.
같은 사업부문의 품목은 한 줄에 쉼표로 나열하고, '기타' 항목은 생략하되 마지막에 '등'을 붙여주세요.
{_supplier_conv}
형식 예시:
- 자회사A: 원재료1 금액{_display_unit}(업체명, 비중%), 원재료2 금액{_display_unit}(업체명, 비중%) 등
- 자회사B: 원재료1 금액{_display_unit}(업체명, 비중%) 등"""
        summary.major_suppliers = _call_llm(client, model, prompt, _sys)
        time.sleep(2)

    # 3. Major customers + revenue breakdown + order backlog (매출수주)
    if 매출수주:
        # Strip trailing summary/aggregate section to prevent LLM from
        # using pre-aggregated totals instead of per-type breakdown.
        매출수주 = _strip_revenue_summary(매출수주)
        # Detect unit for explicit conversion guidance
        unit = _detect_revenue_unit(매출수주)
        conv_rule = _build_conv_rule(unit, _display_unit)

        prompt = f"""다음은 사업보고서의 '매출 및 수주상황' 섹션입니다.

{매출수주[:4000]}

위 내용을 바탕으로 다음 세 가지를 각각 요약해주세요.
반드시 아래 형식을 따라주세요:

[주요 매출처]
자회사/사업부문별로 구분하여 `- ` 불릿 리스트로 작성. 각 항목은 한 줄로 핵심만 간결하게 요약.
형식 예시:
- 사업부문A: 주요 고객과 매출 비중
- 사업부문B: 주요 고객과 매출 비중

[부문별 매출]
가장 최근 기수(期)의 데이터만 사용하여 아래 형식의 마크다운 테이블로 작성하세요.
- 여러 기간의 데이터가 있으면 가장 최근 기수만 사용
- {conv_rule}
- 비중(%): 원문의 비중/% 컬럼은 무시할 것. 반드시 변환된 {_display_unit} 매출액만 사용하여 재계산:
  비중(%) = 해당 품목 {_display_unit} 매출액 ÷ (모든 개별 품목 {_display_unit} 매출액의 합산) × 100, 소수점 첫째자리
  ※ 주의: 합계 행의 비중(100%)이나 원문의 비중 수치로 나누지 말 것
- 동일 품목이 내수/수출로 분리된 경우 합산하여 하나의 행으로 표시 (예: 철탑(내수)+철탑(수출) → 철탑)
- 품목명에서 (내수), (수출) 등 판매경로 표기는 제거
- 매출 유형 구분: 제품매출은 해당 사업부문으로, 상품매출은 '상품', 용역/공사매출은 '기타'로 구분
- 합계 행 포함

| 사업부문 | 품목 | 매출액({_display_unit}) | 비중(%) |
|---|---|---|---|

[수주잔고]
(수주 현황이 있으면 요약, 없으면 "해당사항 없음")
- {conv_rule}"""

        result = _call_llm(client, model, prompt, _sys)
        # Parse the structured response
        sections = _parse_structured_response(result)
        summary.major_customers = sections.get("주요매출처", "")
        raw_breakdown = sections.get("부문별매출", "")
        summary.revenue_breakdown = _fix_revenue_breakdown_scale(
            raw_breakdown, known_revenue_won, _display_unit,
        )
        summary.order_backlog = sections.get("수주잔고", "")

    return summary


def _call_llm(client: OpenAI, model: str, prompt: str, sys_prompt: str = "") -> str:
    """Make an OpenAI API call with error handling and rate limit retry."""
    _sys = sys_prompt or _system_prompt()
    get_llm_limiter().wait()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _sys},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except RateLimitError as e:
        logger.warning("LLM rate limited (model=%s): %s — retrying with backoff", model, e)
        for attempt in range(3):
            wait_time = 2 ** (attempt + 1)
            time.sleep(wait_time)
            get_llm_limiter().wait()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _sys},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except RateLimitError:
                continue
            except Exception:
                break
        logger.warning("LLM rate limit retries exhausted (model=%s)", model)
        return ""
    except Exception as e:
        logger.warning("OpenAI API call failed (model=%s): %s", model, type(e).__name__)
        return ""


def generate_report_tags(
    company_name: str,
    business_model: str,
    revenue_breakdown: str,
    investment_ideas: str,
    conclusion: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> list[str]:
    """Generate 3-5 Obsidian tags based on report content using LLM.

    Tags focus on the company's industry, themes, and key products.

    Returns:
        List of 3-5 Korean tag strings (without '#' prefix).
    """
    if not api_key:
        return []

    if not model:
        model = "gpt-4.1-mini"

    # Build context from available sections
    parts: list[str] = []
    if business_model:
        parts.append(f"[사업 모델]\n{business_model[:1500]}")
    if revenue_breakdown:
        parts.append(f"[부문별 매출]\n{revenue_breakdown[:1000]}")
    if investment_ideas:
        parts.append(f"[투자 아이디어]\n{investment_ideas[:1000]}")
    if conclusion:
        parts.append(f"[결론]\n{conclusion[:500]}")

    if not parts:
        return []

    context = "\n\n".join(parts)

    prompt = f"""다음은 '{company_name}'의 투자 분석 리포트 내용입니다.

{context}

위 내용을 바탕으로, 이 기업이 속한 **산업**, **테마**, **핵심 제품/서비스** 중심으로 옵시디언 태그를 3~5개 생성해주세요.

규칙:
- 한국어로 작성
- '#' 기호 없이 태그명만 작성
- 한 줄에 하나씩, 다른 텍스트 없이 태그만 출력
- 구체적이고 검색에 유용한 태그 (예: 우주산업, 위성제조, K방산수출, 2차전지, AI반도체)
- 너무 일반적인 태그 금지 (예: 주식, 투자, 한국기업)

출력 예시:
우주산업
위성제조
위성영상
K방산수출"""

    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )
    try:
        get_llm_limiter().wait()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        content = (response.choices[0].message.content or "").strip()
    except RateLimitError as e:
        logger.warning("Tag generation rate limited (model=%s): %s", model, e)
        for attempt in range(3):
            time.sleep(2 ** (attempt + 1))
            get_llm_limiter().wait()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200,
                )
                content = (response.choices[0].message.content or "").strip()
                break
            except RateLimitError:
                continue
            except Exception:
                return []
        else:
            return []
    except Exception as e:
        logger.warning("Tag generation failed (model=%s): %s", model, type(e).__name__)
        return []

    # Parse: one tag per line, strip leading '#', '-', whitespace
    tags: list[str] = []
    for line in content.splitlines():
        tag = re.sub(r"^[#\-\s]+", "", line).strip()
        if tag and len(tag) <= 20:
            tags.append(tag)
    # Deduplicate while preserving order
    tags = list(dict.fromkeys(tags))[:5]
    if tags and len(tags) < 3:
        logger.warning(
            "Tag generation returned only %d tags for %s (model=%s)",
            len(tags), company_name, model,
        )
    return tags


_KNOWN_SECTION_HEADERS = {"주요매출처", "부문별매출", "수주잔고"}


def _parse_structured_response(text: str) -> dict[str, str]:
    """Parse a structured LLM response with [section] headers.

    Only recognizes known section headers to avoid false splits on inline
    brackets like [A사], [주1], [단위: 억원] that may appear in content.
    """
    import re

    # Find all [header] patterns at line start
    header_pattern = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)

    # Collect positions of known headers only
    known_positions: list[tuple[int, int, str]] = []
    for match in header_pattern.finditer(text):
        header = match.group(1).replace(" ", "")
        if header in _KNOWN_SECTION_HEADERS:
            known_positions.append((match.start(), match.end(), header))

    # Extract content between known headers
    result: dict[str, str] = {}
    for i, (_, end, header) in enumerate(known_positions):
        next_start = known_positions[i + 1][0] if i + 1 < len(known_positions) else len(text)
        content = text[end:next_start].strip()
        if content:
            result[header] = content

    return result
