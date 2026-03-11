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

    Looks for patterns like '(단위: 천원)', '(단위 : 백만원)' etc.
    Returns '천원', '백만원', '억원', or '백만원' as default.
    """
    m = re.search(r"단위\s*[:：]\s*(천원|백만원|억원)", text)
    if m:
        return m.group(1)
    return "백만원"


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


_SYSTEM_PROMPT = """\
당신은 한국 상장기업의 사업보고서를 분석하는 금융 애널리스트입니다.
사업보고서의 특정 섹션 원문을 받아, 투자 분석에 필요한 핵심 내용만 간결하게 요약합니다.
- 불필요한 수식어 제거, 핵심 팩트 중심
- 테이블 데이터는 마크다운 테이블로 정리
- 금액은 억원 단위로 통일하여 표기 (천원이면 100,000으로, 백만원이면 100으로 나누어 반올림)
- 주요 고객사/매입처 이름이 비공개(A사 등)면 그대로 유지
- 모든 문장은 반드시 명사형 어미로 종결 (예: ~임, ~함, ~중, ~있음, ~됨). 동사형/형용사형 어미(~한다, ~이다, ~하고 있다) 사용 금지
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

    # 1. Business model (사업개요 + 주요제품)
    if 사업개요 or 주요제품:
        prompt = f"""다음은 사업보고서의 '사업의 개요'와 '주요 제품 및 서비스' 섹션입니다.

[사업의 개요]
{사업개요[:3000]}

[주요 제품 및 서비스]
{주요제품[:3000]}

위 내용을 바탕으로 핵심 사업, 주요 제품, 경쟁력을 `- ` 불릿 리스트로 3줄 이내로 요약해주세요.
형식: - (첫째 줄) - (둘째 줄) - (셋째 줄)
각 줄은 한 문장으로, 명사형 어미로 종결."""
        summary.business_model = _call_llm(client, model, prompt)
        time.sleep(2)

    # 2. Major suppliers (원재료)
    if 원재료:
        prompt = f"""다음은 사업보고서의 '원재료 및 생산설비' 섹션입니다.

{원재료[:3000]}

위 내용에서 핵심 원재료와 주요 공급업체를 자회사/사업부문별로 `- ` 불릿 리스트로 작성해주세요.
같은 사업부문의 품목은 한 줄에 쉼표로 나열하고, '기타' 항목은 생략하되 마지막에 '등'을 붙여주세요.
금액은 억원 단위로 변환하여 표기하세요 (천원이면 100,000으로, 백만원이면 100으로 나누어 반올림).
형식 예시:
- 자회사A: 원재료1 금액억원(업체명, 비중%), 원재료2 금액억원(업체명, 비중%) 등
- 자회사B: 원재료1 금액억원(업체명, 비중%) 등"""
        summary.major_suppliers = _call_llm(client, model, prompt)
        time.sleep(2)

    # 3. Major customers + revenue breakdown + order backlog (매출수주)
    if 매출수주:
        # Strip trailing summary/aggregate section to prevent LLM from
        # using pre-aggregated totals instead of per-type breakdown.
        매출수주 = _strip_revenue_summary(매출수주)
        # Detect unit for explicit conversion guidance
        unit = _detect_revenue_unit(매출수주)
        if unit == "천원":
            conv_rule = (
                "원문의 금액 단위: 천원\n"
                "  → 억원 변환법: 숫자 ÷ 100,000 후 반올림\n"
                "  → 예시: 1,192,400 (천원) ÷ 100,000 = 12 (억원)"
            )
        elif unit == "억원":
            conv_rule = "원문의 금액 단위: 억원 (변환 불필요, 그대로 사용)"
        else:  # 백만원
            conv_rule = (
                "원문의 금액 단위: 백만원\n"
                "  → 억원 변환법: 숫자 ÷ 100 후 반올림\n"
                "  → 예시: 1,192 (백만원) ÷ 100 = 12 (억원)"
            )

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
- 비중(%): 원문의 비중/% 컬럼은 무시할 것. 반드시 변환된 억원 매출액만 사용하여 재계산:
  비중(%) = 해당 품목 억원 매출액 ÷ (모든 개별 품목 억원 매출액의 합산) × 100, 소수점 첫째자리
  ※ 주의: 합계 행의 비중(100%)이나 원문의 비중 수치로 나누지 말 것
- 동일 품목이 내수/수출로 분리된 경우 합산하여 하나의 행으로 표시 (예: 철탑(내수)+철탑(수출) → 철탑)
- 품목명에서 (내수), (수출) 등 판매경로 표기는 제거
- 매출 유형 구분: 제품매출은 해당 사업부문으로, 상품매출은 '상품', 용역/공사매출은 '기타'로 구분
- 합계 행 포함

| 사업부문 | 품목 | 매출액(억원) | 비중(%) |
|---|---|---|---|

[수주잔고]
(수주 현황이 있으면 요약, 없으면 "해당사항 없음")
- {conv_rule}"""

        result = _call_llm(client, model, prompt)
        # Parse the structured response
        sections = _parse_structured_response(result)
        summary.major_customers = sections.get("주요매출처", "")
        summary.revenue_breakdown = sections.get("부문별매출", "")
        summary.order_backlog = sections.get("수주잔고", "")

    return summary


def _call_llm(client: OpenAI, model: str, prompt: str) -> str:
    """Make an OpenAI API call with error handling and rate limit retry."""
    get_llm_limiter().wait()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
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
                        {"role": "system", "content": _SYSTEM_PROMPT},
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
