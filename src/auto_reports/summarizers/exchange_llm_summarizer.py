"""LLM-based extraction for KRX exchange disclosure HTML text.

Fallback when the deterministic parser in parsers/exchange_disclosure.py
returns category="unknown" or incomplete data.

Supports:
- 매출액또는손익구조 변동 (performance)
- 단일판매ㆍ공급계약 (contract / backlog)
- 전환청구권행사 / 전환가액의조정 (overhang: exercise or price_adj)
"""

from __future__ import annotations

import json
import logging
import re

import time

from openai import OpenAI, RateLimitError

from auto_reports.fetchers.rate_limiter import get_llm_limiter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.4-mini"
_TEXT_MAX_CHARS = 6000


def extract_exchange_performance(
    html_text: str,
    api_key: str,
    model: str = "",
) -> dict | None:
    """Extract 매출액또는손익구조 변동 data from exchange disclosure text.

    Args:
        html_text: Raw text extracted from disclosure HTML via soup.get_text().
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        Dict matching Performance model structure, or None if extraction fails.
    """
    if not html_text or not api_key:
        return None
    if not model:
        model = _DEFAULT_MODEL

    truncated = html_text[:_TEXT_MAX_CHARS]

    prompt = (
        "다음은 KRX 공시의 '매출액또는손익구조30%(15%)이상변동' 관련 공시 텍스트입니다.\n\n"
        f"{truncated}\n\n"
        "위 내용에서 아래 JSON 구조에 맞게 데이터를 추출하세요.\n"
        "반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.\n\n"
        "응답 JSON 구조:\n"
        "{\n"
        '  "statement_type": "연결" 또는 "별도",\n'
        '  "period": {\n'
        '    "- 시작일": "YYYY년 MM월 DD일",\n'
        '    "- 종료일": "YYYY년 MM월 DD일"\n'
        "  },\n"
        '  "income_changes": {\n'
        '    "- 매출액": {"당해사업연도": "숫자문자열", "직전사업연도": "숫자문자열", "증감비율(%)": "숫자문자열"},\n'
        '    "- 영업이익": {"당해사업연도": "숫자문자열", "직전사업연도": "숫자문자열", "증감비율(%)": "숫자문자열"},\n'
        '    "- 당기순이익": {"당해사업연도": "숫자문자열", "직전사업연도": "숫자문자열", "증감비율(%)": "숫자문자열"}\n'
        "  }\n"
        "}\n\n"
        "데이터를 찾을 수 없으면 null을 응답하세요."
    )

    raw = _call_llm(api_key, model, prompt, max_tokens=600)
    if not raw:
        return None

    result = _parse_json_response(raw)
    if result is None:
        return None

    # Validate top-level keys are present
    if not isinstance(result, dict):
        return None
    if "income_changes" not in result:
        return None

    return result


def extract_exchange_contract(
    html_text: str,
    api_key: str,
    model: str = "",
) -> dict | None:
    """Extract 단일판매ㆍ공급계약 data from exchange disclosure text.

    Args:
        html_text: Raw text extracted from disclosure HTML via soup.get_text().
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        Dict matching exchange_disclosure backlog result structure, or None.
    """
    if not html_text or not api_key:
        return None
    if not model:
        model = _DEFAULT_MODEL

    truncated = html_text[:_TEXT_MAX_CHARS]

    prompt = (
        "다음은 KRX 공시의 '단일판매ㆍ공급계약체결' 또는 '단일판매ㆍ공급계약해지' 관련 공시 텍스트입니다.\n\n"
        f"{truncated}\n\n"
        "위 내용에서 아래 JSON 구조에 맞게 데이터를 추출하세요.\n"
        "반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.\n\n"
        "응답 JSON 구조:\n"
        "{\n"
        '  "type": "단일판매ㆍ공급계약체결" 또는 "단일판매ㆍ공급계약해지",\n'
        '  "description": "계약 내용 또는 구분 (문자열)",\n'
        '  "detail": "계약명 또는 상세내용 (문자열)",\n'
        '  "contract_amount": 계약금액(정수) 또는 null,\n'
        '  "cancel_amount": 해지금액(정수) 또는 null,\n'
        '  "amount_unit": "원" 또는 "천원" 또는 "백만원" 또는 "억원" (공시 본문에 명시된 금액 단위),\n'
        '  "counterparty": "계약상대방 (문자열)" 또는 null,\n'
        '  "revenue_ratio_pct": 매출액대비비율(실수) 또는 null,\n'
        '  "contract_date": "YYYY-MM-DD 또는 YYYY.MM.DD" 또는 null,\n'
        '  "cancel_date": "YYYY-MM-DD 또는 YYYY.MM.DD" 또는 null,\n'
        '  "contract_period_start": "계약기간 시작일 (YYYY-MM-DD)" 또는 null,\n'
        '  "contract_period_end": "계약기간 종료일 (YYYY-MM-DD)" 또는 null\n'
        "}\n\n"
        "데이터를 찾을 수 없으면 null을 응답하세요."
    )

    raw = _call_llm(api_key, model, prompt, max_tokens=500)
    if not raw:
        return None

    result = _parse_json_response(raw)
    if result is None:
        return None

    if not isinstance(result, dict):
        return None
    if "type" not in result:
        return None

    return result


def extract_exchange_overhang(
    html_text: str,
    api_key: str,
    model: str = "",
) -> tuple[str, dict] | None:
    """Extract overhang data (exercise or price_adj) from exchange disclosure text.

    Args:
        html_text: Raw text extracted from disclosure HTML via soup.get_text().
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        Tuple of (category, data) where category is "exercise" or "price_adj",
        or None if extraction fails.
    """
    if not html_text or not api_key:
        return None
    if not model:
        model = _DEFAULT_MODEL

    truncated = html_text[:_TEXT_MAX_CHARS]

    prompt = (
        "다음은 KRX 공시의 전환사채/신주인수권 관련 공시 텍스트입니다 "
        "(전환청구권행사 또는 전환가액의조정 등).\n\n"
        f"{truncated}\n\n"
        "위 내용을 분석하여 공시 유형을 파악하고 아래 JSON 구조 중 하나로 응답하세요.\n"
        "반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.\n\n"
        "전환청구권행사(exercise)인 경우:\n"
        "{\n"
        '  "category": "exercise",\n'
        '  "data": {\n'
        '    "type": "공시 유형명 (예: 전환청구권행사)",\n'
        '    "cumulative_shares": 누적전환주식수(정수) 또는 null,\n'
        '    "total_shares": 발행주식총수(정수) 또는 null,\n'
        '    "ratio_pct": 희석비율퍼센트(실수) 또는 null,\n'
        '    "daily_claims": [\n'
        '      {"date": "YYYY-MM-DD", "shares": 주식수(정수)}\n'
        "    ],\n"
        '    "cb_balance": [\n'
        '      {"series": 회차(정수), "remaining": 잔액(정수), "conversion_price": 전환가액(정수)}\n'
        "    ]\n"
        "  }\n"
        "}\n\n"
        "전환가액조정(price_adj)인 경우:\n"
        "{\n"
        '  "category": "price_adj",\n'
        '  "data": {\n'
        '    "type": "공시 유형명 (예: 전환가액의조정)",\n'
        '    "adjustments": [\n'
        '      {"series": 회차(정수), "price_before": 조정전가액(정수), "price_after": 조정후가액(정수)}\n'
        "    ],\n"
        '    "share_changes": [\n'
        '      {"series": 회차(정수), "shares_before": 조정전주식수(정수), "shares_after": 조정후주식수(정수)}\n'
        "    ],\n"
        '    "reason": "조정사유 (문자열)" 또는 null,\n'
        '    "effective_date": "YYYY-MM-DD" 또는 null\n'
        "  }\n"
        "}\n\n"
        "데이터를 찾을 수 없으면 null을 응답하세요."
    )

    raw = _call_llm(api_key, model, prompt, max_tokens=800)
    if not raw:
        return None

    result = _parse_json_response(raw)
    if result is None:
        return None

    if not isinstance(result, dict):
        return None

    category = result.get("category")
    data = result.get("data")

    if category not in ("exercise", "price_adj") or not isinstance(data, dict):
        logger.warning("extract_exchange_overhang: unexpected structure: %s", result)
        return None

    return (category, data)


def extract_exchange_disclosure_unified(
    html_text: str,
    title: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> tuple[str, dict] | None:
    """Classify and extract KRX exchange disclosure data in a single LLM call.

    Handles all disclosure types in one pass:
    - exercise: 전환/신주인수권/교환 청구권행사
    - price_adj: 전환가액/행사가액/교환가액의 조정
    - backlog: 단일판매ㆍ공급계약 체결/해지
    - sales: 매출액또는손익구조 변동

    Args:
        html_text: Raw text extracted from disclosure HTML via soup.get_text().
        title: Disclosure title from the JSON entry.
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        Tuple of (category, data) or None if extraction fails.
    """
    if not html_text or not api_key:
        return None
    if not model:
        model = _DEFAULT_MODEL

    truncated = html_text[:_TEXT_MAX_CHARS]

    prompt = (
        f'다음은 KRX 거래소 공시 텍스트입니다. 공시 제목: "{title}"\n\n'
        f"{truncated}\n\n"
        "위 공시를 분석하여 유형을 판별하고, 해당 유형에 맞는 JSON을 추출하세요.\n"
        "반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.\n\n"
        "공시 유형별 JSON 구조:\n\n"
        "1. 전환/신주인수권/교환 청구권행사:\n"
        '{"category": "exercise", "data": {"type": "공시유형명", '
        '"cumulative_shares": 누적행사주식수(정수), '
        '"total_shares": 발행주식총수(정수), '
        '"ratio_pct": 비율(실수), '
        '"daily_claims": [{"date": "YYYY-MM-DD", "series": 회차(정수_or_null), '
        '"bond_type": "종류", "amount": 금액(정수), '
        '"conversion_price": 전환또는행사가액(정수), '
        '"shares": 주식수(정수), '
        '"listing_date": "YYYY-MM-DD"(or_null)}], '
        '"cb_balance": [{"series": 회차(정수_or_null), '
        '"face_value": 권면총액(정수), '
        '"remaining": 미전환잔액(정수_or_null), '
        '"conversion_price": 전환가액(정수), '
        '"convertible_shares": 전환가능주식수(정수)}]}}\n\n'
        "2. 전환가액/행사가액/교환가액 조정:\n"
        '{"category": "price_adj", "data": {"type": "공시유형명", '
        '"adjustments": [{"series": 회차(정수_or_null), '
        '"listed": "상장여부"(or_null), '
        '"price_before": 조정전가액(정수), '
        '"price_after": 조정후가액(정수)}], '
        '"share_changes": [{"series": 회차(정수_or_null), '
        '"unconverted": 미전환잔액(정수_or_null), '
        '"currency": "KRW"(or_null), '
        '"shares_before": 조정전주식수(정수), '
        '"shares_after": 조정후주식수(정수)}], '
        '"reason": "조정사유"(or_null), '
        '"effective_date": "YYYY-MM-DD"(or_null)}}\n\n'
        "3. 단일판매ㆍ공급계약체결/해지:\n"
        '{"category": "backlog", "data": {"type": '
        '"단일판매ㆍ공급계약체결"(or"해지"), '
        '"description": "구분", "detail": "상세내용", '
        '"contract_amount": 계약금액(정수,or_null), '
        '"cancel_amount": 해지금액(정수,or_null), '
        '"amount_unit": "원"|"천원"|"백만원"|"억원"(공시본문의금액단위), '
        '"counterparty": "상대방"(or_null), '
        '"revenue_ratio_pct": 매출액대비비율(실수,or_null), '
        '"contract_date": "YYYY-MM-DD"(or_null), '
        '"cancel_date": "YYYY-MM-DD"(or_null), '
        '"contract_period_start": "계약기간시작일YYYY-MM-DD"(or_null), '
        '"contract_period_end": "계약기간종료일YYYY-MM-DD"(or_null)}}\n\n'
        "4. 매출액또는손익구조 변동:\n"
        '{"category": "sales", "data": {"statement_type": "연결"(or"별도"), '
        '"period": {"- 시작일": "YYYY년 MM월 DD일", '
        '"- 종료일": "YYYY년 MM월 DD일"}, '
        '"income_changes": {'
        '"- 매출액": {"당해사업연도": "숫자", '
        '"직전사업연도": "숫자", "증감비율(%)": "숫자"}, '
        '"- 영업이익": {"당해사업연도": "숫자", '
        '"직전사업연도": "숫자", "증감비율(%)": "숫자"}, '
        '"- 당기순이익": {"당해사업연도": "숫자", '
        '"직전사업연도": "숫자", "증감비율(%)": "숫자"}}}}\n\n'
        "위 유형에 해당하지 않으면 null을 응답하세요. "
        "모든 숫자는 쉼표 없이 정수로 표기하세요."
    )

    raw = _call_llm(api_key, model, prompt, max_tokens=1200, base_url=base_url)
    if not raw:
        return None

    result = _parse_json_response(raw)
    if result is None:
        return None

    category = result.get("category")
    data = result.get("data")

    if not category or not isinstance(data, dict):
        logger.warning("extract_exchange_unified: unexpected structure: %s", result)
        return None

    if category not in ("exercise", "price_adj", "backlog", "sales"):
        logger.warning("extract_exchange_unified: unknown category: %s", category)
        return None

    return (category, data)


def _call_llm(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 400,
    base_url: str = "",
) -> str:
    """Make an OpenAI API call with rate limiting and RateLimitError retry."""
    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )
    get_llm_limiter().wait()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except RateLimitError as e:
        logger.warning("LLM rate limited (model=%s): %s — retrying", model, e)
        for attempt in range(3):
            time.sleep(2 ** (attempt + 1))
            get_llm_limiter().wait()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except RateLimitError:
                continue
            except Exception:
                break
        logger.warning("LLM rate limit retries exhausted (model=%s)", model)
        return ""
    except Exception as e:
        logger.warning("LLM extraction failed (model=%s): %s", model, e)
        return ""


def _parse_json_response(text: str) -> dict | None:
    """Parse a JSON response from the LLM, handling markdown code fences.

    Args:
        text: Raw LLM response text, possibly wrapped in ```json ... ``` fences.

    Returns:
        Parsed dict, or None on parse failure.
    """
    if not text:
        return None

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Handle null response
    if text.lower() == "null":
        return None

    try:
        parsed = json.loads(text)
        if parsed is None:
            return None
        if not isinstance(parsed, dict):
            logger.warning("_parse_json_response: expected dict, got %s", type(parsed))
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("_parse_json_response: JSON decode error: %s | text=%r", e, text[:200])
        return None
