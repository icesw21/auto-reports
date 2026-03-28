"""LLM-based extraction for supplementary report sections.

Extracts structured data from raw DART report section text for:
- 주주 구성 (shareholder composition)
- 배당 (dividend information)
- 종속 회사 (subsidiary companies)
"""

from __future__ import annotations

import logging

import time

from openai import OpenAI, RateLimitError

from auto_reports.fetchers.rate_limiter import get_llm_limiter

logger = logging.getLogger(__name__)


def extract_shareholder_info(
    section_text: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> str:
    """Extract largest shareholder info from 주주에 관한 사항 section text.

    Returns formatted string like: "최대주주 이종서 등 13.62%"
    Returns empty string if extraction fails.
    """
    if not section_text or not api_key:
        return ""
    if not model:
        model = "gpt-5.4-mini"

    prompt = (
        "다음은 DART 공시의 '주주에 관한 사항' 섹션입니다.\n\n"
        f"{section_text[:4000]}\n\n"
        "위 내용에서 최대주주 및 특수관계인의 주식 소유 현황을 요약해주세요.\n"
        "응답 규칙:\n"
        '- "최대주주 {이름} 등 {합계 지분율}%" 형식으로만 응답\n'
        "- 합계 지분율은 최대주주 + 특수관계인 전체의 합산 비율\n"
        "- 숫자는 소수점 둘째자리까지\n"
        '- 데이터가 없으면 "없음"이라고만 응답\n'
        "- 다른 텍스트 불필요"
    )

    return _call_llm(api_key, model, prompt, max_completion_tokens=100, base_url=base_url)


def extract_dividend_info(
    section_text: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> str:
    """Extract dividend info from 배당에 관한 사항 section text.

    Returns formatted string like:
    "가장 최근 주당 500원 배당 실시 (현금배당성향 20.5%, 현금배당총액 100억원)"
    Returns empty string if extraction fails.
    """
    if not section_text or not api_key:
        return ""
    if not model:
        model = "gpt-5.4-mini"

    prompt = (
        "다음은 DART 공시의 '배당에 관한 사항' 섹션입니다.\n\n"
        f"{section_text[:4000]}\n\n"
        "위 내용에서 가장 최근 사업연도의 보통주 기준 배당 정보를 추출해주세요.\n"
        "응답 규칙:\n"
        '- "가장 최근 주당 {금액}원 배당 실시 (현금배당성향 {비율}%, 현금배당총액 {금액}억원)" 형식으로만 응답\n'
        "- 현금배당총액은 억원 단위로 변환 (백만원이면 100으로 나눠 반올림)\n"
        "- 배당 실적이 없으면 \"배당 실적 없음\"이라고만 응답\n"
        "- 다른 텍스트 불필요"
    )

    return _call_llm(api_key, model, prompt, max_completion_tokens=150, base_url=base_url)


def extract_subsidiary_info(
    section_text: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> str:
    """Extract subsidiary companies from 연결대상 종속회사 현황 section text.

    Returns formatted string like: "A사, B사, C사" or "해당없음"
    """
    if not section_text or not api_key:
        return ""
    if not model:
        model = "gpt-5.4-mini"

    prompt = (
        "다음은 DART 공시의 '연결대상 종속회사 현황' 섹션입니다.\n\n"
        f"{section_text[:4000]}\n\n"
        "위 내용에서 연결대상 종속회사 목록을 추출해주세요.\n"
        "응답 규칙:\n"
        '- 종속회사명을 쉼표로 구분하여 나열 (예: "A사, B사, C사")\n'
        '- 종속회사가 없으면 "해당없음"이라고만 응답\n'
        "- 다른 텍스트 불필요"
    )

    return _call_llm(api_key, model, prompt, max_completion_tokens=300, base_url=base_url)


def _call_llm(
    api_key: str,
    model: str,
    prompt: str,
    max_completion_tokens: int = 200,
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
            max_completion_tokens=max_completion_tokens,
        )
        result = (response.choices[0].message.content or "").strip()
        # Clean up: remove quotes wrapping the entire response
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        # Return empty if LLM says no data
        if result in ("없음", "데이터 없음", "해당 없음", "해당없음"):
            return ""
        return result
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
                    max_completion_tokens=max_completion_tokens,
                )
                result = (response.choices[0].message.content or "").strip()
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                if result in ("없음", "데이터 없음", "해당 없음", "해당없음"):
                    return ""
                return result
            except RateLimitError:
                continue
            except Exception:
                break
        logger.warning("LLM rate limit retries exhausted (model=%s)", model)
        return ""
    except Exception as e:
        logger.warning("LLM extraction failed (model=%s): %s", model, e)
        return ""
