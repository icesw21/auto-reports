from __future__ import annotations

import json
import logging
import re

import time

from openai import OpenAI, RateLimitError

from auto_reports.fetchers.rate_limiter import get_llm_limiter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.4-mini"
_MAX_INPUT_CHARS = 8000


def _call_llm(api_key: str, model: str, prompt: str, max_completion_tokens: int, base_url: str = "") -> str:
    """Shared LLM call with rate limiting and RateLimitError retry."""
    if not model:
        model = _DEFAULT_MODEL
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
        return (response.choices[0].message.content or "").strip()
    except RateLimitError as exc:
        logger.warning("LLM rate limited: %s — retrying with backoff", exc)
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
                return (response.choices[0].message.content or "").strip()
            except RateLimitError:
                continue
            except Exception:
                break
        logger.warning("LLM rate limit retries exhausted (model=%s)", model)
        return ""
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""


def _parse_json_response(text: str) -> dict | None:
    """Parse JSON from LLM response, handling ```json fences. Returns None on failure."""
    if not text:
        return None
    # Strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    raw = fenced.group(1) if fenced else text
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        logger.warning("LLM JSON response is not a dict: %r", result)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s | raw=%r", exc, raw[:200])
        return None


def extract_rights_issue(text: str, api_key: str, model: str = "", base_url: str = "") -> dict | None:
    """Extract 유상증자결정 structured data from disclosure text via LLM.

    Args:
        text: Raw text from 유상증자 PDF or HTML disclosure.
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        Dict matching parse_rights_issue_pdf output shape, or None on failure.
    """
    truncated = text[:_MAX_INPUT_CHARS]
    prompt = f"""아래는 유상증자결정 공시 원문입니다. 핵심 정보를 추출하여 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

반환할 JSON 구조:
{{
  "type": "유상증자결정",
  "new_shares": {{"보통주식": <int>, "기타주식": <int>}},
  "issue_price": <int 또는 null>,
  "conversion": {{
    "전환가액(원/주)": <int>,
    "전환주식수": <int>,
    "주식총수대비비율(%)": <float>,
    "전환비율(%)": <float>,
    "전환청구기간": {{"시작일": "YYYY-MM-DD", "종료일": "YYYY-MM-DD"}}
  }},
  "funding_purpose": {{"시설자금": <int>, "운영자금": <int>}} 또는 null,
  "share_type": "<str 또는 null>",
  "payment_date": "<YYYY-MM-DD 또는 null>",
  "board_decision_date": "<YYYY-MM-DD 또는 null>"
}}

값을 알 수 없으면 null로 채우세요. 숫자는 쉼표 없이 정수로 표기하세요.
반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

공시 원문:
{truncated}"""

    raw = _call_llm(api_key, model, prompt, max_completion_tokens=1024, base_url=base_url)
    result = _parse_json_response(raw)
    if result is None:
        logger.warning("extract_rights_issue: LLM returned unparseable response")
        return None
    return result


def extract_cb_issuance(text: str, api_key: str, model: str = "", base_url: str = "") -> dict | None:
    """Extract CB/BW 발행결정 structured data from disclosure text via LLM.

    Args:
        text: Raw text from CB/BW 발행결정 disclosure (HTML or PDF).
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        Dict with bond issuance fields, or None on failure.
    """
    truncated = text[:_MAX_INPUT_CHARS]
    prompt = f"""아래는 사채권(CB/BW) 발행결정 공시 원문입니다. 핵심 정보를 추출하여 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

반환할 JSON 구조:
{{
  "issuance_type_name": "<예: '전환사채권 발행결정'>",
  "bond_type": {{"회차": <int>, "종류": "<str>"}},
  "face_value": <int 또는 null>,
  "conversion_price": <int 또는 null>,
  "convertible_shares": <int 또는 null>,
  "share_ratio_pct": <float 또는 null>,
  "exercise_period": {{"시작일": "YYYY-MM-DD", "종료일": "YYYY-MM-DD"}} 또는 null
}}

값을 알 수 없으면 null로 채우세요. 숫자는 쉼표 없이 정수로 표기하세요.
반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

공시 원문:
{truncated}"""

    raw = _call_llm(api_key, model, prompt, max_completion_tokens=512, base_url=base_url)
    result = _parse_json_response(raw)
    if result is None:
        logger.warning("extract_cb_issuance: LLM returned unparseable response")
        return None
    return result


def extract_stock_option(text: str, api_key: str, model: str = "", base_url: str = "") -> dict | None:
    """Extract 주식매수선택권부여결정 structured data from disclosure text via LLM.

    Args:
        text: Raw text from 주식매수선택권 disclosure (HTML or PDF).
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        Dict with stock option fields, or None on failure.
    """
    truncated = text[:_MAX_INPUT_CHARS]
    prompt = f"""아래는 주식매수선택권부여결정 공시 원문입니다. 핵심 정보를 추출하여 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

반환할 JSON 구조:
{{
  "shares": <int 행사시 발행되는 보통주 주식수>,
  "exercise_price": <int 행사가격(원)>,
  "exercise_period": {{"시작일": "YYYY-MM-DD", "종료일": "YYYY-MM-DD"}} 또는 null,
  "grant_date": "<YYYY-MM-DD 또는 null>"
}}

값을 알 수 없으면 null로 채우세요. 숫자는 쉼표 없이 정수로 표기하세요.
반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

공시 원문:
{truncated}"""

    raw = _call_llm(api_key, model, prompt, max_completion_tokens=512, base_url=base_url)
    result = _parse_json_response(raw)
    if result is None:
        logger.warning("extract_stock_option: LLM returned unparseable response")
        return None
    return result


def classify_disclosure_type(text: str, api_key: str, model: str = "", base_url: str = "") -> str:
    """Classify the type of 주요사항보고서 disclosure from raw text via LLM.

    Args:
        text: Raw text from a 주요사항보고서 PDF.
        api_key: OpenAI API key.
        model: OpenAI model name. Defaults to gpt-5.4-mini.

    Returns:
        One of: "유상증자결정", "전환사채권 발행결정",
        "신주인수권부사채권 발행결정", "주식매수선택권부여결정", "기타"
    """
    _VALID_TYPES = {
        "유상증자결정",
        "전환사채권 발행결정",
        "신주인수권부사채권 발행결정",
        "주식매수선택권부여결정",
        "기타",
    }
    truncated = text[:_MAX_INPUT_CHARS]
    prompt = f"""아래는 주요사항보고서 공시 원문입니다. 공시 유형을 분류하여 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

반환할 JSON 구조:
{{"disclosure_type": "<분류 결과>"}}

분류 선택지 (반드시 아래 중 하나):
- "유상증자결정"
- "전환사채권 발행결정"
- "신주인수권부사채권 발행결정"
- "주식매수선택권부여결정"
- "기타"

반드시 JSON 형식으로만 응답하세요. 다른 텍스트 불필요.

공시 원문:
{truncated}"""

    raw = _call_llm(api_key, model, prompt, max_completion_tokens=64, base_url=base_url)
    parsed = _parse_json_response(raw)
    if parsed is None:
        logger.warning("classify_disclosure_type: LLM returned unparseable response; defaulting to '기타'")
        return "기타"
    disclosure_type = parsed.get("disclosure_type", "기타")
    if disclosure_type not in _VALID_TYPES:
        logger.warning(
            "classify_disclosure_type: unexpected value %r; defaulting to '기타'",
            disclosure_type,
        )
        return "기타"
    return disclosure_type
