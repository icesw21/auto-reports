"""LLM-based analysis for investment thesis, risks, and conclusion (sections 5-7)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI, RateLimitError

from auto_reports.fetchers.rate_limiter import get_llm_limiter

logger = logging.getLogger(__name__)


def _llm_call_with_retry(client: OpenAI, model: str, messages: list, **kwargs) -> str:
    """Make an LLM call with rate limiting and RateLimitError retry."""
    get_llm_limiter().wait()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, **kwargs,
        )
        return (response.choices[0].message.content or "").strip()
    except RateLimitError as e:
        logger.warning("LLM rate limited (model=%s): %s — retrying", model, e)
        for attempt in range(3):
            import time
            time.sleep(2 ** (attempt + 1))
            get_llm_limiter().wait()
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, **kwargs,
                )
                return (response.choices[0].message.content or "").strip()
            except RateLimitError:
                continue
            except Exception:
                break
        logger.warning("LLM rate limit retries exhausted (model=%s)", model)
        return ""
    except Exception as e:
        logger.warning("LLM call failed (model=%s): %s", model, type(e).__name__)
        return ""


@dataclass
class AnalysisResult:
    """Result of LLM-based investment analysis."""
    investment_ideas: str = ""  # Section 5 content
    risk_structural: str = ""  # 구조적 위험
    risk_counter: str = ""  # 반론 제기
    risk_tail: str = ""  # Tail Risk
    conclusion: str = ""  # Section 7 content
    value_driver: str = ""  # Value Driver from section 4
    competitor_comparison: str = ""  # 경쟁사 비교 from section 4


def extract_estimated_earnings(
    reports_dir: Path,
    company_name: str,
    target_year: int,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> int | None:
    """Extract estimated net income for target_year from research report PDFs.

    Reads each PDF, asks LLM to find the estimated net income for the
    target year, and returns the average across reports (in KRW 원).

    Args:
        reports_dir: Directory containing research report PDFs.
        company_name: Company name for context.
        target_year: Year to extract estimates for (e.g. 2026).
        api_key: OpenAI API key.
        model: Model to use (default: gpt-4.1-mini).

    Returns:
        Average estimated net income in KRW (원), or None if unavailable.
    """
    if not model:
        model = "gpt-4.1-mini"

    if not reports_dir.is_dir():
        return None

    pdfs = sorted(reports_dir.glob("*.pdf"))
    if not pdfs:
        return None

    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )
    estimates: list[int] = []

    for pdf_path in pdfs:
        text = _read_pdf_text(pdf_path, max_chars=6000)
        if not text:
            continue

        prompt = (
            f"다음은 {company_name}에 대한 증권사 리서치 리포트의 일부입니다.\n\n"
            f"{text}\n\n"
            f"위 리포트에서 {target_year}년 예상(추정/전망) 당기순이익 "
            f"또는 지배주주순이익 금액을 찾아주세요.\n"
            f"응답 규칙:\n"
            f"- 금액을 억원 단위 숫자로만 응답 (예: 150)\n"
            f"- 소수점 허용 (예: 42.5)\n"
            f"- 원문이 백만원 단위면 억원으로 변환 (예: 15,000백만원 → 150)\n"
            f"- 해당 연도의 추정치가 없으면 \"없음\"이라고만 응답\n"
            f"- 숫자 외 다른 텍스트 불필요"
        )

        try:
            answer = _llm_call_with_retry(
                client, model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=50,
            )

            if "없음" in answer or not answer:
                logger.info("No %d estimate in %s", target_year, pdf_path.name)
                continue

            num_match = re.search(r"[\d,.]+", answer)
            if num_match:
                value_eok = float(num_match.group().replace(",", ""))
                value_won = int(value_eok * 1_0000_0000)
                estimates.append(value_won)
                logger.info(
                    "Extracted %d estimate from %s: %.1f억원",
                    target_year, pdf_path.name, value_eok,
                )
        except Exception as e:
            logger.warning(
                "Failed to extract earnings from %s: %s", pdf_path.name, e,
            )

    if not estimates:
        logger.info("No %d earnings estimates found in %d reports", target_year, len(pdfs))
        return None

    avg = sum(estimates) // len(estimates)
    logger.info(
        "Average estimated net income for %d: %d원 (%.1f억원, %d reports)",
        target_year, avg, avg / 1_0000_0000, len(estimates),
    )
    return avg


def _read_pdf_text(path: Path, max_chars: int = 5000) -> str:
    """Extract text from a PDF file, truncated to max_chars."""
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                if len(text) > max_chars:
                    break
            return text[:max_chars]
    except ImportError:
        logger.warning("PyPDF2 not installed, skipping PDF: %s", path.name)
        return ""
    except Exception:
        logger.warning("Failed to read PDF: %s", path.name)
        return ""


def _deduplicate_news(articles: list[dict]) -> list[dict]:
    """Remove duplicate/near-duplicate news articles by title similarity.

    Handles common patterns:
    - Same article from different outlets
    - Updated articles with prefixed tags like [속보], [종합], [단독]
    """
    if not articles:
        return []

    def _norm(t: str) -> str:
        # Remove common bracket tags and collapse whitespace
        t = re.sub(r"\[.*?\]|【.*?】", "", t)
        return re.sub(r"\s+", "", t).strip()

    seen: list[str] = []
    unique: list[dict] = []
    for item in articles:
        title = item.get("title", "")
        norm = _norm(title)
        if not norm:
            continue
        # Substring match catches prefix/suffix variations
        # Min length guard prevents false matches on very short titles
        is_dup = any(
            (norm in s or s in norm) and min(len(norm), len(s)) >= 8
            for s in seen
        )
        if not is_dup:
            unique.append(item)
            seen.append(norm)
    return unique


def _load_news_articles(
    news_file: Path,
    max_chars_per_article: int = 800,
    max_total_chars: int = 15000,
) -> str:
    """Load news articles (title + body) from JSON file, with deduplication.

    Args:
        news_file: Path to the news JSON file.
        max_chars_per_article: Max body characters per article.
        max_total_chars: Max total characters for all news content.

    Returns:
        Formatted news content string with titles and body excerpts.
    """
    if not news_file.exists():
        return ""
    try:
        data = json.loads(news_file.read_text(encoding="utf-8"))
        # Deduplicate before formatting
        data = _deduplicate_news(data)
        lines: list[str] = []
        total_chars = 0
        for item in data:
            if total_chars >= max_total_chars:
                break
            date = item.get("date", "")
            title = item.get("title", "")
            content = item.get("content", "")
            if not title:
                continue
            entry = f"- [{date}] {title}"
            if content:
                truncated = content[:max_chars_per_article]
                if len(content) > max_chars_per_article:
                    truncated += "..."
                entry += f"\n  {truncated}"
            lines.append(entry)
            total_chars += len(entry)
        return "\n".join(lines)
    except Exception:
        logger.warning("Failed to read news file: %s", news_file.name)
        return ""


def _build_sources_content(
    report_text: str,
    research_reports: list[tuple[str, str]],  # [(filename, text), ...]
    news_text: str,
) -> str:
    """Build the {sources_content} for the analysis prompt."""
    sections = []

    if report_text:
        sections.append(f"### 자동 생성된 재무 분석 보고서 (섹션 1~4)\n\n{report_text}")

    for filename, text in research_reports:
        if text:
            sections.append(f"### 리서치 리포트: {filename}\n\n{text}")

    if news_text:
        sections.append(f"### 최근 뉴스\n\n{news_text}")

    return "\n\n---\n\n".join(sections)


def _load_sources(
    report_text: str,
    reports_dir: Path | None,
    news_file: Path | None,
) -> str:
    """Load PDFs + news and build combined sources content."""
    research_reports = []
    if reports_dir and reports_dir.is_dir():
        for pdf_path in sorted(reports_dir.glob("*.pdf")):
            text = _read_pdf_text(pdf_path, max_chars=5000)
            if text:
                research_reports.append((pdf_path.name, text))
    news_text = ""
    if news_file and news_file.exists():
        news_text = _load_news_articles(news_file)
    return _build_sources_content(report_text, research_reports, news_text)


def generate_analysis(
    company_name: str,
    report_sections_1_to_4: str,
    prompt_template: str,
    api_key: str,
    model: str = "",
    reports_dir: Path | None = None,
    news_file: Path | None = None,
    base_url: str = "",
) -> AnalysisResult:
    """Generate investment analysis sections 5-7 using LLM.

    Args:
        company_name: Company name for the analysis.
        report_sections_1_to_4: Text of already-generated sections 1-4.
        prompt_template: The analysis prompt template with {company_name} and {sources_content} placeholders.
        api_key: OpenAI API key.
        model: Model to use (empty string falls back to gpt-4.1-mini).
        reports_dir: Directory containing research report PDFs.
        news_file: Path to news JSON file.

    Returns:
        AnalysisResult with generated content.
    """
    if not model:
        model = "gpt-4.1-mini"

    result = AnalysisResult()

    sources_content = _load_sources(report_sections_1_to_4, reports_dir, news_file)

    # Build final prompt (substitute sources_content first to prevent injection
    # if company_name contains "{sources_content}")
    prompt = prompt_template.replace("{sources_content}", sources_content)
    prompt = prompt.replace("{company_name}", company_name)

    # 5. Call LLM
    client = OpenAI(
        api_key=api_key, max_retries=5,
        **({"base_url": base_url} if base_url else {}),
    )

    logger.info("Calling LLM for analysis (model=%s, prompt=%d chars)", model, len(prompt))
    raw_output = _llm_call_with_retry(
        client, model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4, max_tokens=6000,
    )
    if not raw_output:
        logger.warning("LLM analysis returned empty response (model=%s)", model)
        return result
    logger.info("LLM analysis response: %d chars", len(raw_output))

    # 6. Parse the response - extract sections 4 additions + 5-7
    result = _parse_analysis_response(raw_output)

    # 7. Warn if output appears truncated (section 5 present but conclusion missing)
    if result.investment_ideas and not result.conclusion:
        logger.warning("LLM analysis may be truncated: section 5 present but section 7 (conclusion) missing")

    return result


def supplement_value_driver_and_competitors(
    company_name: str,
    report_sections_1_to_4: str,
    api_key: str,
    model: str = "",
    reports_dir: Path | None = None,
    news_file: Path | None = None,
    base_url: str = "",
) -> tuple[str, str]:
    """Generate Value Driver and 경쟁사 비교 as a feedback-loop supplement.

    Called when the main analysis didn't produce these fields.
    Uses sections 1-4 + research reports + news as context.

    Returns:
        (value_driver, competitor_comparison) tuple of strings.
    """
    if not model:
        model = "gpt-4.1-mini"

    sources_content = _load_sources(report_sections_1_to_4, reports_dir, news_file)

    prompt = (
        f"당신은 전문 금융 분석가입니다. 다음은 {company_name}에 대한 소스 문서입니다.\n\n"
        f"{sources_content}\n\n"
        f"위 소스를 바탕으로 다음 두 가지를 각각 작성해주세요. "
        f"모든 문장은 명사/명사구로 자연스럽게 종결하세요 (예: 수주 확보, 실적 개선). '~임', '~함' 같은 접미사 금지.\n\n"
        f"[Value Driver]\n"
        f"이 회사의 구조적 성장 동인과 해자(Moat)를 분석하세요.\n"
        f"'- '로 시작하는 불릿 리스트로 2-4가지 작성. 각 항목에 정량적 근거 포함.\n"
        f"중첩 불릿(sub-bullet) 금지. 모든 항목을 같은 레벨의 '- '로 작성.\n"
        f"소스에 근거한 내용만 포함.\n\n"
        f"[경쟁사 비교]\n"
        f"동종 업계 경쟁사와 비교 분석하세요.\n"
        f"'- '로 시작하는 불릿 리스트로 작성. Valuation, 매출, 수익성 등 정량적 비교 포함.\n"
        f"중첩 불릿(sub-bullet) 금지. 모든 항목을 같은 레벨의 '- '로 작성.\n"
        f"소스에 근거한 내용만 포함, 없으면 '- 데이터 부족으로 비교 불가'라고 명시.\n"
    )

    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )

    logger.info("Supplementing Value Driver & 경쟁사 비교 (model=%s)", model)
    raw = _llm_call_with_retry(
        client, model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4, max_tokens=2000,
    )
    if not raw:
        return ("", "")

    # Parse [Value Driver] and [경쟁사 비교] sections
    value_driver = ""
    competitor_comparison = ""

    m = re.search(r"\[Value Driver\]\s*(.*?)(?=\[경쟁사 비교\]|\Z)", raw, re.DOTALL)
    if m:
        value_driver = m.group(1).strip()

    m = re.search(r"\[경쟁사 비교\]\s*(.*?)$", raw, re.DOTALL)
    if m:
        competitor_comparison = m.group(1).strip()

    return (value_driver, competitor_comparison)


def generate_momentum_text(
    company_name: str,
    report_sections_1_to_4: str,
    api_key: str,
    model: str = "",
    reports_dir: Path | None = None,
    news_file: Path | None = None,
    base_url: str = "",
) -> str:
    """Generate 주요 모멘텀 text as a feedback-loop supplement.

    Identifies key catalysts/momentum drivers for the stock's price movement
    using sections 1-4, research reports, and news.

    Returns:
        Momentum text string (1-2 sentences), or empty string.
    """
    if not model:
        model = "gpt-4.1-mini"

    sources_content = _load_sources(report_sections_1_to_4, reports_dir, news_file)

    prompt = (
        f"당신은 전문 금융 분석가입니다. 다음은 {company_name}에 대한 소스 문서입니다.\n\n"
        f"{sources_content}\n\n"
        f"위 소스를 바탕으로 이 종목의 최근 1년간 주요 모멘텀(주가 상승/하락 촉매)을 "
        f"3~5개 항목으로 요약해주세요.\n"
        f"- 핵심 이벤트, 수주/계약, 실적 변화, 산업 트렌드 등을 포함\n"
        f"- 각 항목은 반드시 온점(.)으로 종결할 것\n"
        f"- 각 항목은 반드시 줄바꿈(\\n)으로 구분할 것 (한 줄에 한 항목만)\n"
        f"- 각 항목은 명사/명사구로 자연스럽게 종결할 것 (예: 수주 확보, 실적 개선). '~임', '~함' 같은 어색한 접미사 금지\n"
        f"- 항목만 응답 (머리말/꼬리말/번호/불릿 불필요)"
    )

    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )

    logger.info("Generating momentum text for %s (model=%s)", company_name, model)
    result = _llm_call_with_retry(
        client, model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=500,
    )
    if result:
        # Remove any leading "주요 모멘텀:" prefix if LLM included it
        result = re.sub(r'^주요\s*모멘텀\s*[:：]\s*', '', result).strip()
        # Clean up each line: strip bullet/number prefixes, ensure period ending
        lines = []
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Strip leading bullets/numbers: "- ", "1. ", "• ", "1) " etc.
            line = re.sub(r'^[-•·]\s+', '', line)
            line = re.sub(r'^\d+[.)]\s*', '', line)
            line = line.strip()
            if not line:
                continue
            # Ensure line ends with a period
            if not line.endswith('.'):
                line += '.'
            lines.append(line)
        result = '\n'.join(lines)
    return result


def _clean_risk_item(s: str) -> str:
    """Strip leading/trailing list markers and whitespace from a risk item."""
    s = s.strip()
    # Remove leading "- " prefix (not character-set strip)
    if s.startswith("- "):
        s = s[2:]
    # Remove trailing standalone dash (leftover from next bullet's `- `)
    s = re.sub(r'\n-\s*$', '', s).strip()
    # Remove trailing --- separators and code fences
    s = re.sub(r'\n---\s*$', '', s).strip()
    s = re.sub(r'\n```\s*$', '', s).strip()
    return s


def _strip_section(s: str) -> str:
    """Strip trailing --- separators and code fences from a section."""
    s = s.strip()
    s = re.sub(r'\n---\s*$', '', s).strip()
    s = re.sub(r'\n```\s*$', '', s).strip()
    return s


def _parse_analysis_response(text: str) -> AnalysisResult:
    """Parse the LLM analysis response to extract sections.

    The LLM response follows the structure from analysis_prompt.md.
    We extract sections 4 additions (Value Driver, 경쟁사 비교) and sections 5-7.
    """
    result = AnalysisResult()

    # Preprocessing: strip code fences wrapping the whole response
    text = re.sub(r'^```(?:markdown)?\s*\n', '', text.strip())
    text = re.sub(r'\n```\s*$', '', text.strip())

    # Extract section 5: 투자 아이디어
    m = re.search(r"## 5\.\s*투자 아이디어\s*\n(.*?)(?=## 6\.|$)", text, re.DOTALL)
    if m:
        result.investment_ideas = _strip_section(m.group(1))

    # Extract section 6: 리스크
    m = re.search(r"## 6\.\s*리스크.*?\n(.*?)(?=## 7\.|$)", text, re.DOTALL)
    if m:
        risk_text = m.group(1).strip()
        # Strip leading callout block (> [!warning] ... and continuation > lines)
        risk_text = re.sub(r'^>\s*\[!warning\][^\n]*\n?', '', risk_text)
        risk_text = re.sub(r'^>\s*[^\n]*\n?', '', risk_text)  # strip one continuation line
        risk_text = risk_text.strip()

        # Parse sub-items - flexible: both bold (**구조적 위험**:) and plain (구조적 위험:)
        sm = re.search(
            r"\*{0,2}구조적 (?:위험|리스크)\*{0,2}[:\s]*(.+?)"
            r"(?=\n-?\s*\*{0,2}반론|\n-?\s*\*{0,2}Tail|$)",
            risk_text, re.DOTALL,
        )
        if sm:
            result.risk_structural = _clean_risk_item(sm.group(1))
        sm = re.search(
            r"\*{0,2}반론[^\n:]*\*{0,2}[:\s]*(.+?)"
            r"(?=\n-?\s*\*{0,2}Tail|$)",
            risk_text, re.DOTALL,
        )
        if sm:
            result.risk_counter = _clean_risk_item(sm.group(1))
        sm = re.search(r"\*{0,2}Tail Risk\*{0,2}[:\s]*(.+?)$", risk_text, re.DOTALL)
        if sm:
            result.risk_tail = _clean_risk_item(sm.group(1))

        # Fallback: if sub-items not found, use the whole risk text as structural
        if not result.risk_structural and not result.risk_counter and risk_text:
            result.risk_structural = _strip_section(risk_text)

    # Extract section 7: 결론
    m = re.search(r"## 7\.\s*결론\s*\n(.*?)$", text, re.DOTALL)
    if m:
        result.conclusion = _strip_section(m.group(1))

    # Extract Value Driver from section 4
    # Use lazy match [^\n]*? to skip header, then :[ \t]* for the separator colon
    # Stops at 경쟁사 비교 in any format: bold, heading, or plain
    m = re.search(
        r"Value Driver[^\n]*?:[ \t]*(.*?)(?=\n\s*(?:-?\s*\*{0,2}|#{1,4}\s+)경쟁사 비교|\n## 5\.|\Z)",
        text, re.DOTALL,
    )
    # Fallback: if no colon found (e.g. "**Value Driver (해자)**\n"), try skipping whole line
    if not m or not (m.group(1) and len(m.group(1).strip()) > 5):
        m = re.search(
            r"Value Driver[^\n]*\n(.*?)(?=\n\s*(?:-?\s*\*{0,2}|#{1,4}\s+)경쟁사 비교|\n## 5\.|\Z)",
            text, re.DOTALL,
        )
    if m:
        vd = _strip_section(m.group(1))
        if vd and len(vd) > 5:
            result.value_driver = vd

    # Extract 경쟁사 비교 from section 4
    # Capture content after header (same-line after colon + subsequent lines)
    m = re.search(
        r"경쟁사 비교[^\n:]*:?\s*(.*?)(?=\n## 5\.|\Z)",
        text, re.DOTALL,
    )
    if m:
        cc = _strip_section(m.group(1))
        if cc and len(cc) > 5:
            result.competitor_comparison = cc

    return result
