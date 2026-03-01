"""Tests for _parse_analysis_response, _clean_risk_item, and extract_estimated_earnings."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_reports.summarizers.analysis_summarizer import (
    _clean_risk_item,
    _parse_analysis_response,
    extract_estimated_earnings,
    generate_momentum_text,
    supplement_value_driver_and_competitors,
)


def test_clean_risk_item_strips_trailing_dash():
    """Trailing standalone dash from next bullet should be removed."""
    assert _clean_risk_item("  - 리스크 내용\n\n- ") == "리스크 내용"


def test_clean_risk_item_preserves_sub_bullets():
    """Multi-line risk with sub-bullets should preserve them."""
    text = "  매출 집중 리스크\n  - 중국 의존도 85%"
    result = _clean_risk_item(text)
    assert "매출 집중 리스크" in result
    assert "중국 의존도 85%" in result


def test_parse_value_driver_bold_markdown():
    """**Value Driver**: format should strip bold markers and stray dashes."""
    text = (
        "## 4. 사업 모델\n\n"
        "- **Value Driver**:\n"
        "  - AI 인프라 투자 확대에 따른 수요 급증\n"
        "  - 자체 설계 역량 보유\n\n"
        "- **경쟁사 비교**:\n"
        "  - 국내 경쟁사 대비 매출 우위\n\n"
        "## 5. 투자 아이디어\n"
        "1. 첫번째\n"
    )
    result = _parse_analysis_response(text)
    # Should NOT start with ** or :
    assert not result.value_driver.startswith("**")
    assert not result.value_driver.startswith(":")
    assert not result.value_driver.endswith("-")
    assert "AI 인프라" in result.value_driver
    # Competitor comparison should preserve bullet format
    assert not result.competitor_comparison.endswith("-")
    assert "국내 경쟁사" in result.competitor_comparison


def test_parse_value_driver_content_contains_competitor_word():
    """Value Driver content mentioning 경쟁사 should NOT be truncated."""
    text = (
        "- **Value Driver**:\n"
        "  - AI 수요 증가\n"
        "  - 경쟁사 대비 기술 우위\n"
        "  - 파두 등 경쟁사 대비 저평가\n\n"
        "- **경쟁사 비교**:\n"
        "  - 파두 시총 2.5조 vs 엠디 2,200억\n\n"
        "## 5. 투자 아이디어\n"
        "1. 첫번째\n"
    )
    result = _parse_analysis_response(text)
    # Value driver should contain ALL items, not be truncated at 경쟁사
    assert "파두 등 경쟁사 대비 저평가" in result.value_driver
    assert "AI 수요 증가" in result.value_driver
    # Competitor comparison should be separate
    assert "파두 시총" in result.competitor_comparison


def test_parse_sections_5_to_7():
    """Sections 5, 6, 7 should be parsed correctly."""
    text = (
        "## 5. 투자 아이디어\n\n"
        "1. 첫번째 아이디어\n"
        "2. 두번째 아이디어\n\n"
        "## 6. 리스크 (Bear Case)\n\n"
        "- **구조적 위험**: 중국 매출 의존도 과다\n"
        "- **반론 제기**: 영업이익 적자전환 가능성\n"
        "- **Tail Risk**: 글로벌 경기 침체\n\n"
        "## 7. 결론\n\n"
        "종합적으로 성장주로 평가됨\n"
    )
    result = _parse_analysis_response(text)
    assert "첫번째 아이디어" in result.investment_ideas
    assert "중국 매출 의존도" in result.risk_structural
    assert "영업이익 적자전환" in result.risk_counter
    assert "글로벌 경기 침체" in result.risk_tail
    assert "성장주로 평가" in result.conclusion


def test_parse_risk_no_stray_dashes():
    """Risk items should not have stray dashes from adjacent bullets."""
    text = (
        "## 6. 리스크 (Bear Case)\n\n"
        "- **구조적 위험**: 리스크A 내용\n"
        "  - 세부 리스크A-1\n\n"
        "- **반론 제기**: 리스크B 내용\n"
        "  - 세부 리스크B-1\n\n"
        "- **Tail Risk**: 리스크C 내용\n\n"
        "## 7. 결론\n\n결론 내용\n"
    )
    result = _parse_analysis_response(text)
    # No trailing dash or whitespace-only lines
    assert not result.risk_structural.endswith("-")
    assert not result.risk_counter.endswith("-")
    assert not result.risk_tail.endswith("-")
    assert "리스크A 내용" in result.risk_structural
    assert "리스크B 내용" in result.risk_counter
    assert "리스크C 내용" in result.risk_tail


def test_parse_value_driver_same_line():
    """Value Driver content on the same line after colon should be captured."""
    text = (
        "## 4. 사업 모델\n\n"
        "- **Value Driver**: 비메모리 반도체 유통 전문성 강화\n"
        "  - AI 인프라 투자 확대에 따른 수요\n\n"
        "- **경쟁사 비교**: 국내 경쟁사 대비 매출 우위\n\n"
        "## 5. 투자 아이디어\n"
        "1. 첫번째\n"
    )
    result = _parse_analysis_response(text)
    assert "비메모리 반도체" in result.value_driver
    assert "AI 인프라" in result.value_driver
    assert "국내 경쟁사" in result.competitor_comparison


def test_parse_value_driver_heading_competitor():
    """### 경쟁사 비교 heading should not leak into Value Driver content."""
    text = (
        "- **Value Driver**:\n"
        "성장 동인 내용\n"
        "  - 세부 동인 1\n\n"
        "### 경쟁사 비교\n"
        "- 경쟁사 A 대비 우위\n\n"
        "## 5. 투자 아이디어\n"
        "1. 첫번째\n"
    )
    result = _parse_analysis_response(text)
    assert "성장 동인" in result.value_driver
    assert "경쟁사 A" not in result.value_driver
    assert "경쟁사 A" in result.competitor_comparison


def test_parse_risk_no_bold_markers():
    """Risk items without bold markers should still be parsed."""
    text = (
        "## 6. 리스크 (Bear Case)\n\n"
        "> [!warning] 주요 리스크 요약\n\n"
        "- 구조적 위험: 공급망 불확실성\n"
        "- 반론 제기: 실적 안정성 우려\n"
        "- Tail Risk: 경기 침체\n\n"
        "## 7. 결론\n\n결론 내용\n"
    )
    result = _parse_analysis_response(text)
    assert "공급망 불확실성" in result.risk_structural
    assert "실적 안정성" in result.risk_counter
    assert "경기 침체" in result.risk_tail


def test_parse_value_driver_no_colon():
    """Value Driver without colon separator should still be captured."""
    text = (
        "- **Value Driver (해자)**\n"
        "글로벌 고객사 확보\n"
        "자체 설계 역량\n\n"
        "- **경쟁사 비교 (벤치마크)**\n"
        "파두 대비 저평가\n\n"
        "## 5. 투자 아이디어\n"
        "1. 첫번째\n"
    )
    result = _parse_analysis_response(text)
    assert "글로벌 고객사" in result.value_driver
    assert "파두 대비" in result.competitor_comparison


# ── supplement_value_driver_and_competitors tests ──


@patch("auto_reports.summarizers.analysis_summarizer.OpenAI")
def test_supplement_value_driver_basic(mock_openai_cls):
    """Should parse [Value Driver] and [경쟁사 비교] from LLM response."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_llm_response(
        "[Value Driver]\nAI 인프라 확대로 매출 성장\n\n"
        "[경쟁사 비교]\n파두 대비 매출 2배 수준"
    )
    vd, cc = supplement_value_driver_and_competitors(
        company_name="테스트",
        report_sections_1_to_4="테스트 내용",
        api_key="fake-key",
    )
    assert "AI 인프라" in vd
    assert "파두 대비" in cc


@patch("auto_reports.summarizers.analysis_summarizer.OpenAI")
def test_supplement_returns_empty_on_failure(mock_openai_cls):
    """Should return empty strings when LLM call fails."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API error")
    vd, cc = supplement_value_driver_and_competitors(
        company_name="테스트",
        report_sections_1_to_4="테스트 내용",
        api_key="fake-key",
    )
    assert vd == ""
    assert cc == ""


# ── generate_momentum_text tests ──


@patch("auto_reports.summarizers.analysis_summarizer.OpenAI")
def test_generate_momentum_text_basic(mock_openai_cls):
    """Should return cleaned momentum text from LLM response."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_llm_response(
        "주요 모멘텀: AI 인프라 투자 확대에 따른 SSD 수요 급증 및 대규모 수주 확인."
    )
    result = generate_momentum_text(
        company_name="테스트",
        report_sections_1_to_4="테스트 내용",
        api_key="fake-key",
    )
    # Should strip the "주요 모멘텀:" prefix
    assert not result.startswith("주요 모멘텀")
    assert "AI 인프라" in result


@patch("auto_reports.summarizers.analysis_summarizer.OpenAI")
def test_generate_momentum_text_returns_empty_on_failure(mock_openai_cls):
    """Should return empty string when LLM call fails."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API error")
    result = generate_momentum_text(
        company_name="테스트",
        report_sections_1_to_4="테스트 내용",
        api_key="fake-key",
    )
    assert result == ""


# ── extract_estimated_earnings tests ──


def _mock_llm_response(content: str):
    """Create a mock OpenAI response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("auto_reports.summarizers.analysis_summarizer._read_pdf_text")
@patch("auto_reports.summarizers.analysis_summarizer.OpenAI")
def test_extract_earnings_averages_multiple(mock_openai_cls, mock_read_pdf, tmp_path):
    """Should average estimates from multiple reports."""
    # Create dummy PDFs
    (tmp_path / "report_a.pdf").write_bytes(b"dummy")
    (tmp_path / "report_b.pdf").write_bytes(b"dummy")

    mock_read_pdf.return_value = "some report text"

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = [
        _mock_llm_response("200"),   # report_a: 200억원
        _mock_llm_response("300"),   # report_b: 300억원
    ]

    result = extract_estimated_earnings(
        reports_dir=tmp_path,
        company_name="테스트",
        target_year=2026,
        api_key="fake-key",
    )

    # Average of 200억 and 300억 = 250억 = 25,000,000,000원
    assert result == 250 * 1_0000_0000


@patch("auto_reports.summarizers.analysis_summarizer._read_pdf_text")
@patch("auto_reports.summarizers.analysis_summarizer.OpenAI")
def test_extract_earnings_returns_none_when_no_estimates(mock_openai_cls, mock_read_pdf, tmp_path):
    """Should return None when no reports have estimates."""
    (tmp_path / "report.pdf").write_bytes(b"dummy")

    mock_read_pdf.return_value = "some text"

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_llm_response("없음")

    result = extract_estimated_earnings(
        reports_dir=tmp_path,
        company_name="테스트",
        target_year=2026,
        api_key="fake-key",
    )

    assert result is None


@patch("auto_reports.summarizers.analysis_summarizer._read_pdf_text")
@patch("auto_reports.summarizers.analysis_summarizer.OpenAI")
def test_extract_earnings_handles_decimal(mock_openai_cls, mock_read_pdf, tmp_path):
    """Should handle decimal estimates (e.g. 42.5억원)."""
    (tmp_path / "report.pdf").write_bytes(b"dummy")

    mock_read_pdf.return_value = "some text"

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_llm_response("42.5")

    result = extract_estimated_earnings(
        reports_dir=tmp_path,
        company_name="테스트",
        target_year=2026,
        api_key="fake-key",
    )

    assert result == int(42.5 * 1_0000_0000)


def test_extract_earnings_empty_dir(tmp_path):
    """Should return None for empty directory (no PDFs)."""
    result = extract_estimated_earnings(
        reports_dir=tmp_path,
        company_name="테스트",
        target_year=2026,
        api_key="fake-key",
    )
    assert result is None


def test_extract_earnings_nonexistent_dir():
    """Should return None for nonexistent directory."""
    result = extract_estimated_earnings(
        reports_dir=Path("/nonexistent/dir"),
        company_name="테스트",
        target_year=2026,
        api_key="fake-key",
    )
    assert result is None
