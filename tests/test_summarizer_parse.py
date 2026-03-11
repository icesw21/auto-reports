"""Tests for _parse_structured_response, _detect_revenue_unit, and generate_report_tags in openai_summarizer."""

from unittest.mock import MagicMock, patch

from auto_reports.summarizers.openai_summarizer import (
    _detect_revenue_unit,
    _parse_structured_response,
    generate_report_tags,
)


def test_normal_multiline():
    """Multiple sections with multiline content."""
    text = (
        "[주요 매출처]\n"
        "A사 30%, B사 20%\n"
        "\n"
        "[부문별 매출]\n"
        "|부문|비율|\n"
        "제품A 60%, 서비스B 40%\n"
        "\n"
        "[수주잔고]\n"
        "해당사항 없음"
    )
    result = _parse_structured_response(text)
    assert len(result) == 3
    assert "주요매출처" in result
    assert "부문별매출" in result
    assert "수주잔고" in result
    assert "A사 30%" in result["주요매출처"]


def test_inline_brackets_not_split():
    """Inline [brackets] should NOT be treated as section headers."""
    text = (
        "[주요 매출처]\n"
        "[A사] 매출비중 30%, [B사] 20%\n"
        "기타 50%\n"
        "\n"
        "[부문별 매출]\n"
        "제품X 60%\n"
        "\n"
        "[수주잔고]\n"
        "해당사항 없음"
    )
    result = _parse_structured_response(text)
    assert len(result) == 3
    assert "[A사]" in result["주요매출처"]
    assert "[B사]" in result["주요매출처"]


def test_footnotes_preserved():
    """Footnotes like [주1] in middle of content should not split."""
    text = (
        "[주요 매출처]\n"
        "상위 매출처 [주1] 비공개\n"
        "합계 100%\n"
        "\n"
        "[수주잔고]\n"
        "없음"
    )
    result = _parse_structured_response(text)
    assert len(result) == 2
    assert "[주1]" in result["주요매출처"]


def test_unit_annotation_preserved():
    """Unit annotations like [단위: 억원] should not split."""
    text = (
        "[부문별 매출]\n"
        "매출 현황 [단위: 억원]\n"
        "제품A: 100, 제품B: 200\n"
        "\n"
        "[수주잔고]\n"
        "해당사항 없음"
    )
    result = _parse_structured_response(text)
    assert len(result) == 2
    assert "[단위: 억원]" in result["부문별매출"]


def test_empty_input():
    """Empty string returns empty dict."""
    assert _parse_structured_response("") == {}


def test_no_sections():
    """Text without [headers] returns empty dict."""
    assert _parse_structured_response("just plain text") == {}


# ── generate_report_tags tests ──


class TestGenerateReportTags:
    """Tests for generate_report_tags."""

    def _mock_response(self, content: str):
        """Create a mock OpenAI response."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @patch("auto_reports.summarizers.openai_summarizer.OpenAI")
    def test_basic_tag_generation(self, mock_openai_cls):
        """LLM returns tags, one per line."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_response(
            "우주산업\n위성제조\n위성영상\nK방산수출"
        )

        tags = generate_report_tags(
            company_name="쎄트렉아이",
            business_model="- 위성 시스템 개발/제조",
            revenue_breakdown="위성영상 60%, 위성제조 40%",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert tags == ["우주산업", "위성제조", "위성영상", "K방산수출"]
        mock_client.chat.completions.create.assert_called_once()

    @patch("auto_reports.summarizers.openai_summarizer.OpenAI")
    def test_strips_hash_and_dash_prefix(self, mock_openai_cls):
        """Tags with '#' or '- ' prefix are cleaned."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_response(
            "#AI반도체\n- 2차전지\n# 전기차"
        )

        tags = generate_report_tags(
            company_name="테스트사",
            business_model="반도체",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert tags == ["AI반도체", "2차전지", "전기차"]

    @patch("auto_reports.summarizers.openai_summarizer.OpenAI")
    def test_max_5_tags(self, mock_openai_cls):
        """At most 5 tags returned even if LLM outputs more."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_response(
            "태그1\n태그2\n태그3\n태그4\n태그5\n태그6\n태그7"
        )

        tags = generate_report_tags(
            company_name="테스트사",
            business_model="사업모델",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert len(tags) == 5

    def test_no_api_key_returns_empty(self):
        """No API key returns empty list without calling LLM."""
        tags = generate_report_tags(
            company_name="테스트사",
            business_model="사업모델",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="",
        )
        assert tags == []

    def test_no_content_returns_empty(self):
        """No content sections returns empty list without calling LLM."""
        tags = generate_report_tags(
            company_name="테스트사",
            business_model="",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert tags == []

    @patch("auto_reports.summarizers.openai_summarizer.OpenAI")
    def test_api_error_returns_empty(self, mock_openai_cls):
        """API error returns empty list gracefully."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        tags = generate_report_tags(
            company_name="테스트사",
            business_model="사업모델",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert tags == []

    @patch("auto_reports.summarizers.openai_summarizer.OpenAI")
    def test_duplicate_tags_deduplicated(self, mock_openai_cls):
        """Duplicate tags from LLM are deduplicated."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_response(
            "우주산업\n위성제조\n우주산업\n위성영상\n위성제조"
        )

        tags = generate_report_tags(
            company_name="테스트사",
            business_model="위성 사업",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert tags == ["우주산업", "위성제조", "위성영상"]

    @patch("auto_reports.summarizers.openai_summarizer.OpenAI")
    def test_empty_lines_filtered(self, mock_openai_cls):
        """Empty lines in LLM output are ignored."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_response(
            "우주산업\n\n위성제조\n\n위성영상"
        )

        tags = generate_report_tags(
            company_name="테스트사",
            business_model="위성 사업",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert tags == ["우주산업", "위성제조", "위성영상"]

    @patch("auto_reports.summarizers.openai_summarizer.OpenAI")
    def test_long_tags_filtered(self, mock_openai_cls):
        """Tags longer than 20 chars are filtered out."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_response(
            "AI반도체\n이것은매우긴태그이름이라서필터링되어야합니다스물자넘음\n2차전지"
        )

        tags = generate_report_tags(
            company_name="테스트사",
            business_model="반도체",
            revenue_breakdown="",
            investment_ideas="",
            conclusion="",
            api_key="test-key",
        )
        assert "AI반도체" in tags
        assert "2차전지" in tags
        assert len(tags) == 2


# ── _detect_revenue_unit tests ──


class TestDetectRevenueUnit:
    """Tests for _detect_revenue_unit."""

    def test_detect_cheonwon(self):
        """Detects 천원 unit."""
        text = "매출 현황\n(단위: 천원)\n전선퓨즈 1,192,400"
        assert _detect_revenue_unit(text) == "천원"

    def test_detect_baekmawon(self):
        """Detects 백만원 unit."""
        text = "매출 현황\n(단위: 백만원)\n전선퓨즈 1,192"
        assert _detect_revenue_unit(text) == "백만원"

    def test_detect_eokwon(self):
        """Detects 억원 unit."""
        text = "매출 현황\n(단위: 억원)\n전선퓨즈 12"
        assert _detect_revenue_unit(text) == "억원"

    def test_detect_with_spaces(self):
        """Detects unit with extra spaces around colon."""
        text = "매출 현황\n(단위 : 천원)\n데이터"
        assert _detect_revenue_unit(text) == "천원"

    def test_detect_fullwidth_colon(self):
        """Detects unit with fullwidth colon."""
        text = "매출 현황\n(단위：백만원)\n데이터"
        assert _detect_revenue_unit(text) == "백만원"

    def test_default_when_no_unit(self):
        """Defaults to 백만원 when no unit found."""
        text = "매출 현황\n전선퓨즈 1,192"
        assert _detect_revenue_unit(text) == "백만원"

    def test_default_on_empty(self):
        """Defaults to 백만원 on empty string."""
        assert _detect_revenue_unit("") == "백만원"

    def test_first_match_wins(self):
        """Takes first matching unit in text."""
        text = "(단위: 천원)\n...\n참고: 단위 백만원 기준"
        assert _detect_revenue_unit(text) == "천원"

    def test_no_colon_returns_default(self):
        """Bare space without colon should NOT match — returns default."""
        text = "부문별 단위 천원 기준 합계"
        assert _detect_revenue_unit(text) == "백만원"
