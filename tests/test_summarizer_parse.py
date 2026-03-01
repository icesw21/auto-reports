"""Tests for _parse_structured_response in openai_summarizer."""

from auto_reports.summarizers.openai_summarizer import _parse_structured_response


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
