"""Tests for LLM-based overhang extraction from financial statement notes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from auto_reports.parsers.notes_overhang_llm import (
    _normalize_date,
    _normalize_instrument,
    _parse_llm_response,
    _safe_int,
    extract_overhang_context,
    parse_notes_overhang_llm,
)


# ------------------------------------------------------------------
# extract_overhang_context
# ------------------------------------------------------------------


class TestExtractOverhangContext:
    """Test HTML context extraction around overhang keywords."""

    def test_finds_section_with_cb_keyword(self):
        html = """<html><body>
        <p>12. 기타</p>
        <p>내용</p>
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table><tr><td>발행총액</td><td>10,000,000,000원</td></tr></table>
        <p>14. 종업원급여</p>
        </body></html>"""
        ctx = extract_overhang_context(html)
        assert "전환사채" in ctx
        assert "발행총액" in ctx
        assert "종업원급여" not in ctx

    def test_finds_subsection_with_preferred_stock(self):
        html = """<html><body>
        <p>13. 자본</p>
        <p>(1) 보통주</p>
        <p>내용</p>
        <p>(2) 전환우선주</p>
        <table><tr><td>발행주식수</td><td>100,000주</td></tr></table>
        <p>(3) 기타</p>
        </body></html>"""
        ctx = extract_overhang_context(html)
        assert "전환우선주" in ctx
        assert "발행주식수" in ctx

    def test_returns_empty_when_no_keywords(self):
        html = """<html><body>
        <p>1. 회사의 개요</p>
        <p>내용</p>
        <table><tr><td>자산</td><td>100</td></tr></table>
        </body></html>"""
        ctx = extract_overhang_context(html)
        assert ctx == ""

    def test_finds_keyword_in_table(self):
        html = """<html><body>
        <p>12. 차입금</p>
        <table><tr><td>구분</td><td>금액</td></tr>
        <tr><td>전환사채</td><td>15,000,000,000</td></tr></table>
        </body></html>"""
        ctx = extract_overhang_context(html)
        assert "전환사채" in ctx
        assert "15,000,000,000" in ctx

    def test_collects_multiple_tables_in_section(self):
        """Multiple tables between paragraphs should all be captured."""
        html = """<html><body>
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table><tr><td>발행총액</td><td>10,000,000,000원</td></tr></table>
        <table><tr><td>전환가격</td><td>7,875원</td></tr></table>
        <table><tr><td>공정가치</td><td>9,500,000,000원</td></tr></table>
        <p>14. 기타</p>
        </body></html>"""
        ctx = extract_overhang_context(html)
        assert "발행총액" in ctx
        assert "전환가격" in ctx
        assert "공정가치" in ctx

    def test_finds_stock_option_keyword(self):
        html = """<html><body>
        <p>20. 주식기준보상</p>
        <p>당사는 임직원에게 주식매수선택권을 부여하였습니다.</p>
        <table>
        <tr><td>구분</td><td>행사가격</td><td>잔여수량</td></tr>
        <tr><td>1차</td><td>5,000</td><td>100,000</td></tr>
        </table>
        <p>21. 기타</p>
        </body></html>"""
        ctx = extract_overhang_context(html)
        assert "주식매수선택권" in ctx or "주식기준보상" in ctx
        assert "행사가격" in ctx
        assert "잔여수량" in ctx

    def test_multiple_keywords_found(self):
        html = """<html><body>
        <p>13. 전환사채</p>
        <table><tr><td>발행총액</td><td>10,000,000,000</td></tr></table>
        <p>14. 신주인수권부사채</p>
        <table><tr><td>발행총액</td><td>5,000,000,000</td></tr></table>
        <p>15. 기타</p>
        </body></html>"""
        ctx = extract_overhang_context(html)
        assert "전환사채" in ctx
        assert "신주인수권부사채" in ctx


# ------------------------------------------------------------------
# _parse_llm_response
# ------------------------------------------------------------------


class TestParseLlmResponse:
    """Test LLM JSON response parsing."""

    def test_parses_instruments_wrapper(self):
        content = json.dumps({
            "instruments": [
                {
                    "category": "CB",
                    "series": 1,
                    "kind": "제1회 전환사채",
                    "face_value": 15_000_000_000,
                    "convertible_shares": 1_955_555,
                    "conversion_price": 7_875,
                    "exercise_start": "2024.09.28",
                    "exercise_end": "2027.08.28",
                    "active": True,
                },
            ],
        })
        results = _parse_llm_response(content)
        assert len(results) == 1
        assert results[0]["category"] == "CB"
        assert results[0]["series"] == 1
        assert results[0]["face_value"] == 15_000_000_000
        assert results[0]["convertible_shares"] == 1_955_555
        assert results[0]["conversion_price"] == 7_875
        assert results[0]["exercise_start"] == "2024.09.28"
        assert results[0]["active"] is True

    def test_parses_direct_array(self):
        content = json.dumps([
            {
                "category": "BW",
                "series": 2,
                "kind": "제2회 신주인수권부사채",
                "face_value": 5_000_000_000,
                "convertible_shares": 500_000,
                "conversion_price": 10_000,
                "exercise_start": "2025.01.01",
                "exercise_end": "2028.12.31",
                "active": True,
            },
        ])
        results = _parse_llm_response(content)
        assert len(results) == 1
        assert results[0]["category"] == "BW"

    def test_normalizes_korean_category(self):
        content = json.dumps({
            "instruments": [
                {
                    "category": "전환사채",
                    "series": 3,
                    "kind": "제3회 전환사채",
                    "face_value": 4_000_000_000,
                    "convertible_shares": 0,
                    "conversion_price": 9_714,
                    "exercise_start": "",
                    "exercise_end": "",
                    "active": True,
                },
            ],
        })
        results = _parse_llm_response(content)
        assert len(results) == 1
        assert results[0]["category"] == "CB"
        # convertible_shares should be calculated
        assert results[0]["convertible_shares"] == 4_000_000_000 // 9_714

    def test_normalizes_preferred_stock_variants(self):
        for cat_name in ("전환우선주", "전환상환우선주", "상환전환우선주", "RCPS"):
            content = json.dumps({
                "instruments": [{
                    "category": cat_name,
                    "series": 0,
                    "kind": "전환우선주",
                    "face_value": 1_000_000_000,
                    "convertible_shares": 100_000,
                    "conversion_price": 10_000,
                    "exercise_start": "",
                    "exercise_end": "",
                    "active": True,
                }],
            })
            results = _parse_llm_response(content)
            assert results[0]["category"] == "전환우선주", f"Failed for {cat_name}"

    def test_calculates_shares_from_face_value_and_price(self):
        content = json.dumps({
            "instruments": [{
                "category": "CB",
                "series": 1,
                "kind": "제1회 전환사채",
                "face_value": 10_000_000_000,
                "convertible_shares": 0,
                "conversion_price": 5_000,
                "exercise_start": "",
                "exercise_end": "",
                "active": True,
            }],
        })
        results = _parse_llm_response(content)
        assert results[0]["convertible_shares"] == 2_000_000

    def test_skips_instruments_with_no_data(self):
        content = json.dumps({
            "instruments": [{
                "category": "CB",
                "series": 1,
                "kind": "제1회 전환사채",
                "face_value": 0,
                "convertible_shares": 0,
                "conversion_price": 0,
                "exercise_start": "",
                "exercise_end": "",
                "active": True,
            }],
        })
        results = _parse_llm_response(content)
        assert len(results) == 0

    def test_normalizes_stock_option_category(self):
        for cat_name in ("SO", "주식매수선택권", "주식기준보상", "스톡옵션"):
            content = json.dumps({
                "instruments": [{
                    "category": cat_name,
                    "series": 1,
                    "kind": "주식매수선택권 1차",
                    "face_value": 0,
                    "convertible_shares": 50_000,
                    "conversion_price": 8_000,
                    "exercise_start": "2024.01.01",
                    "exercise_end": "2028.12.31",
                    "active": True,
                }],
            })
            results = _parse_llm_response(content)
            assert len(results) == 1, f"Failed for {cat_name}"
            assert results[0]["category"] == "SO", f"Failed for {cat_name}"
            assert results[0]["face_value"] == 0
            assert results[0]["convertible_shares"] == 50_000
            assert results[0]["conversion_price"] == 8_000

    def test_so_skipped_when_zero_shares(self):
        content = json.dumps({
            "instruments": [{
                "category": "SO",
                "series": 1,
                "kind": "주식매수선택권 1차",
                "face_value": 0,
                "convertible_shares": 0,
                "conversion_price": 5_000,
                "exercise_start": "",
                "exercise_end": "",
                "active": True,
            }],
        })
        results = _parse_llm_response(content)
        assert len(results) == 0

    def test_so_with_multiple_grants(self):
        content = json.dumps({
            "instruments": [
                {
                    "category": "SO",
                    "series": 1,
                    "kind": "주식매수선택권 1차",
                    "face_value": 0,
                    "convertible_shares": 30_000,
                    "conversion_price": 5_000,
                    "exercise_start": "2023.06.01",
                    "exercise_end": "2027.05.31",
                    "active": True,
                },
                {
                    "category": "SO",
                    "series": 2,
                    "kind": "주식매수선택권 2차",
                    "face_value": 0,
                    "convertible_shares": 50_000,
                    "conversion_price": 8_000,
                    "exercise_start": "2024.03.01",
                    "exercise_end": "2028.02.28",
                    "active": True,
                },
            ],
        })
        results = _parse_llm_response(content)
        assert len(results) == 2
        assert results[0]["series"] == 1
        assert results[1]["series"] == 2

    def test_so_active_false_preserved(self):
        """SO with convertible_shares > 0 but active=False should be returned with active=False."""
        content = json.dumps({
            "instruments": [{
                "category": "SO",
                "series": 1,
                "kind": "주식매수선택권 1차",
                "face_value": 0,
                "convertible_shares": 50_000,
                "conversion_price": 8_000,
                "exercise_start": "2023.01.01",
                "exercise_end": "2027.12.31",
                "active": False,
            }],
        })
        results = _parse_llm_response(content)
        assert len(results) == 1
        assert results[0]["category"] == "SO"
        assert results[0]["convertible_shares"] == 50_000
        assert results[0]["active"] is False

    def test_skips_invalid_category(self):
        content = json.dumps({
            "instruments": [{
                "category": "주식",
                "series": 1,
                "kind": "보통주",
                "face_value": 1_000_000,
                "convertible_shares": 100,
                "conversion_price": 10_000,
                "exercise_start": "",
                "exercise_end": "",
                "active": True,
            }],
        })
        results = _parse_llm_response(content)
        assert len(results) == 0

    def test_empty_response(self):
        assert _parse_llm_response("") == []

    def test_invalid_json(self):
        assert _parse_llm_response("not json at all") == []

    def test_normalizes_dates(self):
        content = json.dumps({
            "instruments": [{
                "category": "CB",
                "series": 1,
                "kind": "제1회 전환사채",
                "face_value": 10_000_000_000,
                "convertible_shares": 1_000_000,
                "conversion_price": 10_000,
                "exercise_start": "2024-9-1",
                "exercise_end": "2027/12/31",
                "active": True,
            }],
        })
        results = _parse_llm_response(content)
        assert results[0]["exercise_start"] == "2024.09.01"
        assert results[0]["exercise_end"] == "2027.12.31"

    def test_handles_string_numbers(self):
        content = json.dumps({
            "instruments": [{
                "category": "CB",
                "series": 1,
                "kind": "제1회 전환사채",
                "face_value": "10,000,000,000원",
                "convertible_shares": "1,000,000주",
                "conversion_price": "10,000",
                "exercise_start": "",
                "exercise_end": "",
                "active": True,
            }],
        })
        results = _parse_llm_response(content)
        assert results[0]["face_value"] == 10_000_000_000
        assert results[0]["convertible_shares"] == 1_000_000
        assert results[0]["conversion_price"] == 10_000

    def test_multiple_instruments(self):
        content = json.dumps({
            "instruments": [
                {
                    "category": "CB",
                    "series": 1,
                    "kind": "제1회 전환사채",
                    "face_value": 15_400_000_000,
                    "convertible_shares": 1_955_555,
                    "conversion_price": 7_875,
                    "exercise_start": "2024.09.28",
                    "exercise_end": "2027.08.28",
                    "active": True,
                },
                {
                    "category": "CB",
                    "series": 2,
                    "kind": "제2회 전환사채",
                    "face_value": 6_000_000_000,
                    "convertible_shares": 564_227,
                    "conversion_price": 10_634,
                    "exercise_start": "2025.02.21",
                    "exercise_end": "2028.01.21",
                    "active": True,
                },
                {
                    "category": "CB",
                    "series": 3,
                    "kind": "제3회 전환사채",
                    "face_value": 4_000_000_000,
                    "convertible_shares": 411_776,
                    "conversion_price": 9_714,
                    "exercise_start": "2025.08.28",
                    "exercise_end": "2028.07.28",
                    "active": True,
                },
                {
                    "category": "전환우선주",
                    "series": 0,
                    "kind": "전환우선주",
                    "face_value": 1_000_000_000,
                    "convertible_shares": 102_943,
                    "conversion_price": 9_714,
                    "exercise_start": "2025.12.14",
                    "exercise_end": "2029.12.14",
                    "active": True,
                },
            ],
        })
        results = _parse_llm_response(content)
        assert len(results) == 4
        categories = [r["category"] for r in results]
        assert categories == ["CB", "CB", "CB", "전환우선주"]


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


class TestSafeInt:
    def test_int(self):
        assert _safe_int(42) == 42

    def test_float(self):
        assert _safe_int(42.7) == 42

    def test_string_with_commas(self):
        assert _safe_int("10,000,000") == 10_000_000

    def test_string_with_unit(self):
        assert _safe_int("7,875원") == 7_875

    def test_string_with_shares(self):
        assert _safe_int("100,000주") == 100_000

    def test_empty_string(self):
        assert _safe_int("") == 0

    def test_none(self):
        assert _safe_int(None) == 0

    def test_non_numeric_string(self):
        assert _safe_int("abc") == 0


class TestNormalizeDate:
    def test_dot_format(self):
        assert _normalize_date("2024.09.28") == "2024.09.28"

    def test_dash_format(self):
        assert _normalize_date("2024-9-1") == "2024.09.01"

    def test_slash_format(self):
        assert _normalize_date("2024/12/31") == "2024.12.31"

    def test_korean_format(self):
        assert _normalize_date("2024년 9월 28일") == "2024.09.28"

    def test_empty(self):
        assert _normalize_date("") == ""

    def test_none_string(self):
        assert _normalize_date("None") == ""

    def test_null_string(self):
        assert _normalize_date("null") == ""


# ------------------------------------------------------------------
# _normalize_instrument
# ------------------------------------------------------------------


class TestNormalizeInstrument:
    def test_valid_cb(self):
        item = {
            "category": "CB",
            "series": 1,
            "kind": "제1회 전환사채",
            "face_value": 10_000_000_000,
            "convertible_shares": 1_000_000,
            "conversion_price": 10_000,
            "exercise_start": "2024.01.01",
            "exercise_end": "2027.12.31",
            "active": True,
        }
        result = _normalize_instrument(item)
        assert result is not None
        assert result["category"] == "CB"

    def test_maps_korean_cb_category(self):
        item = {
            "category": "전환사채",
            "series": 1,
            "kind": "제1회",
            "face_value": 1_000_000,
            "convertible_shares": 100,
            "conversion_price": 10_000,
        }
        result = _normalize_instrument(item)
        assert result["category"] == "CB"

    def test_returns_none_for_unknown_category(self):
        item = {"category": "채권", "face_value": 1000, "convertible_shares": 10}
        assert _normalize_instrument(item) is None

    def test_returns_none_for_zero_values(self):
        item = {
            "category": "CB",
            "series": 1,
            "kind": "test",
            "face_value": 0,
            "convertible_shares": 0,
        }
        assert _normalize_instrument(item) is None

    def test_calculates_shares_when_missing(self):
        item = {
            "category": "CB",
            "series": 1,
            "kind": "제1회 전환사채",
            "face_value": 10_000_000_000,
            "convertible_shares": 0,
            "conversion_price": 5_000,
        }
        result = _normalize_instrument(item)
        assert result["convertible_shares"] == 2_000_000

    def test_valid_so(self):
        item = {
            "category": "SO",
            "series": 3,
            "kind": "주식매수선택권 3차",
            "face_value": 0,
            "convertible_shares": 100_000,
            "conversion_price": 12_000,
            "exercise_start": "2024.06.01",
            "exercise_end": "2028.05.31",
            "active": True,
        }
        result = _normalize_instrument(item)
        assert result is not None
        assert result["category"] == "SO"
        assert result["face_value"] == 0
        assert result["convertible_shares"] == 100_000
        assert result["conversion_price"] == 12_000

    def test_so_zero_shares_returns_none(self):
        item = {
            "category": "SO",
            "series": 1,
            "kind": "주식매수선택권 1차",
            "face_value": 0,
            "convertible_shares": 0,
            "conversion_price": 5_000,
        }
        assert _normalize_instrument(item) is None

    def test_maps_korean_so_category(self):
        item = {
            "category": "주식매수선택권",
            "series": 2,
            "kind": "주식매수선택권 2차",
            "face_value": 0,
            "convertible_shares": 75_000,
            "conversion_price": 6_000,
        }
        result = _normalize_instrument(item)
        assert result["category"] == "SO"


# ------------------------------------------------------------------
# parse_notes_overhang_llm (mocked LLM)
# ------------------------------------------------------------------


class TestParseNotesOverhangLlm:
    """Integration test with mocked OpenAI client."""

    def _make_html(self) -> str:
        return """<html><body>
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table>
        <tr><td>발행총액</td><td>15,400,000,000원</td></tr>
        <tr><td>전환가격(원/주)</td><td>7,875</td></tr>
        <tr><td>전환에 따라 발행할 주식수</td><td>1,955,555주</td></tr>
        <tr><td>전환청구기간</td><td>2024.09.28 ~ 2027.08.28</td></tr>
        </table>
        <p>14. 기타</p>
        </body></html>"""

    @patch("auto_reports.parsers.notes_overhang_llm.OpenAI")
    def test_successful_extraction(self, mock_openai_cls):
        llm_response = json.dumps({
            "instruments": [{
                "category": "CB",
                "series": 1,
                "kind": "제1회 전환사채",
                "face_value": 15_400_000_000,
                "convertible_shares": 1_955_555,
                "conversion_price": 7_875,
                "exercise_start": "2024.09.28",
                "exercise_end": "2027.08.28",
                "active": True,
            }],
        })
        mock_msg = MagicMock()
        mock_msg.content = llm_response
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        results = parse_notes_overhang_llm(
            self._make_html(), api_key="test-key", model="gpt-4.1-mini",
        )

        assert len(results) == 1
        assert results[0]["category"] == "CB"
        assert results[0]["series"] == 1
        assert results[0]["face_value"] == 15_400_000_000
        assert results[0]["convertible_shares"] == 1_955_555

    @patch("auto_reports.parsers.notes_overhang_llm.OpenAI")
    def test_llm_error_returns_empty(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_cls.return_value = mock_client

        results = parse_notes_overhang_llm(
            self._make_html(), api_key="test-key",
        )
        assert results == []

    def test_no_context_returns_empty(self):
        html = "<html><body><p>1. 회사의 개요</p></body></html>"
        # Even with a valid api_key, no overhang context → empty result
        # (won't reach LLM because extract_overhang_context returns "")
        results = parse_notes_overhang_llm(html, api_key="test-key")
        assert results == []


# ------------------------------------------------------------------
# parse_notes_overhang with LLM integration
# ------------------------------------------------------------------


class TestParseNotesOverhangWithLlm:
    """Test the main parse_notes_overhang function with LLM integration."""

    def _make_html(self) -> str:
        return """<html><body>
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table>
        <tr><td>발행총액</td><td>10,000,000,000원</td></tr>
        <tr><td>전환가격(원/주)</td><td>10,000</td></tr>
        <tr><td>전환청구기간</td><td>2024.01.01 ~ 2027.12.31</td></tr>
        </table>
        <p>14. 기타</p>
        </body></html>"""

    def test_without_api_key_returns_empty(self):
        """Without api_key, LLM-only mode returns empty list."""
        from auto_reports.parsers.notes_overhang import parse_notes_overhang

        results = parse_notes_overhang(self._make_html())
        # LLM-only mode: no api_key → empty list (regex fallback removed)
        assert results == []

    @patch("auto_reports.parsers.notes_overhang_llm.OpenAI")
    def test_with_api_key_tries_llm_first(self, mock_openai_cls):
        """With api_key, should try LLM first."""
        from auto_reports.parsers.notes_overhang import parse_notes_overhang

        llm_response = json.dumps({
            "instruments": [{
                "category": "CB",
                "series": 1,
                "kind": "제1회 전환사채 (LLM)",
                "face_value": 10_000_000_000,
                "convertible_shares": 1_000_000,
                "conversion_price": 10_000,
                "exercise_start": "2024.01.01",
                "exercise_end": "2027.12.31",
                "active": True,
            }],
        })
        mock_msg = MagicMock()
        mock_msg.content = llm_response
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        results = parse_notes_overhang(
            self._make_html(), api_key="test-key", model="gpt-4.1-mini",
        )

        assert len(results) == 1
        # Should have the LLM-specific kind text
        assert "LLM" in results[0]["kind"]

    @patch("auto_reports.parsers.notes_overhang_llm.OpenAI")
    def test_llm_failure_returns_empty(self, mock_openai_cls):
        """When LLM fails, LLM-only mode returns empty (no regex fallback)."""
        from auto_reports.parsers.notes_overhang import parse_notes_overhang

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_cls.return_value = mock_client

        results = parse_notes_overhang(
            self._make_html(), api_key="test-key",
        )

        # LLM-only mode: no regex fallback, returns empty
        assert results == []

    @patch("auto_reports.parsers.notes_overhang_llm.OpenAI")
    def test_llm_empty_result_returns_empty(self, mock_openai_cls):
        """When LLM returns empty instruments, LLM-only mode returns empty."""
        from auto_reports.parsers.notes_overhang import parse_notes_overhang

        llm_response = json.dumps({"instruments": []})
        mock_msg = MagicMock()
        mock_msg.content = llm_response
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        results = parse_notes_overhang(
            self._make_html(), api_key="test-key",
        )

        # LLM-only mode: no regex fallback
        assert results == []
