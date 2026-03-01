"""Tests for LLM-based disclosure summarizers and pipeline integration helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

from auto_reports.models.disclosure import CBIssuance, ConversionTerms


# ──────────────────────────────────────────────────────────────────
# exchange_llm_summarizer tests
# ──────────────────────────────────────────────────────────────────


class TestParseJsonResponse:
    """Tests for _parse_json_response in exchange_llm_summarizer."""

    def test_plain_json(self):
        from auto_reports.summarizers.exchange_llm_summarizer import _parse_json_response
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fence(self):
        from auto_reports.summarizers.exchange_llm_summarizer import _parse_json_response
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_null_response(self):
        from auto_reports.summarizers.exchange_llm_summarizer import _parse_json_response
        assert _parse_json_response("null") is None

    def test_empty_string(self):
        from auto_reports.summarizers.exchange_llm_summarizer import _parse_json_response
        assert _parse_json_response("") is None

    def test_non_dict_response(self):
        from auto_reports.summarizers.exchange_llm_summarizer import _parse_json_response
        assert _parse_json_response("[1, 2, 3]") is None

    def test_invalid_json(self):
        from auto_reports.summarizers.exchange_llm_summarizer import _parse_json_response
        assert _parse_json_response("not json at all") is None


class TestExtractExchangePerformance:
    """Tests for extract_exchange_performance."""

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_success(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_performance
        mock_llm.return_value = json.dumps({
            "statement_type": "연결",
            "period": {"- 시작일": "2024년 01월 01일", "- 종료일": "2024년 12월 31일"},
            "income_changes": {
                "- 매출액": {"당해사업연도": "100000000", "직전사업연도": "80000000", "증감비율(%)": "25.0"},
                "- 영업이익": {"당해사업연도": "20000000", "직전사업연도": "15000000", "증감비율(%)": "33.3"},
                "- 당기순이익": {"당해사업연도": "15000000", "직전사업연도": "10000000", "증감비율(%)": "50.0"},
            },
        })
        result = extract_exchange_performance("sample text", "api_key")
        assert result is not None
        assert "income_changes" in result
        assert result["statement_type"] == "연결"

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_returns_none_on_empty_response(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_performance
        mock_llm.return_value = ""
        assert extract_exchange_performance("text", "key") is None

    def test_returns_none_without_api_key(self):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_performance
        assert extract_exchange_performance("text", "") is None

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_returns_none_without_income_changes(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_performance
        mock_llm.return_value = json.dumps({"statement_type": "연결"})
        assert extract_exchange_performance("text", "key") is None


class TestExtractExchangeContract:
    """Tests for extract_exchange_contract."""

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_success(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_contract
        mock_llm.return_value = json.dumps({
            "type": "단일판매ㆍ공급계약체결",
            "description": "반도체 장비 공급",
            "detail": "식각 장비",
            "contract_amount": 50000000000,
            "counterparty": "삼성전자",
            "revenue_ratio_pct": 15.3,
            "contract_date": "2024-06-15",
        })
        result = extract_exchange_contract("sample text", "api_key")
        assert result is not None
        assert result["type"] == "단일판매ㆍ공급계약체결"
        assert result["contract_amount"] == 50000000000

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_returns_none_without_type(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_contract
        mock_llm.return_value = json.dumps({"description": "something"})
        assert extract_exchange_contract("text", "key") is None


class TestExtractExchangeOverhang:
    """Tests for extract_exchange_overhang."""

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_exercise(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_overhang
        mock_llm.return_value = json.dumps({
            "category": "exercise",
            "data": {
                "type": "전환청구권행사",
                "cumulative_shares": 1000000,
                "total_shares": 50000000,
                "ratio_pct": 2.0,
            },
        })
        result = extract_exchange_overhang("text", "key")
        assert result is not None
        cat, data = result
        assert cat == "exercise"
        assert data["cumulative_shares"] == 1000000

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_price_adj(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_overhang
        mock_llm.return_value = json.dumps({
            "category": "price_adj",
            "data": {
                "type": "전환가액의조정",
                "adjustments": [
                    {"series": 1, "price_before": 5000, "price_after": 4500},
                ],
                "reason": "시가하락",
            },
        })
        result = extract_exchange_overhang("text", "key")
        assert result is not None
        cat, data = result
        assert cat == "price_adj"
        assert data["adjustments"][0]["price_after"] == 4500

    @patch("auto_reports.summarizers.exchange_llm_summarizer._call_llm")
    def test_invalid_category(self, mock_llm):
        from auto_reports.summarizers.exchange_llm_summarizer import extract_exchange_overhang
        mock_llm.return_value = json.dumps({
            "category": "invalid",
            "data": {"type": "something"},
        })
        assert extract_exchange_overhang("text", "key") is None


# ──────────────────────────────────────────────────────────────────
# disclosure_llm_summarizer tests
# ──────────────────────────────────────────────────────────────────


class TestDisclosureParseJsonResponse:
    """Tests for _parse_json_response in disclosure_llm_summarizer."""

    def test_plain_json(self):
        from auto_reports.summarizers.disclosure_llm_summarizer import _parse_json_response
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fence(self):
        from auto_reports.summarizers.disclosure_llm_summarizer import _parse_json_response
        result = _parse_json_response('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_empty(self):
        from auto_reports.summarizers.disclosure_llm_summarizer import _parse_json_response
        assert _parse_json_response("") is None


class TestExtractRightsIssue:
    """Tests for extract_rights_issue."""

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_success(self, mock_llm):
        from auto_reports.summarizers.disclosure_llm_summarizer import extract_rights_issue
        mock_llm.return_value = json.dumps({
            "type": "유상증자결정",
            "new_shares": {"보통주식": 5000000, "기타주식": 0},
            "issue_price": 3000,
            "payment_date": "2024-08-15",
            "board_decision_date": "2024-06-01",
        })
        result = extract_rights_issue("pdf text", "api_key")
        assert result is not None
        assert result["type"] == "유상증자결정"
        assert result["issue_price"] == 3000

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_returns_none_on_failure(self, mock_llm):
        from auto_reports.summarizers.disclosure_llm_summarizer import extract_rights_issue
        mock_llm.return_value = ""
        assert extract_rights_issue("text", "key") is None


class TestExtractCbIssuance:
    """Tests for extract_cb_issuance."""

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_success(self, mock_llm):
        from auto_reports.summarizers.disclosure_llm_summarizer import extract_cb_issuance
        mock_llm.return_value = json.dumps({
            "issuance_type_name": "전환사채권 발행결정",
            "bond_type": {"회차": 3, "종류": "무기명식 무보증 사모 전환사채"},
            "face_value": 10000000000,
            "conversion_price": 5000,
            "convertible_shares": 2000000,
            "share_ratio_pct": 4.0,
            "exercise_period": {"시작일": "2025-01-01", "종료일": "2027-12-31"},
        })
        result = extract_cb_issuance("text", "key")
        assert result is not None
        assert result["face_value"] == 10000000000
        assert result["conversion_price"] == 5000


class TestClassifyDisclosureType:
    """Tests for classify_disclosure_type."""

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_rights_issue(self, mock_llm):
        from auto_reports.summarizers.disclosure_llm_summarizer import classify_disclosure_type
        mock_llm.return_value = json.dumps({"disclosure_type": "유상증자결정"})
        assert classify_disclosure_type("text", "key") == "유상증자결정"

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_cb_issuance(self, mock_llm):
        from auto_reports.summarizers.disclosure_llm_summarizer import classify_disclosure_type
        mock_llm.return_value = json.dumps({"disclosure_type": "전환사채권 발행결정"})
        assert classify_disclosure_type("text", "key") == "전환사채권 발행결정"

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_invalid_returns_default(self, mock_llm):
        from auto_reports.summarizers.disclosure_llm_summarizer import classify_disclosure_type
        mock_llm.return_value = json.dumps({"disclosure_type": "알수없음"})
        assert classify_disclosure_type("text", "key") == "기타"

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_empty_returns_default(self, mock_llm):
        from auto_reports.summarizers.disclosure_llm_summarizer import classify_disclosure_type
        mock_llm.return_value = ""
        assert classify_disclosure_type("text", "key") == "기타"


# ──────────────────────────────────────────────────────────────────
# Pipeline helper tests
# ──────────────────────────────────────────────────────────────────


class TestIsSparseRightsIssue:
    """Tests for _is_sparse_rights_issue helper."""

    def test_empty_dict(self):
        from auto_reports.pipeline import _is_sparse_rights_issue
        assert _is_sparse_rights_issue({}) is True

    def test_none(self):
        from auto_reports.pipeline import _is_sparse_rights_issue
        assert _is_sparse_rights_issue(None) is True

    def test_sparse(self):
        from auto_reports.pipeline import _is_sparse_rights_issue
        assert _is_sparse_rights_issue({"type": "유상증자결정"}) is True

    def test_has_one_key(self):
        from auto_reports.pipeline import _is_sparse_rights_issue
        assert _is_sparse_rights_issue({"issue_price": 3000}) is True

    def test_not_sparse(self):
        from auto_reports.pipeline import _is_sparse_rights_issue
        data = {
            "new_shares": {"보통주식": 5000000},
            "issue_price": 3000,
            "payment_date": "2024-08-15",
        }
        assert _is_sparse_rights_issue(data) is False


class TestIsSparseIssuance:
    """Tests for _is_sparse_cb_issuance helper."""

    def test_empty_issuance(self):
        from auto_reports.pipeline import _is_sparse_cb_issuance
        cb = CBIssuance()
        assert _is_sparse_cb_issuance(cb) is True

    def test_has_face_value(self):
        from auto_reports.pipeline import _is_sparse_cb_issuance
        cb = CBIssuance(face_value=10000000000)
        assert _is_sparse_cb_issuance(cb) is False

    def test_has_conversion_terms(self):
        from auto_reports.pipeline import _is_sparse_cb_issuance
        cb = CBIssuance(
            conversion_terms=ConversionTerms(conversion_price=5000),
        )
        assert _is_sparse_cb_issuance(cb) is False


class TestTryCbIssuanceLlmFallback:
    """Tests for _try_cb_issuance_llm_fallback helper."""

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_success(self, mock_llm):
        from auto_reports.pipeline import _try_cb_issuance_llm_fallback
        mock_llm.return_value = json.dumps({
            "issuance_type_name": "전환사채권 발행결정",
            "bond_type": {"회차": 2, "종류": "무기명식 사모 전환사채"},
            "face_value": 5000000000,
            "conversion_price": 4000,
            "convertible_shares": 1250000,
            "share_ratio_pct": 3.5,
            "exercise_period": {"시작일": "2025-06-01", "종료일": "2028-05-31"},
        })
        result = _try_cb_issuance_llm_fallback("text", "key", "")
        assert isinstance(result, CBIssuance)
        assert result.issuance_type_name == "전환사채권 발행결정"
        assert result.face_value == 5000000000
        assert result.bond_type is not None
        assert result.bond_type.series == 2
        assert result.conversion_terms is not None
        assert result.conversion_terms.conversion_price == 4000
        assert result.conversion_terms.share_ratio == 3.5

    @patch("auto_reports.summarizers.disclosure_llm_summarizer._call_llm")
    def test_returns_none_on_failure(self, mock_llm):
        from auto_reports.pipeline import _try_cb_issuance_llm_fallback
        mock_llm.return_value = ""
        assert _try_cb_issuance_llm_fallback("text", "key", "") is None


class TestTryExchangeLlmFallback:
    """Tests for _try_exchange_llm_fallback helper."""

    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_performance")
    def test_performance_match(self, mock_perf):
        from auto_reports.pipeline import _try_exchange_llm_fallback
        mock_perf.return_value = {
            "income_changes": {"- 매출액": {"당해사업연도": "100"}},
        }
        result = _try_exchange_llm_fallback("text", "key", "")
        assert result is not None
        cat, data = result
        assert cat == "sales"

    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_performance")
    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_contract")
    def test_contract_match(self, mock_contract, mock_perf):
        from auto_reports.pipeline import _try_exchange_llm_fallback
        mock_perf.return_value = None
        mock_contract.return_value = {"type": "단일판매ㆍ공급계약체결"}
        result = _try_exchange_llm_fallback("text", "key", "")
        assert result is not None
        cat, data = result
        assert cat == "backlog"

    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_performance")
    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_contract")
    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_overhang")
    def test_overhang_match(self, mock_oh, mock_contract, mock_perf):
        from auto_reports.pipeline import _try_exchange_llm_fallback
        mock_perf.return_value = None
        mock_contract.return_value = None
        mock_oh.return_value = ("exercise", {"type": "전환청구권행사"})
        result = _try_exchange_llm_fallback("text", "key", "")
        assert result is not None
        cat, data = result
        assert cat == "exercise"

    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_performance")
    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_contract")
    @patch("auto_reports.summarizers.exchange_llm_summarizer.extract_exchange_overhang")
    def test_no_match(self, mock_oh, mock_contract, mock_perf):
        from auto_reports.pipeline import _try_exchange_llm_fallback
        mock_perf.return_value = None
        mock_contract.return_value = None
        mock_oh.return_value = None
        result = _try_exchange_llm_fallback("text", "key", "")
        assert result is None
