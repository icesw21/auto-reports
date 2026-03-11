"""Tests for exchange disclosure sales parsers (잠정실적, 실적전망)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_reports.parsers.exchange_disclosure import (
    parse_exchange_preliminary_results,
    parse_exchange_forecast,
    parse_exchange_disclosure_from_soup,
    _classify,
)
from bs4 import BeautifulSoup

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / (
    "src/auto_reports/parsers/example/exchange_disclosure/sales"
)


def _load_soup(filename: str) -> BeautifulSoup:
    """Load an HTML example and return decoded soup."""
    html_path = EXAMPLE_DIR / filename
    html_bytes = html_path.read_bytes()
    decoded = html_bytes.decode("euc-kr", errors="replace")
    return BeautifulSoup(decoded, "html.parser")


def _load_expected(filename: str) -> dict:
    """Load expected JSON output."""
    json_path = EXAMPLE_DIR / filename
    return json.loads(json_path.read_text("utf-8"))


# ------------------------------------------------------------------
# Classification tests
# ------------------------------------------------------------------


class TestClassification:
    """Test _classify routing for preliminary and forecast."""

    def test_classify_preliminary(self):
        soup = _load_soup("연결재무제표기준영업(잠정)실적.html")
        assert _classify(soup) == "preliminary"

    def test_classify_forecast(self):
        soup = _load_soup("연결재무제표기준영업실적등에대한전망(공정공시).html")
        assert _classify(soup) == "forecast"

    def test_from_soup_routes_preliminary(self):
        soup = _load_soup("연결재무제표기준영업(잠정)실적.html")
        category, result = parse_exchange_disclosure_from_soup(soup)
        assert category == "preliminary"
        assert result["type"] == "잠정실적"

    def test_from_soup_routes_forecast(self):
        soup = _load_soup("연결재무제표기준영업실적등에대한전망(공정공시).html")
        category, result = parse_exchange_disclosure_from_soup(soup)
        assert category == "forecast"
        assert result["type"] == "실적전망"


# ------------------------------------------------------------------
# 잠정실적 parser tests
# ------------------------------------------------------------------


class TestPreliminaryResults:
    """Test parse_exchange_preliminary_results against example HTML."""

    @pytest.fixture()
    def result(self) -> dict:
        soup = _load_soup("연결재무제표기준영업(잠정)실적.html")
        return parse_exchange_preliminary_results(soup)

    @pytest.fixture()
    def expected(self) -> dict:
        return _load_expected("연결재무제표기준영업(잠정)실적.json")

    def test_type_and_statement(self, result):
        assert result["type"] == "잠정실적"
        assert result["statement_type"] == "연결"

    def test_unit(self, result):
        assert result["unit"] == "백만원"

    def test_all_periods_present(self, result, expected):
        for key in expected["periods"]:
            assert key in result["periods"], f"Missing period: {key}"

    def test_period_dates(self, result, expected):
        for key, exp_period in expected["periods"].items():
            actual = result["periods"][key]
            assert actual["start"] == exp_period["start"]
            assert actual["end"] == exp_period["end"]

    def test_period_labels(self, result):
        assert result["periods"]["당해실적"]["label"] == "2025.4Q"
        assert result["periods"]["전분기실적"]["label"] == "2025.3Q"
        assert result["periods"]["전년동기실적"]["label"] == "2024.4Q"

    def test_quarterly_items_count(self, result):
        assert len(result["quarterly"]) == 5

    def test_quarterly_revenue(self, result):
        rev = result["quarterly"]["매출액"]
        assert rev["당해실적"] == 86643
        assert rev["전분기실적"] == 82833
        assert rev["전분기대비증감율"] == 4.60
        assert rev["전년동기실적"] == 79596
        assert rev["전년동기대비증감율"] == 8.85

    def test_quarterly_operating_income(self, result):
        op = result["quarterly"]["영업이익"]
        assert op["당해실적"] == 16270
        assert op["전년동기대비증감율"] == 29.18

    def test_quarterly_net_income_turnaround(self, result):
        ni = result["quarterly"]["당기순이익"]
        assert ni["당해실적"] == 14084
        assert ni["전년동기실적"] == -439
        assert ni["전년동기대비증감율"] is None
        assert ni["흑자적자전환"] == "흑자전환"

    def test_quarterly_controlling_shareholder(self, result):
        assert "지배주주당기순이익" in result["quarterly"]
        csi = result["quarterly"]["지배주주당기순이익"]
        assert csi["당해실적"] == 14084

    def test_cumulative_items_count(self, result):
        assert len(result["cumulative"]) == 5

    def test_cumulative_revenue(self, result):
        rev = result["cumulative"]["매출액"]
        assert rev["당해실적"] == 322467
        assert rev["전년동기실적"] == 310715
        assert rev["전년동기대비증감율"] == 3.78

    def test_cumulative_net_income(self, result):
        ni = result["cumulative"]["당기순이익"]
        assert ni["당해실적"] == 48701
        assert ni["전년동기대비증감율"] == 53.11

    def test_all_values_match_expected(self, result, expected):
        """Cross-check all quarterly and cumulative values against expected JSON."""
        for section in ["quarterly", "cumulative"]:
            for item, exp_data in expected[section].items():
                assert item in result[section], f"Missing {section}.{item}"
                for field, exp_val in exp_data.items():
                    actual_val = result[section][item].get(field)
                    assert actual_val == exp_val, (
                        f"{section}.{item}.{field}: {actual_val} != {exp_val}"
                    )


# ------------------------------------------------------------------
# 실적전망 parser tests
# ------------------------------------------------------------------


class TestForecast:
    """Test parse_exchange_forecast against example HTML."""

    @pytest.fixture()
    def result(self) -> dict:
        soup = _load_soup("연결재무제표기준영업실적등에대한전망(공정공시).html")
        return parse_exchange_forecast(soup)

    @pytest.fixture()
    def expected(self) -> dict:
        return _load_expected("연결재무제표기준영업실적등에대한전망(공정공시).json")

    def test_type_and_statement(self, result):
        assert result["type"] == "실적전망"
        assert result["statement_type"] == "연결"

    def test_unit(self, result):
        assert result["unit"] == "억원"

    def test_forecasts_count(self, result):
        assert len(result["forecasts"]) == 3

    def test_forecast_years(self, result):
        years = [f["year"] for f in result["forecasts"]]
        assert years == [2026, 2027, 2028]

    def test_forecast_2026_revenue(self, result):
        fc = result["forecasts"][0]
        assert fc["year"] == 2026
        assert fc["매출액"] == 6122

    def test_forecast_2026_period(self, result):
        fc = result["forecasts"][0]
        assert fc["period"]["start"] == "2026-01-01"
        assert fc["period"]["end"] == "2026-12-31"

    def test_forecast_null_items(self, result):
        fc = result["forecasts"][0]
        assert fc["영업이익"] is None
        assert fc["당기순이익"] is None

    def test_assumptions(self, result):
        assert result["assumptions"] is not None
        assert len(result["assumptions"]) > 10

    def test_comparison_exists(self, result):
        assert "prior_forecast_comparison" in result
        comp = result["prior_forecast_comparison"]
        assert comp["period"]["start"] == "2025-01-01"
        assert comp["period"]["end"] == "2025-12-31"

    def test_comparison_revenue(self, result):
        comp = result["prior_forecast_comparison"]
        rev = comp["items"]["매출액"]
        assert rev["전망"] == 5329
        assert rev["실적"] == 4912
        assert rev["괴리율"] == -7.8

    def test_comparison_null_items(self, result):
        comp = result["prior_forecast_comparison"]
        op = comp["items"]["영업이익"]
        assert op["전망"] is None
        assert op["실적"] is None
        assert op["괴리율"] is None

    def test_all_forecasts_match_expected(self, result, expected):
        """Cross-check all forecast values against expected JSON."""
        for i, exp_fc in enumerate(expected["forecasts"]):
            actual_fc = result["forecasts"][i]
            for key, exp_val in exp_fc.items():
                actual_val = actual_fc.get(key)
                assert actual_val == exp_val, (
                    f"forecasts[{i}].{key}: {actual_val} != {exp_val}"
                )


# ------------------------------------------------------------------
# Pipeline integration test
# ------------------------------------------------------------------


class TestPreliminaryIntegration:
    """Test _integrate_preliminary_results conversion to IncomeStatementItem."""

    def test_quarterly_conversion(self):
        from auto_reports.pipeline import _integrate_preliminary_results

        prelim = {
            "unit": "백만원",
            "periods": {
                "당해실적": {"start": "2025-10-01", "end": "2025-12-31", "label": "2025.4Q"},
            },
            "quarterly": {
                "매출액": {"당해실적": 86643},
                "영업이익": {"당해실적": 16270},
                "당기순이익": {"당해실적": 14084},
            },
            "cumulative": {},
        }
        quarterly: list = []
        annual: list = []
        _integrate_preliminary_results([prelim], annual, quarterly)

        assert len(quarterly) == 1
        q = quarterly[0]
        assert q.period == "2025.Q4"
        assert q.revenue == 86_643_000_000
        assert q.operating_income == 16_270_000_000
        assert q.net_income == 14_084_000_000

    def test_annual_conversion(self):
        from auto_reports.pipeline import _integrate_preliminary_results

        prelim = {
            "unit": "억원",
            "periods": {
                "당해실적": {"start": "2025-10-01", "end": "2025-12-31", "label": "2025.4Q"},
                "누적당해실적": {"start": "2025-01-01", "end": "2025-12-31"},
            },
            "quarterly": {
                "매출액": {"당해실적": 1000},
                "영업이익": {"당해실적": 200},
                "당기순이익": {"당해실적": 150},
            },
            "cumulative": {
                "매출액": {"당해실적": 4000},
                "영업이익": {"당해실적": 800},
                "당기순이익": {"당해실적": 600},
            },
        }
        quarterly: list = []
        annual: list = []
        _integrate_preliminary_results([prelim], annual, quarterly)

        assert len(annual) == 1
        a = annual[0]
        assert a.period == "2025"
        # 억원 → 원: × 100,000,000
        assert a.revenue == 400_000_000_000
        assert a.operating_income == 80_000_000_000
        assert a.net_income == 60_000_000_000

    def test_skip_existing(self):
        from auto_reports.pipeline import _integrate_preliminary_results
        from auto_reports.models.financial import IncomeStatementItem

        # Pre-existing quarterly statement
        existing = IncomeStatementItem(
            period="2025.Q4", revenue=999, operating_income=111, net_income=222,
        )
        prelim = {
            "unit": "백만원",
            "periods": {
                "당해실적": {"start": "2025-10-01", "end": "2025-12-31", "label": "2025.4Q"},
            },
            "quarterly": {
                "매출액": {"당해실적": 86643},
                "영업이익": {"당해실적": 16270},
                "당기순이익": {"당해실적": 14084},
            },
            "cumulative": {},
        }
        quarterly = [existing]
        annual: list = []
        _integrate_preliminary_results([prelim], annual, quarterly)

        # Should not overwrite existing
        assert len(quarterly) == 1
        assert quarterly[0].revenue == 999
