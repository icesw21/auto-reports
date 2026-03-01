"""Tests for financial analyzers."""

import pytest
from bs4 import BeautifulSoup

from auto_reports.analyzers.financial import (
    build_cumulative_annual_row,
    calc_yoy_change,
    format_eok,
    format_income_cell,
    to_eok,
)
from auto_reports.analyzers.overhang import OverhangAnalyzer
from auto_reports.fetchers.dart_report import _IS_FIELDS
from auto_reports.fetchers.opendart import OpenDartFetcher, _IS_FIELD_NAMES, _detect_unit_scale, _parse_amount, _subtract_income
from auto_reports.models.disclosure import CBIssuance, Performance
from auto_reports.parsers.classifier import DisclosureType
from auto_reports.models.financial import IncomeStatementItem
from auto_reports.parsers.base import table_to_dict
from auto_reports.pipeline import (
    _detect_performance_period,
    _extract_date_from_url,
    _integrate_performance_disclosures,
)


class TestToEok:
    def test_normal(self):
        assert to_eok(109_500_000_000) == 1095

    def test_rounding(self):
        assert to_eok(150_000_000) == 2  # rounds to nearest

    def test_none(self):
        assert to_eok(None) is None


class TestFormatEok:
    def test_with_comma(self):
        assert format_eok(109_500_000_000) == "1,095"

    def test_none(self):
        assert format_eok(None) == "-"


class TestYoYChange:
    def test_positive_growth(self):
        assert calc_yoy_change(4245_0000_0000, 1129_0000_0000) == "+276%"

    def test_negative_growth(self):
        assert calc_yoy_change(624_0000_0000, 741_0000_0000) == "-16%"

    def test_turnaround_to_profit(self):
        assert calc_yoy_change(996_0000_0000, -281_0000_0000) == "흑자전환"

    def test_turnaround_to_loss(self):
        assert calc_yoy_change(-92_0000_0000, 44_0000_0000) == "적자전환"

    def test_loss_reduction(self):
        assert calc_yoy_change(-92_0000_0000, -102_0000_0000) == "적자축소"

    def test_loss_expansion(self):
        assert calc_yoy_change(-281_0000_0000, -92_0000_0000) == "적자확대"

    def test_none_input(self):
        assert calc_yoy_change(None, 100) == "-"
        assert calc_yoy_change(100, None) == "-"


class TestFormatIncomeCell:
    def test_normal(self):
        assert format_income_cell(4245_0000_0000, "+276%") == "4,245 (+276%)"

    def test_none(self):
        assert format_income_cell(None, "+10%") == "-"


class TestOverhangAnalyzer:
    def test_issuance_and_items(self):
        cb = CBIssuance(
            issuance_type_name="전환사채권 발행결정",
            **{
                "1. 사채의 종류": {"회차": 1, "종류": "무기명식 사모 전환사채"},
                "2. 사채의 권면(전자등록)총액 (원)": 25_900_000_000,
                "9. 전환에 관한 사항": {
                    "전환가액 (원/주)": 49_591,
                    "전환에_따라_발행할_주식": {
                        "종류": "보통주",
                        "주식수": 522_279,
                        "주식총수 대비 비율(%)": 5.44,
                    },
                    "전환청구기간": {"시작일": "2025-01-01", "종료일": "2027-12-31"},
                },
            },
        )
        analyzer = OverhangAnalyzer(total_shares=9_602_955)
        analyzer.process_issuance(cb)
        items = analyzer.get_overhang_items()

        assert len(items) == 1
        assert "전환사채(CB)" in items[0].category
        assert items[0].exercise_price == "49,591원"
        assert items[0].dilution_ratio is not None

    def test_total_dilution(self):
        cb = CBIssuance(
            issuance_type_name="전환사채권 발행결정",
            **{
                "1. 사채의 종류": {"회차": 1, "종류": "전환사채"},
                "2. 사채의 권면(전자등록)총액 (원)": 10_000_000_000,
                "9. 전환에 관한 사항": {
                    "전환가액 (원/주)": 10_000,
                    "전환에_따라_발행할_주식": {"종류": "보통주", "주식수": 1_000_000, "주식총수 대비 비율(%)": 10.0},
                    "전환청구기간": {"시작일": "2025-01-01", "종료일": "2027-12-31"},
                },
            },
        )
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        analyzer.process_issuance(cb)
        assert analyzer.get_total_dilution() == pytest.approx(10.0)

    def test_process_event_cb(self):
        """OpenDART event API CB data should create overhang instrument."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        event = {
            "category": "CB",
            "series": 3,
            "kind": "무기명식 사모 전환사채",
            "face_value": 5_000_000_000,
            "conversion_price": 10_000,
            "convertible_shares": 500_000,
            "share_type": "보통주",
        }
        analyzer.process_event(event)
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert "전환사채(CB)" in items[0].category
        assert items[0].exercise_price == "10,000원"
        assert analyzer.get_total_dilution() == pytest.approx(5.0)

    def test_process_event_skips_existing(self):
        """Event API data should not overwrite existing HTML disclosure data."""
        cb = CBIssuance(
            issuance_type_name="전환사채권 발행결정",
            **{
                "1. 사채의 종류": {"회차": 1, "종류": "전환사채"},
                "2. 사채의 권면(전자등록)총액 (원)": 10_000_000_000,
                "9. 전환에 관한 사항": {
                    "전환가액 (원/주)": 10_000,
                    "전환에_따라_발행할_주식": {"종류": "보통주", "주식수": 1_000_000, "주식총수 대비 비율(%)": 10.0},
                    "전환청구기간": {"시작일": "2025-01-01", "종료일": "2027-12-31"},
                },
            },
        )
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        analyzer.process_issuance(cb)

        # Now process an event for the same CB series - should be skipped
        event = {
            "category": "CB",
            "series": 1,
            "face_value": 5_000_000_000,
            "conversion_price": 5_000,
            "convertible_shares": 1_000_000,
        }
        analyzer.process_event(event)

        # Should still have original data, not event data
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert items[0].exercise_price == "10,000원"

    def test_process_event_enriches_incomplete_html(self):
        """API data should enrich existing instrument when HTML had incomplete data."""
        # Create CB from HTML with face_value but no conversion terms
        cb = CBIssuance(
            issuance_type_name="전환사채권 발행결정",
            **{
                "1. 사채의 종류": {"회차": 3, "종류": "전환사채"},
                "2. 사채의 권면(전자등록)총액 (원)": 25_200_000_000,
            },
        )
        analyzer = OverhangAnalyzer(total_shares=19_925_510)
        analyzer.process_issuance(cb)

        # Instrument created but with 0 conversion_price and 0 shares
        items_before = analyzer.get_overhang_items()
        assert len(items_before) == 1
        assert items_before[0].exercise_price is None  # no price
        assert items_before[0].dilution_ratio is None  # no shares → no dilution

        # API provides the missing conversion data
        event = {
            "category": "CB",
            "series": 3,
            "face_value": 25_200_000_000,
            "conversion_price": 18_223,
            "convertible_shares": 1_382_867,
            "cv_start": "2026년 10월 28일",
            "cv_end": "2055년 09월 28일",
            "disclosure_date": "20251001",
        }
        analyzer.process_event(event)

        # Now the instrument should be enriched
        items_after = analyzer.get_overhang_items()
        assert len(items_after) == 1
        assert items_after[0].exercise_price == "18,223원"
        assert items_after[0].dilution_ratio is not None
        assert "2026.10.28" in items_after[0].exercise_period
        assert "2055.09.28" in items_after[0].exercise_period

    def test_process_event_rights_issue(self):
        """Capital increase event should create overhang item with 기타주식 label."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        event = {
            "category": "RIGHTS_ISSUE",
            "shares": 2_000_000,
            "issue_price": 5_000,
            "share_type": "전환우선주",
            "face_value": 10_000_000_000,
        }
        analyzer.process_event(event)
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert "전환우선주" in items[0].category
        assert items[0].exercise_price == "5,000원"
        assert "100 억원" in items[0].remaining_amount
        assert analyzer.get_total_dilution() == pytest.approx(20.0)

    def test_rights_issue_enrichment_by_rcept_no(self):
        """LLM PDF event should enrich existing API-sourced RIGHTS_ISSUE by rcept_no."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        # Step 1: API creates instrument (no conversion terms)
        api_event = {
            "category": "RIGHTS_ISSUE",
            "rcept_no": "20251001000123",
            "shares": 592_655,
            "share_type": "전환우선주",
        }
        analyzer.process_event(api_event)
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert items[0].exercise_price is None  # no conversion price yet

        # Step 2: LLM enrichment with same rcept_no adds conversion terms
        llm_event = {
            "category": "RIGHTS_ISSUE",
            "rcept_no": "20251001000123",
            "shares": 592_655,
            "conversion_price": 18_223,
            "cv_start": "2026.11.07",
            "cv_end": "2030.10.28",
        }
        analyzer.process_event(llm_event)
        items = analyzer.get_overhang_items()
        assert len(items) == 1  # no duplicate
        assert items[0].exercise_price == "18,223원"
        assert "2026.11.07" in items[0].exercise_period

    def test_rights_issue_enrichment_by_shares_match(self):
        """LLM PDF event should enrich existing API-sourced RIGHTS_ISSUE by shares match."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        # Step 1: API creates instrument (no conversion terms)
        api_event = {
            "category": "RIGHTS_ISSUE",
            "rcept_no": "20251001000456",
            "shares": 500_000,
        }
        analyzer.process_event(api_event)

        # Step 2: LLM event without rcept_no matches by shares
        llm_event = {
            "category": "RIGHTS_ISSUE",
            "shares": 500_000,
            "conversion_price": 15_000,
            "cv_start": "2026.01.01",
            "cv_end": "2030.12.31",
        }
        analyzer.process_event(llm_event)
        items = analyzer.get_overhang_items()
        assert len(items) == 1  # enriched, not duplicated
        assert items[0].exercise_price == "15,000원"
        assert "2026.01.01" in items[0].exercise_period

    def test_process_event_computes_shares_from_face_value(self):
        """If convertible_shares missing, compute from face_value / conversion_price."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        event = {
            "category": "CB",
            "series": 2,
            "kind": "전환사채",
            "face_value": 10_000_000_000,
            "conversion_price": 50_000,
            "convertible_shares": None,
            "share_type": "보통주",
        }
        analyzer.process_event(event)
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        # 10B / 50K = 200,000 shares
        assert analyzer.get_total_dilution() == pytest.approx(2.0)

    def test_issuance_uses_share_count_from_disclosure(self):
        """process_issuance should use share_count from conversion terms, not calculate."""
        cb = CBIssuance(
            issuance_type_name="전환사채권 발행결정",
            **{
                "1. 사채의 종류": {"회차": 1, "종류": "전환사채"},
                "2. 사채의 권면(전자등록)총액 (원)": 10_000_000_000,
                "9. 전환에 관한 사항": {
                    "전환가액 (원/주)": 11_398,
                    "전환에_따라_발행할_주식": {
                        "종류": "보통주",
                        "주식수": 877_346,  # Actual from disclosure
                    },
                    "전환청구기간": {"시작일": "2023년 09월 28일", "종료일": "2027년 08월 28일"},
                },
            },
        )
        analyzer = OverhangAnalyzer(total_shares=19_925_510)
        analyzer.process_issuance(cb)
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        # Should use 877,346 from disclosure, not 10B // 11,398 = 877,347
        assert "877,346" in items[0].remaining_amount

    def test_issuance_falls_back_to_calculation_without_share_count(self):
        """Without share_count, should calculate from face_value / price."""
        cb = CBIssuance(
            issuance_type_name="전환사채권 발행결정",
            **{
                "1. 사채의 종류": {"회차": 1, "종류": "전환사채"},
                "2. 사채의 권면(전자등록)총액 (원)": 10_000_000_000,
                "9. 전환에 관한 사항": {
                    "전환가액 (원/주)": 10_000,
                    "전환청구기간": {"시작일": "2025-01-01", "종료일": "2027-12-31"},
                },
            },
        )
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        analyzer.process_issuance(cb)
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        # 10B / 10K = 1,000,000
        assert "1,000,000" in items[0].remaining_amount

    def test_process_rights_issue_disclosure(self):
        """process_rights_issue_disclosure should create instrument with conversion details."""
        analyzer = OverhangAnalyzer(total_shares=19_925_510)
        data = {
            "type": "유상증자결정",
            "new_shares": {"기타주식": 276492},
            "issue_price": 10850,
            "conversion": {
                "전환가액(원/주)": 12177,
                "전환주식수": 246361,
                "전환청구기간": {
                    "시작일": "2025-01-03",
                    "종료일": "2028-12-21",
                },
            },
        }
        analyzer.process_rights_issue_disclosure(data, "20231213")
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        item = items[0]
        assert "전환우선주" in item.category
        assert item.exercise_price == "12,177원"
        assert "30 억원" in item.remaining_amount
        assert "246,361주" in item.remaining_amount
        assert "2025.01.03~2028.12.21" == item.exercise_period

    def test_rights_issue_no_estk_skipped(self):
        """Disclosure without 기타주식 should be silently skipped."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        data = {
            "type": "유상증자결정",
            "new_shares": {"보통주식": 500_000},
        }
        analyzer.process_rights_issue_disclosure(data, "20240101")
        assert len(analyzer.get_overhang_items()) == 0


class TestParseAmount:
    """Tests for _parse_amount handling of whitespace and type coercion."""

    def test_normal_int(self):
        assert _parse_amount(12345) == 12345

    def test_string_with_commas(self):
        assert _parse_amount("1,234,567") == 1234567

    def test_string_with_internal_spaces(self):
        assert _parse_amount("1 234 567") == 1234567

    def test_string_with_nbsp(self):
        assert _parse_amount("1\xa0234\xa0567") == 1234567

    def test_string_with_mixed_whitespace(self):
        assert _parse_amount(" 1 234,567 ") == 1234567

    def test_negative_with_spaces(self):
        assert _parse_amount("-1 234 567") == -1234567

    def test_none(self):
        assert _parse_amount(None) is None

    def test_dash(self):
        assert _parse_amount("-") is None

    def test_empty(self):
        assert _parse_amount("") is None

    def test_float_string(self):
        assert _parse_amount("1234.56") == 1234

    def test_non_numeric_string_returns_none(self):
        assert _parse_amount("영업이익") is None

    def test_large_value_passes_through(self):
        """_parse_amount does not auto-correct; inflation is detected at DataFrame level."""
        assert _parse_amount(8_559_696_389_000_000) == 8_559_696_389_000_000


class TestDetectUnitScale:
    """Tests for DataFrame-level 백만원 inflation detection."""

    def _make_df(self, amounts: list, sj_div: str = "CIS"):
        import pandas as pd
        return pd.DataFrame({
            "sj_div": [sj_div] * len(amounts),
            "account_nm": [f"account_{i}" for i in range(len(amounts))],
            "thstrm_add_amount": amounts,
        })

    def test_inflated_all_divisible(self):
        """All significant amounts divisible by 1M → scale = 1,000,000."""
        df = self._make_df([
            233_654_482_000_000,
            -5_281_932_709_000_000,
            8_559_696_389_000_000,
            -805,  # EPS, should be skipped
        ])
        assert _detect_unit_scale(df, "thstrm_add_amount") == 1_000_000

    def test_normal_not_divisible(self):
        """Normal amounts not divisible by 1M → scale = 1."""
        df = self._make_df([
            124_648_789,
            -5_390_276_938,
            11_320_316_023,
            -781,  # EPS
        ])
        assert _detect_unit_scale(df, "thstrm_add_amount") == 1

    def test_too_few_significant_values(self):
        """Need at least 2 significant values for detection."""
        df = self._make_df([1_000_000, -50])  # only 1 significant
        assert _detect_unit_scale(df, "thstrm_add_amount") == 1

    def test_no_sj_div_column(self):
        """Missing sj_div column → returns 1."""
        import pandas as pd
        df = pd.DataFrame({"thstrm_add_amount": [1_000_000_000_000]})
        assert _detect_unit_scale(df, "thstrm_add_amount") == 1

    def test_mixed_divisibility_no_detection(self):
        """One divisible, one not → scale = 1 (no false positive)."""
        df = self._make_df([
            8_559_696_389_000_000,  # divisible by 1M
            124_648_789,            # not divisible
        ])
        assert _detect_unit_scale(df, "thstrm_add_amount") == 1

    def test_bs_rows_ignored(self):
        """Only IS/CIS rows are checked, not BS."""
        df = self._make_df([500_000_000_000_000, 300_000_000_000_000], sj_div="BS")
        assert _detect_unit_scale(df, "thstrm_add_amount") == 1


class TestISFieldNames:
    """Ensure operating_income variants include 영업손익."""

    def test_operating_income_includes_영업손익(self):
        names = _IS_FIELD_NAMES["operating_income"]
        assert "영업손익" in names

    def test_operating_income_includes_영업손실(self):
        names = _IS_FIELD_NAMES["operating_income"]
        assert "영업손실" in names

    def test_dart_report_includes_영업손익(self):
        names = _IS_FIELDS["operating_income"]
        assert "영업손익" in names


    def test_net_income_includes_당기순손실_이익(self):
        """net_income keyword list should include 당기순손실(이익)."""
        names = _IS_FIELD_NAMES["net_income"]
        assert "당기순손실(이익)" in names

    def test_dart_report_net_income_includes_당기순손실_이익(self):
        """dart_report _IS_FIELDS should include 당기순손실(이익)."""
        names = _IS_FIELDS["net_income"]
        assert "당기순손실(이익)" in names


class TestSubtractIncome:
    """Tests for cumulative subtraction to get standalone quarter data."""

    def test_basic_subtraction(self):
        """H1 cumulative - Q1 = Q2 standalone."""
        h1 = IncomeStatementItem(
            period="2025.Q2",
            revenue=500_0000_0000,
            operating_income=100_0000_0000,
            net_income=80_0000_0000,
        )
        q1 = IncomeStatementItem(
            period="2025.Q1",
            revenue=200_0000_0000,
            operating_income=30_0000_0000,
            net_income=20_0000_0000,
        )
        result = _subtract_income(h1, q1, "2025.Q2")
        assert result.period == "2025.Q2"
        assert result.revenue == 300_0000_0000
        assert result.operating_income == 70_0000_0000
        assert result.net_income == 60_0000_0000

    def test_missing_earlier_returns_none(self):
        """If earlier data is None, result field should be None."""
        later = IncomeStatementItem(
            period="2025.Q3",
            revenue=600_0000_0000,
            operating_income=None,
            net_income=50_0000_0000,
        )
        earlier = IncomeStatementItem(
            period="2025.Q2",
            revenue=None,
            operating_income=None,
            net_income=30_0000_0000,
        )
        result = _subtract_income(later, earlier, "2025.Q3")
        assert result.revenue is None  # later has value but earlier doesn't
        assert result.operating_income is None  # both None
        assert result.net_income == 20_0000_0000

    def test_q4_from_annual_minus_q3(self):
        """Q4 = Annual - Q3 cumulative."""
        annual = IncomeStatementItem(
            period="2024",
            revenue=1000_0000_0000,
            operating_income=200_0000_0000,
            net_income=150_0000_0000,
        )
        q3_cum = IncomeStatementItem(
            period="2024.Q3",
            revenue=700_0000_0000,
            operating_income=130_0000_0000,
            net_income=100_0000_0000,
        )
        result = _subtract_income(annual, q3_cum, "2024.Q4")
        assert result.period == "2024.Q4"
        assert result.revenue == 300_0000_0000
        assert result.operating_income == 70_0000_0000
        assert result.net_income == 50_0000_0000

    def test_negative_result_allowed(self):
        """Negative standalone values (e.g. Q4 loss after Q3 profit) are valid."""
        annual = IncomeStatementItem(
            period="2024",
            revenue=100_0000_0000,
            operating_income=-10_0000_0000,
            net_income=-20_0000_0000,
        )
        q3_cum = IncomeStatementItem(
            period="2024.Q3",
            revenue=120_0000_0000,
            operating_income=5_0000_0000,
            net_income=10_0000_0000,
        )
        result = _subtract_income(annual, q3_cum, "2024.Q4")
        assert result.revenue == -20_0000_0000
        assert result.operating_income == -15_0000_0000
        assert result.net_income == -30_0000_0000


class TestBuildCumulativeAnnualRow:
    """Tests for build_cumulative_annual_row."""

    def _make_q(self, year: int, q: int, rev: int, op: int, ni: int) -> IncomeStatementItem:
        return IncomeStatementItem(
            period=f"{year}.Q{q}",
            revenue=rev,
            operating_income=op,
            net_income=ni,
        )

    def test_three_quarters_cumulative(self):
        """Q1+Q2+Q3 should be summed and YoY vs prior year same period."""
        stmts = [
            # 2025 quarters (newest first)
            self._make_q(2025, 3, 679_0000_0000, 82_0000_0000, 45_0000_0000),
            self._make_q(2025, 2, 613_0000_0000, 49_0000_0000, 9_0000_0000),
            self._make_q(2025, 1, 759_0000_0000, 94_0000_0000, 67_0000_0000),
            # 2024 quarters (for YoY)
            self._make_q(2024, 3, 770_0000_0000, 81_0000_0000, 31_0000_0000),
            self._make_q(2024, 2, 578_0000_0000, 36_0000_0000, 4_0000_0000),
            self._make_q(2024, 1, 497_0000_0000, 17_0000_0000, 8_0000_0000),
        ]
        row = build_cumulative_annual_row(stmts)
        assert row is not None
        assert row.year == "**2025.3Q**"
        # Cumulative 2025: 679+613+759 = 2051 억
        assert "2,051" in row.revenue
        # Cumulative 2024 Q1-Q3: 770+578+497 = 1845 억
        # YoY = (2051-1845)/1845 * 100 ≈ +11%
        assert "+11%" in row.revenue

    def test_returns_none_for_empty_list(self):
        assert build_cumulative_annual_row([]) is None

    def test_returns_none_for_q4_latest(self):
        """If the latest quarter is Q4, return None (full year exists in annual)."""
        stmts = [
            self._make_q(2024, 4, 926_0000_0000, 108_0000_0000, 102_0000_0000),
            self._make_q(2024, 3, 770_0000_0000, 81_0000_0000, 31_0000_0000),
            self._make_q(2024, 2, 578_0000_0000, 36_0000_0000, 4_0000_0000),
            self._make_q(2024, 1, 497_0000_0000, 17_0000_0000, 8_0000_0000),
        ]
        row = build_cumulative_annual_row(stmts)
        assert row is None

    def test_q1_only(self):
        """Single Q1 should still produce a row (cumulative = Q1 standalone)."""
        stmts = [
            self._make_q(2025, 1, 759_0000_0000, 94_0000_0000, 67_0000_0000),
            self._make_q(2024, 1, 497_0000_0000, 17_0000_0000, 8_0000_0000),
        ]
        row = build_cumulative_annual_row(stmts)
        assert row is not None
        assert row.year == "**2025.1Q**"
        assert "759" in row.revenue

    def test_no_prior_year_shows_dash_yoy(self):
        """Without prior year data, YoY should be '-'."""
        stmts = [
            self._make_q(2025, 2, 613_0000_0000, 49_0000_0000, 9_0000_0000),
            self._make_q(2025, 1, 759_0000_0000, 94_0000_0000, 67_0000_0000),
        ]
        row = build_cumulative_annual_row(stmts)
        assert row is not None
        # format_income_cell wraps YoY in parens: "1,372 (-)"
        assert "(-)" in row.revenue


class TestTableToDict3Column:
    """Tests for table_to_dict handling 3+ column DART table rows."""

    def _make_table(self, rows_html: str) -> BeautifulSoup:
        html = f"<table>{rows_html}</table>"
        return BeautifulSoup(html, "html.parser").find("table")

    def test_two_column_row(self):
        """Standard 2-column row: key=cells[0], value=cells[1]."""
        table = self._make_table(
            "<tr><td>전환가액 (원/주)</td><td>11,398</td></tr>"
        )
        result = table_to_dict(table)
        assert result["전환가액 (원/주)"] == "11,398"

    def test_three_column_row_captures_last_pair(self):
        """3-column row: also capture cells[-2]→cells[-1]."""
        table = self._make_table(
            "<tr><td>전환에 따라 발행할 주식</td><td>주식수</td><td>877,346</td></tr>"
        )
        result = table_to_dict(table)
        # cells[0]→cells[1] captured as well
        assert result["전환에 따라 발행할 주식"] == "주식수"
        # cells[-2]→cells[-1] also captured
        assert result["주식수"] == "877,346"

    def test_three_column_row_empty_first_cell(self):
        """3-column row with empty first cell (rowspan sub-row)."""
        table = self._make_table(
            "<tr><td></td><td>주식수</td><td>877,346</td></tr>"
        )
        result = table_to_dict(table)
        assert result["주식수"] == "877,346"

    def test_three_column_does_not_override_existing(self):
        """Last-pair extraction should not override existing entries."""
        table = self._make_table(
            "<tr><td>주식수</td><td>1,000,000</td></tr>"
            "<tr><td>전환에 따라 발행할 주식</td><td>주식수</td><td>877,346</td></tr>"
        )
        result = table_to_dict(table)
        # First row sets 주식수=1,000,000; second row's sub-pair should not override
        assert result["주식수"] == "1,000,000"

    def test_four_column_row(self):
        """4-column row: captures last pair."""
        table = self._make_table(
            "<tr><td>9. 전환에 관한 사항</td><td>전환에 따라 발행할 주식</td>"
            "<td>주식수</td><td>877,346</td></tr>"
        )
        result = table_to_dict(table)
        assert result["주식수"] == "877,346"

    def test_multiline_cell_text_normalized(self):
        """Cell text with newlines should be normalized by clean_text."""
        table = self._make_table(
            "<tr><td>전환에 따라\n발행할 주식</td><td>종류</td><td>기명식 보통주</td></tr>"
        )
        result = table_to_dict(table)
        assert "전환에 따라 발행할 주식" in result
        assert result["종류"] == "기명식 보통주"


class TestExtractDateFromUrl:
    """Tests for _extract_date_from_url."""

    def test_standard_dart_url(self):
        url = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20240628901050"
        assert _extract_date_from_url(url) == "2024.06.28"

    def test_no_rcpno(self):
        assert _extract_date_from_url("https://example.com") == ""

    def test_empty(self):
        assert _extract_date_from_url("") == ""


class TestParseEventRow:
    """Tests for OpenDartFetcher._parse_event_row extracting disclosure_date from rcept_no."""

    def _make_fetcher(self):
        """Create an OpenDartFetcher without calling __init__ (avoids API key)."""
        return object.__new__(OpenDartFetcher)

    def test_cb_extracts_date_from_rcept_no(self):
        """rcept_no first 8 digits should become disclosure_date."""
        fetcher = self._make_fetcher()
        row = {
            "rcept_no": "20220920000336",
            "bd_tm": "1",
            "bd_knd": "전환사채",
            "bd_fta": "10,000,000,000",
            "cv_prc": "11,398",
            "cvisstk_cnt": "877,346",
            "cvisstk_knd": "보통주",
            "cvrqpd_bgd": "2023년 09월 28일",
            "cvrqpd_edd": "2027년 08월 28일",
        }
        result = fetcher._parse_event_row(row, "CB")
        assert result["disclosure_date"] == "20220920"

    def test_rights_issue_extracts_date_from_rcept_no(self):
        """Rights issue events should also extract date from rcept_no."""
        fetcher = self._make_fetcher()
        row = {
            "rcept_no": "20241213000100",
            "nstk_ostk_cnt": "0",
            "nstk_estk_cnt": "500,000",
        }
        result = fetcher._parse_event_row(row, "RIGHTS_ISSUE")
        assert result["disclosure_date"] == "20241213"

    def test_rights_issue_extracts_funding_as_face_value(self):
        """Rights issue should compute face_value from fdpp_* funding amounts."""
        fetcher = self._make_fetcher()
        row = {
            "rcept_no": "20231213000100",
            "nstk_ostk_cnt": "0",
            "nstk_estk_cnt": "276,492",
            "fdpp_op": "2,999,938,200",
        }
        result = fetcher._parse_event_row(row, "RIGHTS_ISSUE")
        assert result["shares"] == 276_492
        assert result["face_value"] == 2_999_938_200

    def test_rights_issue_extracts_issue_price(self):
        """Rights issue should extract stk_estk_issu_prc as issue_price."""
        fetcher = self._make_fetcher()
        row = {
            "rcept_no": "20231213000100",
            "nstk_ostk_cnt": "0",
            "nstk_estk_cnt": "276,492",
            "stk_estk_issu_prc": "10,850",
        }
        result = fetcher._parse_event_row(row, "RIGHTS_ISSUE")
        assert result["issue_price"] == 10_850

    def test_missing_rcept_no_gives_empty_date(self):
        """Missing rcept_no should yield empty disclosure_date."""
        fetcher = self._make_fetcher()
        row = {
            "bd_tm": "1",
            "bd_knd": "전환사채",
            "bd_fta": "5,000,000,000",
            "cv_prc": "10,000",
            "cvisstk_cnt": "500,000",
            "cvisstk_knd": "보통주",
            "cvrqpd_bgd": "",
            "cvrqpd_edd": "",
        }
        result = fetcher._parse_event_row(row, "CB")
        assert result["disclosure_date"] == ""

    def test_malformed_rcept_no_gives_empty_date(self):
        """Truncated or non-digit rcept_no should yield empty disclosure_date."""
        fetcher = self._make_fetcher()
        base_row = {
            "bd_tm": "1", "bd_knd": "전환사채", "bd_fta": "0",
            "cv_prc": "0", "cvisstk_cnt": "0", "cvisstk_knd": "",
            "cvrqpd_bgd": "", "cvrqpd_edd": "",
        }
        for bad in ("2022092", "ABCD0920000336", "0", "12345678"):
            row = {**base_row, "rcept_no": bad}
            result = fetcher._parse_event_row(row, "CB")
            assert result["disclosure_date"] == "", f"expected empty for {bad!r}"

class TestDetectPerformancePeriod:
    """Tests for _detect_performance_period helper."""

    def test_annual_period(self):
        """12-month period → annual (quarter=0)."""
        period = {"- 시작일": "2025.01.01", "- 종료일": "2025.12.31"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 2025
        assert quarter == 0
        assert is_cum is False

    def test_q1_standalone(self):
        """3-month Jan-Mar → Q1 standalone."""
        period = {"- 시작일": "2026.01.01", "- 종료일": "2026.03.31"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 2026
        assert quarter == 1
        assert is_cum is False

    def test_q2_standalone(self):
        """3-month Apr-Jun → Q2 standalone."""
        period = {"- 시작일": "2026.04.01", "- 종료일": "2026.06.30"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 2026
        assert quarter == 2
        assert is_cum is False

    def test_q3_standalone(self):
        """3-month Jul-Sep → Q3 standalone."""
        period = {"- 시작일": "2026.07.01", "- 종료일": "2026.09.30"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 2026
        assert quarter == 3
        assert is_cum is False

    def test_h1_cumulative(self):
        """6-month Jan-Jun → Q2 cumulative."""
        period = {"- 시작일": "2026.01.01", "- 종료일": "2026.06.30"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 2026
        assert quarter == 2
        assert is_cum is True

    def test_9m_cumulative(self):
        """9-month Jan-Sep → Q3 cumulative."""
        period = {"- 시작일": "2026.01.01", "- 종료일": "2026.09.30"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 2026
        assert quarter == 3
        assert is_cum is True

    def test_period_keys_without_dash(self):
        """Period dict without '- ' prefix in keys."""
        period = {"시작일": "2026.01.01", "종료일": "2026.03.31"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 2026
        assert quarter == 1

    def test_no_end_date_returns_zero(self):
        period = {"- 시작일": "2026.01.01"}
        year, quarter, is_cum = _detect_performance_period(period)
        assert year == 0


def _make_performance(start: str, end: str, rev: int, op: int, ni: int) -> Performance:
    """Helper to build a Performance object matching parse_performance() output.

    The parser produces period as flat dict and income items keyed by
    당해사업연도/직전사업연도 (IncomeItem aliases).
    """
    return Performance(**{
        "1. 재무제표의 종류": "연결",
        "2. 결산기간": {"- 시작일": start, "- 종료일": end},
        "3. 매출액 또는 손익구조변동내용(단위: 원)": {
            "- 매출액": {"당해사업연도": str(rev), "직전사업연도": "0"},
            "- 영업이익": {"당해사업연도": str(op), "직전사업연도": "0"},
            "- 당기순이익": {"당해사업연도": str(ni), "직전사업연도": "0"},
        },
    })


class TestIntegratePerformanceDisclosures:
    """Tests for _integrate_performance_disclosures routing logic."""

    def test_annual_goes_to_annual_statements(self):
        """12-month performance → annual_statements."""
        perf = _make_performance(
            "2025.01.01", "2025.12.31",
            100_0000_0000, 20_0000_0000, 10_0000_0000,
        )
        annual = [IncomeStatementItem(period="2025")]
        quarterly = []
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        assert annual[0].revenue == 100_0000_0000
        assert annual[0].period == "2025"
        assert len(quarterly) == 0

    def test_q1_goes_to_quarterly_statements(self):
        """Q1 flash earnings → quarterly_statements as YYYY.Q1."""
        perf = _make_performance(
            "2026.01.01", "2026.03.31",
            50_0000_0000, 8_0000_0000, 3_0000_0000,
        )
        annual = []
        quarterly = []
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        assert len(annual) == 0
        assert len(quarterly) == 1
        assert quarterly[0].period == "2026.Q1"
        assert quarterly[0].revenue == 50_0000_0000

    def test_q1_not_misrouted_to_annual(self):
        """Q1 flash earnings must NOT appear in annual_statements."""
        perf = _make_performance(
            "2026.01.01", "2026.03.31",
            50_0000_0000, 8_0000_0000, 3_0000_0000,
        )
        annual = [IncomeStatementItem(period="2026")]
        quarterly = []
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        # The annual 2026 placeholder should remain empty
        assert annual[0].revenue is None
        # The quarterly should have the data
        assert quarterly[0].period == "2026.Q1"

    def test_q2_standalone_goes_to_quarterly(self):
        """Standalone Q2 (Apr-Jun) → quarterly_statements."""
        perf = _make_performance(
            "2026.04.01", "2026.06.30",
            60_0000_0000, 10_0000_0000, 5_0000_0000,
        )
        annual = []
        quarterly = []
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        assert quarterly[0].period == "2026.Q2"
        assert quarterly[0].revenue == 60_0000_0000

    def test_h1_cumulative_derives_q2(self):
        """H1 cumulative with existing Q1 → derives Q2 via subtraction."""
        perf = _make_performance(
            "2026.01.01", "2026.06.30",
            110_0000_0000, 18_0000_0000, 8_0000_0000,
        )
        annual = []
        quarterly = [
            IncomeStatementItem(
                period="2026.Q1",
                revenue=50_0000_0000,
                operating_income=8_0000_0000,
                net_income=3_0000_0000,
            ),
        ]
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        assert len(quarterly) == 2
        q2 = next(s for s in quarterly if s.period == "2026.Q2")
        assert q2.revenue == 60_0000_0000  # 110 - 50
        assert q2.operating_income == 10_0000_0000  # 18 - 8
        assert q2.net_income == 5_0000_0000  # 8 - 3

    def test_h1_cumulative_without_q1_skips(self):
        """H1 cumulative without Q1 data → cannot derive Q2, skips."""
        perf = _make_performance(
            "2026.01.01", "2026.06.30",
            110_0000_0000, 18_0000_0000, 8_0000_0000,
        )
        annual = []
        quarterly = []
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        assert len(quarterly) == 0

    def test_skip_if_existing_nonempty_data(self):
        """If period already has non-empty data, skip integration."""
        perf = _make_performance(
            "2026.01.01", "2026.03.31",
            50_0000_0000, 8_0000_0000, 3_0000_0000,
        )
        existing_q1 = IncomeStatementItem(
            period="2026.Q1",
            revenue=55_0000_0000,
            operating_income=9_0000_0000,
            net_income=4_0000_0000,
        )
        annual = []
        quarterly = [existing_q1]
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        # Existing data should be preserved, not overwritten
        assert quarterly[0].revenue == 55_0000_0000

    def test_quarterly_flows_into_cumulative_annual_row(self):
        """Q1 flash earnings in quarterly should be picked up by
        build_cumulative_annual_row for yearly accumulation."""
        perf = _make_performance(
            "2026.01.01", "2026.03.31",
            50_0000_0000, 8_0000_0000, 3_0000_0000,
        )
        annual = []
        quarterly = [
            # Prior year Q1 for YoY
            IncomeStatementItem(
                period="2025.Q1",
                revenue=40_0000_0000,
                operating_income=6_0000_0000,
                net_income=2_0000_0000,
            ),
        ]
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        # Now build cumulative annual row from the quarterly list
        row = build_cumulative_annual_row(quarterly)
        assert row is not None
        assert row.year == "**2026.1Q**"
        assert "50" in row.revenue  # 50_0000_0000 = 50억

    def test_h1_cumulative_does_not_overwrite_existing_q2(self):
        """H1 cumulative should not overwrite Q2 that already has API data."""
        perf = _make_performance(
            "2026.01.01", "2026.06.30",
            110_0000_0000, 18_0000_0000, 8_0000_0000,
        )
        existing_q2 = IncomeStatementItem(
            period="2026.Q2",
            revenue=65_0000_0000,
            operating_income=12_0000_0000,
            net_income=6_0000_0000,
        )
        annual = []
        quarterly = [
            IncomeStatementItem(
                period="2026.Q1",
                revenue=50_0000_0000,
                operating_income=8_0000_0000,
                net_income=3_0000_0000,
            ),
            existing_q2,
        ]
        _integrate_performance_disclosures(
            {DisclosureType.PERFORMANCE: [perf]}, annual, quarterly,
        )
        q2 = next(s for s in quarterly if s.period == "2026.Q2")
        # Original API data preserved, not overwritten by derived value
        assert q2.revenue == 65_0000_0000


class TestOverhangDilutiveShares:
    """Tests for get_total_dilutive_shares."""

    def test_empty(self):
        analyzer = OverhangAnalyzer(total_shares=1_000_000)
        assert analyzer.get_total_dilutive_shares() == 0

    def test_with_instruments(self):
        analyzer = OverhangAnalyzer(total_shares=1_000_000)
        analyzer.process_event({
            "category": "CB",
            "series": 1,
            "face_value": 100_0000_0000,
            "conversion_price": 10000,
            "convertible_shares": 100_000,
            "disclosure_date": "20240101",
        })
        analyzer.process_event({
            "category": "CB",
            "series": 2,
            "face_value": 50_0000_0000,
            "conversion_price": 10000,
            "convertible_shares": 50_000,
            "disclosure_date": "20240601",
        })
        assert analyzer.get_total_dilutive_shares() == 150_000

    def test_excludes_expired(self):
        analyzer = OverhangAnalyzer(total_shares=1_000_000)
        analyzer.process_event({
            "category": "CB",
            "series": 1,
            "face_value": 100_0000_0000,
            "conversion_price": 10000,
            "convertible_shares": 100_000,
            "cv_end": "2020.01.01",
            "disclosure_date": "20190101",
        })
        assert analyzer.get_total_dilutive_shares() == 0

    def test_mixes_active_and_expired(self):
        analyzer = OverhangAnalyzer(total_shares=1_000_000)
        analyzer.process_event({
            "category": "CB",
            "series": 1,
            "face_value": 100_0000_0000,
            "conversion_price": 10000,
            "convertible_shares": 100_000,
            "cv_end": "2020.01.01",
            "disclosure_date": "20190101",
        })
        analyzer.process_event({
            "category": "CB",
            "series": 2,
            "face_value": 50_0000_0000,
            "conversion_price": 10000,
            "convertible_shares": 50_000,
            "cv_end": "2099.12.31",
            "disclosure_date": "20240601",
        })
        assert analyzer.get_total_dilutive_shares() == 50_000

    def test_consistent_with_dilution_ratio(self):
        analyzer = OverhangAnalyzer(total_shares=1_000_000)
        analyzer.process_event({
            "category": "CB",
            "series": 1,
            "face_value": 100_0000_0000,
            "conversion_price": 10000,
            "convertible_shares": 100_000,
            "disclosure_date": "20240101",
        })
        dilutive = analyzer.get_total_dilutive_shares()
        ratio = analyzer.get_total_dilution()
        expected_ratio = (dilutive / 1_000_000) * 100
        assert abs(ratio - expected_ratio) < 0.01


class TestFullyDilutedCalculation:
    """Tests for fully-diluted market cap calculation logic."""

    def test_basic_calculation(self):
        """Fully-diluted = (shares + dilutive) * price."""
        total_shares = 10_000_000
        dilutive_shares = 2_000_000
        stock_price = 50_000
        fd_total = total_shares + dilutive_shares
        fd_market_cap = fd_total * stock_price
        fd_mc_eok = round(fd_market_cap / 1_0000_0000)
        result = f"{fd_mc_eok:,}억원 (희석주식 포함 총 {fd_total:,}주 기준)"
        assert result == "6,000억원 (희석주식 포함 총 12,000,000주 기준)"

    def test_no_dilution(self):
        """No dilutive shares → same as regular market cap."""
        total_shares = 10_000_000
        dilutive_shares = 0
        stock_price = 50_000
        fd_total = total_shares + dilutive_shares
        fd_market_cap = fd_total * stock_price
        fd_mc_eok = round(fd_market_cap / 1_0000_0000)
        assert fd_mc_eok == 5_000


class TestReportDataNewFields:
    """Tests for new ReportData fields."""

    def test_defaults(self):
        from auto_reports.models.report import ReportData
        data = ReportData()
        assert data.fully_diluted_market_cap_str == ""
        assert data.shareholder_info == ""
        assert data.dividend_info == ""
        assert data.subsidiary_info == ""

    def test_populated(self):
        from auto_reports.models.report import ReportData
        data = ReportData(
            fully_diluted_market_cap_str="6,000억원 (희석주식 포함 총 12,000,000주 기준)",
            shareholder_info="최대주주 이종서 등 13.62%",
            dividend_info="가장 최근 주당 500원 배당 실시 (현금배당성향 20.5%, 현금배당총액 100억원)",
            subsidiary_info="해당없음",
        )
        assert "6,000억원" in data.fully_diluted_market_cap_str
        assert "이종서" in data.shareholder_info
        assert "500원" in data.dividend_info
        assert data.subsidiary_info == "해당없음"


class TestReportTemplateNewSections:
    """Tests for new sections in report template rendering."""

    def test_template_renders_new_sections(self):
        from auto_reports.generators.report import generate_report
        from auto_reports.models.report import ReportData, ReportFrontmatter

        data = ReportData(
            frontmatter=ReportFrontmatter(created="2026-03-01"),
            fully_diluted_market_cap_str="6,000억원 (희석주식 포함 총 12,000,000주 기준)",
            shareholder_info="최대주주 이종서 등 13.62%",
            dividend_info="가장 최근 주당 500원 배당 실시 (현금배당성향 20.5%, 현금배당총액 100억원)",
            subsidiary_info="해당없음",
        )
        md = generate_report(data)
        assert "**Fully-Diluted 시가총액**" in md
        assert "6,000억원" in md
        assert "**주주 구성**" in md
        assert "이종서" in md
        assert "**배당**" in md
        assert "500원" in md
        assert "**종속 회사**" in md
        assert "해당없음" in md

    def test_template_omits_empty_sections(self):
        from auto_reports.generators.report import generate_report
        from auto_reports.models.report import ReportData, ReportFrontmatter

        data = ReportData(
            frontmatter=ReportFrontmatter(created="2026-03-01"),
        )
        md = generate_report(data)
        # Fully-diluted and shareholder are conditional
        assert "**Fully-Diluted 시가총액**" not in md
        assert "**주주 구성**" not in md
        # Subsidiary always shows
        assert "**종속 회사**: 해당없음" in md

    def test_sections_appear_after_overhang(self):
        from auto_reports.generators.report import generate_report
        from auto_reports.models.report import ReportData, ReportFrontmatter

        data = ReportData(
            frontmatter=ReportFrontmatter(created="2026-03-01"),
            fully_diluted_market_cap_str="1,000억원 (희석주식 포함 총 5,000,000주 기준)",
            subsidiary_info="A사, B사",
        )
        md = generate_report(data)
        # New sections must appear BEFORE "## 2. 재무상태표"
        fd_pos = md.index("Fully-Diluted 시가총액")
        section2_pos = md.index("## 2. 재무상태표")
        assert fd_pos < section2_pos
        sub_pos = md.index("**종속 회사**")
        assert sub_pos < section2_pos
