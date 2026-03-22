"""Tests for notes-based overhang extraction and OverhangAnalyzer integration."""

from __future__ import annotations

import pandas as pd
from unittest.mock import patch, MagicMock

from auto_reports.analyzers.overhang import OverhangAnalyzer
from auto_reports.fetchers.opendart import OpenDartFetcher
from auto_reports.models.report import OverhangItem
from auto_reports.parsers.notes_overhang import (
    _parse_notes_overhang_regex as parse_notes_overhang,
    _parse_notes_overhang_regex,
)


def _make_fetcher() -> OpenDartFetcher:
    """Create a fetcher with a dummy API key (no real calls)."""
    with patch("auto_reports.fetchers.opendart.OpenDartReader"):
        return OpenDartFetcher(api_key="test_key_12345")


def _pad_html(html: str, min_length: int = 1500) -> str:
    """Pad HTML with a comment to exceed the 1KB placeholder threshold."""
    if len(html.encode("utf-8")) >= min_length:
        return html
    needed = min_length - len(html.encode("utf-8"))
    padding = f"<!-- {'x' * needed} -->"
    return html.replace("</body>", padding + "</body>")


# ------------------------------------------------------------------
# parse_notes_overhang unit tests
# ------------------------------------------------------------------

class TestParseNotesOverhang:
    """Test the notes parser with synthetic HTML fragments."""

    def _wrap_html(self, body: str) -> str:
        return f"<html><body>{body}</body></html>"

    def test_cb_section_basic(self):
        """Parse a minimal CB section with one series."""
        html = self._wrap_html("""
        <p>13. 전환사채</p>
        <p>① 제1회 무기명식 사모 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>10,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>877,346주</td></tr>
          <tr><td rowspan="3">전환에 관한 사항</td><td>전환가격: 11,398원 / 주</td></tr>
          <tr><td>전환비율: 100%</td></tr>
          <tr><td>전환청구가능기간: 2023년 09월 28일 부터 2027년 08월 28일 까지</td></tr>
        </table>
        <p>14. 다른항목</p>
        """)
        results = parse_notes_overhang(html)
        assert len(results) == 1
        r = results[0]
        assert r["category"] == "CB"
        assert r["series"] == 1
        assert r["face_value"] == 10_000_000_000
        assert r["convertible_shares"] == 877_346
        assert r["conversion_price"] == 11_398
        assert r["exercise_start"] == "2023.09.28"
        assert r["exercise_end"] == "2027.08.28"
        assert r["active"] is True  # no inactive footnote

    def test_cb_inactive_footnote(self):
        """CB with footnote indicating full conversion → inactive."""
        html = self._wrap_html("""
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>5,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>100,000주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 50,000원 / 주</td></tr>
        </table>
        <p>(*1) 전기 중 모두 보통주로 전환되었습니다.</p>
        <p>14. 기타</p>
        """)
        results = parse_notes_overhang(html)
        assert len(results) == 1
        assert results[0]["active"] is False

    def test_cb_redeemed_inactive(self):
        """CB with footnote indicating full redemption → inactive."""
        html = self._wrap_html("""
        <p>13. 전환사채</p>
        <p>① 제2회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>3,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>50,000주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 60,000원 / 주</td></tr>
        </table>
        <p>(*1) 당분기 중 전액 상환되었습니다.</p>
        <p>14. 기타</p>
        """)
        results = parse_notes_overhang(html)
        assert len(results) == 1
        assert results[0]["active"] is False

    def test_multiple_cb_series(self):
        """Parse multiple CB series in one section."""
        html = self._wrap_html("""
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>10,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>200,000주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 50,000원 / 주</td></tr>
        </table>
        <p>② 제2회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>5,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>100,000주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 50,000원 / 주</td></tr>
        </table>
        <p>14. 기타</p>
        """)
        results = parse_notes_overhang(html)
        assert len(results) == 2
        assert results[0]["series"] == 1
        assert results[1]["series"] == 2

    def test_preferred_stock_section(self):
        """Parse a 전환우선주 section."""
        html = self._wrap_html("""
        <p>(4) 전환우선주</p>
        <p>① 2024년 12월 14일 발행 전환우선주</p>
        <table>
          <tr><td>발행주식수</td><td>607,669주</td></tr>
          <tr><td>1주당 발행금액</td><td>14,646원</td></tr>
          <tr><td>총발행가액</td><td>8,899,920,174원</td></tr>
          <tr><td>전환조건</td><td>전환우선주 1주당 보통주 1주</td></tr>
          <tr><td>전환기간</td><td>최초발행일 후 1년이 경과한 날부터 5년이 되는 날</td></tr>
          <tr><td>발행일</td><td>2024년 12월 14일</td></tr>
        </table>
        <p>15. 기타</p>
        """)
        results = parse_notes_overhang(html)
        assert len(results) == 1
        r = results[0]
        assert r["category"] == "전환우선주"
        assert r["convertible_shares"] == 607_669
        assert r["conversion_price"] == 14_646
        assert r["exercise_start"] == "2025.12.14"
        assert r["exercise_end"] == "2029.12.14"
        assert r["active"] is True

    def test_preferred_stock_inactive(self):
        """전환우선주 fully converted → inactive."""
        html = self._wrap_html("""
        <p>(4) 전환우선주</p>
        <p>① 2022년 발행 전환우선주</p>
        <table>
          <tr><td>발행주식수</td><td>100,000주</td></tr>
          <tr><td>1주당 발행금액</td><td>10,000원</td></tr>
          <tr><td>총발행가액</td><td>1,000,000,000원</td></tr>
          <tr><td>전환조건</td><td>전환우선주 1주당 보통주 1주</td></tr>
          <tr><td>전환기간</td><td>최초발행일 후 1년이 경과한 날부터 5년이 되는 날</td></tr>
          <tr><td>발행일</td><td>2022년 01월 01일</td></tr>
        </table>
        <p>(*1) 전기 중 모두 보통주로 전환되었습니다.</p>
        <p>15. 기타</p>
        """)
        results = parse_notes_overhang(html)
        assert len(results) == 1
        assert results[0]["active"] is False

    def test_no_overhang_sections(self):
        """HTML with no overhang sections returns empty."""
        html = self._wrap_html("""
        <p>1. 일반사항</p>
        <p>회사 개요 내용</p>
        <p>2. 재무제표 작성기준</p>
        """)
        results = parse_notes_overhang(html)
        assert results == []

    def test_bw_section(self):
        """Parse a 신주인수권부사채 section."""
        html = self._wrap_html("""
        <p>14. 신주인수권부사채</p>
        <p>① 제1회 신주인수권부사채</p>
        <table>
          <tr><td>발행총액</td><td>5,000,000,000원</td></tr>
          <tr><td>행사시 발행 주식수</td><td>200,000주</td></tr>
          <tr><td>신주인수권에 관한 사항</td><td>행사가격: 25,000원 / 주</td></tr>
        </table>
        <p>15. 기타</p>
        """)
        results = parse_notes_overhang(html)
        assert len(results) == 1
        r = results[0]
        assert r["category"] == "BW"
        assert r["face_value"] == 5_000_000_000
        assert r["convertible_shares"] == 200_000
        assert r["conversion_price"] == 25_000


# ------------------------------------------------------------------
# OverhangAnalyzer.process_notes_instrument tests
# ------------------------------------------------------------------

class TestProcessNotesInstrument:
    """Test OverhangAnalyzer.process_notes_instrument()."""

    def test_cb_instrument(self):
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        analyzer.process_notes_instrument({
            "category": "CB",
            "series": 1,
            "kind": "제1회 전환사채",
            "face_value": 10_000_000_000,
            "convertible_shares": 200_000,
            "conversion_price": 50_000,
            "exercise_start": "2025.01.01",
            "exercise_end": "2028.12.31",
            "active": True,
        })

        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert "전환사채(CB)" in items[0].category
        assert items[0].exercise_price == "50,000원"
        assert items[0].dilution_ratio == "2.00%"
        assert items[0].exercise_period == "2025.01.01~2028.12.31"

    def test_preferred_stock_instrument(self):
        analyzer = OverhangAnalyzer(total_shares=20_000_000)
        analyzer.process_notes_instrument({
            "category": "전환우선주",
            "series": 0,
            "kind": "2024년 12월 14일 발행 전환우선주",
            "face_value": 8_899_920_174,
            "convertible_shares": 607_669,
            "conversion_price": 14_646,
            "exercise_start": "2025.12.14",
            "exercise_end": "2029.12.14",
            "active": True,
        })

        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert "전환우선주" in items[0].category
        assert items[0].exercise_price == "14,646원"
        assert items[0].dilution_ratio == "3.04%"

    def test_multiple_instruments(self):
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        analyzer.process_notes_instrument({
            "category": "CB",
            "series": 3,
            "kind": "제3회 전환사채",
            "face_value": 25_000_000_000,
            "convertible_shares": 1_000_000,
            "conversion_price": 25_000,
            "exercise_start": "2026.01.01",
            "exercise_end": "2030.12.31",
            "active": True,
        })
        analyzer.process_notes_instrument({
            "category": "전환우선주",
            "series": 0,
            "kind": "2024년 발행 전환우선주",
            "face_value": 5_000_000_000,
            "convertible_shares": 500_000,
            "conversion_price": 10_000,
            "exercise_start": "2025.01.01",
            "exercise_end": "2029.12.31",
            "active": True,
        })

        items = analyzer.get_overhang_items()
        assert len(items) == 2
        total_dilutive = analyzer.get_total_dilutive_shares()
        assert total_dilutive == 1_500_000

    def test_overhang_item_no_disclosure_date(self):
        """OverhangItem should not have disclosure_date field."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        analyzer.process_notes_instrument({
            "category": "CB",
            "series": 1,
            "kind": "제1회 전환사채",
            "face_value": 1_000_000_000,
            "convertible_shares": 100_000,
            "conversion_price": 10_000,
            "active": True,
        })
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        # disclosure_date should not be in the model fields
        assert "disclosure_date" not in OverhangItem.model_fields


# ------------------------------------------------------------------
# get_overhang_from_notes integration tests (mocked)
# ------------------------------------------------------------------

@patch("auto_reports.parsers.notes_overhang.parse_notes_overhang",
       side_effect=lambda html, **kw: _parse_notes_overhang_regex(html))
class TestGetOverhangFromNotes:
    """Test get_overhang_from_notes() with mocked DART API."""

    def _mock_dart_list(self, reports: list[tuple[str, str]]):
        """Create a mock dart.list() result DataFrame."""
        if not reports:
            return pd.DataFrame()
        return pd.DataFrame([
            {"rcept_no": rn, "report_nm": nm, "rcept_dt": rn[:8]}
            for rn, nm in reports
        ])

    def _mock_sub_docs(self, sections: list[tuple[str, str]]):
        """Create a mock dart.sub_docs() result DataFrame."""
        if not sections:
            return pd.DataFrame()
        return pd.DataFrame([
            {"title": title, "url": url}
            for title, url in sections
        ])

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_full_flow(self, mock_requests_get, _mock_parser):
        """Full flow: find report → fetch notes → parse → return active."""
        fetcher = _make_fetcher()

        # Mock dart.list → returns one report
        fetcher.dart.list.return_value = self._mock_dart_list([
            ("20251110000124", "분기보고서 (2025.09)"),
        ])

        # Mock dart.sub_docs → returns notes section
        fetcher.dart.sub_docs.return_value = self._mock_sub_docs([
            ("5.재무제표 주석", "http://dart.fss.or.kr/notes/12345"),
        ])

        # Mock requests.get → returns notes HTML (padded to exceed 1KB threshold)
        notes_html = _pad_html("""<html><body>
        <p>13. 전환사채</p>
        <p>① 제3회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>25,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>1,382,867주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 18,223원 / 주</td></tr>
        </table>
        <p>14. 기타</p>
        </body></html>""")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = notes_html
        mock_resp.encoding = "utf-8"
        mock_requests_get.return_value = mock_resp

        results, ref_date = fetcher.get_overhang_from_notes("00126380")

        assert len(results) == 1
        assert results[0]["category"] == "CB"
        assert results[0]["series"] == 3
        assert results[0]["convertible_shares"] == 1_382_867
        assert results[0]["conversion_price"] == 18_223
        # 분기보고서 (2025.09) → Format B parses to "20250930"

    def test_no_periodic_report(self, _mock_parser):
        """No periodic report found → return empty."""
        fetcher = _make_fetcher()
        fetcher.dart.list.return_value = pd.DataFrame()

        results, ref_date = fetcher.get_overhang_from_notes("00126380")
        assert results == []
        assert ref_date == ""

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_no_notes_section(self, mock_requests_get, _mock_parser):
        """Report exists but no notes section → return empty."""
        fetcher = _make_fetcher()
        fetcher.dart.list.return_value = self._mock_dart_list([
            ("20251110000124", "분기보고서 (2025.09)"),
        ])
        fetcher.dart.sub_docs.return_value = self._mock_sub_docs([
            ("1.회사의 개요", "http://dart.fss.or.kr/overview/12345"),
        ])

        results, ref_date = fetcher.get_overhang_from_notes("00126380")
        assert results == []

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_filters_inactive_instruments(self, mock_requests_get, _mock_parser):
        """Only active instruments are returned."""
        fetcher = _make_fetcher()
        fetcher.dart.list.return_value = self._mock_dart_list([
            ("20251110000124", "분기보고서 (2025.09)"),
        ])
        fetcher.dart.sub_docs.return_value = self._mock_sub_docs([
            ("5.재무제표 주석", "http://dart.fss.or.kr/notes/12345"),
        ])

        notes_html = _pad_html("""<html><body>
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>10,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>200,000주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 50,000원 / 주</td></tr>
        </table>
        <p>(*1) 전기 중 모두 보통주로 전환되었습니다.</p>
        <p>② 제2회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>5,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>100,000주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 50,000원 / 주</td></tr>
        </table>
        <p>14. 기타</p>
        </body></html>""")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = notes_html
        mock_requests_get.return_value = mock_resp

        results, ref_date = fetcher.get_overhang_from_notes("00126380")

        # Series 1 is inactive (converted), only series 2 should remain
        assert len(results) == 1
        assert results[0]["series"] == 2
        assert results[0]["active"] is True

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_api_error_graceful(self, mock_requests_get, _mock_parser):
        """DART API error → return empty."""
        fetcher = _make_fetcher()
        fetcher.dart.list.side_effect = Exception("Connection timeout")

        results, ref_date = fetcher.get_overhang_from_notes("00126380")
        assert results == []
        assert ref_date == ""

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_prefers_standalone_notes(self, mock_requests_get, _mock_parser):
        """When both consolidated and standalone notes exist, prefer standalone."""
        fetcher = _make_fetcher()
        fetcher.dart.list.return_value = self._mock_dart_list([
            ("20251110000124", "분기보고서 (2025.09)"),
        ])
        # Both sections available
        fetcher.dart.sub_docs.return_value = self._mock_sub_docs([
            ("3.연결재무제표 주석", "http://dart.fss.or.kr/consolidated/12345"),
            ("5.재무제표 주석", "http://dart.fss.or.kr/standalone/12345"),
        ])

        # Standalone has content, consolidated is a placeholder
        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.encoding = "utf-8"
            if "standalone" in url:
                resp.text = _pad_html("""<html><body>
                <p>13. 전환사채</p>
                <p>① 제1회 전환사채</p>
                <table>
                  <tr><td>발행총액</td><td>5,000,000,000원</td></tr>
                  <tr><td>전환시 전환 주식수</td><td>100,000주</td></tr>
                  <tr><td>전환에 관한 사항</td><td>전환가격: 50,000원 / 주</td></tr>
                </table>
                <p>14. 기타</p>
                </body></html>""")
            else:
                resp.text = "<html><body>placeholder</body></html>"  # < 1KB
            return resp

        mock_requests_get.side_effect = side_effect

        results, ref_date = fetcher.get_overhang_from_notes("00126380")
        assert len(results) == 1
        assert results[0]["category"] == "CB"


# ------------------------------------------------------------------
# Full integration: notes → analyzer → OverhangItem
# ------------------------------------------------------------------

@patch("auto_reports.parsers.notes_overhang.parse_notes_overhang",
       side_effect=lambda html, **kw: _parse_notes_overhang_regex(html))
class TestFullIntegration:
    """End-to-end: notes HTML → parse → analyzer → report items."""

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_notes_to_report_items(self, mock_requests_get, _mock_parser):
        """Full pipeline: DART notes → OverhangAnalyzer → OverhangItem."""
        fetcher = _make_fetcher()
        fetcher.dart.list.return_value = pd.DataFrame([
            {"rcept_no": "20251110000124", "report_nm": "분기보고서 (2025.09)", "rcept_dt": "20251110"},
        ])
        fetcher.dart.sub_docs.return_value = pd.DataFrame([
            {"title": "5.재무제표 주석", "url": "http://dart.fss.or.kr/notes/12345"},
        ])

        notes_html = _pad_html("""<html><body>
        <p>13. 전환사채</p>
        <p>① 제3회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>25,200,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>1,382,867주</td></tr>
          <tr><td rowspan="3">전환에 관한 사항</td><td>전환가격: 18,223원 / 주</td></tr>
          <tr><td>전환비율: 100%</td></tr>
          <tr><td>전환청구가능기간: 2026년 10월 01일 부터 2030년 09월 01일 까지</td></tr>
        </table>
        <p>14. 기타</p>
        </body></html>""")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = notes_html
        mock_resp.encoding = "utf-8"
        mock_requests_get.return_value = mock_resp

        results, ref_date = fetcher.get_overhang_from_notes("00126380")

        analyzer = OverhangAnalyzer(total_shares=20_000_000)
        for inst in results:
            analyzer.process_notes_instrument(inst)

        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert "전환사채(CB)" in items[0].category
        assert items[0].exercise_price == "18,223원"
        assert items[0].dilution_ratio == "6.91%"
        assert "2026.10.01" in items[0].exercise_period
        assert "2030.09.01" in items[0].exercise_period


# ------------------------------------------------------------------
# _parse_reference_date tests
# ------------------------------------------------------------------

class TestParseReferenceDate:
    """Test _parse_reference_date() for various report name formats."""

    # --- Format A: "2025년 3분기 분기보고서" ---

    def test_annual_report(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("2024년 사업보고서") == "20241231"

    def test_q3_report(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("2025년 3분기 분기보고서") == "20250930"

    def test_q1_report(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("2025년 1분기 분기보고서") == "20250331"

    def test_half_year_report(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("2025년 반기보고서") == "20250630"

    def test_corrected_report(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("[기재정정]2025년 3분기 분기보고서") == "20250930"

    def test_unknown_format(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("unknown report") == ""

    def test_q2_report(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("2025년 2분기 분기보고서") == "20250630"

    # --- Format B: "분기보고서 (2025.09)" (actual OpenDART API format) ---

    def test_format_b_annual(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("사업보고서 (2024.12)") == "20241231"

    def test_format_b_q3(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("분기보고서 (2025.09)") == "20250930"

    def test_format_b_half_year(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("반기보고서 (2025.06)") == "20250630"

    def test_format_b_q1(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("분기보고서 (2025.03)") == "20250331"

    def test_format_b_corrected(self):
        from auto_reports.fetchers.opendart import _parse_reference_date
        assert _parse_reference_date("[기재정정]분기보고서 (2025.09)") == "20250930"


# ------------------------------------------------------------------
# get_overhang_from_notes returns reference_date
# ------------------------------------------------------------------

@patch("auto_reports.parsers.notes_overhang.parse_notes_overhang",
       side_effect=lambda html, **kw: _parse_notes_overhang_regex(html))
class TestGetOverhangFromNotesReferenceDate:
    """Test that get_overhang_from_notes returns correct reference_date."""

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_returns_reference_date_for_q3(self, mock_requests_get, _mock_parser):
        fetcher = _make_fetcher()
        fetcher.dart.list.return_value = pd.DataFrame([
            {"rcept_no": "20251110000124", "report_nm": "2025년 3분기 분기보고서", "rcept_dt": "20251110"},
        ])
        fetcher.dart.sub_docs.return_value = pd.DataFrame([
            {"title": "5.재무제표 주석", "url": "http://dart.fss.or.kr/notes/12345"},
        ])
        notes_html = _pad_html("""<html><body>
        <p>13. 전환사채</p>
        <p>① 제3회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>25,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>1,382,867주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 18,223원 / 주</td></tr>
        </table>
        <p>14. 기타</p>
        </body></html>""")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = notes_html
        mock_resp.encoding = "utf-8"
        mock_requests_get.return_value = mock_resp

        results, ref_date = fetcher.get_overhang_from_notes("00126380")
        assert ref_date == "20250930"
        assert len(results) == 1

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_returns_reference_date_for_annual(self, mock_requests_get, _mock_parser):
        fetcher = _make_fetcher()
        fetcher.dart.list.return_value = pd.DataFrame([
            {"rcept_no": "20250315000100", "report_nm": "2024년 사업보고서", "rcept_dt": "20250315"},
        ])
        fetcher.dart.sub_docs.return_value = pd.DataFrame([
            {"title": "5.재무제표 주석", "url": "http://dart.fss.or.kr/notes/12345"},
        ])
        notes_html = _pad_html("""<html><body>
        <p>13. 전환사채</p>
        <p>① 제1회 전환사채</p>
        <table>
          <tr><td>발행총액</td><td>10,000,000,000원</td></tr>
          <tr><td>전환시 전환 주식수</td><td>500,000주</td></tr>
          <tr><td>전환에 관한 사항</td><td>전환가격: 20,000원 / 주</td></tr>
        </table>
        <p>14. 기타</p>
        </body></html>""")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = notes_html
        mock_resp.encoding = "utf-8"
        mock_requests_get.return_value = mock_resp

        results, ref_date = fetcher.get_overhang_from_notes("00126380")
        assert ref_date == "20241231"
        assert len(results) == 1


# ------------------------------------------------------------------
# get_overhang_issuances after_date filtering
# ------------------------------------------------------------------

class TestGetOverhangIssuancesAfterDate:
    """Test that get_overhang_issuances uses after_date to set bgn_de."""

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_after_date_sets_bgn_de(self, mock_requests_get):
        """When after_date is provided, bgn_de is set to after_date + 1 day."""
        fetcher = _make_fetcher()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"status": "013", "message": "no data"}
        mock_requests_get.return_value = mock_resp

        fetcher.get_overhang_issuances("00126380", after_date="20250930")

        # Check the first API call's params
        calls = mock_requests_get.call_args_list
        assert len(calls) >= 1
        first_params = calls[0].kwargs.get("params") or calls[0][1].get("params", {})
        assert first_params["bgn_de"] == "20251001"

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_no_after_date_uses_default(self, mock_requests_get):
        """When after_date is empty, uses start_year-based bgn_de."""
        fetcher = _make_fetcher()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"status": "013", "message": "no data"}
        mock_requests_get.return_value = mock_resp

        fetcher.get_overhang_issuances("00126380", start_year=2023)

        calls = mock_requests_get.call_args_list
        assert len(calls) >= 1
        first_params = calls[0].kwargs.get("params") or calls[0][1].get("params", {})
        assert first_params["bgn_de"] == "20230101"


# ------------------------------------------------------------------
# Phase 3b: _parse_issuance_item for RIGHTS_ISSUE (piicDecsn)
# ------------------------------------------------------------------


class TestParseIssuanceItemRightsIssue:
    """Test _parse_issuance_item for RIGHTS_ISSUE (piicDecsn API responses)."""

    def test_preferred_shares_only(self):
        """piicDecsn with 기타주식 only → valid RIGHTS_ISSUE."""
        fetcher = _make_fetcher()
        item = {
            "rcept_no": "20251001000123",
            "nstk_ostk_cnt": "0",
            "nstk_estk_cnt": "592,655",
            "stk_estk_issu_prc": "18,223",
            "fdpp_op": "10,806,247,465",
        }
        result = fetcher._parse_issuance_item(item, "RIGHTS_ISSUE")
        assert result is not None
        assert result["category"] == "RIGHTS_ISSUE"
        assert result["shares"] == 592_655
        assert result["issue_price"] == 18_223
        assert result["rcept_no"] == "20251001000123"
        assert result["face_value"] == 10_806_247_465

    def test_preferred_shares_with_dash_ordinary(self):
        """piicDecsn with '-' for ordinary → valid (treat as 0)."""
        fetcher = _make_fetcher()
        item = {
            "rcept_no": "20251001000456",
            "nstk_ostk_cnt": "-",
            "nstk_estk_cnt": "100,000",
            "stk_estk_issu_prc": "20,000",
        }
        result = fetcher._parse_issuance_item(item, "RIGHTS_ISSUE")
        assert result is not None
        assert result["shares"] == 100_000

    def test_ordinary_shares_nonzero_filtered(self):
        """piicDecsn with nonzero ordinary shares → filtered out."""
        fetcher = _make_fetcher()
        item = {
            "rcept_no": "20251001000789",
            "nstk_ostk_cnt": "500,000",
            "nstk_estk_cnt": "0",
        }
        result = fetcher._parse_issuance_item(item, "RIGHTS_ISSUE")
        assert result is None

    def test_no_preferred_shares_filtered(self):
        """piicDecsn with 0 preferred shares → filtered out."""
        fetcher = _make_fetcher()
        item = {
            "rcept_no": "20251001000000",
            "nstk_ostk_cnt": "0",
            "nstk_estk_cnt": "0",
        }
        result = fetcher._parse_issuance_item(item, "RIGHTS_ISSUE")
        assert result is None

    def test_missing_fields_handled(self):
        """piicDecsn with missing optional fields."""
        fetcher = _make_fetcher()
        item = {
            "rcept_no": "20251001000111",
            "nstk_estk_cnt": "200,000",
            # nstk_ostk_cnt missing (treated as None → passes filter)
            # stk_estk_issu_prc missing
        }
        result = fetcher._parse_issuance_item(item, "RIGHTS_ISSUE")
        assert result is not None
        assert result["shares"] == 200_000
        assert "issue_price" not in result  # not set when missing


# ------------------------------------------------------------------
# Phase 3b: _parse_issuance_item for CB (cvbdIsDecsn)
# ------------------------------------------------------------------


class TestParseIssuanceItemCB:
    """Test _parse_issuance_item for CB (cvbdIsDecsn API responses)."""

    def test_cb_basic(self):
        """cvbdIsDecsn with standard CB data."""
        fetcher = _make_fetcher()
        item = {
            "rcept_no": "20251001000222",
            "bd_tm": "3",
            "bd_knd": "무기명식 이권부 무보증 사모 전환사채",
            "bd_fta": "25,200,000,000",
            "cv_prc": "18,223",
            "cvisstk_cnt": "1,382,867",
            "cvrqpd_bgd": "2026.10.28",
            "cvrqpd_edd": "2055.09.28",
        }
        result = fetcher._parse_issuance_item(item, "CB")
        assert result is not None
        assert result["category"] == "CB"
        assert result["series"] == 3
        assert result["face_value"] == 25_200_000_000
        assert result["conversion_price"] == 18_223
        assert result["convertible_shares"] == 1_382_867
        assert result["cv_start"] == "2026.10.28"
        assert result["cv_end"] == "2055.09.28"


# ------------------------------------------------------------------
# Phase 3b: Full flow — DS005 APIs → OverhangAnalyzer → overhang items
# ------------------------------------------------------------------


class TestPhase3bFullFlow:
    """Test the full Phase 3b flow with mocked API responses (앱클론-like data)."""

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_piic_and_cb_produce_overhang_items(self, mock_requests_get):
        """piicDecsn + cvbdIsDecsn → OverhangAnalyzer produces correct items."""
        fetcher = _make_fetcher()

        # Mock API responses per category
        def mock_api_response(url, params=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()

            if "piicDecsn" in url:
                resp.json.return_value = {
                    "status": "000",
                    "list": [{
                        "rcept_no": "20251001800123",
                        "nstk_ostk_cnt": "0",
                        "nstk_estk_cnt": "592,655",
                        "stk_estk_issu_prc": "18,223",
                        "fdpp_op": "10,806,247,465",
                    }],
                }
            elif "cvbdIsDecsn" in url:
                resp.json.return_value = {
                    "status": "000",
                    "list": [{
                        "rcept_no": "20251001800456",
                        "bd_tm": "3",
                        "bd_knd": "전환사채",
                        "bd_fta": "25,200,000,000",
                        "cv_prc": "18,223",
                        "cvisstk_cnt": "1,382,867",
                        "cvrqpd_bgd": "2026.10.28",
                        "cvrqpd_edd": "2055.09.28",
                    }],
                }
            else:
                resp.json.return_value = {"status": "013", "message": "no data"}
            return resp

        mock_requests_get.side_effect = mock_api_response

        # Step 1: Fetch issuances
        events = fetcher.get_overhang_issuances("00174900", after_date="20250930")
        assert len(events) == 2

        ri_events = [e for e in events if e["category"] == "RIGHTS_ISSUE"]
        cb_events = [e for e in events if e["category"] == "CB"]
        assert len(ri_events) == 1
        assert len(cb_events) == 1
        assert ri_events[0]["shares"] == 592_655
        assert cb_events[0]["convertible_shares"] == 1_382_867

        # Step 2: Process through OverhangAnalyzer
        analyzer = OverhangAnalyzer(total_shares=19_925_510)
        for event in events:
            analyzer.process_event(event)

        items = analyzer.get_overhang_items()
        assert len(items) == 2

        # Check labels and data
        labels = {item.category for item in items}
        assert "전환우선주" in labels
        assert "제3회 전환사채(CB)" in labels

        ri_item = next(i for i in items if i.category == "전환우선주")
        assert "592,655" in (ri_item.remaining_amount or "")

        cb_item = next(i for i in items if "전환사채" in i.category)
        assert "1,382,867" in (cb_item.remaining_amount or "")
        assert cb_item.exercise_price == "18,223원"
        assert "2026.10.28" in (cb_item.exercise_period or "")

    @patch("auto_reports.fetchers.opendart.dart_get_with_retry")
    def test_phase3a_and_3b_combined(self, mock_requests_get):
        """Phase 3a notes + Phase 3b DS005 → 4 distinct overhang items."""
        analyzer = OverhangAnalyzer(total_shares=19_925_510)

        # Phase 3a: Add baseline from notes (전환우선주 607,669주 + SO 144,700주)
        analyzer.process_notes_instrument({
            "category": "PREF",
            "series": 0,
            "kind": "제1회 전환우선주",
            "face_value": 8_900_000_000,
            "convertible_shares": 607_669,
            "conversion_price": 14_646,
            "exercise_start": "2025.12.14",
            "exercise_end": "2029.12.14",
            "active": True,
        })
        analyzer.process_notes_instrument({
            "category": "SO",
            "series": 0,
            "kind": "",
            "face_value": 0,
            "convertible_shares": 144_700,
            "conversion_price": 8_640,
            "exercise_start": "",
            "exercise_end": "",
            "active": True,
        })

        items_after_3a = analyzer.get_overhang_items()
        assert len(items_after_3a) == 2

        # Phase 3b: Add post-기준일 issuances
        analyzer.process_event({
            "category": "RIGHTS_ISSUE",
            "rcept_no": "20251001800123",
            "disclosure_date": "20251001",
            "shares": 592_655,
            "share_type": "전환우선주",
            "face_value": 10_806_247_465,
            "issue_price": 18_223,
        })
        analyzer.process_event({
            "category": "CB",
            "rcept_no": "20251001800456",
            "disclosure_date": "20251001",
            "series": 3,
            "kind": "전환사채",
            "face_value": 25_200_000_000,
            "conversion_price": 18_223,
            "convertible_shares": 1_382_867,
            "cv_start": "2026.10.28",
            "cv_end": "2055.09.28",
        })

        items_after_3b = analyzer.get_overhang_items()
        assert len(items_after_3b) == 4, (
            f"Expected 4 items (2 from notes + 2 from DS005), got {len(items_after_3b)}: "
            f"{[i.category for i in items_after_3b]}"
        )

        # Verify all 4 categories present
        labels = sorted(i.category for i in items_after_3b)
        assert "전환우선주" in labels  # from Phase 3b RIGHTS_ISSUE
        assert "제3회 전환사채(CB)" in labels  # from Phase 3b CB
        assert "주식매수선택권" in labels  # from Phase 3a SO

        # Phase 3a 전환우선주 uses notes key "PREF_제1회_전환우선주"
        # Phase 3b 전환우선주 uses key "RIGHTS_20251001800123"
        # Both should appear separately (different instruments)
        cps_items = [i for i in items_after_3b if "전환우선주" in i.category or "PREF" in i.category]
        assert len(cps_items) >= 2, (
            f"Expected 2 전환우선주 items (notes + DS005), got {len(cps_items)}"
        )
