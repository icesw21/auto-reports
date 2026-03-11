"""Tests for overhang parsing fixes (direct KV table, 전환가액 variant, cb_balance remaining)."""

from __future__ import annotations

from bs4 import BeautifulSoup

from auto_reports.analyzers.overhang import OverhangAnalyzer, _infer_category_from_exercise
from auto_reports.parsers.exchange_disclosure import _parse_cb_balance
from auto_reports.parsers.notes_overhang import (
    parse_notes_overhang,
    _extract_conv_price_smart,
    _parse_korean_text_number,
    _extract_amount,
    _extract_amount_with_unit,
)


def _pad_html(html: str, min_length: int = 1500) -> str:
    """Pad HTML to exceed placeholder threshold."""
    if len(html.encode("utf-8")) >= min_length:
        return html
    needed = min_length - len(html.encode("utf-8"))
    padding = f"<!-- {'x' * needed} -->"
    return html.replace("</body>", padding + "</body>")


def _wrap(body: str) -> str:
    return f"<html><body>{body}</body></html>"


# ------------------------------------------------------------------
# notes_overhang: direct KV table format (no circled subsections)
# ------------------------------------------------------------------


class TestParseBondDirect:
    """Test _parse_bond_direct fallback for reports without ①②③ headers."""

    def test_single_cb_direct_kv(self):
        """Parse a CB section with direct KV table (no ① header)."""
        html = _pad_html(_wrap("""
        <p>15. 전환사채</p>
        <table>
          <tr><th>구분</th><th>제1회 무기명식 사모 전환사채</th></tr>
          <tr><td>발행일</td><td>2024.06.11</td></tr>
          <tr><td>만기일</td><td>2029.06.11</td></tr>
          <tr><td>권면(전자등록)총액(원)</td><td>25,000,000,000</td></tr>
          <tr><td>전환가액(원/주)</td><td>12,508</td></tr>
          <tr><td>전환에 따라 발행할 주식수</td><td>1,998,720</td></tr>
          <tr><td>전환청구기간</td><td>2025.07.11~2029.06.11</td></tr>
        </table>
        <p>16. 다른항목</p>
        """))
        results = parse_notes_overhang(html)
        assert len(results) == 1
        r = results[0]
        assert r["category"] == "CB"
        assert r["series"] == 1
        assert r["face_value"] == 25_000_000_000
        assert r["conversion_price"] == 12_508
        assert r["convertible_shares"] == 1_998_720
        assert r["exercise_start"] == "2025.07.11"
        assert r["exercise_end"] == "2029.06.11"
        assert r["active"] is True

    def test_multi_column_direct(self):
        """Parse a multi-column table (구분 | 제1회 | 제2회)."""
        html = _pad_html(_wrap("""
        <p>12. 전환사채</p>
        <table>
          <tr><th>구분</th><th>제1회 CB</th><th>제2회 CB</th></tr>
          <tr><td>권면총액(원)</td><td>10,000,000,000</td><td>5,000,000,000</td></tr>
          <tr><td>전환가격(원/주)</td><td>10,000</td><td>8,000</td></tr>
          <tr><td>전환에 따라 발행할 주식수</td><td>1,000,000</td><td>625,000</td></tr>
          <tr><td>전환청구기간</td><td>2024.01.01~2027.12.31</td><td>2025.06.01~2028.05.31</td></tr>
        </table>
        <p>13. 다른항목</p>
        """))
        results = parse_notes_overhang(html)
        assert len(results) == 2
        assert results[0]["series"] == 1
        assert results[0]["face_value"] == 10_000_000_000
        assert results[0]["conversion_price"] == 10_000
        assert results[1]["series"] == 2
        assert results[1]["face_value"] == 5_000_000_000
        assert results[1]["conversion_price"] == 8_000
        assert results[1]["convertible_shares"] == 625_000


# ------------------------------------------------------------------
# notes_overhang: 전환가액 vs 전환가격 key variant
# ------------------------------------------------------------------


class TestConvPriceKeyVariant:
    """Test that both 전환가격 and 전환가액 are recognized."""

    def test_conv_price_with_가액(self):
        """전환가액 (with 액) should be recognized for CB."""
        kv = {"전환가액(원/주)": "12,508"}
        result = _extract_conv_price_smart(kv, "CB")
        assert result == 12_508

    def test_conv_price_with_가격(self):
        """전환가격 (with 격) should also work."""
        kv = {"전환가격(원/주)": "10,000"}
        result = _extract_conv_price_smart(kv, "CB")
        assert result == 10_000

    def test_exercise_price_with_가액_for_bw(self):
        """행사가액 should be recognized for BW."""
        kv = {"행사가액(원/주)": "5,000"}
        result = _extract_conv_price_smart(kv, "BW")
        assert result == 5_000

    def test_exercise_price_with_가격_for_bw(self):
        """행사가격 should also work for BW."""
        kv = {"행사가격(원/주)": "5,000"}
        result = _extract_conv_price_smart(kv, "BW")
        assert result == 5_000


# ------------------------------------------------------------------
# exchange_disclosure: _parse_cb_balance remaining between two KRW markers
# ------------------------------------------------------------------


class TestParseCbBalanceRemaining:
    """Test _parse_cb_balance correctly extracts remaining between currency markers."""

    def _make_table(self, rows_html: str) -> BeautifulSoup:
        html = f"<table>{rows_html}</table>"
        return BeautifulSoup(html, "html.parser").find("table")

    def test_remaining_between_two_krw_markers(self):
        """Table: | 회차 | 권면총액 | KRW | 미잔액 | KRW | 전환가액 | 주식수 |"""
        table = self._make_table("""
        <tr><th>회차</th><th>권면총액</th><th>통화</th><th>미잔액</th><th>통화</th><th>전환가액</th><th>전환가능주식수</th></tr>
        <tr><td>1</td><td>25,000,000,000</td><td>KRW</td><td>12,273,243,000</td><td>KRW</td><td>10,172</td><td>1,206,571</td></tr>
        """)
        balances = _parse_cb_balance(table)
        assert len(balances) == 1
        bal = balances[0]
        assert bal["series"] == 1
        assert bal["face_value"] == 25_000_000_000
        assert bal["remaining"] == 12_273_243_000
        assert bal["conversion_price"] == 10_172
        assert bal["convertible_shares"] == 1_206_571

    def test_remaining_before_single_krw(self):
        """Table: | 회차 | 발행일 | 미잔액 | KRW | 전환가액 | 주식수 |"""
        table = self._make_table("""
        <tr><th>회차</th><th>발행일</th><th>미잔액</th><th>통화</th><th>전환가액</th><th>전환가능주식수</th></tr>
        <tr><td>1</td><td>2024-06-11</td><td>12,273,243,000</td><td>KRW</td><td>10,172</td><td>1,206,571</td></tr>
        """)
        balances = _parse_cb_balance(table)
        assert len(balances) == 1
        bal = balances[0]
        assert bal["remaining"] == 12_273_243_000

    def test_no_remaining_when_no_currency(self):
        """Table without currency markers should still parse face_value."""
        table = self._make_table("""
        <tr><th>회차</th><th>권면총액</th><th>전환가액</th></tr>
        <tr><td>1</td><td>25,000,000,000</td><td>10,172</td></tr>
        """)
        balances = _parse_cb_balance(table)
        assert len(balances) == 1
        assert balances[0]["face_value"] == 25_000_000_000
        assert "remaining" not in balances[0]


# ------------------------------------------------------------------
# overhang.py: process_exchange_exercise ordering & fallback creation
# ------------------------------------------------------------------


class TestExerciseOrdering:
    """Test that exercises applied in chronological order produce correct result."""

    def _seed_instrument(self, analyzer: OverhangAnalyzer) -> None:
        """Add a CB instrument via process_notes_instrument."""
        analyzer.process_notes_instrument({
            "category": "CB",
            "series": 1,
            "kind": "제1회",
            "face_value": 25_000_000_000,
            "convertible_shares": 1_998_720,
            "conversion_price": 12_508,
            "exercise_start": "2025.07.11",
            "exercise_end": "2029.06.11",
            "active": True,
        })

    def test_latest_exercise_wins(self):
        """When multiple exercises are applied oldest→newest, latest remaining wins."""
        analyzer = OverhangAnalyzer(total_shares=12_010_222)
        self._seed_instrument(analyzer)

        # Apply exercises in chronological order (oldest first)
        exercises = [
            {"cb_balance": [{"series": 1, "face_value": 25_000_000_000, "remaining": 22_305_000_000,
                             "conversion_price": 10_172, "convertible_shares": 2_192_784, "currency": "KRW"}],
             "type": "전환청구권행사", "_entry_date": "2025-10-13"},
            {"cb_balance": [{"series": 1, "face_value": 25_000_000_000, "remaining": 17_717_764_000,
                             "conversion_price": 10_172, "convertible_shares": 1_741_817, "currency": "KRW"}],
             "type": "전환청구권행사", "_entry_date": "2025-10-29"},
            {"cb_balance": [{"series": 1, "face_value": 25_000_000_000, "remaining": 12_273_243_000,
                             "conversion_price": 10_172, "convertible_shares": 1_206_571, "currency": "KRW"}],
             "type": "전환청구권행사", "_entry_date": "2026-01-07"},
        ]

        for ex in exercises:
            analyzer.process_exchange_exercise(ex)

        # Check internal state: the latest exercise (2026-01-07) should win
        state = analyzer._instruments["CB_1"]
        assert state.remaining == 12_273_243_000
        assert state.convertible_shares == 1_206_571
        assert state.conversion_price == 10_172

    def test_reverse_order_latest_wins(self):
        """Demonstrates the sort fix: unsorted (newest first) → oldest wins."""
        analyzer = OverhangAnalyzer(total_shares=12_010_222)
        self._seed_instrument(analyzer)

        # Reverse order (newest first, as stored in JSON) — without sorting
        exercises_unsorted = [
            {"cb_balance": [{"series": 1, "remaining": 12_273_243_000,
                             "conversion_price": 10_172, "convertible_shares": 1_206_571}],
             "_entry_date": "2026-01-07"},
            {"cb_balance": [{"series": 1, "remaining": 22_305_000_000,
                             "conversion_price": 10_172, "convertible_shares": 2_192_784}],
             "_entry_date": "2025-10-13"},
        ]

        # Without sorting, the oldest (2025-10-13) is applied last and wins
        for ex in exercises_unsorted:
            analyzer.process_exchange_exercise(ex)

        state = analyzer._instruments["CB_1"]
        assert state.remaining == 22_305_000_000  # Bug: oldest wins

        # Now test with sorting (as pipeline does after fix)
        analyzer2 = OverhangAnalyzer(total_shares=12_010_222)
        self._seed_instrument(analyzer2)

        exercises_sorted = sorted(exercises_unsorted, key=lambda d: d.get("_entry_date", ""))
        for ex in exercises_sorted:
            analyzer2.process_exchange_exercise(ex)

        state2 = analyzer2._instruments["CB_1"]
        assert state2.remaining == 12_273_243_000  # Fixed: latest wins


class TestExerciseFallbackCreation:
    """Test fallback instrument creation from exchange exercise."""

    def test_creates_instrument_when_notes_missed(self):
        """When no matching instrument exists, one should be created from cb_balance."""
        analyzer = OverhangAnalyzer(total_shares=10_000_000)
        # No instruments added from notes

        exercise = {
            "cb_balance": [{"series": 1, "face_value": 10_000_000_000,
                            "remaining": 8_000_000_000,
                            "conversion_price": 10_000, "convertible_shares": 800_000}],
            "type": "전환청구권행사",
        }
        analyzer.process_exchange_exercise(exercise)

        # Check internal state
        assert "CB_1" in analyzer._instruments
        state = analyzer._instruments["CB_1"]
        assert state.remaining == 8_000_000_000
        assert state.conversion_price == 10_000
        assert state.convertible_shares == 800_000

        # Also verify formatted output
        items = analyzer.get_overhang_items()
        assert len(items) == 1
        assert "전환사채(CB)" in items[0].category


class TestPriceAdjShareRecalculation:
    """Test that process_exchange_price_adj recalculates shares."""

    def _seed_cb(self, analyzer, series, remaining, conv_price):
        """Seed a CB instrument into the analyzer."""
        from auto_reports.analyzers.overhang import _InstrumentState
        key = f"CB_{series}"
        analyzer._instruments[key] = _InstrumentState(
            category="CB", series=series, kind="전환사채",
            face_value=remaining, remaining=remaining,
            conversion_price=conv_price,
            convertible_shares=remaining // conv_price,
            exercise_start="2022.01.01", exercise_end="2027.01.01",
        )

    def _seed_pref(self, analyzer, remaining, conv_price):
        """Seed a preferred stock instrument."""
        from auto_reports.analyzers.overhang import _InstrumentState
        key = "PREF_전환우선주부채"
        analyzer._instruments[key] = _InstrumentState(
            category="RIGHTS_PREFERRED", series=0, kind="전환우선주",
            face_value=remaining, remaining=remaining,
            conversion_price=conv_price,
            convertible_shares=remaining // conv_price,
            exercise_start="2026.01.01", exercise_end="2035.01.01",
        )

    def test_series_based_recalculation(self):
        """CB price adj recalculates shares from remaining / new_price."""
        analyzer = OverhangAnalyzer(total_shares=22_000_000)
        self._seed_cb(analyzer, 3, 4_000_000_000, 12_197)
        assert analyzer._instruments["CB_3"].convertible_shares == 327_949

        adj_data = {
            "adjustments": [{"series": 3, "price_before": 12_197, "price_after": 9_714}],
            "share_changes": [],
        }
        analyzer.process_exchange_price_adj(adj_data)

        state = analyzer._instruments["CB_3"]
        assert state.conversion_price == 9_714
        assert state.convertible_shares == 4_000_000_000 // 9_714  # 411_776

    def test_no_series_price_before_match(self):
        """전환우선주 (no series) matches by price_before and recalculates."""
        analyzer = OverhangAnalyzer(total_shares=22_000_000)
        self._seed_pref(analyzer, 1_000_000_310, 12_197)
        assert analyzer._instruments["PREF_전환우선주부채"].convertible_shares == 81_987

        adj_data = {
            "adjustments": [{"listed": "비상장", "price_before": 12_197, "price_after": 9_714}],
            "share_changes": [],
        }
        analyzer.process_exchange_price_adj(adj_data)

        state = analyzer._instruments["PREF_전환우선주부채"]
        assert state.conversion_price == 9_714
        assert state.convertible_shares == 1_000_000_310 // 9_714  # 102_943

    def test_explicit_share_changes_override_recalculation(self):
        """Explicit share_changes should override the recalculated value."""
        analyzer = OverhangAnalyzer(total_shares=22_000_000)
        self._seed_cb(analyzer, 3, 4_000_000_000, 12_197)

        adj_data = {
            "adjustments": [{"series": 3, "price_before": 12_197, "price_after": 9_714}],
            "share_changes": [{"series": 3, "shares_before": 327_949, "shares_after": 999_999}],
        }
        analyzer.process_exchange_price_adj(adj_data)

        state = analyzer._instruments["CB_3"]
        assert state.conversion_price == 9_714
        # share_changes override, not the recalculated value
        assert state.convertible_shares == 999_999

    def test_no_match_leaves_instruments_unchanged(self):
        """Price adj with no matching instrument leaves all untouched."""
        analyzer = OverhangAnalyzer(total_shares=22_000_000)
        self._seed_cb(analyzer, 1, 15_400_000_000, 7_875)
        orig_shares = analyzer._instruments["CB_1"].convertible_shares

        adj_data = {
            "adjustments": [{"series": 99, "price_before": 12_197, "price_after": 9_714}],
        }
        analyzer.process_exchange_price_adj(adj_data)

        assert analyzer._instruments["CB_1"].convertible_shares == orig_shares
        assert analyzer._instruments["CB_1"].conversion_price == 7_875


class TestInferCategoryFromExercise:
    """Test _infer_category_from_exercise helper."""

    def test_cb_from_전환청구권행사(self):
        assert _infer_category_from_exercise("전환청구권행사") == "CB"

    def test_bw_from_신주인수권행사(self):
        assert _infer_category_from_exercise("신주인수권행사") == "BW"

    def test_default_cb(self):
        assert _infer_category_from_exercise("unknown") == "CB"


# ------------------------------------------------------------------
# Korean text number parsing
# ------------------------------------------------------------------


class TestKoreanTextNumber:
    """Test _parse_korean_text_number helper."""

    def test_삼백억원(self):
        assert _parse_korean_text_number("금 삼백억원") == 30_000_000_000

    def test_오백억원(self):
        assert _parse_korean_text_number("오백억원") == 50_000_000_000

    def test_이십억원(self):
        assert _parse_korean_text_number("이십억원") == 2_000_000_000

    def test_일천오백억원(self):
        assert _parse_korean_text_number("일천오백억원") == 150_000_000_000

    def test_삼십억원(self):
        assert _parse_korean_text_number("삼십억원") == 3_000_000_000

    def test_non_korean_returns_none(self):
        assert _parse_korean_text_number("25,000,000,000") is None

    def test_empty_returns_none(self):
        assert _parse_korean_text_number("") is None


class TestExtractAmountPrefix:
    """Test _extract_amount handles 1주당 prefix."""

    def test_1주당_prefix_stripped(self):
        assert _extract_amount("1주당 16,672원") == 16_672

    def test_normal_amount_unchanged(self):
        assert _extract_amount("10,000,000,000원") == 10_000_000_000


class TestExtractAmountWithUnit:
    """Test _extract_amount_with_unit handles numeric+unit and Korean text."""

    def test_천원_unit(self):
        assert _extract_amount_with_unit("25,000,000천원") == 25_000_000_000

    def test_억원_unit_numeric(self):
        assert _extract_amount_with_unit("500억원") == 50_000_000_000

    def test_korean_text_삼백억원(self):
        assert _extract_amount_with_unit("금 삼백억원") == 30_000_000_000

    def test_plain_numeric(self):
        assert _extract_amount_with_unit("10,000,000,000") == 10_000_000_000


# ------------------------------------------------------------------
# Embedded CB in 전환상환우선주부채 section
# ------------------------------------------------------------------


class TestEmbeddedCBInRCPSSection:
    """Test CB detection when embedded inside 전환상환우선주부채 section."""

    def test_cb_in_전환상환우선주부채_section(self):
        """CB found via (5) subsection in 전환상환우선주부채 등 section."""
        html = _pad_html(_wrap("""
        <p>14. 전환상환우선주부채 등</p>
        <p>(1) 전환상환우선주 관련 내용</p>
        <table>
          <tr><td>항목</td><td>내용</td></tr>
          <tr><td>발행주식수</td><td>100,000</td></tr>
        </table>
        <p>(5) 당사가 발행한 전환사채 중 전환청구기간이 종료되지 않은 전환사채의 내역은 다음과 같습니다.</p>
        <table>
          <tr><td>구분</td><td>9회차 무기명식 이권부 무보증 무담보 사모 전환사채</td></tr>
          <tr><td>권면총액</td><td>금 삼백억원</td></tr>
          <tr><td>액면가액/발행가액(원)</td><td>1주당 16,672원</td></tr>
          <tr><td>전환청구기간</td><td>시작일 : 2026년 07월 18일\n종료일 : 2028년 06월 18일</td></tr>
        </table>
        <p>15. 다른항목</p>
        """))
        results = parse_notes_overhang(html)
        cb_results = [r for r in results if r["category"] == "CB"]
        assert len(cb_results) == 1
        r = cb_results[0]
        assert r["series"] == 9
        assert r["face_value"] == 30_000_000_000
        assert r["conversion_price"] == 16_672
        assert r["convertible_shares"] == 30_000_000_000 // 16_672
        assert r["exercise_start"] == "2026.07.18"
        assert r["exercise_end"] == "2028.06.18"
        assert r["active"] is True

    def test_conv_price_발행가액_key(self):
        """발행가액 key should be recognized for CB conversion price."""
        kv = {"액면가액/발행가액(원)": "1주당 16,672원"}
        result = _extract_conv_price_smart(kv, "CB")
        assert result == 16_672


# ── CB under "차입금 및 전환사채" subsection tests ──


class TestCBUnderBorrowingsSection:
    """Test CB parsing when CB is a (N) subsection under broader section.

    Structure: "16. 차입금 및 전환사채" → "(3) 전환사채" → ①②③ per-series.
    The main CB pattern won't match the top-level section, so the embedded
    CB scanner should find "(3) 전환사채" and delegate to _parse_bond_section.
    """

    PUNGWON_HTML = _pad_html(_wrap("""
        <p>16. 차입금 및 전환사채</p>
        <p>(1) 단기차입금</p>
        <table><tr><td>은행</td><td>1,000,000,000</td></tr></table>
        <p>(2) 장기차입금</p>
        <table><tr><td>은행</td><td>2,000,000,000</td></tr></table>
        <p>(3) 전환사채</p>
        <p>당기 중 전환사채의 발행조건은 다음과 같습니다.</p>
        <p>① 제1회 무기명식 이권부 무보증 사모 전환사채</p>
        <table>
            <tr><td>사채의 액면총액</td><td>15,400,000,000원</td></tr>
            <tr><td>전환가격(원/주)</td><td>11,398원</td></tr>
            <tr><td>전환청구기간</td><td>2022년 09월 11일 ~ 2027년 09월 10일</td></tr>
        </table>
        <p>② 제2회 무기명식 이권부 무보증 사모 전환사채</p>
        <table>
            <tr><td>사채의 액면총액</td><td>6,000,000,000원</td></tr>
            <tr><td>전환가격(원/주)</td><td>12,500원</td></tr>
            <tr><td>전환청구기간</td><td>2025년 06월 21일 ~ 2029년 05월 21일</td></tr>
        </table>
        <p>③ 제3회 무기명식 이권부 무보증 사모 전환사채</p>
        <table>
            <tr><td>사채의 액면총액</td><td>4,000,000,000원</td></tr>
            <tr><td>전환가격(원/주)</td><td>11,500원</td></tr>
            <tr><td>전환청구기간</td><td>2026년 05월 16일 ~ 2030년 04월 16일</td></tr>
        </table>
        <p>(4) 전환우선주부채</p>
        <p>전환우선주의 발행조건은 다음과 같습니다.</p>
        <table>
            <tr><td>발행일</td><td>2025년 04월 16일</td></tr>
            <tr><td>발행주식수</td><td>86,881주</td></tr>
            <tr><td>1주당 발행금액</td><td>11,510원</td></tr>
            <tr><td>총발행가액</td><td>1,000,000,310원</td></tr>
            <tr><td>전환기간</td><td>최초발행일 후 1년이 경과한 날부터 10년이 되는 날</td></tr>
            <tr><td>전환조건</td><td>전환우선주 1주당 보통주 1주</td></tr>
        </table>
        <p>17. 종업원급여</p>
    """))

    def test_finds_three_cbs(self):
        """Should find all 3 CBs from ①②③ subsections under (3) 전환사채."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = [r for r in results if r["category"] == "CB"]
        assert len(cbs) == 3

    def test_cb_series_numbers(self):
        """Each CB should have correct series number."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = sorted([r for r in results if r["category"] == "CB"], key=lambda r: r["series"])
        assert [c["series"] for c in cbs] == [1, 2, 3]

    def test_cb1_face_value(self):
        """CB #1 face value should be 15,400,000,000원."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = sorted([r for r in results if r["category"] == "CB"], key=lambda r: r["series"])
        assert cbs[0]["face_value"] == 15_400_000_000

    def test_cb2_face_value(self):
        """CB #2 face value should be 6,000,000,000원."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = sorted([r for r in results if r["category"] == "CB"], key=lambda r: r["series"])
        assert cbs[1]["face_value"] == 6_000_000_000

    def test_cb3_face_value(self):
        """CB #3 face value should be 4,000,000,000원."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = sorted([r for r in results if r["category"] == "CB"], key=lambda r: r["series"])
        assert cbs[2]["face_value"] == 4_000_000_000

    def test_cb1_conversion_price(self):
        """CB #1 conversion price should be 11,398원."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = sorted([r for r in results if r["category"] == "CB"], key=lambda r: r["series"])
        assert cbs[0]["conversion_price"] == 11_398

    def test_cb1_exercise_period(self):
        """CB #1 exercise period should be 2022.09.11 ~ 2027.09.10."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = sorted([r for r in results if r["category"] == "CB"], key=lambda r: r["series"])
        assert cbs[0]["exercise_start"] == "2022.09.11"
        assert cbs[0]["exercise_end"] == "2027.09.10"

    def test_cb3_exercise_period(self):
        """CB #3 exercise period should be 2026.05.16 ~ 2030.04.16."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = sorted([r for r in results if r["category"] == "CB"], key=lambda r: r["series"])
        assert cbs[2]["exercise_start"] == "2026.05.16"
        assert cbs[2]["exercise_end"] == "2030.04.16"

    def test_all_cbs_active(self):
        """All CBs should be active."""
        results = parse_notes_overhang(self.PUNGWON_HTML)
        cbs = [r for r in results if r["category"] == "CB"]
        assert all(c["active"] for c in cbs)


class TestPreferredStockDirectTable:
    """Test preferred stock parsing from direct KV table (no ①②③).

    Structure: "(4) 전환우선주부채" → direct KV table with issue details.
    """

    def test_finds_preferred_stock(self):
        """Should find 1 preferred stock from (4) 전환우선주부채."""
        results = parse_notes_overhang(TestCBUnderBorrowingsSection.PUNGWON_HTML)
        rcps = [r for r in results if r["category"] == "전환우선주"]
        assert len(rcps) == 1

    def test_preferred_face_value(self):
        """Preferred stock face value should be ~1,000,000,310원."""
        results = parse_notes_overhang(TestCBUnderBorrowingsSection.PUNGWON_HTML)
        rcps = [r for r in results if r["category"] == "전환우선주"]
        assert rcps[0]["face_value"] == 1_000_000_310

    def test_preferred_shares(self):
        """Preferred stock convertible shares should be 86,881."""
        results = parse_notes_overhang(TestCBUnderBorrowingsSection.PUNGWON_HTML)
        rcps = [r for r in results if r["category"] == "전환우선주"]
        assert rcps[0]["convertible_shares"] == 86_881

    def test_preferred_conversion_price(self):
        """Preferred stock conversion price should be 11,510원 (1:1 ratio)."""
        results = parse_notes_overhang(TestCBUnderBorrowingsSection.PUNGWON_HTML)
        rcps = [r for r in results if r["category"] == "전환우선주"]
        assert rcps[0]["conversion_price"] == 11_510

    def test_preferred_exercise_period(self):
        """Preferred stock: 1 year after 2025.04.16 ~ 10 years after."""
        results = parse_notes_overhang(TestCBUnderBorrowingsSection.PUNGWON_HTML)
        rcps = [r for r in results if r["category"] == "전환우선주"]
        assert rcps[0]["exercise_start"] == "2026.04.16"
        assert rcps[0]["exercise_end"] == "2035.04.16"

    def test_preferred_active(self):
        """Preferred stock should be active."""
        results = parse_notes_overhang(TestCBUnderBorrowingsSection.PUNGWON_HTML)
        rcps = [r for r in results if r["category"] == "전환우선주"]
        assert rcps[0]["active"] is True

    def test_no_loan_data_in_results(self):
        """Loan tables should not appear in overhang results."""
        results = parse_notes_overhang(TestCBUnderBorrowingsSection.PUNGWON_HTML)
        categories = {r["category"] for r in results}
        assert categories == {"CB", "전환우선주"}
