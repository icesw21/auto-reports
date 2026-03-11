"""Tests for Mark Minervini Trend Template screener."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from auto_reports.screeners.minervini import (
    ScreenParams,
    check_liquidity,
    check_trend_template,
    compute_composite_score,
    compute_indicators,
    compute_rs_ratings,
    detect_vcp,
    fetch_universe,
    screen,
    write_stocks_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(
    prices: list[float],
    days: int | None = None,
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Create a DataFrame with OHLCV columns and DatetimeIndex."""
    if days is None:
        days = len(prices)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=days)[-len(prices):]
    data: dict[str, list[float]] = {"Close": prices}
    if volumes is not None:
        data["Volume"] = volumes
    else:
        data["Volume"] = [100_000] * len(prices)
    if highs is not None:
        data["High"] = highs
    else:
        data["High"] = [p * 1.02 for p in prices]
    if lows is not None:
        data["Low"] = lows
    else:
        data["Low"] = [p * 0.98 for p in prices]
    return pd.DataFrame(data, index=idx)


def _trending_up_prices(n: int = 260, start: float = 1000.0, growth: float = 0.003) -> list[float]:
    """Generate a steadily rising price series (all criteria likely pass)."""
    prices = []
    price = start
    for _ in range(n):
        price *= (1 + growth)
        prices.append(round(price, 2))
    return prices


def _flat_prices(n: int = 260, value: float = 1000.0) -> list[float]:
    """Generate a flat price series."""
    return [value] * n


def _declining_prices(n: int = 260, start: float = 2000.0, decay: float = 0.003) -> list[float]:
    """Generate a steadily declining price series."""
    prices = []
    price = start
    for _ in range(n):
        price *= (1 - decay)
        prices.append(round(price, 2))
    return prices


# ---------------------------------------------------------------------------
# ScreenParams
# ---------------------------------------------------------------------------

class TestScreenParams:

    def test_defaults(self):
        p = ScreenParams()
        assert p.rs_min_percentile == 80
        assert p.min_above_52w_low_pct == 25.0
        assert p.max_below_52w_high_pct == 25.0
        assert p.min_market_cap_krw == 100_000_000_000
        assert p.min_avg_turnover_krw == 1_000_000_000
        assert p.min_price == 1_000
        assert p.check_vcp is False

    def test_strict(self):
        p = ScreenParams.strict()
        assert p.rs_min_percentile == 90
        assert p.min_above_52w_low_pct == 30.0
        assert p.max_below_52w_high_pct == 15.0
        assert p.min_market_cap_krw == 200_000_000_000
        assert p.min_avg_turnover_krw == 1_000_000_000
        assert p.check_vcp is True

    def test_custom(self):
        p = ScreenParams(rs_min_percentile=85, min_price=5_000)
        assert p.rs_min_percentile == 85
        assert p.min_price == 5_000


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------

class TestComputeIndicators:

    _EXPECTED_KEYS = {
        "close", "sma50", "sma150", "sma200",
        "sma200_1m_ago", "week52_high", "week52_low",
        "ma200_slope_pct", "pct_above_52w_low", "pct_below_52w_high",
        "avg_volume_50d", "avg_turnover_50d",
        "return_1m", "return_3m", "return_6m", "return_12m",
        "volatility_20d",
    }

    def test_sufficient_data_returns_dict(self):
        prices = _trending_up_prices(260)
        df = _make_price_df(prices)
        result = compute_indicators(df)

        assert result is not None
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_insufficient_data_returns_none(self):
        prices = _trending_up_prices(100)  # Less than 221
        df = _make_price_df(prices)
        result = compute_indicators(df)
        assert result is None

    def test_sma_values_are_correct(self):
        # Use flat prices — all SMAs should equal the price
        prices = _flat_prices(260, value=5000.0)
        df = _make_price_df(prices)
        result = compute_indicators(df)

        assert result is not None
        assert result["close"] == 5000.0
        assert result["sma50"] == pytest.approx(5000.0)
        assert result["sma150"] == pytest.approx(5000.0)
        assert result["sma200"] == pytest.approx(5000.0)
        assert result["sma200_1m_ago"] == pytest.approx(5000.0)

    def test_52wk_high_low_flat(self):
        prices = _flat_prices(260, value=3000.0)
        df = _make_price_df(prices)
        result = compute_indicators(df)

        assert result is not None
        assert result["week52_high"] == 3000.0
        assert result["week52_low"] == 3000.0

    def test_trending_up_sma_order(self):
        """In an uptrend, shorter SMAs should be above longer ones."""
        prices = _trending_up_prices(260)
        df = _make_price_df(prices)
        result = compute_indicators(df)

        assert result is not None
        assert result["sma50"] > result["sma150"]
        assert result["sma150"] > result["sma200"]

    def test_nan_in_close_handled(self):
        """NaN values in Close are dropped; 258 rows remain (>= 221), so result is valid."""
        prices = _trending_up_prices(260)
        prices[100] = np.nan
        prices[200] = np.nan
        df = _make_price_df(prices)
        result = compute_indicators(df)
        # 258 rows after dropna >= _MIN_TRADING_DAYS (221), so must return a dict
        assert result is not None
        assert result["close"] > 0

    def test_extended_fields_present(self):
        prices = _trending_up_prices(260)
        df = _make_price_df(prices, volumes=[50_000] * 260)
        result = compute_indicators(df)

        assert result is not None
        assert result["ma200_slope_pct"] > 0  # Uptrend → positive slope
        assert result["pct_above_52w_low"] > 0
        assert result["pct_below_52w_high"] >= 0
        assert result["avg_volume_50d"] == pytest.approx(50_000)
        assert result["avg_turnover_50d"] > 0
        assert result["return_3m"] > 0  # Uptrend → positive return
        assert result["volatility_20d"] >= 0

    def test_no_volume_column(self):
        """If Volume is missing, volume fields default to 0."""
        prices = _flat_prices(260, value=5000.0)
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=260)[-260:]
        df = pd.DataFrame({"Close": prices}, index=idx)
        result = compute_indicators(df)

        assert result is not None
        assert result["avg_volume_50d"] == 0
        assert result["avg_turnover_50d"] == 0

    def test_actual_turnover_column_used(self):
        """When Amount column exists, use it instead of Close * Volume."""
        prices = _flat_prices(260, value=1000.0)
        volumes = [100] * 260
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=260)[-260:]
        df = pd.DataFrame({
            "Close": prices,
            "Volume": volumes,
            "Amount": [999_999] * 260,  # Actual turnover
        }, index=idx)
        result = compute_indicators(df)

        assert result is not None
        # Should use Amount column (999_999), not Close*Volume (1000*100=100_000)
        assert result["avg_turnover_50d"] == pytest.approx(999_999)


# ---------------------------------------------------------------------------
# check_trend_template
# ---------------------------------------------------------------------------

class TestCheckTrendTemplate:

    def _passing_indicators(self):
        """Indicators where all criteria should pass."""
        return {
            "close": 1500.0,
            "sma50": 1400.0,
            "sma150": 1300.0,
            "sma200": 1200.0,
            "sma200_1m_ago": 1150.0,
            "week52_high": 1600.0,
            "week52_low": 1000.0,
        }

    def test_all_pass(self):
        ind = self._passing_indicators()
        result = check_trend_template(ind, rs_rating=85.0)

        assert result["all_pass"] is True
        assert result["c1_price_above_150_200"] is True
        assert result["c2_sma150_above_sma200"] is True
        assert result["c3_sma200_trending_up"] is True
        assert result["c4_sma50_above_150_200"] is True
        assert result["c5_price_above_sma50"] is True
        assert result["c6_above_52wk_low"] is True
        assert result["c7_near_52wk_high"] is True
        assert result["c8_rs_rating"] is True

    def test_c1_fails_below_sma150(self):
        ind = self._passing_indicators()
        ind["close"] = 1250.0  # Below sma150 (1300)
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c1_price_above_150_200"] is False
        assert result["all_pass"] is False

    def test_c1_fails_below_sma200(self):
        ind = self._passing_indicators()
        ind["close"] = 1100.0  # Below sma200 (1200)
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c1_price_above_150_200"] is False

    def test_c2_fails(self):
        ind = self._passing_indicators()
        ind["sma150"] = 1100.0  # Below sma200 (1200)
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c2_sma150_above_sma200"] is False
        assert result["all_pass"] is False

    def test_c3_fails_sma200_declining(self):
        ind = self._passing_indicators()
        ind["sma200_1m_ago"] = 1250.0  # sma200 was higher 1 month ago
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c3_sma200_trending_up"] is False
        assert result["all_pass"] is False

    def test_c4_fails_sma50_below_sma150(self):
        ind = self._passing_indicators()
        ind["sma50"] = 1250.0  # Below sma150 (1300)
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c4_sma50_above_150_200"] is False
        assert result["all_pass"] is False

    def test_c5_fails_below_sma50(self):
        ind = self._passing_indicators()
        ind["close"] = 1350.0  # Below sma50 (1400)
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c5_price_above_sma50"] is False
        assert result["all_pass"] is False

    def test_c6_fails_not_25pct_above_low(self):
        ind = self._passing_indicators()
        ind["week52_low"] = 1300.0  # 1300 * 1.25 = 1625 > close (1500)
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c6_above_52wk_low"] is False
        assert result["all_pass"] is False

    def test_c6_boundary_exact_25pct(self):
        ind = self._passing_indicators()
        ind["close"] = 1250.0
        ind["week52_low"] = 1000.0  # 1000 * 1.25 = 1250 == close
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c6_above_52wk_low"] is True

    def test_c6_custom_threshold(self):
        ind = self._passing_indicators()
        ind["close"] = 1250.0
        ind["week52_low"] = 1000.0
        # 30% threshold: 1000 * 1.30 = 1300 > 1250 → fails
        result = check_trend_template(ind, rs_rating=85.0, min_above_52w_low_pct=30.0)
        assert result["c6_above_52wk_low"] is False

    def test_c7_fails_too_far_from_high(self):
        ind = self._passing_indicators()
        ind["week52_high"] = 2200.0  # 2200 * 0.75 = 1650 > close (1500)
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c7_near_52wk_high"] is False
        assert result["all_pass"] is False

    def test_c7_boundary_exact_75pct(self):
        ind = self._passing_indicators()
        ind["close"] = 1500.0
        ind["week52_high"] = 2000.0  # 2000 * 0.75 = 1500 == close
        result = check_trend_template(ind, rs_rating=85.0)
        assert result["c7_near_52wk_high"] is True

    def test_c7_custom_threshold(self):
        ind = self._passing_indicators()
        ind["close"] = 1500.0
        ind["week52_high"] = 1800.0  # 1800 * 0.85 = 1530 > 1500 → fails
        result = check_trend_template(ind, rs_rating=85.0, max_below_52w_high_pct=15.0)
        assert result["c7_near_52wk_high"] is False

    def test_c8_fails_low_rs(self):
        ind = self._passing_indicators()
        result = check_trend_template(ind, rs_rating=50.0)
        assert result["c8_rs_rating"] is False
        assert result["all_pass"] is False

    def test_c8_custom_min_rs(self):
        ind = self._passing_indicators()
        result = check_trend_template(ind, rs_rating=75.0, min_rs=80)
        assert result["c8_rs_rating"] is False

    def test_c8_boundary_exactly_min(self):
        ind = self._passing_indicators()
        result = check_trend_template(ind, rs_rating=80.0, min_rs=80)
        assert result["c8_rs_rating"] is True


# ---------------------------------------------------------------------------
# check_liquidity
# ---------------------------------------------------------------------------

class TestCheckLiquidity:

    def _base_indicators(self):
        return {
            "close": 5000.0,
            "avg_volume_50d": 100_000,
            "avg_turnover_50d": 1_000_000_000,  # 10억
        }

    def test_all_pass(self):
        ind = self._base_indicators()
        result = check_liquidity(ind, market_cap=200_000_000_000)
        assert all(result.values())

    def test_penny_stock_fails(self):
        ind = self._base_indicators()
        ind["close"] = 500.0  # Below 1000
        result = check_liquidity(ind)
        assert result["f1_min_price"] is False

    def test_low_volume_fails(self):
        ind = self._base_indicators()
        ind["avg_volume_50d"] = 10_000  # Below 50000
        result = check_liquidity(ind)
        assert result["f2_avg_volume"] is False

    def test_low_turnover_fails(self):
        ind = self._base_indicators()
        ind["avg_turnover_50d"] = 100_000_000  # Below 5억
        result = check_liquidity(ind)
        assert result["f3_avg_turnover"] is False

    def test_low_market_cap_fails(self):
        ind = self._base_indicators()
        result = check_liquidity(ind, market_cap=50_000_000_000)  # 500억 < 1000억
        assert result["f4_market_cap"] is False

    def test_market_cap_none_skipped(self):
        """When market cap is None (unavailable), filter is skipped."""
        ind = self._base_indicators()
        result = check_liquidity(ind, market_cap=None)
        assert result["f4_market_cap"] is True

    def test_market_cap_zero_fails(self):
        """Explicit zero market cap should NOT bypass the filter."""
        ind = self._base_indicators()
        result = check_liquidity(ind, market_cap=0)
        assert result["f4_market_cap"] is False

    def test_custom_params(self):
        ind = self._base_indicators()
        ind["close"] = 500.0
        p = ScreenParams(min_price=100)
        result = check_liquidity(ind, params=p)
        assert result["f1_min_price"] is True  # 500 >= 100


# ---------------------------------------------------------------------------
# compute_rs_ratings
# ---------------------------------------------------------------------------

class TestComputeRsRatings:

    def test_empty_input(self):
        assert compute_rs_ratings({}) == {}

    def test_single_stock(self):
        prices = _trending_up_prices(260)
        df = _make_price_df(prices)
        ratings = compute_rs_ratings({"000001": df})
        assert "000001" in ratings
        # Single stock should be ranked at ~50 (midpoint) or 99 (only one)
        assert 1 <= ratings["000001"] <= 99

    def test_ranking_order(self):
        """Stock with higher returns should have higher RS rating."""
        strong = _make_price_df(_trending_up_prices(260, growth=0.005))
        weak = _make_price_df(_trending_up_prices(260, growth=0.001))
        flat = _make_price_df(_flat_prices(260))

        ratings = compute_rs_ratings({
            "STRONG": strong,
            "WEAK": weak,
            "FLAT": flat,
        })

        assert ratings["STRONG"] > ratings["WEAK"]
        assert ratings["WEAK"] > ratings["FLAT"]

    def test_declining_stock_low_rating(self):
        """Declining stock should have lowest RS among peers."""
        up = _make_price_df(_trending_up_prices(260))
        down = _make_price_df(_declining_prices(260))
        flat = _make_price_df(_flat_prices(260))

        ratings = compute_rs_ratings({"UP": up, "DOWN": down, "FLAT": flat})
        assert ratings["DOWN"] < ratings["UP"]
        assert ratings["DOWN"] < ratings["FLAT"]

    def test_insufficient_data_excluded(self):
        short = _make_price_df(_trending_up_prices(30))  # < 63 days
        ok = _make_price_df(_trending_up_prices(260))

        ratings = compute_rs_ratings({"SHORT": short, "OK": ok})
        assert "SHORT" not in ratings
        assert "OK" in ratings

    def test_ratings_in_range(self):
        data = {}
        for i in range(20):
            growth = 0.001 + i * 0.0005
            data[f"S{i:03d}"] = _make_price_df(_trending_up_prices(260, growth=growth))

        ratings = compute_rs_ratings(data)
        for ticker, rating in ratings.items():
            assert 1 <= rating <= 99, f"{ticker} rating {rating} out of range"


# ---------------------------------------------------------------------------
# detect_vcp
# ---------------------------------------------------------------------------

class TestDetectVcp:

    def test_contracting_ranges_detected(self):
        """Progressively narrower 20-day ranges should be detected as VCP."""
        n = 132
        closes = []
        highs = []
        lows = []
        base = 1000.0
        for i in range(n):
            segment = i // 20
            # Each segment has narrower amplitude
            amplitude = 0.15 - segment * 0.02  # 15%, 13%, 11%, 9%, 7%, 5%, 3%
            amplitude = max(amplitude, 0.01)
            closes.append(base)
            highs.append(base * (1 + amplitude / 2))
            lows.append(base * (1 - amplitude / 2))

        df = _make_price_df(closes, highs=highs, lows=lows)
        result = detect_vcp(df)
        assert result is True

    def test_expanding_ranges_not_detected(self):
        """Expanding ranges should NOT be detected as VCP."""
        n = 132
        closes = []
        highs = []
        lows = []
        base = 1000.0
        for i in range(n):
            segment = i // 20
            amplitude = 0.05 + segment * 0.03  # Expanding
            closes.append(base)
            highs.append(base * (1 + amplitude / 2))
            lows.append(base * (1 - amplitude / 2))

        df = _make_price_df(closes, highs=highs, lows=lows)
        result = detect_vcp(df)
        assert result is False

    def test_insufficient_data(self):
        prices = _flat_prices(30)
        df = _make_price_df(prices)
        assert detect_vcp(df) is False

    def test_no_high_low_columns(self):
        prices = _flat_prices(132)
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=132)[-132:]
        df = pd.DataFrame({"Close": prices}, index=idx)
        assert detect_vcp(df) is False

    def test_base_too_deep(self):
        """Base depth exceeding threshold should reject VCP."""
        n = 132
        closes = [1000.0] * n
        highs = [1500.0] * n   # Very wide range
        lows = [500.0] * n
        df = _make_price_df(closes, highs=highs, lows=lows)
        p = ScreenParams(vcp_max_base_depth_pct=20.0)
        assert detect_vcp(df, p) is False


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------

class TestComputeCompositeScore:

    def test_perfect_stock(self):
        """High RS, near 52w high, strong MA alignment, positive slope and momentum."""
        ind = {
            "pct_below_52w_high": 2.0,   # Very close to high
            "sma50": 1500.0,
            "sma200": 1200.0,            # 25% spread
            "ma200_slope_pct": 5.0,       # Strong uptrend
            "return_1m": 10.0,            # Good momentum
        }
        score = compute_composite_score(ind, rs_percentile=95.0)
        assert score > 60.0

    def test_weak_stock(self):
        """Low RS, far from high, weak alignment."""
        ind = {
            "pct_below_52w_high": 20.0,
            "sma50": 1050.0,
            "sma200": 1000.0,            # 5% spread
            "ma200_slope_pct": 0.5,
            "return_1m": 1.0,
        }
        score = compute_composite_score(ind, rs_percentile=30.0)
        assert score < 30.0

    def test_score_in_range(self):
        ind = {
            "pct_below_52w_high": 10.0,
            "sma50": 1200.0,
            "sma200": 1100.0,
            "ma200_slope_pct": 2.0,
            "return_1m": 5.0,
        }
        score = compute_composite_score(ind, rs_percentile=80.0)
        assert 0 <= score <= 100

    def test_missing_fields_handled(self):
        """Missing optional fields should not crash."""
        ind = {}
        score = compute_composite_score(ind, rs_percentile=50.0)
        assert score >= 0


# ---------------------------------------------------------------------------
# screen (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestScreen:

    @patch("auto_reports.screeners.minervini.fetch_price_data")
    @patch("auto_reports.screeners.minervini.fetch_universe")
    def test_screen_returns_passing_stocks(self, mock_universe, mock_prices):
        # Universe: 3 stocks
        mock_universe.return_value = pd.DataFrame({
            "Code": ["000001", "000002", "000003"],
            "Name": ["StrongCo", "WeakCo", "FlatCo"],
        })

        # Price data: strong stock passes, weak/flat don't
        strong_prices = _trending_up_prices(260, growth=0.005)
        weak_prices = _declining_prices(260)
        flat_prices = _flat_prices(260)

        mock_prices.return_value = {
            "000001": _make_price_df(strong_prices),
            "000002": _make_price_df(weak_prices),
            "000003": _make_price_df(flat_prices),
        }

        # Use low thresholds to isolate trend criteria from liquidity filters
        params = ScreenParams(
            rs_min_percentile=1,
            min_avg_volume=0,
            min_avg_turnover_krw=0,
            min_market_cap_krw=0,
            min_price=0,
        )
        results = screen(market="ALL", params=params)

        # At least the strong stock should pass (all SMAs aligned)
        tickers = [r["ticker"] for r in results]
        assert "000001" in tickers

        # Declining stock should NOT pass
        assert "000002" not in tickers

        # Results should have score field
        if results:
            assert "score" in results[0]

    @patch("auto_reports.screeners.minervini.fetch_price_data")
    @patch("auto_reports.screeners.minervini.fetch_universe")
    def test_screen_empty_universe(self, mock_universe, mock_prices):
        mock_universe.return_value = pd.DataFrame({"Code": [], "Name": []})
        mock_prices.return_value = {}

        results = screen()
        assert results == []

    @patch("auto_reports.screeners.minervini.fetch_price_data")
    @patch("auto_reports.screeners.minervini.fetch_universe")
    def test_screen_no_sufficient_data(self, mock_universe, mock_prices):
        mock_universe.return_value = pd.DataFrame({
            "Code": ["000001"],
            "Name": ["ShortCo"],
        })
        # Only 100 days of data — insufficient
        mock_prices.return_value = {
            "000001": _make_price_df(_trending_up_prices(100)),
        }

        results = screen()
        assert results == []

    @patch("auto_reports.screeners.minervini.fetch_price_data")
    @patch("auto_reports.screeners.minervini.fetch_universe")
    def test_screen_results_sorted_by_score(self, mock_universe, mock_prices):
        mock_universe.return_value = pd.DataFrame({
            "Code": ["S1", "S2", "S3"],
            "Name": ["Strong", "Medium", "Moderate"],
        })

        # All trending up but at different rates
        mock_prices.return_value = {
            "S1": _make_price_df(_trending_up_prices(260, growth=0.006)),
            "S2": _make_price_df(_trending_up_prices(260, growth=0.004)),
            "S3": _make_price_df(_trending_up_prices(260, growth=0.005)),
        }

        params = ScreenParams(
            rs_min_percentile=1,
            min_avg_volume=0,
            min_avg_turnover_krw=0,
            min_market_cap_krw=0,
            min_price=0,
        )
        results = screen(params=params)
        if len(results) >= 2:
            # Should be sorted by score descending
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]

    @patch("auto_reports.screeners.minervini.fetch_price_data")
    @patch("auto_reports.screeners.minervini.fetch_universe")
    def test_screen_liquidity_filter_rejects(self, mock_universe, mock_prices):
        """Stocks that pass trend but fail liquidity should be excluded."""
        mock_universe.return_value = pd.DataFrame({
            "Code": ["000001"],
            "Name": ["LowVol"],
        })

        # Strong trend but low volume
        prices = _trending_up_prices(260, growth=0.005)
        mock_prices.return_value = {
            "000001": _make_price_df(prices, volumes=[100] * 260),
        }

        # Default params have min_avg_volume=50000 which 100 fails
        results = screen(params=ScreenParams(rs_min_percentile=1, min_price=0,
                                             min_avg_turnover_krw=0, min_market_cap_krw=0))
        tickers = [r["ticker"] for r in results]
        assert "000001" not in tickers


# ---------------------------------------------------------------------------
# fetch_universe (retry logic)
# ---------------------------------------------------------------------------

class TestFetchUniverse:

    @patch("auto_reports.screeners.minervini.fdr.StockListing")
    def test_desc_endpoint_used_first(self, mock_listing):
        """DESC endpoint is tried first; success means no Marcap fallback."""
        mock_listing.return_value = pd.DataFrame({
            "Code": ["005930", "000660"],
            "Name": ["Samsung", "SK Hynix"],
        })

        df = fetch_universe("KOSPI")
        assert len(df) == 2
        # Should call KOSPI-DESC, not KOSPI
        mock_listing.assert_called_once_with("KOSPI-DESC")

    @patch("auto_reports.screeners.minervini.fdr.StockListing")
    def test_all_market_uses_krx_desc(self, mock_listing):
        mock_listing.return_value = pd.DataFrame({
            "Code": ["005930"], "Name": ["Samsung"],
        })

        fetch_universe("ALL")
        mock_listing.assert_called_once_with("KRX-DESC")

    @patch("auto_reports.screeners.minervini.time.sleep")
    @patch("auto_reports.screeners.minervini.fdr.StockListing")
    def test_fallback_to_marcap_with_retry(self, mock_listing, mock_sleep):
        """When DESC fails, falls back to Marcap with retry."""
        import json
        mock_listing.side_effect = [
            # DESC call fails
            json.JSONDecodeError("Expecting value", "", 0),
            # Marcap attempt 1 fails
            json.JSONDecodeError("Expecting value", "", 0),
            # Marcap attempt 2 succeeds
            pd.DataFrame({"Code": ["005930"], "Name": ["Samsung"]}),
        ]

        df = fetch_universe("ALL", max_retries=5)
        assert len(df) == 1
        # 1 DESC + 2 Marcap = 3 total
        assert mock_listing.call_count == 3
        # Only 1 sleep (between Marcap retries)
        assert mock_sleep.call_count == 1

    @patch("auto_reports.screeners.minervini.time.sleep")
    @patch("auto_reports.screeners.minervini.fdr.StockListing")
    def test_raises_after_max_retries(self, mock_listing, mock_sleep):
        """Should raise after exhausting DESC + all Marcap retries."""
        import json
        mock_listing.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        with pytest.raises(json.JSONDecodeError):
            fetch_universe("ALL", max_retries=3)
        # 1 DESC + 3 Marcap = 4 total
        assert mock_listing.call_count == 4

    @patch("auto_reports.screeners.minervini.fdr.StockListing")
    def test_empty_df_falls_back(self, mock_listing):
        """Empty DESC DataFrame triggers Marcap fallback; empty Marcap raises."""
        mock_listing.return_value = pd.DataFrame()

        with pytest.raises(RuntimeError, match="Failed to fetch"):
            fetch_universe("ALL", max_retries=2)

    @patch("auto_reports.screeners.minervini.fdr.StockListing")
    def test_etf_spac_excluded(self, mock_listing):
        """ETF, ETN, 리츠, 스팩, SPAC should be excluded from universe."""
        mock_listing.return_value = pd.DataFrame({
            "Code": ["005930", "069500", "100000", "200000", "300000"],
            "Name": ["삼성전자", "KODEX 200 ETF", "스팩1호", "맥쿼리리츠", "일반기업"],
        })

        df = fetch_universe("ALL")
        names = df["Name"].tolist()
        assert "삼성전자" in names
        assert "일반기업" in names
        assert "KODEX 200 ETF" not in names
        assert "스팩1호" not in names
        assert "맥쿼리리츠" not in names

    @patch("auto_reports.screeners.minervini.fdr.StockListing")
    def test_preferred_stocks_excluded(self, mock_listing):
        """Preferred stocks (우선주) with codes not ending in '0' are excluded."""
        mock_listing.return_value = pd.DataFrame({
            "Code": ["005930", "005935", "005937", "000660", "000665"],
            "Name": ["삼성전자", "삼성전자우", "삼성전자우B", "SK하이닉스", "SK하이닉스우"],
        })

        df = fetch_universe("ALL")
        codes = df["Code"].tolist()
        assert "005930" in codes   # 삼성전자 (보통주)
        assert "000660" in codes   # SK하이닉스 (보통주)
        assert "005935" not in codes  # 삼성전자우
        assert "005937" not in codes  # 삼성전자우B
        assert "000665" not in codes  # SK하이닉스우


# ---------------------------------------------------------------------------
# write_stocks_json
# ---------------------------------------------------------------------------

class TestWriteStocksJson:

    def test_write_basic(self, tmp_path):
        output = tmp_path / "stocks.json"
        results = [
            {"ticker": "005930", "name": "Samsung", "close": 70000, "rs_rating": 95, "score": 80.0},
            {"ticker": "000660", "name": "SK Hynix", "close": 150000, "rs_rating": 88, "score": 75.0},
        ]

        path = write_stocks_json(results, str(output))

        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"005930": "Samsung", "000660": "SK Hynix"}

    def test_write_empty(self, tmp_path):
        output = tmp_path / "stocks.json"
        path = write_stocks_json([], str(output))

        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {}

    def test_write_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "sub" / "dir" / "stocks.json"
        results = [{"ticker": "000001", "name": "Test", "close": 1000, "rs_rating": 80, "score": 50.0}]

        path = write_stocks_json(results, str(output))
        assert path.exists()

    def test_write_korean_names(self, tmp_path):
        output = tmp_path / "stocks.json"
        results = [
            {"ticker": "101490", "name": "에스앤에스텍", "close": 50000, "rs_rating": 90, "score": 70.0},
        ]

        path = write_stocks_json(results, str(output))

        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["101490"] == "에스앤에스텍"
