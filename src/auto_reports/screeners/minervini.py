"""Mark Minervini Trend Template screener for Korean stocks.

Core criteria (8 conditions):
1. Price > 150-day SMA AND 200-day SMA
2. 150-day SMA > 200-day SMA
3. 200-day SMA trending up for >= 1 month
4. 50-day SMA > 150-day SMA AND 200-day SMA
5. Price > 50-day SMA
6. Price >= 52-week low * 1.25 (25% above)
7. Price >= 52-week high * 0.75 (within 25% of high)
8. RS Rating >= 80 (relative strength percentile)

Domestic market filters:
- Market cap >= 1000억원
- 50-day avg turnover >= 10억원
- Min price >= 1,000원 (penny stock exclusion)
- ETF/ETN/SPAC/리츠/우선주 excluded

RS: 2×Q3 + Q6 + Q12 weighted return, percentile ranked across universe.
VCP: Optional 20-day volatility contraction pattern detection.
Score (0-100): RS 30% + high proximity 20% + MA alignment 20%
               + 200d slope 15% + momentum 15%.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum trading days needed for SMA200 + 21-day shift
_MIN_TRADING_DAYS = 221

# Name keywords to exclude from universe
_EXCLUDE_KEYWORDS = ("ETF", "ETN", "리츠", "스팩", "SPAC")


# ─── Parameters ───────────────────────────────────────────────


@dataclass
class ScreenParams:
    """Configurable parameters for the Minervini screener.

    Adapts the original SEPA template to the Korean market with
    liquidity and market-cap filters.
    """

    # 52-week range thresholds
    min_above_52w_low_pct: float = 25.0
    max_below_52w_high_pct: float = 25.0

    # 200-day uptrend check period (trading days, ~1 month)
    ma_long_uptrend_days: int = 22

    # RS minimum percentile
    rs_min_percentile: int = 80

    # Domestic market liquidity filters
    min_avg_volume: int = 50_000
    min_avg_turnover_krw: int = 1_000_000_000      # 10억원
    min_market_cap_krw: int = 100_000_000_000      # 1000억원
    min_price: int = 1_000

    # VCP (Volatility Contraction Pattern)
    check_vcp: bool = False
    vcp_contraction_count: int = 3
    vcp_max_base_depth_pct: float = 35.0

    @classmethod
    def strict(cls) -> ScreenParams:
        """Strict mode: higher thresholds, VCP enabled."""
        return cls(
            min_above_52w_low_pct=30.0,
            max_below_52w_high_pct=15.0,
            rs_min_percentile=90,
            min_avg_volume=100_000,
            min_avg_turnover_krw=1_000_000_000,    # 10억
            min_market_cap_krw=200_000_000_000,     # 2000억
            check_vcp=True,
        )


# ─── Universe ─────────────────────────────────────────────────


def fetch_universe(market: str = "ALL", max_retries: int = 5) -> pd.DataFrame:
    """Fetch all KOSPI/KOSDAQ tickers from FinanceDataReader.

    Uses the DESC listing endpoint (``KRX-DESC``) which is more reliable
    than the Marcap endpoint (``KRX``).  Falls back to Marcap with retry
    if DESC also fails.  The KRX API sometimes returns empty/invalid JSON
    ('LOGOUT' response), so retries with exponential backoff.

    Args:
        market: "KOSPI", "KOSDAQ", or "ALL" (default).
        max_retries: Maximum retry attempts (default 5).

    Returns:
        DataFrame with at least 'Code' and 'Name' columns.
    """
    market_key = market.upper()

    # DESC endpoints are more reliable (description-based, not Marcap)
    desc_map = {"KOSPI": "KOSPI-DESC", "KOSDAQ": "KOSDAQ-DESC"}
    marcap_map = {"KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}

    if market_key in ("KOSPI", "KOSDAQ"):
        desc_arg = desc_map[market_key]
        marcap_arg = marcap_map[market_key]
    else:
        desc_arg = "KRX-DESC"
        marcap_arg = "KRX"

    # Try DESC endpoint first (more reliable)
    try:
        df = fdr.StockListing(desc_arg)
        if df is not None and not df.empty:
            logger.info("Universe loaded via %s: %d tickers", desc_arg, len(df))

            # Filter by market if DESC returns all and we want a subset
            if market_key in ("KOSPI", "KOSDAQ") and "Market" in df.columns:
                df = df[df["Market"] == market_key]

            return _normalise_universe(df, market_key)
    except Exception:
        logger.warning("DESC listing (%s) failed, falling back to Marcap", desc_arg)

    # Fallback: Marcap endpoint with retry
    df = None
    for attempt in range(max_retries):
        try:
            df = fdr.StockListing(marcap_arg)
            if df is not None and not df.empty:
                break
        except Exception:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "KRX listing attempt %d/%d failed, retrying in %ds...",
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)

    if df is None or df.empty:
        raise RuntimeError(f"Failed to fetch {marcap_arg} listing after {max_retries} attempts")

    return _normalise_universe(df, market_key)


def _normalise_universe(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Normalise universe DataFrame columns, filter markets, exclude ETF/SPAC."""

    # Normalise column names — fdr versions differ
    if "Symbol" in df.columns and "Code" not in df.columns:
        df = df.rename(columns={"Symbol": "Code"})

    # Ensure 6-digit zero-padded codes
    df["Code"] = df["Code"].astype(str).str.strip().str.zfill(6)

    # Drop rows without a name
    df = df.dropna(subset=["Name"])
    df = df[df["Name"].str.strip() != ""]

    # Filter to KOSPI/KOSDAQ only (exclude KONEX, KOSDAQ GLOBAL, etc.)
    if "Market" in df.columns and market == "ALL":
        df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])]

    # Exclude ETF, ETN, 리츠, 스팩, SPAC by name
    name_mask = df["Name"].apply(
        lambda n: not any(kw in n for kw in _EXCLUDE_KEYWORDS)
    )
    excluded_name = len(df) - name_mask.sum()
    df = df[name_mask]
    if excluded_name:
        logger.info("Excluded %d ETF/ETN/SPAC/리츠 from universe", excluded_name)

    # Exclude preferred stocks (우선주): common stocks end in '0'
    pref_mask = df["Code"].str[-1] == "0"
    excluded_pref = len(df) - pref_mask.sum()
    df = df[pref_mask]
    if excluded_pref:
        logger.info("Excluded %d preferred stocks (우선주) from universe", excluded_pref)

    logger.info("Universe loaded: %d tickers (market=%s)", len(df), market)
    return df


# ─── Price data ───────────────────────────────────────────────


def fetch_price_data(
    tickers: list[str],
    days: int = 400,
    max_workers: int = 4,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for multiple tickers in parallel.

    Args:
        tickers: List of 6-digit ticker codes.
        days: Calendar days of history to fetch (default 400, ~280 trading days).
        max_workers: Thread pool size.
        progress_callback: Optional callable(done, total) for progress updates.

    Returns:
        Dict mapping ticker -> DataFrame with at least a 'Close' column.
    """
    end = date.today()
    start = end - timedelta(days=days)
    start_str = start.isoformat()
    end_str = end.isoformat()

    results: dict[str, pd.DataFrame] = {}
    done_count = 0

    def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame | None]:
        try:
            df = fdr.DataReader(ticker, start_str, end_str)
            if df is not None and not df.empty and "Close" in df.columns:
                closes = df["Close"].dropna()
                if len(closes) >= _MIN_TRADING_DAYS:
                    return ticker, df
            return ticker, None
        except Exception:
            logger.debug("Price fetch failed for %s", ticker)
            return ticker, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, df = future.result()
            if df is not None:
                results[ticker] = df
            done_count += 1
            if progress_callback and done_count % 100 == 0:
                progress_callback(done_count, len(tickers))

    if progress_callback:
        progress_callback(len(tickers), len(tickers))

    logger.info(
        "Price data fetched: %d/%d tickers have sufficient history",
        len(results), len(tickers),
    )
    return results


# ─── Technical indicators ─────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> dict | None:
    """Compute technical indicators for a single stock.

    Returns dict with SMA, 52-week range, volume, returns, and volatility.
    Returns None if data is insufficient.
    """
    close = df["Close"].dropna()
    if len(close) < _MIN_TRADING_DAYS:
        return None

    # Volume (fallback to zeros if absent)
    if "Volume" in df.columns:
        volume = df["Volume"].reindex(close.index).fillna(0)
    else:
        volume = pd.Series(0, index=close.index)

    # Actual turnover column (거래대금) if available; else approximate
    amount_col = next(
        (c for c in ("Amount", "Turnover", "거래대금") if c in df.columns),
        None,
    )
    if amount_col is not None:
        turnover = df[amount_col].reindex(close.index).fillna(0)
    else:
        turnover = close * volume

    # Moving averages
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    sma200_1m_ago = sma200.shift(21)

    # 52-week high/low
    week52_high = close.rolling(252, min_periods=200).max()
    week52_low = close.rolling(252, min_periods=200).min()

    latest = close.index[-1]

    current = close[latest]
    s50 = sma50[latest]
    s150 = sma150[latest]
    s200 = sma200[latest]
    s200_1m = sma200_1m_ago[latest]
    h52 = week52_high[latest]
    l52 = week52_low[latest]

    core = {
        "close": current,
        "sma50": s50,
        "sma150": s150,
        "sma200": s200,
        "sma200_1m_ago": s200_1m,
        "week52_high": h52,
        "week52_low": l52,
    }

    # Any NaN in core means insufficient data
    if any(pd.isna(v) for v in core.values()):
        return None

    # 200-day slope (%)
    ma200_slope_pct = (s200 / s200_1m - 1) * 100 if s200_1m > 0 else 0.0

    # 52-week position (%)
    pct_above_low = (current / l52 - 1) * 100 if l52 > 0 else 0.0
    pct_below_high = (1 - current / h52) * 100 if h52 > 0 else 0.0

    # Period returns (%)
    n = len(close)

    def _ret(days: int) -> float:
        return (current / close.iloc[-days] - 1) * 100 if n >= days else 0.0

    # Volume / turnover (50-day average)
    tail = min(50, len(volume))
    avg_vol = float(volume.iloc[-tail:].mean())
    avg_turn = float(turnover.iloc[-tail:].mean())

    # 20-day annualised volatility
    daily_ret = close.pct_change().iloc[-20:]
    vol_20d = float(daily_ret.std() * np.sqrt(252) * 100) if len(daily_ret) >= 10 else 0.0

    return {
        **core,
        "ma200_slope_pct": ma200_slope_pct,
        "pct_above_52w_low": pct_above_low,
        "pct_below_52w_high": pct_below_high,
        "avg_volume_50d": avg_vol,
        "avg_turnover_50d": avg_turn,
        "return_1m": _ret(22),
        "return_3m": _ret(63),
        "return_6m": _ret(126),
        "return_12m": _ret(252),
        "volatility_20d": vol_20d,
    }


# ─── RS Ratings ───────────────────────────────────────────────


def compute_rs_ratings(price_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute Relative Strength ratings for all stocks.

    Uses weighted composite: ``2×Q3_return + Q6_return + Q12_return``,
    then percentile-ranked across the universe (1-99 scale).
    """
    scores: dict[str, float] = {}

    for ticker, df in price_data.items():
        close = df["Close"].dropna()
        n = len(close)
        if n < 63:  # Need at least 1 quarter
            continue

        current = close.iloc[-1]

        r3 = current / close.iloc[-63] - 1
        r6 = (current / close.iloc[-126] - 1) if n >= 126 else r3
        r12 = (current / close.iloc[-252] - 1) if n >= 252 else r6

        score = 2 * r3 + r6 + r12
        scores[ticker] = score

    if not scores:
        return {}

    # Percentile rank (1-99)
    s = pd.Series(scores)
    ranked = s.rank(pct=True).mul(98).add(1).round().clip(1, 99)
    return ranked.to_dict()


# ─── Trend Template ───────────────────────────────────────────


def check_trend_template(
    indicators: dict,
    rs_rating: float,
    min_rs: int = 80,
    min_above_52w_low_pct: float = 25.0,
    max_below_52w_high_pct: float = 25.0,
) -> dict:
    """Evaluate all 8 Minervini Trend Template criteria.

    Args:
        indicators: Output of compute_indicators().
        rs_rating: Pre-computed RS percentile rating (1-99).
        min_rs: Minimum RS rating threshold (default 80).
        min_above_52w_low_pct: Minimum % above 52-week low (default 25).
        max_below_52w_high_pct: Maximum % below 52-week high (default 25).

    Returns:
        Dict with each criterion result and 'all_pass' boolean.
    """
    c = indicators["close"]

    results = {
        "c1_price_above_150_200": (c > indicators["sma150"]) and (c > indicators["sma200"]),
        "c2_sma150_above_sma200": indicators["sma150"] > indicators["sma200"],
        "c3_sma200_trending_up": indicators["sma200"] > indicators["sma200_1m_ago"],
        "c4_sma50_above_150_200": (
            indicators["sma50"] > indicators["sma150"]
            and indicators["sma50"] > indicators["sma200"]
        ),
        "c5_price_above_sma50": c > indicators["sma50"],
        "c6_above_52wk_low": c >= indicators["week52_low"] * (1 + min_above_52w_low_pct / 100),
        "c7_near_52wk_high": c >= indicators["week52_high"] * (1 - max_below_52w_high_pct / 100),
        "c8_rs_rating": rs_rating >= min_rs,
    }
    results["all_pass"] = all(results.values())
    return results


# ─── Liquidity filter ─────────────────────────────────────────


def check_liquidity(
    indicators: dict,
    market_cap: int | None = None,
    params: ScreenParams | None = None,
) -> dict:
    """Check domestic market liquidity filters.

    Returns dict mapping filter name -> bool (True = passes).
    market_cap=None means data unavailable (filter skipped).
    """
    p = params or ScreenParams()
    return {
        "f1_min_price": indicators["close"] >= p.min_price,
        "f2_avg_volume": indicators.get("avg_volume_50d", 0) >= p.min_avg_volume,
        "f3_avg_turnover": indicators.get("avg_turnover_50d", 0) >= p.min_avg_turnover_krw,
        "f4_market_cap": market_cap >= p.min_market_cap_krw if market_cap is not None else True,
    }


# ─── VCP ──────────────────────────────────────────────────────


def detect_vcp(df: pd.DataFrame, params: ScreenParams | None = None) -> bool:
    """Detect Volatility Contraction Pattern (VCP).

    Splits the last 6 months into 20-day windows and checks whether
    the high-low range contracts progressively.
    """
    p = params or ScreenParams()

    if "High" not in df.columns or "Low" not in df.columns:
        return False

    high = df["High"].iloc[-132:]
    low = df["Low"].iloc[-132:]

    if len(high) < 60:
        return False

    window = 20
    ranges: list[float] = []
    for i in range(0, len(high) - window + 1, window):
        seg = high.iloc[i:i + window]
        if len(seg) < window:
            break  # Skip partial trailing segment
        seg_high = seg.max()
        seg_low = low.iloc[i:i + window].min()
        if seg_low > 0:
            ranges.append((seg_high / seg_low - 1) * 100)

    if len(ranges) < 3:
        return False

    # Max base depth check
    base_depth = (1 - low.min() / high.max()) * 100
    if base_depth > p.vcp_max_base_depth_pct:
        return False

    # Count contractions (range shrinks vs previous window)
    contractions = sum(1 for i in range(1, len(ranges)) if ranges[i] < ranges[i - 1])
    return contractions >= p.vcp_contraction_count


# ─── Composite Score ──────────────────────────────────────────


def compute_composite_score(indicators: dict, rs_percentile: float) -> float:
    """Composite score (0-100).

    Weights:
        RS percentile        30%
        52-week high proximity  20%
        MA alignment strength  20%
        200-day slope          15%
        1-month momentum       15%
    """
    score = 0.0

    # RS (30%)
    score += min(rs_percentile, 100) * 0.30

    # 52-week high proximity (20%) — closer to high → higher
    pct_below = indicators.get("pct_below_52w_high", 25.0)
    proximity = max(0.0, 100 - pct_below * 4)
    score += proximity * 0.20

    # MA alignment strength (20%) — 50d/200d spread
    sma50 = indicators.get("sma50", 0)
    sma200 = indicators.get("sma200", 0)
    if sma50 > 0 and sma200 > 0:
        spread = (sma50 / sma200 - 1) * 100
        score += min(max(spread * 5, 0), 100) * 0.20

    # 200-day slope (15%)
    slope = indicators.get("ma200_slope_pct", 0)
    score += min(max(slope * 10, 0), 100) * 0.15

    # 1-month momentum (15%)
    mom = indicators.get("return_1m", 0)
    score += min(max(mom * 5, 0), 100) * 0.15

    return round(score, 1)


# ─── Main screen pipeline ────────────────────────────────────


def screen(
    market: str = "ALL",
    min_rs: int = 80,
    max_workers: int = 4,
    params: ScreenParams | None = None,
    progress_callback=None,
) -> list[dict]:
    """Run the full Minervini Trend Template screen.

    Args:
        market: "KOSPI", "KOSDAQ", or "ALL".
        min_rs: Minimum RS Rating threshold (default 80).
        max_workers: Parallel workers for price fetching.
        params: Optional ScreenParams (overrides min_rs if provided).
        progress_callback: Optional callable(phase, done, total).

    Returns:
        List of dicts sorted by composite score descending.
    """
    p = params or ScreenParams(rs_min_percentile=min_rs)

    # Phase 1: Fetch universe
    if progress_callback:
        progress_callback("universe", 0, 1)

    universe_df = fetch_universe(market)
    ticker_to_name = dict(zip(universe_df["Code"], universe_df["Name"]))

    # Try to extract market cap from universe DataFrame (None = unavailable)
    ticker_to_mcap: dict[str, int | None] = {}
    mcap_col = next(
        (c for c in ("MarCap", "Marcap", "MKTCAP") if c in universe_df.columns),
        None,
    )
    shares_col = next(
        (c for c in ("Stocks", "LIST_SHRS", "ListedShares") if c in universe_df.columns),
        None,
    )
    _shares_map: dict[str, int] = {}

    if mcap_col:
        for code, val in zip(universe_df["Code"], universe_df[mcap_col]):
            try:
                ticker_to_mcap[code] = int(val)
            except (ValueError, TypeError):
                pass
        logger.info("Market cap loaded from %s column (%d tickers)", mcap_col, len(ticker_to_mcap))
    elif shares_col:
        for code, val in zip(universe_df["Code"], universe_df[shares_col]):
            try:
                _shares_map[code] = int(val)
            except (ValueError, TypeError):
                pass
        logger.info("Listed shares loaded from %s column (%d tickers)", shares_col, len(_shares_map))
    else:
        logger.warning("No market cap or shares data in universe — market cap filter disabled")

    tickers = list(ticker_to_name.keys())

    if progress_callback:
        progress_callback("universe", 1, 1)

    # Phase 2: Fetch price data
    def _price_progress(done, total):
        if progress_callback:
            progress_callback("prices", done, total)

    price_data = fetch_price_data(
        tickers, max_workers=max_workers, progress_callback=_price_progress,
    )

    # Compute market cap from shares × latest close when needed
    if _shares_map and not ticker_to_mcap:
        for ticker, shares in _shares_map.items():
            if ticker in price_data:
                latest_close = price_data[ticker]["Close"].dropna().iloc[-1]
                ticker_to_mcap[ticker] = int(shares * latest_close)
        logger.info("Computed market cap for %d tickers from shares × close", len(ticker_to_mcap))

    # Phase 3: Compute RS ratings
    if progress_callback:
        progress_callback("rs_rating", 0, 1)

    rs_ratings = compute_rs_ratings(price_data)

    if progress_callback:
        progress_callback("rs_rating", 1, 1)

    # Phase 4: Screen each stock
    screened: list[dict] = []
    total = len(price_data)

    for i, (ticker, df) in enumerate(price_data.items()):
        indicators = compute_indicators(df)
        if indicators is None:
            continue

        # Liquidity filter (fast reject)
        mcap = ticker_to_mcap.get(ticker)
        liq = check_liquidity(indicators, market_cap=mcap, params=p)
        if not all(liq.values()):
            continue

        # Trend template
        rs = rs_ratings.get(ticker, 0.0)
        template = check_trend_template(
            indicators, rs,
            min_rs=p.rs_min_percentile,
            min_above_52w_low_pct=p.min_above_52w_low_pct,
            max_below_52w_high_pct=p.max_below_52w_high_pct,
        )

        if not template["all_pass"]:
            continue

        # VCP (optional)
        vcp = detect_vcp(df, p) if p.check_vcp else False
        if p.check_vcp and not vcp:
            continue

        # Composite score
        score = compute_composite_score(indicators, rs)

        screened.append({
            "ticker": ticker,
            "name": ticker_to_name.get(ticker, ""),
            "close": int(indicators["close"]),
            "rs_rating": rs,
            "score": score,
            "market_cap": mcap or 0,
            "vcp_detected": vcp,
            **template,
        })

        if progress_callback and (i + 1) % 200 == 0:
            progress_callback("screening", i + 1, total)

    if progress_callback:
        progress_callback("screening", total, total)

    # Sort by composite score descending, RS as tiebreaker
    screened.sort(key=lambda x: (x["score"], x["rs_rating"]), reverse=True)

    logger.info(
        "Screening complete: %d/%d stocks pass (min_rs=%d)",
        len(screened), total, p.rs_min_percentile,
    )
    return screened


# ─── Output ───────────────────────────────────────────────────


def write_stocks_json(results: list[dict], output_path: str) -> Path:
    """Write screening results to stocks.json format.

    Args:
        results: Output of screen().
        output_path: File path for output.

    Returns:
        Path to the written file.
    """
    stocks = {r["ticker"]: r["name"] for r in results}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stocks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Wrote %d stocks to %s", len(stocks), path)
    return path
