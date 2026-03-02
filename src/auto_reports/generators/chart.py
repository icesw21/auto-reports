"""Stock price chart generator using mplfinance."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (thread-safe, must be before pyplot import)

logger = logging.getLogger(__name__)


def generate_stock_chart(
    ticker: str,
    output_dir: str | Path,
    days: int = 365,
) -> str | None:
    """Generate a candlestick chart with MA20/MA60 and volume.

    Args:
        ticker: KRX stock ticker (e.g. "005930").
        output_dir: Directory to save the chart PNG.
        days: Number of trading days to display (default 365).

    Returns:
        Filename of the generated chart (e.g. "chart_005930_20260228.png"),
        or None on failure.
    """
    try:
        import FinanceDataReader as fdr
        import matplotlib.pyplot as plt
        import mplfinance as mpf
    except ImportError as e:
        logger.warning("Chart dependencies not installed: %s", e)
        return None

    try:
        # Fetch extra rows for MA warm-up, then slice for display
        df_full = fdr.DataReader(ticker).tail(days + 60)
        if df_full is None or df_full.empty:
            logger.warning("No price data for chart: %s", ticker)
            return None

        # Ensure OHLCV columns exist
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df_full.columns):
            logger.warning("Missing OHLCV columns for %s: %s", ticker, df_full.columns.tolist())
            return None

        # Moving averages on full series, then slice for display
        ma20_full = df_full["Close"].rolling(window=20).mean()
        ma60_full = df_full["Close"].rolling(window=60).mean()
        df = df_full.tail(days)
        ma20 = ma20_full.tail(days)
        ma60 = ma60_full.tail(days)

        apds = [
            mpf.make_addplot(ma20, label="20MA"),
            mpf.make_addplot(ma60, label="60MA"),
        ]

        # Chart style: red up, blue down (Korean convention)
        mc = mpf.make_marketcolors(up="red", down="blue", inherit=True)
        style = mpf.make_mpf_style(marketcolors=mc, gridstyle="--", y_on_right=False)

        fig, axlist = mpf.plot(
            df,
            type="candle",
            addplot=apds,
            volume=True,
            style=style,
            figsize=(12, 6),
            returnfig=True,
        )

        axlist[0].legend(["20MA", "60MA"], loc="upper left")

        # Save
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        filename = f"chart_{ticker}_{datetime.now().strftime('%Y%m%d')}.png"
        full_path = out_path / filename

        fig.savefig(str(full_path), bbox_inches="tight", dpi=150)
        plt.close(fig)

        logger.info("Chart saved: %s", full_path)
        return filename

    except Exception:
        logger.exception("Failed to generate chart for %s", ticker)
        return None
