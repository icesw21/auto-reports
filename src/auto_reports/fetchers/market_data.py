"""Market data fetcher using FinanceDataReader."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import re
import urllib.request

import FinanceDataReader as fdr

from auto_reports.models.financial import MarketData

logger = logging.getLogger(__name__)


@dataclass
class PriceHistory:
    """Stock price history summary (1-year or since listing)."""
    period_start: str = ""  # YYYY.MM.DD
    period_end: str = ""
    start_price: int = 0
    current_price: int = 0
    low_price: int = 0
    low_date: str = ""
    high_price: int = 0
    high_date: str = ""
    return_1y: float = 0.0  # percentage
    return_label: str = "1년수익률"  # "1년수익률" or "상장후수익률"


class MarketDataFetcher:
    """Fetch market data for Korean stocks via FinanceDataReader."""

    def _fetch_krx_listing(self, max_retries: int = 3):
        """Fetch KRX listing with retry logic for transient failures.

        The KRX API sometimes returns 'LOGOUT' instead of JSON data,
        causing a JSONDecodeError. This retries with exponential backoff.
        """
        import time as _time

        for attempt in range(max_retries):
            try:
                listing = fdr.StockListing("KRX")
                if listing is not None and not listing.empty:
                    return listing
            except Exception:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "KRX listing attempt %d/%d failed, retrying in %ds...",
                    attempt + 1, max_retries, wait,
                )
                _time.sleep(wait)
        return None

    def get_market_data(self, ticker: str) -> MarketData:
        """Fetch current market data for a stock.

        Retrieves stock price, market cap, and shares outstanding.
        Market cap and shares outstanding are sourced from the KRX listing
        snapshot; price is the latest close from DataReader.
        """
        logger.info("Fetching market data for ticker: %s", ticker)

        price = self.get_stock_price(ticker)
        market_cap: int | None = None
        shares: int | None = None
        data_date = date.today().isoformat()

        try:
            listing = self._fetch_krx_listing()
            if listing is None:
                logger.warning("KRX listing returned empty after retries for %s", ticker)
                return MarketData(date=data_date, stock_price=price)
            row = listing[listing["Code"] == ticker]
            if row.empty:
                # Some listings use 'Symbol' column
                sym_col = "Symbol" if "Symbol" in listing.columns else None
                if sym_col:
                    row = listing[listing[sym_col] == ticker]

            if not row.empty:
                r = row.iloc[0]
                # Market cap column names vary across fdr versions
                for cap_col in ("Marcap", "MarketCap", "시가총액"):
                    if cap_col in r.index and r[cap_col]:
                        raw = r[cap_col]
                        market_cap = int(raw) if raw else None
                        break

                for shares_col in ("Stocks", "shares", "상장주식수"):
                    if shares_col in r.index and r[shares_col]:
                        raw = r[shares_col]
                        shares = int(raw) if raw else None
                        break

                # If market cap missing but price and shares are available, calculate
                if market_cap is None and price is not None and shares is not None:
                    market_cap = price * shares
        except Exception:
            logger.exception("Failed to fetch KRX listing for %s", ticker)

        # Fallback: if shares still missing, try Naver Finance
        if shares is None:
            naver_shares = self._fetch_shares_naver(ticker)
            if naver_shares:
                shares = naver_shares
                logger.info("Got shares_outstanding from Naver fallback: %d", shares)
                if market_cap is None and price is not None:
                    market_cap = price * shares

        return MarketData(
            date=data_date,
            stock_price=price,
            market_cap=market_cap,
            shares_outstanding=shares,
        )

    def _fetch_shares_naver(self, ticker: str) -> int | None:
        """Fallback: scrape 상장주식수 from Naver Finance.

        Used when fdr.StockListing("KRX") fails (KRX API returns invalid JSON).
        """
        try:
            url = f"https://finance.naver.com/item/coinfo.naver?code={ticker}&target=finsum_main"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            html = urllib.request.urlopen(req, timeout=10).read().decode("euc-kr", errors="replace")
            m = re.search(r"상장주식수[^0-9]*([0-9,]+)", html)
            if m:
                return int(m.group(1).replace(",", ""))
        except Exception:
            logger.warning("Naver shares fallback failed for %s", ticker)
        return None

    def get_price_history(self, ticker: str, years: int = 1) -> PriceHistory | None:
        """Fetch 1-year price history summary for a stock.

        Returns PriceHistory with start/current/high/low prices and dates, or None.
        """
        try:
            end = date.today()
            start = end - timedelta(days=years * 365)

            df = fdr.DataReader(ticker, start.isoformat(), end.isoformat())
            if df is None or df.empty:
                logger.warning("No price history for %s", ticker)
                return None

            close_col = "Close" if "Close" in df.columns else df.columns[-1]
            closes = df[close_col].dropna()
            if closes.empty:
                return None

            start_price = int(closes.iloc[0])
            current_price = int(closes.iloc[-1])
            low_idx = closes.idxmin()
            high_idx = closes.idxmax()
            return_pct = (current_price - start_price) / start_price * 100

            # Determine if stock has less than 1 year of history
            actual_days = (closes.index[-1] - closes.index[0]).days
            requested_days = years * 365
            if actual_days < requested_days * 0.9:
                return_label = "상장후수익률"
                logger.info(
                    "Stock %s listed %d days ago (< %d), using listing date as start",
                    ticker, actual_days, requested_days,
                )
            else:
                return_label = "1년수익률"

            return PriceHistory(
                period_start=closes.index[0].strftime("%Y.%m.%d"),
                period_end=closes.index[-1].strftime("%Y.%m.%d"),
                start_price=start_price,
                current_price=current_price,
                low_price=int(closes[low_idx]),
                low_date=low_idx.strftime("%Y.%m.%d"),
                high_price=int(closes[high_idx]),
                high_date=high_idx.strftime("%Y.%m.%d"),
                return_1y=return_pct,
                return_label=return_label,
            )
        except Exception:
            logger.exception("Failed to fetch price history for %s", ticker)
            return None

    def get_stock_price(self, ticker: str, date_str: str = "") -> int | None:
        """Get stock price for a specific date (or latest if date_str is empty).

        Returns the closing price as an integer (won), or None on failure.
        """
        try:
            if date_str:
                start = date_str
                end = date_str
            else:
                today = date.today()
                # Go back 7 days to ensure we catch the last trading day
                start = (today - timedelta(days=7)).isoformat()
                end = today.isoformat()

            df = fdr.DataReader(ticker, start, end)
            if df is None or df.empty:
                logger.warning("No price data returned for %s", ticker)
                return None

            close_col = "Close" if "Close" in df.columns else df.columns[-1]
            price = df[close_col].dropna().iloc[-1]
            return int(price)
        except Exception:
            logger.exception("Failed to fetch stock price for %s", ticker)
            return None
