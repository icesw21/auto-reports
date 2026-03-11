"""Market data fetcher using Naver Finance API."""

from __future__ import annotations

import json as _json
import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from auto_reports.models.financial import ConsensusItem, MarketData

logger = logging.getLogger(__name__)

_NAVER_UA = {"User-Agent": "Mozilla/5.0"}


def _parse_eok(val_str: str) -> int | None:
    """Parse 억원 float string (e.g. '1,877.50') to 원 int."""
    if not val_str or val_str == "-":
        return None
    cleaned = val_str.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return round(float(cleaned)) * 1_0000_0000
    except (ValueError, OverflowError):
        return None


def _parse_market_value(text: str) -> int | None:
    """Parse Naver market cap string to 원.

    Handles formats:
      - "3,432억"           → 3432 * 1_0000_0000
      - "1조 3,293억"       → (1*10000 + 3293) * 1_0000_0000
      - "22조 2,725억"      → (22*10000 + 2725) * 1_0000_0000
      - "1조"               → 1 * 10000 * 1_0000_0000
    """
    if not text:
        return None
    # "N조 M억" pattern
    m = re.match(r"([0-9,]+)\s*조\s*([0-9,]+)\s*억", text)
    if m:
        jo = int(m.group(1).replace(",", ""))
        eok = int(m.group(2).replace(",", ""))
        return (jo * 10_000 + eok) * 1_0000_0000
    # "N조" only (no 억 part)
    m = re.match(r"([0-9,]+)\s*조", text)
    if m:
        jo = int(m.group(1).replace(",", ""))
        return jo * 10_000 * 1_0000_0000
    # "N억" pattern
    m = re.match(r"([0-9,]+)\s*억", text)
    if m:
        return int(m.group(1).replace(",", "")) * 1_0000_0000
    return None


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
    """Fetch market data for Korean stocks via Naver Finance API."""

    def _naver_api_get(self, url: str) -> dict | list | None:
        """Fetch JSON from Naver mobile stock API."""
        try:
            req = urllib.request.Request(url, headers=_NAVER_UA)
            raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
            return _json.loads(raw)
        except Exception:
            logger.warning("Naver API request failed: %s", url)
            return None

    def get_market_data(self, ticker: str) -> MarketData:
        """Fetch current market data via Naver Finance API.

        Retrieves stock price, market cap, and shares outstanding from
        the Naver mobile stock integration API.
        """
        logger.info("Fetching market data for ticker: %s", ticker)

        data_date = date.today().isoformat()
        price: int | None = None
        market_cap: int | None = None
        shares: int | None = None

        # Primary: Naver mobile stock integration API
        try:
            data = self._naver_api_get(
                f"https://m.stock.naver.com/api/stock/{ticker}/integration"
            )
            if data and isinstance(data, dict):
                for item in data.get("totalInfos", []):
                    code = item.get("code", "")
                    value = item.get("value", "")
                    if code == "marketValue" and value:
                        market_cap = _parse_market_value(value)
        except Exception:
            logger.warning("Naver integration API failed for %s", ticker)

        # Get current price
        price = self.get_stock_price(ticker)

        # Get shares from HTML (exact 상장주식수, more accurate than derivation)
        naver_data = self._fetch_market_data_naver(ticker)
        if naver_data:
            if "shares" in naver_data:
                shares = naver_data["shares"]
            if market_cap is None and "market_cap" in naver_data:
                market_cap = naver_data["market_cap"]

        # Fallback: derive shares from market_cap / price
        if shares is None and market_cap is not None and market_cap > 0 and price and price > 0:
            shares = market_cap // price
            logger.info("Derived shares from market_cap/price for %s: %d", ticker, shares)

        # Fallback: derive market_cap from price * shares
        if market_cap is None and price is not None and shares is not None:
            market_cap = price * shares

        return MarketData(
            date=data_date,
            stock_price=price,
            market_cap=market_cap,
            shares_outstanding=shares,
        )

    def _fetch_market_data_naver(self, ticker: str) -> dict | None:
        """Scrape market cap and shares from Naver Finance HTML.

        Returns dict with 'market_cap' and 'shares' keys, or None on failure.
        Used as fallback when API-based methods fail.
        """
        try:
            url = f"https://finance.naver.com/item/sise.naver?code={ticker}"
            req = urllib.request.Request(url, headers=_NAVER_UA)
            html = urllib.request.urlopen(req, timeout=10).read().decode("euc-kr", errors="replace")

            # Strip HTML tags to avoid matching numbers inside attributes
            text = re.sub(r"<[^>]+>", " ", html)

            result = {}

            # 시가총액 (억원 units on sise page)
            m = re.search(r"시가총액\s+([0-9,]+)\s*억", text)
            if m:
                result["market_cap"] = int(m.group(1).replace(",", "")) * 1_0000_0000

            # 상장주식수
            m = re.search(r"상장주식수\s+([0-9,]+)", text)
            if m:
                shares_val = int(m.group(1).replace(",", ""))
                if shares_val >= 1_000:
                    result["shares"] = shares_val
                else:
                    logger.warning("Naver shares value %d out of range for %s", shares_val, ticker)

            return result if result else None
        except Exception:
            logger.warning("Naver HTML market data fallback failed for %s", ticker)
            return None

    def get_consensus(self, ticker: str) -> list[ConsensusItem]:
        """Fetch Naver consensus estimates (up to 3 years) for a stock.

        Uses the Naver Company (wisereport) AJAX endpoint which provides
        multi-year consensus data.  Falls back to the Naver mobile stock API
        if the primary source fails.

        Returns list of ConsensusItem sorted oldest-first (e.g. 2025E, 2026E, 2027E).
        """
        items = self._get_consensus_wisereport(ticker)
        if not items:
            items = self._get_consensus_mobile(ticker)
        if items:
            logger.info("Fetched %d consensus estimate(s) for %s", len(items), ticker)
        return items

    def _get_consensus_wisereport(self, ticker: str) -> list[ConsensusItem]:
        """Fetch multi-year consensus from Naver Company (wisereport) AJAX."""
        try:
            # Step 1: Get encparam token from main page
            main_url = (
                f"https://navercomp.wisereport.co.kr/v2/company/"
                f"c1010001.aspx?cmp_cd={ticker}"
            )
            req = urllib.request.Request(main_url, headers=_NAVER_UA)
            main_html = (
                urllib.request.urlopen(req, timeout=10)
                .read()
                .decode("utf-8", errors="replace")
            )
            m = re.search(r"encparam:\s*'([^']+)'", main_html)
            if not m:
                logger.debug("No encparam found for %s", ticker)
                return []
            encparam = m.group(1)

            # Step 2: POST to AJAX endpoint (fin_typ=0: 주요재무정보, freq_typ=Y: 연간)
            ajax_url = (
                "https://navercomp.wisereport.co.kr/v2/company/ajax/cF1001.aspx"
            )
            params = (
                f"cmp_cd={ticker}&fin_typ=0&freq_typ=Y"
                f"&encparam={encparam}&asktype=0"
            )
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": main_url,
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            req2 = urllib.request.Request(
                ajax_url, data=params.encode(), headers=headers,
            )
            resp_html = (
                urllib.request.urlopen(req2, timeout=10)
                .read()
                .decode("utf-8", errors="replace")
            )

            # Step 3: Isolate the real data table (skip decoy first table)
            # The AJAX response contains a decoy table (all identical values)
            # followed by the real table with actual data in title attributes.
            table_parts = resp_html.split("</table>")
            if len(table_parts) < 2:
                return []
            # Use everything after the first </table> (the real table)
            real_table_html = "</table>".join(table_parts[1:])

            # Step 4: Parse column headers from the real table
            year_headers = re.findall(
                r"(20\d{2}/\d{2}(?:\(E\))?)", real_table_html,
            )
            if not year_headers:
                return []

            estimate_indices: list[tuple[int, str]] = []
            for idx, hdr in enumerate(year_headers):
                if hdr.endswith("(E)"):
                    estimate_indices.append((idx, hdr))
            if not estimate_indices:
                return []

            # Step 5: Parse data rows from the real table only
            row_map: dict[str, list[str]] = {}
            all_rows = re.findall(
                r"<tr>(.*?)</tr>", real_table_html, re.DOTALL,
            )
            for row in all_rows:
                th = re.search(
                    r'<th[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</th>',
                    row, re.DOTALL,
                )
                if not th:
                    continue
                label = re.sub(r"<[^>]+>", "", th.group(1)).strip()
                label = label.replace("&nbsp;", "").strip()
                td_titles = re.findall(r'<td[^>]*title="([^"]*)"', row)
                if td_titles:
                    row_map[label] = td_titles

            def _get_val(label: str, col_idx: int) -> int | None:
                vals = row_map.get(label, [])
                if col_idx < len(vals):
                    return _parse_eok(vals[col_idx])
                return None

            # Use 영업이익(발표기준) for operating income (has consensus values
            # where plain 영업이익 is often empty for estimate columns)
            op_label = "영업이익(발표기준)"
            if op_label not in row_map:
                op_label = "영업이익"

            items = []
            for col_idx, hdr in estimate_indices:
                # "2025/12(E)" → "2025(E)"
                year_m = re.match(r"(\d{4})", hdr)
                year_part = year_m.group(1) if year_m else hdr
                period_display = f"{year_part}(E)"

                rev = _get_val("매출액", col_idx)
                op = _get_val(op_label, col_idx)
                ni = _get_val("당기순이익", col_idx)

                # Skip if all values are None
                if rev is None and op is None and ni is None:
                    continue

                items.append(ConsensusItem(
                    period=period_display,
                    revenue=rev,
                    operating_income=op,
                    net_income=ni,
                ))

            return items
        except Exception:
            logger.warning("Wisereport consensus fetch failed for %s", ticker)
            return []

    def _get_consensus_mobile(self, ticker: str) -> list[ConsensusItem]:
        """Fallback: fetch consensus from Naver mobile stock API (1 year only)."""
        try:
            url = f"https://m.stock.naver.com/api/stock/{ticker}/finance/annual"
            req = urllib.request.Request(url, headers=_NAVER_UA)
            raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
            data = _json.loads(raw)

            fi = data.get("financeInfo", {})
            tr_titles = fi.get("trTitleList", [])
            consensus_cols = [
                (t.get("key", ""), t.get("title", ""))
                for t in tr_titles
                if t.get("isConsensus") == "Y"
            ]
            if not consensus_cols:
                return []

            row_map: dict[str, dict] = {}
            for row in fi.get("rowList", []):
                row_map[row.get("title", "")] = row.get("columns", {})

            def _get_val(row_title: str, period_key: str) -> int | None:
                cols = row_map.get(row_title, {})
                cell = cols.get(period_key, {})
                return _parse_eok(cell.get("value", ""))

            items = []
            for period_key, period_label in consensus_cols:
                m = re.match(r"(\d{4})", period_label)
                year_part = m.group(1) if m else period_label
                period_display = f"{year_part}(E)"
                items.append(ConsensusItem(
                    period=period_display,
                    revenue=_get_val("매출액", period_key),
                    operating_income=_get_val("영업이익", period_key),
                    net_income=_get_val("당기순이익", period_key),
                ))
            return items
        except Exception:
            logger.warning("Naver mobile consensus fetch failed for %s", ticker)
            return []

    def get_price_history(self, ticker: str, years: int = 1) -> PriceHistory | None:
        """Fetch price history summary via Naver mobile stock API.

        Returns PriceHistory with start/current/high/low prices and dates, or None.
        """
        try:
            # Naver API limits pageSize to ~60; paginate to cover ~260 trading days/year
            page_size = 60
            pages_needed = (years * 260 // page_size) + 2  # extra margin
            all_entries: list[dict] = []
            for page in range(1, pages_needed + 1):
                data = self._naver_api_get(
                    f"https://m.stock.naver.com/api/stock/{ticker}/price"
                    f"?pageSize={page_size}&page={page}"
                )
                if not data or not isinstance(data, list):
                    break
                all_entries.extend(data)
                if len(data) < page_size:
                    break  # last page

            if len(all_entries) < 2:
                logger.warning("No price history for %s", ticker)
                return None

            # Data is newest-first; reverse for chronological order
            entries = list(reversed(all_entries))

            # Filter to requested time range
            cutoff = (date.today() - timedelta(days=years * 365)).isoformat()
            entries = [e for e in entries if e.get("localTradedAt", "") >= cutoff]
            if len(entries) < 2:
                return None

            def _parse_price(val) -> int | None:
                if not val:
                    return None
                return int(str(val).replace(",", ""))

            start_price = _parse_price(entries[0].get("closePrice"))
            current_price = _parse_price(entries[-1].get("closePrice"))
            if not start_price or not current_price:
                return None

            # Find high and low
            low_price = float("inf")
            low_date_str = ""
            high_price = 0
            high_date_str = ""
            for e in entries:
                p = _parse_price(e.get("closePrice"))
                if p is None:
                    continue
                d = e["localTradedAt"]
                if p < low_price:
                    low_price = p
                    low_date_str = d
                if p > high_price:
                    high_price = p
                    high_date_str = d

            return_pct = (current_price - start_price) / start_price * 100

            # Determine if stock has less than 1 year of history
            actual_start = datetime.strptime(entries[0]["localTradedAt"], "%Y-%m-%d")
            actual_end = datetime.strptime(entries[-1]["localTradedAt"], "%Y-%m-%d")
            actual_days = (actual_end - actual_start).days
            requested_days = years * 365
            if actual_days < requested_days * 0.9:
                return_label = "상장후수익률"
                logger.info(
                    "Stock %s listed %d days ago (< %d), using listing date as start",
                    ticker, actual_days, requested_days,
                )
            else:
                return_label = "1년수익률"

            def _fmt_date(d: str) -> str:
                return d.replace("-", ".")

            return PriceHistory(
                period_start=_fmt_date(entries[0]["localTradedAt"]),
                period_end=_fmt_date(entries[-1]["localTradedAt"]),
                start_price=start_price,
                current_price=current_price,
                low_price=int(low_price),
                low_date=_fmt_date(low_date_str),
                high_price=high_price,
                high_date=_fmt_date(high_date_str),
                return_1y=return_pct,
                return_label=return_label,
            )
        except Exception:
            logger.exception("Failed to fetch price history for %s", ticker)
            return None

    def get_stock_price(self, ticker: str, date_str: str = "") -> int | None:
        """Get latest stock price via Naver mobile stock API.

        Returns the closing price as an integer (won), or None on failure.
        """
        try:
            data = self._naver_api_get(
                f"https://m.stock.naver.com/api/stock/{ticker}/basic"
            )
            if data and isinstance(data, dict):
                close_str = data.get("closePrice", "")
                if close_str:
                    return int(close_str.replace(",", ""))
            return None
        except Exception:
            logger.exception("Failed to fetch stock price for %s", ticker)
            return None
