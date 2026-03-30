"""OpenDART API wrapper for financial statement data."""

from __future__ import annotations

import calendar
import logging
import re as _re_module
from datetime import date, timedelta
from typing import Any

import OpenDartReader
from auto_reports.fetchers.rate_limiter import dart_call_with_retry, dart_get_with_retry
from auto_reports.models.financial import BalanceSheet, IncomeStatementItem

logger = logging.getLogger(__name__)

# Report code constants
_REPRT_ANNUAL = "11011"
_REPRT_Q1 = "11013"
_REPRT_HALF = "11012"
_REPRT_Q3 = "11014"

_QUARTER_TO_REPRT = {
    0: _REPRT_ANNUAL,
    1: _REPRT_Q1,
    2: _REPRT_HALF,
    3: _REPRT_Q3,
}

# Balance sheet field -> Korean account name variants (first match wins)
_BS_FIELD_NAMES: dict[str, list[str]] = {
    "total_assets": ["자산총계"],
    "cash_and_equivalents": ["현금및현금성자산"],
    "short_term_investments": ["단기금융상품"],
    "total_liabilities": ["부채총계"],
    "short_term_debt_and_bonds": ["단기차입금및사채", "유동 차입금", "유동차입금", "유동성 차입금", "유동성차입금"],
    "long_term_debt_and_bonds": ["장기차입금및사채", "비유동성 차입금", "비유동성차입금", "비유동 차입금", "비유동차입금"],
    "short_term_borrowings": ["단기차입금"],
    "current_long_term_debt": ["유동성장기부채", "유동성장기차입금"],
    "current_bonds": ["유동성사채"],
    "long_term_borrowings": ["장기차입금"],
    "bonds": ["사채"],
    "total_equity": ["자본총계"],
}

# Substring exclusions: when doing substring matching, if the account name
# contains any of the exclusion strings, skip it. Prevents e.g. "사채"
# from matching "전환사채" or "유동성전환사채".
_BS_SUBSTRING_EXCLUSIONS: dict[str, list[str]] = {
    "사채": ["유동성", "전환", "할인", "할증", "상환", "교환", "차입금"],
    "차입금": ["장기", "유동성"],
    "장기차입금": ["유동성", "유동"],
    "자본총계": ["부채"],
    "부채총계": ["자본"],
    "매출": ["원가", "총이익", "총손실"],
}

# Income statement field -> Korean account name variants
_IS_FIELD_NAMES: dict[str, list[str]] = {
    "revenue": ["매출액", "매출", "수익(매출액)", "영업수익", "수익"],
    "operating_income": ["영업이익", "영업이익(손실)", "영업손익", "영업손실"],
    "net_income": [
        "당기순이익",
        "당기순이익(손실)",
        "당기순손실(이익)",
        "당기순손실",
        "당(반)기순이익",
        "당(반)기순이익(손실)",
        "당(반)기순손실(이익)",
        "당(분)기순이익",
        "당(분)기순이익(손실)",
        "당(분)기순손실(이익)",
        "반기순이익",
        "반기순이익(손실)",
        "반기순손실(이익)",
        "반기순손실",
        "분기순이익",
        "분기순이익(손실)",
        "분기순손실(이익)",
        "분기순손실",
    ],
}


def _parse_amount(value: Any) -> int | None:
    """Convert API string/int amounts to int (won). Returns None on failure."""
    if value is None:
        return None
    try:
        cleaned = str(value).replace(",", "").replace("\xa0", "").replace(" ", "").strip()
        if cleaned in ("", "-", "N/A"):
            return None
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def _detect_unit_scale(df: Any, amount_col: str) -> int:
    """Detect if IS/CIS amounts in a DataFrame are inflated by 백만원 factor.

    Some DART filings declare 백만원 as the unit, but the API returns values
    already multiplied by 1,000,000.  Detection: if ALL significant (non-EPS,
    non-zero) IS/CIS amounts are exactly divisible by 1,000,000, the column
    is inflated.

    Returns 1_000_000 if inflated, else 1.
    """
    if "sj_div" not in df.columns or amount_col not in df.columns:
        return 1

    is_df = df[df["sj_div"].isin(["IS", "CIS"])]
    if not _is_valid_df(is_df):
        return 1

    significant: list[int] = []
    for val in is_df[amount_col]:
        parsed = _parse_amount(val)
        if parsed is not None and abs(parsed) > 100_000:  # skip EPS / ratios
            significant.append(parsed)

    if len(significant) < 2:
        return 1

    if all(v % 1_000_000 == 0 for v in significant):
        logger.warning(
            "Detected 백만원 unit inflation in column '%s' "
            "(%d values all divisible by 1,000,000). Will normalize.",
            amount_col, len(significant),
        )
        return 1_000_000

    return 1


def _format_yoy(current: int | None, previous: int | None) -> str | None:
    """Return a YoY change string with sign-transition handling."""
    if current is None or previous is None:
        return None

    if current > 0 and previous < 0:
        return "흑자전환"
    if current < 0 and previous > 0:
        return "적자전환"
    if current < 0 and previous < 0:
        if abs(current) < abs(previous):
            return "적자축소"
        elif abs(current) > abs(previous):
            return "적자확대"
        return "적자지속"

    if previous == 0:
        return None

    pct = (current - previous) / abs(previous) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _is_valid_df(df: Any) -> bool:
    """Check if df is a non-empty DataFrame."""
    if df is None:
        return False
    try:
        return not df.empty
    except AttributeError:
        return False


def _has_meaningful_amounts(df: Any) -> bool:
    """Check if a DataFrame has at least one non-zero amount value.

    Used after currency filtering to ensure KRW rows contain real data,
    not just placeholder rows with zero amounts (e.g. 코오롱티슈진 older years).
    """
    found_col = False
    for col in ("thstrm_amount", "당기금액", "amount", "thstrm_add_amount"):
        if col not in df.columns:
            continue
        found_col = True
        for val in df[col]:
            parsed = _parse_amount(val)
            if parsed is not None and parsed != 0:
                return True
    if not found_col:
        logger.warning(
            "_has_meaningful_amounts: no expected amount columns in df. "
            "Columns: %s", list(df.columns),
        )
    return False


class OpenDartFetcher:
    """Wrapper around the OpenDartReader library."""

    def __init__(self, api_key: str, delay: float = 0.5, fs_div: str = "CFS") -> None:
        self.dart = OpenDartReader(api_key)
        self._api_key = api_key  # Store for direct API calls
        self.delay = delay
        self.preferred_fs_div = fs_div  # "CFS" (연결) or "OFS" (별도)
        self._cfs_succeeded = False  # Track whether CFS ever returned data
        self._last_currency = "KRW"  # Track currency of last fetched data

    @property
    def effective_fs_pref(self) -> str:
        """Return the actual statement type that produced data.

        If preferred CFS but it never returned data (always fell back to OFS),
        returns "별도" so downstream consumers can match the accounting basis.
        """
        if self.preferred_fs_div == "CFS" and not self._cfs_succeeded:
            return "별도"
        return "연결" if self.preferred_fs_div == "CFS" else "별도"

    # ------------------------------------------------------------------
    # Internal: fetch with CFS → OFS fallback
    # ------------------------------------------------------------------

    def _fetch_finstate(
        self, corp_code: str, year: int, reprt_code: str
    ) -> tuple[Any, str]:
        """Fetch finstate_all with automatic CFS → OFS fallback.

        Returns (DataFrame_or_None, statement_type_label).
        """
        try:
            df = dart_call_with_retry(self.dart.finstate_all, corp_code, year, reprt_code, self.preferred_fs_div)
        except Exception:
            logger.exception("finstate_all failed: %s %d %s %s", corp_code, year, reprt_code, self.preferred_fs_div)
            df = None

        stmt_label = "연결" if self.preferred_fs_div == "CFS" else "별도"

        # Note: _cfs_succeeded is set AFTER the KRW filter below,
        # not here — otherwise a CFS response with only USD rows
        # would incorrectly mark CFS as successful.

        # Fallback: CFS → OFS when consolidated data is unavailable
        if not _is_valid_df(df) and self.preferred_fs_div == "CFS":
            logger.info("CFS data empty, falling back to OFS: %s %d", corp_code, year)
            try:
                df = dart_call_with_retry(self.dart.finstate_all, corp_code, year, reprt_code, "OFS")
            except Exception:
                logger.exception("OFS fallback also failed: %s %d", corp_code, year)
                return None, ""
            stmt_label = "별도"

        if not _is_valid_df(df):
            return None, stmt_label

        # Filter currency — prefer KRW; if not available, use the single
        # foreign currency (e.g. CNY for Chinese subsidiaries).
        if "currency" in df.columns:
            krw_df = df[df["currency"] == "KRW"]
            if _is_valid_df(krw_df) and _has_meaningful_amounts(krw_df):
                df = krw_df
                self._last_currency = "KRW"
                if self.preferred_fs_div == "CFS":
                    self._cfs_succeeded = True
            else:
                # No KRW rows — check if there's a single foreign currency
                currencies = df["currency"].dropna().unique().tolist()
                if len(currencies) == 1 and _has_meaningful_amounts(df):
                    self._last_currency = currencies[0]
                    logger.info(
                        "Using %s currency for %s %d (no KRW data)",
                        self._last_currency, corp_code, year,
                    )
                    if self.preferred_fs_div == "CFS":
                        self._cfs_succeeded = True
                else:
                    logger.warning(
                        "No meaningful amounts in API for %s %d (currencies: %s); "
                        "returning None to trigger HTML report fallback",
                        corp_code, year, currencies,
                    )
                    return None, stmt_label
        else:
            # No currency column — standard KRW-only company
            self._last_currency = "KRW"
            if self.preferred_fs_div == "CFS":
                self._cfs_succeeded = True

        return df, stmt_label

    def _find_amount(
        self, df: Any, sj_div: str, field_names: list[str], amount_col: str,
    ) -> int | None:
        """Find an amount value from DataFrame rows filtered by sj_div.

        Tries each account name variant; exact match first, then substring.
        """
        if "sj_div" in df.columns:
            filtered = df[df["sj_div"] == sj_div]
        else:
            filtered = df

        if not _is_valid_df(filtered):
            return None

        account_col = (
            "account_nm" if "account_nm" in filtered.columns
            else self._detect_account_col(filtered)
        )

        for name in field_names:
            # Exact match
            mask = filtered[account_col] == name
            if mask.any():
                return _parse_amount(filtered[mask].iloc[0][amount_col])
            # Substring match (with exclusions for ambiguous names)
            mask = filtered[account_col].str.contains(name, na=False, regex=False)
            if mask.any():
                exclusions = _BS_SUBSTRING_EXCLUSIONS.get(name, [])
                if exclusions:
                    # Filter out rows where account name contains any exclusion
                    for excl in exclusions:
                        mask = mask & ~filtered[account_col].str.contains(excl, na=False, regex=False)
                if mask.any():
                    return _parse_amount(filtered[mask].iloc[0][amount_col])

        return None

    def _find_amount_sum(
        self, df: Any, sj_div: str, field_names: list[str], amount_col: str,
    ) -> int | None:
        """Find and SUM all matching amounts (not just first match).

        Used for bare "차입금" which may appear as separate rows for
        유동 and 비유동 liabilities with the same account name.
        """
        if "sj_div" in df.columns:
            filtered = df[df["sj_div"] == sj_div]
        else:
            filtered = df

        if not _is_valid_df(filtered):
            return None

        account_col = (
            "account_nm" if "account_nm" in filtered.columns
            else self._detect_account_col(filtered)
        )

        total = 0
        found = False
        for name in field_names:
            # Exact match — sum all rows
            mask = filtered[account_col] == name
            if mask.any():
                for _, row in filtered[mask].iterrows():
                    val = _parse_amount(row[amount_col])
                    if val is not None:
                        total += val
                        found = True
                continue
            # Substring match with exclusions — sum all rows
            mask = filtered[account_col].str.contains(name, na=False, regex=False)
            if mask.any():
                exclusions = _BS_SUBSTRING_EXCLUSIONS.get(name, [])
                if exclusions:
                    for excl in exclusions:
                        mask = mask & ~filtered[account_col].str.contains(
                            excl, na=False, regex=False,
                        )
                if mask.any():
                    for _, row in filtered[mask].iterrows():
                        val = _parse_amount(row[amount_col])
                        if val is not None:
                            total += val
                            found = True

        return total if found else None

    # ------------------------------------------------------------------
    # Corp code resolution
    # ------------------------------------------------------------------

    def resolve_corp_code(self, ticker: str, company_name: str = "") -> str:
        """Resolve stock ticker to DART corp_code.

        Falls back to company name lookup when ticker-based lookup fails
        (e.g. KONEX tickers like '0004V0' that contain letters).
        """
        try:
            code = self.dart.find_corp_code(ticker)
            if code:
                logger.info("Resolved ticker %s -> corp_code %s", ticker, code)
                return code
        except Exception:
            logger.debug("find_corp_code failed for ticker: %s", ticker)

        # Fallback: lookup by company name (handles KONEX tickers etc.)
        if company_name:
            try:
                results = self.dart.company_by_name(company_name)
                if results:
                    code = results[0]["corp_code"]
                    logger.info(
                        "Resolved company name '%s' -> corp_code %s",
                        company_name, code,
                    )
                    return code
            except Exception:
                logger.debug("company_by_name failed for: %s", company_name)

        raise ValueError(f"No corp_code found for ticker: {ticker}")

    # ------------------------------------------------------------------
    # Balance sheet
    # ------------------------------------------------------------------

    def get_balance_sheet(
        self, corp_code: str, year: int, quarter: int = 0,
    ) -> BalanceSheet:
        """Fetch balance sheet for a specific period.

        quarter=0 means annual; 1-3 for quarterly.
        """
        reprt_code = _QUARTER_TO_REPRT.get(quarter, _REPRT_ANNUAL)
        period_label = str(year) if quarter == 0 else f"{year}.Q{quarter}"

        logger.info(
            "Fetching balance sheet: corp=%s year=%d quarter=%d",
            corp_code, year, quarter,
        )

        df, stmt_label = self._fetch_finstate(corp_code, year, reprt_code)

        if not _is_valid_df(df):
            logger.warning("Empty balance sheet data: %s %d", corp_code, year)
            return BalanceSheet(period=period_label)

        bs = BalanceSheet(period=period_label, statement_type=stmt_label, currency=self._last_currency)
        amount_col = self._detect_amount_col(df)

        for field, names in _BS_FIELD_NAMES.items():
            val = self._find_amount(df, "BS", names, amount_col)
            if val is not None:
                setattr(bs, field, val)

        # Sum all bare "차입금" rows (유동+비유동) for companies that only
        # report "차입금" without 단기/장기 prefix.
        borrowings_sum = self._find_amount_sum(df, "BS", ["차입금"], amount_col)
        if borrowings_sum is not None:
            bs.borrowings_total = borrowings_sum

        return bs

    # ------------------------------------------------------------------
    # Income statement
    # ------------------------------------------------------------------

    def get_income_statements(
        self, corp_code: str, year: int, quarter: int = 0,
        cumulative: bool = False,
    ) -> IncomeStatementItem:
        """Fetch income statement for a specific period.

        When cumulative=True and quarter>0, prefer the cumulative amount column
        (thstrm_add_amount) to get year-to-date figures for quarterly reports.
        """
        reprt_code = _QUARTER_TO_REPRT.get(quarter, _REPRT_ANNUAL)
        period_label = str(year) if quarter == 0 else f"{year}.Q{quarter}"

        logger.info(
            "Fetching income statement: corp=%s year=%d quarter=%d cumulative=%s",
            corp_code, year, quarter, cumulative,
        )

        df, stmt_label = self._fetch_finstate(corp_code, year, reprt_code)

        if not _is_valid_df(df):
            return IncomeStatementItem(period=period_label)

        item = IncomeStatementItem(period=period_label, currency=self._last_currency)
        # For quarterly cumulative, prefer thstrm_add_amount (누적)
        if cumulative and quarter > 0:
            amount_col = self._detect_cumulative_amount_col(df)
        else:
            amount_col = self._detect_amount_col(df)

        # Debug: log all account names in the income statement divisions
        for sj_div in ("CIS", "IS"):
            if "sj_div" in df.columns:
                filtered = df[df["sj_div"] == sj_div]
            else:
                filtered = df
            if _is_valid_df(filtered):
                account_col = (
                    "account_nm" if "account_nm" in filtered.columns
                    else self._detect_account_col(filtered)
                )
                accounts = filtered[account_col].unique().tolist()
                logger.debug(
                    "IS accounts [%s] for %s %d Q%d: %s",
                    sj_div, corp_code, year, quarter, accounts,
                )

        # Detect 백만원 unit inflation at DataFrame level
        unit_scale = _detect_unit_scale(df, amount_col)

        # Income statement: try 'CIS' (K-IFRS 포괄손익계산서), fall back to 'IS' (K-GAAP 손익계산서)
        for sj_div in ("CIS", "IS"):
            for field, names in _IS_FIELD_NAMES.items():
                if getattr(item, field) is None:
                    val = self._find_amount(df, sj_div, names, amount_col)
                    if val is not None:
                        if unit_scale > 1:
                            val = val // unit_scale
                        setattr(item, field, val)

        return item

    # ------------------------------------------------------------------
    # Multi-year financials
    # ------------------------------------------------------------------

    def get_multi_year_financials(
        self, corp_code: str, years: int = 5,
    ) -> dict[str, list[IncomeStatementItem]]:
        """Fetch financial data for multiple years.

        Returns dict with 'annual' and 'quarterly' lists of IncomeStatementItem.

        Quarterly data uses cumulative subtraction:
        - Q1 standalone = Q1 cumulative (same for first quarter)
        - Q2 standalone = H1 cumulative - Q1 cumulative
        - Q3 standalone = 9M cumulative - H1 cumulative
        - Q4 standalone = Annual - 9M cumulative
        """
        current_year = date.today().year
        annual: list[IncomeStatementItem] = []
        quarterly: list[IncomeStatementItem] = []

        for yr in range(current_year - years, current_year):
            # Fetch annual
            annual_item = self.get_income_statements(corp_code, yr, quarter=0)
            annual.append(annual_item)

            # Fetch cumulative quarterly data
            cum_q1 = self.get_income_statements(corp_code, yr, quarter=1, cumulative=True)
            cum_h1 = self.get_income_statements(corp_code, yr, quarter=2, cumulative=True)
            cum_q3 = self.get_income_statements(corp_code, yr, quarter=3, cumulative=True)

            # Compute standalone quarters by subtraction
            q1 = cum_q1  # Q1 cumulative = Q1 standalone
            q2 = _subtract_income(cum_h1, cum_q1, f"{yr}.Q2")
            q3 = _subtract_income(cum_q3, cum_h1, f"{yr}.Q3")
            # Q4 = Annual - Q3 cumulative (only if both available)
            q4 = _subtract_income(annual_item, cum_q3, f"{yr}.Q4")

            quarterly.extend([q1, q2, q3, q4])

        # Calculate YoY for quarterly data (compare same quarter across years)
        quarterly_by_q: dict[int, list[IncomeStatementItem]] = {}
        for q_item in quarterly:
            if ".Q" in q_item.period:
                q_num = int(q_item.period.split(".Q")[1])
                quarterly_by_q.setdefault(q_num, []).append(q_item)

        for items in quarterly_by_q.values():
            for i in range(1, len(items)):
                curr = items[i]
                prev = items[i - 1]
                curr.revenue_yoy = _format_yoy(curr.revenue, prev.revenue)
                curr.operating_income_yoy = _format_yoy(
                    curr.operating_income, prev.operating_income,
                )
                curr.net_income_yoy = _format_yoy(curr.net_income, prev.net_income)

        return {"annual": annual, "quarterly": quarterly}

    # ------------------------------------------------------------------
    # Overhang events (주요사항보고서 via event API)
    # ------------------------------------------------------------------

    # Event keyword -> overhang category
    _OVERHANG_EVENT_KEYWORDS: dict[str, str] = {
        "전환사채발행": "CB",
        "신주인수권부사채발행": "BW",

        "조건부자본증권발행": "PERPETUAL",
        "유상증자": "RIGHTS_ISSUE",
        "유무상증자": "MIXED_ISSUE",
        "주식매수선택권부여": "STOCK_OPTION",
    }

    def get_overhang_events(
        self, corp_code: str, start_year: int = 0,
    ) -> list[dict]:
        """Fetch overhang-relevant 주요사항보고서 events from OpenDART.

        Fetches: 전환사채, 신주인수권부사채, 영구채, 유상증자, 유무상증자
        Filters out 우선주 for capital increase events.

        Returns list of dicts with keys: category, series, face_value,
        conversion_price, convertible_shares, share_type, etc.
        """
        if not start_year:
            start_year = date.today().year - 5

        events: list[dict] = []
        start_date = f"{start_year}0101"

        for keyword, category in self._OVERHANG_EVENT_KEYWORDS.items():
            try:
                df = dart_call_with_retry(self.dart.event, corp_code, keyword, start=start_date)
            except Exception:
                logger.debug("Event fetch failed for %s/%s after retries", keyword, corp_code)
                continue

            if not _is_valid_df(df):
                continue

            for _, row in df.iterrows():
                event = self._parse_event_row(row, category)
                if event:
                    events.append(event)
                    logger.info(
                        "Found overhang event: %s %s (series=%s)",
                        category, corp_code, event.get("series"),
                    )

        return events

    def _parse_event_row(self, row: Any, category: str) -> dict | None:
        """Parse an event API row into a structured dict."""
        result: dict = {"category": category}
        # Event API doesn't return rcept_dt; extract filing date from rcept_no
        # rcept_no format: YYYYMMDD + 6-digit sequence (e.g. 20220920000336)
        rcept_no = str(row.get("rcept_no", "") or "")
        if len(rcept_no) == 14 and rcept_no.isdigit():
            result["disclosure_date"] = rcept_no[:8]
        else:
            result["disclosure_date"] = ""

        if category in ("CB", "BW", "PERPETUAL"):
            result["series"] = _parse_amount(row.get("bd_tm", "")) or 0
            result["kind"] = str(row.get("bd_knd", ""))
            result["face_value"] = _parse_amount(row.get("bd_fta", ""))
            result["rcept_no"] = str(row.get("rcept_no", ""))

            if category == "BW":
                # BW uses different field names: ex_prc, nstk_isstk_cnt, expd_bgd/edd
                result["conversion_price"] = _parse_amount(row.get("ex_prc", ""))
                result["convertible_shares"] = _parse_amount(row.get("nstk_isstk_cnt", ""))
                result["share_type"] = str(row.get("nstk_isstk_knd", ""))
                result["cv_start"] = str(row.get("expd_bgd", ""))
                result["cv_end"] = str(row.get("expd_edd", ""))
            else:
                # CB, PERPETUAL use cv_prc, cvisstk_cnt, cvrqpd_bgd/edd
                result["conversion_price"] = _parse_amount(row.get("cv_prc", ""))
                result["convertible_shares"] = _parse_amount(row.get("cvisstk_cnt", ""))
                result["share_type"] = str(row.get("cvisstk_knd", ""))
                result["cv_start"] = str(row.get("cvrqpd_bgd", ""))
                result["cv_end"] = str(row.get("cvrqpd_edd", ""))

        elif category == "STOCK_OPTION":
            # 주식매수선택권부여: stk_ostk_cnt or stk_isstk_ostk_cnt = 보통주
            shares = _parse_amount(row.get("stk_ostk_cnt", ""))
            if not shares:
                shares = _parse_amount(row.get("stk_isstk_ostk_cnt", ""))
            if not shares:
                shares = _parse_amount(row.get("nstk_ostk_cnt", ""))
            exercise_price = _parse_amount(row.get("ex_prc", ""))
            result["shares"] = shares or 0
            result["convertible_shares"] = shares or 0
            result["exercise_price"] = exercise_price or 0
            result["cv_start"] = str(row.get("expd_bgd", ""))
            result["cv_end"] = str(row.get("expd_edd", ""))
            if not shares or shares <= 0:
                logger.debug("Filtering out stock option with no shares")
                return None

        elif category in ("RIGHTS_ISSUE", "MIXED_ISSUE"):
            # nstk_ostk_cnt = 보통주, nstk_estk_cnt = 기타주(전환우선주 등)
            ordinary_shares = _parse_amount(row.get("nstk_ostk_cnt", ""))
            preferred_shares = _parse_amount(row.get("nstk_estk_cnt", ""))
            result["rcept_no"] = str(row.get("rcept_no", ""))

            # Filter out 보통주-only 유상증자; keep only 전환우선주
            if not preferred_shares or preferred_shares <= 0:
                logger.debug(
                    "Filtering out common stock rights issue: ostk=%s, estk=%s",
                    ordinary_shares, preferred_shares,
                )
                return None

            result["shares"] = preferred_shares
            result["share_type"] = "전환우선주"

            # Extract issue price per share for 기타주식
            issue_price = _parse_amount(row.get("stk_estk_issu_prc", ""))
            if issue_price:
                result["issue_price"] = issue_price

            # Compute face_value from funding purpose amounts
            total_funding = 0
            for field in ("fdpp_fclt", "fdpp_bsninh", "fdpp_op", "fdpp_dae", "fdpp_etc"):
                amt = _parse_amount(row.get(field, ""))
                if amt:
                    total_funding += amt
            if total_funding > 0:
                result["face_value"] = total_funding

        return result

    # ------------------------------------------------------------------
    # Overhang issuance decisions (dedicated DS005 APIs)
    # ------------------------------------------------------------------

    # Dedicated issuance decision API endpoints (structured, reliable)
    _ISSUANCE_APIS: dict[str, str] = {
        "CB": "https://opendart.fss.or.kr/api/cvbdIsDecsn.json",
        "BW": "https://opendart.fss.or.kr/api/bdwtIsDecsn.json",
        "RIGHTS_ISSUE": "https://opendart.fss.or.kr/api/piicDecsn.json",
        "STOCK_OPTION": "https://opendart.fss.or.kr/api/stkOptGrDecsn.json",
    }

    def get_overhang_issuances(
        self, corp_code: str, start_year: int = 0, after_date: str = "",
    ) -> list[dict]:
        """Fetch overhang issuance decisions from dedicated OpenDART DS005 APIs.

        Uses structured APIs instead of keyword-based event search:
        - 전환사채 (CB): cvbdIsDecsn
        - 신주인수권부사채 (BW): bdwtIsDecsn
        - 유상증자: piicDecsn

        For 유상증자: only keeps records where nstk_ostk_cnt == 0
        and nstk_estk_cnt != 0 (기타주식 only, e.g. 전환우선주).

        Args:
            corp_code: DART corp_code.
            start_year: Fallback start year if after_date is not provided.
            after_date: YYYYMMDD string. When provided, bgn_de is set to the
                day after this date so only post-기준일 issuances are returned.

        Returns list of dicts compatible with OverhangAnalyzer.process_event().
        """
        if after_date:
            # Start from the day after the reference date
            try:
                ref = date(int(after_date[:4]), int(after_date[4:6]), int(after_date[6:8]))
                bgn_de = (ref + timedelta(days=1)).strftime("%Y%m%d")
            except (ValueError, IndexError):
                bgn_de = after_date
        elif start_year:
            bgn_de = f"{start_year}0101"
        else:
            bgn_de = f"{date.today().year - 5}0101"
        end_de = date.today().strftime("%Y%m%d")

        results: list[dict] = []
        for category, url in self._ISSUANCE_APIS.items():
            params = {
                "crtfc_key": self._api_key,
                "corp_code": corp_code,
                "bgn_de": bgn_de,
                "end_de": end_de,
            }
            try:
                resp = dart_get_with_retry(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                logger.warning(
                    "Issuance API fetch failed: %s %s", category, corp_code,
                )
                continue

            status = data.get("status")
            if status != "000":
                logger.info(
                    "Issuance API %s: status=%s (%s)",
                    category, status, data.get("message", "no message"),
                )
                continue

            for item in data.get("list", []):
                parsed = self._parse_issuance_item(item, category)
                if parsed:
                    results.append(parsed)
                    logger.info(
                        "Found issuance: %s %s (series=%s)",
                        category, corp_code, parsed.get("series"),
                    )

        return results

    def _parse_issuance_item(self, item: dict, category: str) -> dict | None:
        """Parse a single issuance decision API response item.

        Returns dict compatible with OverhangAnalyzer.process_event().
        """
        rcept_no = str(item.get("rcept_no", "") or "")
        disclosure_date = rcept_no[:8] if len(rcept_no) >= 8 and rcept_no[:8].isdigit() else ""

        if category in ("CB", "BW"):
            result: dict = {
                "category": category,
                "rcept_no": rcept_no,
                "disclosure_date": disclosure_date,
                "series": _parse_amount(item.get("bd_tm")) or 0,
                "kind": str(item.get("bd_knd", "")),
                "face_value": _parse_amount(item.get("bd_fta")),
            }

            if category == "CB":
                result["conversion_price"] = _parse_amount(item.get("cv_prc"))
                result["convertible_shares"] = _parse_amount(item.get("cvisstk_cnt"))
                result["cv_start"] = str(item.get("cvrqpd_bgd", ""))
                result["cv_end"] = str(item.get("cvrqpd_edd", ""))
            elif category == "BW":
                result["conversion_price"] = _parse_amount(item.get("ex_prc"))
                result["convertible_shares"] = _parse_amount(item.get("nstk_isstk_cnt"))
                result["cv_start"] = str(item.get("expd_bgd", ""))
                result["cv_end"] = str(item.get("expd_edd", ""))

            return result

        elif category == "RIGHTS_ISSUE":
            ordinary = _parse_amount(item.get("nstk_ostk_cnt"))
            preferred = _parse_amount(item.get("nstk_estk_cnt"))

            # Filter: nstk_ostk_cnt == 0 AND nstk_estk_cnt != 0
            # Treat absent field (None) as zero — field may be omitted when not applicable
            if ordinary is not None and ordinary != 0:
                logger.debug(
                    "Filtering out rights issue with ordinary shares: ostk=%s",
                    ordinary,
                )
                return None
            if not preferred or preferred <= 0:
                logger.debug(
                    "Filtering out rights issue with no preferred shares: estk=%s",
                    preferred,
                )
                return None

            # Compute face_value from funding purpose amounts
            total_funding = 0
            for field in (
                "fdpp_fclt", "fdpp_bsninh", "fdpp_op",
                "fdpp_dtrp", "fdpp_ocsa", "fdpp_etc",
            ):
                amt = _parse_amount(item.get(field))
                if amt:
                    total_funding += amt

            # issue_price from API if available; conversion period is not
            # provided by piicDecsn — enriched later via process_rights_issue_disclosure
            issue_price = _parse_amount(item.get("stk_estk_issu_prc"))

            result_ri: dict = {
                "category": "RIGHTS_ISSUE",
                "rcept_no": rcept_no,
                "disclosure_date": disclosure_date,
                "shares": preferred,
                "share_type": "전환우선주",
                "face_value": total_funding if total_funding > 0 else None,
            }
            if issue_price:
                result_ri["issue_price"] = issue_price
            return result_ri

        elif category == "STOCK_OPTION":
            # 주식매수선택권부여결정 DS005 API
            shares = _parse_amount(item.get("stk_ostk_cnt"))
            if not shares:
                shares = _parse_amount(item.get("stk_isstk_ostk_cnt"))
            exercise_price = _parse_amount(item.get("ex_prc"))

            if not shares or shares <= 0:
                logger.debug("Filtering out stock option with no shares from DS005")
                return None

            return {
                "category": "STOCK_OPTION",
                "rcept_no": rcept_no,
                "disclosure_date": disclosure_date,
                "shares": shares,
                "convertible_shares": shares,
                "exercise_price": exercise_price or 0,
                "conversion_price": exercise_price or 0,
                "cv_start": str(item.get("expd_bgd", "")),
                "cv_end": str(item.get("expd_edd", "")),
            }

        return None

    # ------------------------------------------------------------------
    # Latest balance sheet (try quarterly first)
    # ------------------------------------------------------------------

    def get_latest_balance_sheet(
        self, corp_code: str, settlement_month: int = 12,
    ) -> tuple[BalanceSheet, BalanceSheet | None]:
        """Fetch the latest available balance sheet.

        Tries: Q3 → Q2 → Q1 of ongoing FY, then annual of completed FY,
        then Q3 → Q2 → Q1 of completed FY (if annual not yet filed).
        Returns (current_bs, previous_bs_or_None).

        Args:
            corp_code: DART corp code.
            settlement_month: Fiscal year-end month (default 12=December).
        """
        today = date.today()
        current_year = today.year
        current_month = today.month

        # Determine the ongoing and last completed fiscal years
        if current_month <= settlement_month:
            ongoing_fy = current_year
            completed_fy = current_year - 1
        else:
            ongoing_fy = current_year + 1
            completed_fy = current_year

        # Try quarterly reports for the ongoing fiscal year (newest first)
        for q in (3, 2, 1):
            bs = self.get_balance_sheet(corp_code, ongoing_fy, quarter=q)
            if bs.total_assets is not None:
                prev_bs = self.get_balance_sheet(corp_code, completed_fy)
                if prev_bs.total_assets is None:
                    prev_bs = None
                logger.info("Latest BS from FY%d Q%d", ongoing_fy, q)
                return bs, prev_bs

        # Try annual of completed fiscal year first (사업보고서 is the most
        # authoritative; once filed, it should take priority over quarterly).
        bs = self.get_balance_sheet(corp_code, completed_fy)
        if bs.total_assets is not None:
            prev_bs = self.get_balance_sheet(corp_code, completed_fy - 1)
            if prev_bs.total_assets is None:
                prev_bs = None
            logger.info("Latest BS from FY%d annual", completed_fy)
            return bs, prev_bs

        # Fall back to quarterly of completed FY if annual not yet filed
        # e.g. settlement_month=12, today=2026-02: FY2025 Q3 (2025.09)
        for q in (3, 2, 1):
            bs = self.get_balance_sheet(corp_code, completed_fy, quarter=q)
            if bs.total_assets is not None:
                prev_bs = self.get_balance_sheet(corp_code, completed_fy - 1)
                if prev_bs.total_assets is None:
                    prev_bs = None
                logger.info("Latest BS from FY%d Q%d", completed_fy, q)
                return bs, prev_bs

        # Try year before completed FY
        bs = self.get_balance_sheet(corp_code, completed_fy - 1)
        prev_bs = None
        if bs.total_assets is not None:
            prev_bs_candidate = self.get_balance_sheet(corp_code, completed_fy - 2)
            prev_bs = prev_bs_candidate if prev_bs_candidate.total_assets is not None else None

        return bs, prev_bs

    # ------------------------------------------------------------------
    # Overhang from financial statement notes (재무제표 주석)
    # ------------------------------------------------------------------

    def get_overhang_from_notes(
        self,
        corp_code: str,
        *,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ) -> tuple[list[dict], str]:
        """Fetch overhang instruments from the latest periodic report's 재무제표 주석.

        Finds the most recent periodic report (분기/반기/사업보고서),
        navigates to the financial statement notes section, fetches the HTML,
        and extracts CB, BW, and convertible preferred stock data.

        When *api_key* is provided, LLM-based extraction is attempted first,
        falling back to regex-based parsing on failure.

        Returns (active_instruments, reference_date) where reference_date is
        the period end date in YYYYMMDD format (e.g. "20250930").
        """
        from auto_reports.parsers.notes_overhang import parse_notes_overhang

        rcept_no, report_name, ref_date = self._find_latest_periodic_report(corp_code)
        if not rcept_no:
            logger.warning("No periodic report found for overhang notes: %s", corp_code)
            return [], ""

        logger.info(
            "Using report for overhang notes: %s (%s), 기준일=%s",
            report_name, rcept_no, ref_date,
        )

        html = self._fetch_notes_section_html(rcept_no)
        if not html:
            logger.warning("No notes section HTML found in report %s", rcept_no)
            return [], ref_date

        instruments = parse_notes_overhang(html, api_key=api_key, model=model, base_url=base_url)
        active = [inst for inst in instruments if inst.get("active", False)]
        logger.info(
            "Overhang from notes: %d active / %d total instruments",
            len(active), len(instruments),
        )
        return active, ref_date

    def _find_latest_periodic_report(self, corp_code: str) -> tuple[str, str, str]:
        """Find the most recent periodic report (분기/반기/사업보고서).

        Returns (rcept_no, report_name, reference_date) or ("", "", "").
        reference_date is the period end date in YYYYMMDD format, e.g. "20250930".
        """
        today = date.today()
        start = f"{today.year - 2}0101"
        end = today.strftime("%Y%m%d")

        try:
            df = dart_call_with_retry(self.dart.list, corp_code, start=start, end=end, kind="A")
        except Exception:
            logger.exception("dart.list failed for %s", corp_code)
            return ("", "", "")

        if not _is_valid_df(df):
            return ("", "", "")

        # Sort by receipt date descending (most recent first)
        if "rcept_dt" in df.columns:
            df = df.sort_values("rcept_dt", ascending=False)

        # Filter to periodic reports only
        mask = df["report_nm"].str.contains(
            r"분기보고서|반기보고서|사업보고서", na=False, regex=True,
        )
        filtered = df[mask]

        if filtered.empty:
            return ("", "", "")

        row = filtered.iloc[0]
        report_name = str(row["report_nm"])
        rcept_no = str(row["rcept_no"])
        ref_date = _parse_reference_date(report_name)
        return (rcept_no, report_name, ref_date)

    def _fetch_notes_section_html(self, rcept_no: str) -> str:
        """Fetch the 재무제표 주석 section HTML from a periodic report.

        Prefers '5.재무제표주석' (별도) over '3.연결재무제표주석' because
        분기보고서 often has only a placeholder for the consolidated notes.
        Falls back to whichever has more content.
        """
        import re as _re

        try:
            docs = dart_call_with_retry(self.dart.sub_docs, rcept_no)
        except Exception:
            logger.exception("sub_docs failed for %s after retries", rcept_no)
            return ""

        if not _is_valid_df(docs):
            return ""

        _ws = _re.compile(r"[\s\xa0]+")
        candidates: list[tuple[str, str, str]] = []  # (priority, title, url)

        for _, row in docs.iterrows():
            title = str(row.get("title", "")).strip()
            url = str(row.get("url", ""))
            title_norm = _ws.sub("", title)

            # Prefer 재무제표주석 (standalone) — usually has actual content
            if "재무제표주석" in title_norm and "연결" not in title_norm:
                candidates.append(("A", title, url))
            elif "연결재무제표주석" in title_norm:
                candidates.append(("B", title, url))

        if not candidates:
            logger.debug("No notes section found in sub_docs for %s", rcept_no)
            return ""

        # Sort by priority (A=standalone before B=consolidated)
        candidates.sort(key=lambda x: x[0])

        # Try candidates in order; pick the one with substantial content
        for _, title, url in candidates:
            try:
                resp = dart_get_with_retry(url, timeout=30)
                resp.raise_for_status()
                resp.encoding = "utf-8"
                html = resp.text
            except Exception:
                logger.warning("Failed to fetch notes section: %s", title)
                continue

            if len(html) > 1000:  # Skip placeholders (< 1KB)
                logger.info("Fetched notes section: %s (%d bytes)", title, len(html))
                return html
            else:
                logger.debug(
                    "Notes section too small (placeholder?): %s (%d bytes)",
                    title, len(html),
                )

        return ""

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_account_col(df: Any) -> str:
        """Return the column name that holds Korean account names."""
        for candidate in ("account_nm", "account_name", "계정명", "항목명"):
            if candidate in df.columns:
                return candidate
        for col in df.columns:
            if df[col].dtype == object:
                return col
        return df.columns[0]

    @staticmethod
    def _detect_amount_col(df: Any) -> str:
        """Return the column name that holds the current-period amount."""
        for candidate in ("thstrm_amount", "당기금액", "amount", "thstrm_add_amount"):
            if candidate in df.columns:
                return candidate
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols):
            return numeric_cols[-1]
        return df.columns[-1]

    @staticmethod
    def _detect_cumulative_amount_col(df: Any) -> str:
        """Return the column that holds cumulative (누적) amount for quarterly IS.

        For quarterly reports, thstrm_add_amount typically contains the
        year-to-date cumulative figure. Falls back to thstrm_amount.
        """
        for candidate in ("thstrm_add_amount", "thstrm_amount", "당기금액"):
            if candidate in df.columns:
                return candidate
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols):
            return numeric_cols[-1]
        return df.columns[-1]

    def get_finstate_text(
        self, corp_code: str, year: int, reprt_code: str = "11011",
    ) -> tuple[str, str]:
        """Return full BS and IS as formatted text for LLM highlight generation.

        Returns (bs_text, is_text) tuple. Empty strings if data unavailable.
        """
        df, stmt_label = self._fetch_finstate(corp_code, year, reprt_code)
        if df is None or df.empty:
            return "", ""

        account_col = (
            "account_nm" if "account_nm" in df.columns
            else self._detect_account_col(df)
        )
        amount_col = self._detect_amount_col(df)
        prev_col = "frmtrm_amount" if "frmtrm_amount" in df.columns else None

        def _format_section(sj_div: str) -> str:
            subset = df[df["sj_div"] == sj_div] if "sj_div" in df.columns else df
            if subset.empty:
                return ""
            lines = []
            for _, row in subset.iterrows():
                name = str(row.get(account_col, "")).strip()
                curr = str(row.get(amount_col, "")).strip()
                prev = str(row.get(prev_col, "")).strip() if prev_col else ""
                if name and curr:
                    line = f"{name}: 당기 {curr}"
                    if prev and prev != curr:
                        line += f", 전기 {prev}"
                    lines.append(line)
            return "\n".join(lines)

        bs_text = _format_section("BS")
        is_text = _format_section("IS")
        return bs_text, is_text


def _parse_reference_date(report_name: str) -> str:
    """Extract period end date (기준일) from a periodic report name.

    Handles two formats returned by OpenDART:
      Format A (older): "2025년 3분기 분기보고서"
      Format B (common): "분기보고서 (2025.09)"

    Examples:
        "2025년 3분기 분기보고서"           → "20250930"
        "2025년 반기보고서"                 → "20250630"
        "2024년 사업보고서"                 → "20241231"
        "2025년 1분기 분기보고서"           → "20250331"
        "[기재정정]2025년 3분기 분기보고서" → "20250930"
        "분기보고서 (2025.09)"             → "20250930"
        "반기보고서 (2025.06)"             → "20250630"
        "사업보고서 (2024.12)"             → "20241231"
        "[기재정정]분기보고서 (2025.09)"   → "20250930"

    Returns YYYYMMDD string, or "" if parsing fails.
    """
    # --- Format A: "2025년 3분기 분기보고서" ---
    m_year = _re_module.search(r"(\d{4})\s*년", report_name)
    if m_year:
        year = int(m_year.group(1))
        if "사업보고서" in report_name:
            return f"{year}1231"
        elif "반기보고서" in report_name:
            return f"{year}0630"
        elif "분기보고서" in report_name:
            m_q = _re_module.search(r"(\d)\s*분기", report_name)
            if m_q:
                quarter = int(m_q.group(1))
                month = quarter * 3
                last_day = calendar.monthrange(year, month)[1]
                return f"{year}{month:02d}{last_day:02d}"

    # --- Format B: "분기보고서 (2025.09)" ---
    m_paren = _re_module.search(r"\((\d{4})[./](\d{2})\)", report_name)
    if m_paren:
        year = int(m_paren.group(1))
        month = int(m_paren.group(2))
        last_day = calendar.monthrange(year, month)[1]
        return f"{year}{month:02d}{last_day:02d}"

    return ""


def _subtract_income(
    later: IncomeStatementItem,
    earlier: IncomeStatementItem,
    period: str,
) -> IncomeStatementItem:
    """Compute standalone quarter by subtracting earlier cumulative from later.

    later - earlier = standalone period.
    If either is missing data, returns what's available.
    """
    result = IncomeStatementItem(period=period, currency=later.currency)

    # Check if earlier period has any data at all
    earlier_has_data = any(
        getattr(earlier, f) is not None
        for f in ("revenue", "operating_income", "net_income")
    )

    for field in ("revenue", "operating_income", "net_income"):
        later_val = getattr(later, field)
        earlier_val = getattr(earlier, field)
        if later_val is not None and earlier_val is not None:
            setattr(result, field, later_val - earlier_val)
        elif later_val is not None and not earlier_has_data:
            # Earlier period entirely missing (e.g. newly listed company):
            # use cumulative value as-is rather than discarding
            setattr(result, field, later_val)

    return result
