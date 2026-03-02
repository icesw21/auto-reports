"""Main pipeline orchestrator - fetch, parse, analyze, generate."""

from __future__ import annotations

import io
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from auto_reports.analyzers.financial import (
    build_annual_rows,
    build_balance_sheet_rows,
    build_cumulative_annual_row,
    build_quarterly_rows,
)
from auto_reports.analyzers.overhang import OverhangAnalyzer
from auto_reports.config import CompanyYamlConfig, Settings
from auto_reports.fetchers.dart_business import DartBusinessFetcher
from auto_reports.fetchers.dart_html import DartHtmlFetcher
from auto_reports.fetchers.dart_report import DartReportFetcher
from auto_reports.fetchers.market_data import MarketDataFetcher
from auto_reports.fetchers.opendart import OpenDartFetcher, _format_yoy, _subtract_income
from auto_reports.models.financial import IncomeStatementItem
from auto_reports.generators.report import generate_report, write_report
from auto_reports.summarizers.openai_summarizer import generate_report_tags
from auto_reports.models.disclosure import (
    BondType,
    CBIssuance,
    ConversionTerms,
    Performance,
)
from auto_reports.models.report import (
    ExchangeContract,
    ReportData,
    ReportFrontmatter,
)
from auto_reports.parsers.classifier import DisclosureType, classify_disclosure
from auto_reports.parsers.contract import parse_contract
from auto_reports.parsers.convert import parse_convert
from auto_reports.parsers.convert_price import parse_convert_price
from auto_reports.parsers.exchange_disclosure import parse_exchange_disclosure_from_soup
from auto_reports.parsers.issue import parse_issue, parse_rights_issue_html
from auto_reports.parsers.performance import parse_performance

logger = logging.getLogger(__name__)

if sys.platform == "win32" and not (sys.stdout.encoding or "").startswith("utf"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_RCPNO_DATE_RE = re.compile(r"rcpNo=(\d{8})")


def _extract_date_from_url(url: str) -> str:
    """Extract disclosure date from DART URL's rcpNo parameter.

    rcpNo format: YYYYMMDD + sequence digits (e.g. 20240628901050).
    The first 8 digits are the filing date. Returns 'YYYY.MM.DD' or empty string.
    """
    m = _RCPNO_DATE_RE.search(url)
    if m:
        raw = m.group(1)
        return f"{raw[:4]}.{raw[4:6]}.{raw[6:8]}"
    return ""

console = Console(force_terminal=True)

# Map disclosure types to parser functions
def _is_empty_is(item: "IncomeStatementItem") -> bool:
    """Check if an income statement item has no data."""
    return item.revenue is None and item.operating_income is None and item.net_income is None


def _parse_quarter_from_period(period: str) -> tuple[int, int]:
    """Extract (year, quarter) from period string like '2023' or '2023.Q1'."""
    if ".Q" in period:
        parts = period.split(".Q")
        return int(parts[0]), int(parts[1])
    return int(period), 0


def _merge_none_fields(target: IncomeStatementItem, source: IncomeStatementItem) -> None:
    """Merge IS fields from source into target where target field is None."""
    for field in ("revenue", "operating_income", "net_income"):
        if getattr(target, field) is None and getattr(source, field) is not None:
            setattr(target, field, getattr(source, field))


def _fill_quarterly_cumulative(
    quarterly: list,
    report_fetcher: "DartReportFetcher",
    corp: str,
) -> None:
    """Fill remaining None fields in quarterly IS using cumulative subtraction.

    For quarterly HTML reports, the standalone 3-month column may lack certain
    fields (especially net_income).  This pass fetches cumulative (누적) figures
    and applies: Q2 = H1_cum - Q1, Q3 = 9M_cum - H1_cum.
    """
    yearly: dict[int, dict[int, IncomeStatementItem]] = {}
    for item in quarterly:
        if ".Q" in item.period:
            year, q = _parse_quarter_from_period(item.period)
            yearly.setdefault(year, {})[q] = item

    for year, quarters in yearly.items():
        # Check if any Q1-Q3 still has None fields
        needs = False
        for q in (1, 2, 3):
            if q in quarters:
                item = quarters[q]
                for field in ("revenue", "operating_income", "net_income"):
                    if getattr(item, field) is None:
                        needs = True
                        break
            if needs:
                break

        if not needs:
            continue

        logger.info("Cumulative subtraction pass for year %d (%s)", year, corp)

        # Fill Q1 missing fields from HTML report (Q1 cum = standalone)
        if 1 in quarters:
            q1 = quarters[1]
            q1_has_none = any(
                getattr(q1, f) is None
                for f in ("revenue", "operating_income", "net_income")
            )
            if q1_has_none:
                q1_html = report_fetcher.get_income_statement(
                    corp, year, quarter=1,
                )
                if not _is_empty_is(q1_html):
                    _merge_none_fields(q1, q1_html)
                    logger.debug(
                        "Filled Q1 %d from HTML: rev=%s, op=%s, ni=%s",
                        year, q1.revenue, q1.operating_income, q1.net_income,
                    )

        # Q1 cumulative = Q1 standalone (now with HTML-filled fields)
        cum_q1 = quarters.get(1, IncomeStatementItem(period=f"{year}.Q1"))

        # Fetch H1 and Q3 cumulative from HTML reports
        cum_h1 = report_fetcher.get_income_statement(
            corp, year, quarter=2, cumulative=True,
        )
        cum_q3 = report_fetcher.get_income_statement(
            corp, year, quarter=3, cumulative=True,
        )
        logger.debug(
            "Cumulative H1 %d: rev=%s, op=%s, ni=%s",
            year, cum_h1.revenue, cum_h1.operating_income, cum_h1.net_income,
        )
        logger.debug(
            "Cumulative 9M %d: rev=%s, op=%s, ni=%s",
            year, cum_q3.revenue, cum_q3.operating_income, cum_q3.net_income,
        )

        # Fill Q2 = H1 cumulative - Q1 cumulative
        if 2 in quarters and not _is_empty_is(cum_h1):
            q2_comp = _subtract_income(cum_h1, cum_q1, f"{year}.Q2")
            _merge_none_fields(quarters[2], q2_comp)
            logger.debug(
                "Q2 %d after cumulative: rev=%s, op=%s, ni=%s",
                year, quarters[2].revenue, quarters[2].operating_income,
                quarters[2].net_income,
            )

        # Fill Q3 = 9M cumulative - H1 cumulative
        if 3 in quarters and not _is_empty_is(cum_q3) and not _is_empty_is(cum_h1):
            q3_comp = _subtract_income(cum_q3, cum_h1, f"{year}.Q3")
            _merge_none_fields(quarters[3], q3_comp)
            logger.debug(
                "Q3 %d after cumulative: rev=%s, op=%s, ni=%s",
                year, quarters[3].revenue, quarters[3].operating_income,
                quarters[3].net_income,
            )


def _detect_performance_period(period: dict) -> tuple[int, int, bool]:
    """Detect period type from Performance disclosure period dict.

    Returns ``(year, quarter, is_cumulative)`` where:
    - quarter=0 means annual (12-month period)
    - quarter=1..4 means the quarter the period ends in
    - is_cumulative=True when the period starts in January and spans
      more than one quarter (e.g. H1 or 9-month cumulative).
    """
    period_start = period.get("- 시작일", "") or period.get("시작일", "")
    period_end = period.get("- 종료일", "") or period.get("종료일", "")

    start_m = re.search(r"(\d{4})\D?(\d{2})", period_start)
    end_m = re.search(r"(\d{4})\D?(\d{2})", period_end)

    if not end_m:
        return (0, 0, False)

    end_year = int(end_m.group(1))
    end_month = int(end_m.group(2))
    start_month = int(start_m.group(2)) if start_m else 1

    month_span = end_month - start_month + 1
    if month_span <= 0:
        month_span += 12

    # 11-12 months → annual
    if month_span >= 11:
        return (end_year, 0, False)

    # Determine quarter from end month
    if end_month <= 3:
        quarter = 1
    elif end_month <= 6:
        quarter = 2
    elif end_month <= 9:
        quarter = 3
    else:
        quarter = 4

    # Cumulative if starts in January and spans more than one quarter
    is_cumulative = (start_month <= 1 and month_span > 4)

    return (end_year, quarter, is_cumulative)


def _upsert_statement(statements: list, item: IncomeStatementItem) -> None:
    """Insert or replace an IncomeStatementItem in a statements list."""
    for i, s in enumerate(statements):
        if s.period == item.period:
            statements[i] = item
            return
    statements.append(item)


def _build_is_from_performance(perf: Performance) -> IncomeStatementItem:
    """Build an IncomeStatementItem from a Performance disclosure."""
    is_item = IncomeStatementItem(period="")
    if perf.revenue and perf.revenue.current_int is not None:
        is_item.revenue = perf.revenue.current_int
    if perf.operating_income and perf.operating_income.current_int is not None:
        is_item.operating_income = perf.operating_income.current_int
    if perf.net_income and perf.net_income.current_int is not None:
        is_item.net_income = perf.net_income.current_int
    return is_item


def _integrate_performance_disclosures(
    parsed_disclosures: dict,
    annual_statements: list,
    quarterly_statements: list,
) -> None:
    """Integrate 매출액또는손익구조 변동 disclosures into IS.

    Routes data by period length:
    - 12-month (annual) → annual_statements
    - 3-month (standalone quarter) → quarterly_statements
    - 6/9-month (cumulative) → derive standalone quarter via subtraction

    Quarterly data in quarterly_statements is automatically picked up by
    ``build_cumulative_annual_row`` for yearly accumulation.
    """
    for perf in parsed_disclosures.get(DisclosureType.PERFORMANCE, []):
        if not isinstance(perf, Performance):
            continue
        if not perf.income_changes:
            logger.debug("Performance disclosure skipped: no income_changes data")
            continue
        if not perf.period:
            logger.debug("Performance disclosure skipped: no period data")
            continue

        year, quarter, is_cumulative = _detect_performance_period(perf.period)
        if year == 0:
            logger.debug("Performance disclosure skipped: could not detect period")
            continue

        is_item = _build_is_from_performance(perf)
        if _is_empty_is(is_item):
            continue

        logger.debug(
            "Performance period: year=%d, quarter=%d, cumulative=%s, "
            "revenue=%s, op=%s, ni=%s",
            year, quarter, is_cumulative,
            is_item.revenue, is_item.operating_income, is_item.net_income,
        )

        if quarter == 0:
            # ── Annual ──
            year_str = str(year)
            is_item.period = year_str
            existing = next(
                (s for s in annual_statements
                 if s.period == year_str and not _is_empty_is(s)),
                None,
            )
            if existing:
                continue
            _upsert_statement(annual_statements, is_item)
            logger.info("Added annual IS from performance disclosure: %s", year_str)

        elif not is_cumulative:
            # ── Standalone quarter (e.g. Q1 = 3 months) ──
            period_str = f"{year}.Q{quarter}"
            is_item.period = period_str
            existing = next(
                (s for s in quarterly_statements
                 if s.period == period_str and not _is_empty_is(s)),
                None,
            )
            if existing:
                continue
            _upsert_statement(quarterly_statements, is_item)
            logger.info("Added quarterly IS from performance disclosure: %s", period_str)

        else:
            # ── Cumulative period (H1 or 9M) → derive standalone quarter ──
            existing_qs: dict[int, IncomeStatementItem] = {}
            for s in quarterly_statements:
                if ".Q" in s.period:
                    y, q = _parse_quarter_from_period(s.period)
                    if y == year and not _is_empty_is(s):
                        existing_qs[q] = s

            if quarter == 2 and 1 in existing_qs:
                # Q2 = H1_cumulative - Q1 (skip if Q2 already has data)
                if 2 not in existing_qs:
                    q2 = _subtract_income(is_item, existing_qs[1], f"{year}.Q2")
                    if not _is_empty_is(q2):
                        _upsert_statement(quarterly_statements, q2)
                        logger.info("Added Q2 IS from H1 cumulative performance: %s.Q2", year)
            elif quarter == 3 and 1 in existing_qs and 2 in existing_qs:
                # Q3 = 9M_cumulative - (Q1 + Q2) (skip if Q3 already has data)
                if 3 not in existing_qs:
                    h1_sum = IncomeStatementItem(period="_tmp")
                    for field in ("revenue", "operating_income", "net_income"):
                        v1 = getattr(existing_qs[1], field)
                        v2 = getattr(existing_qs[2], field)
                        if v1 is not None and v2 is not None:
                            setattr(h1_sum, field, v1 + v2)
                    q3 = _subtract_income(is_item, h1_sum, f"{year}.Q3")
                    if not _is_empty_is(q3):
                        _upsert_statement(quarterly_statements, q3)
                        logger.info("Added Q3 IS from 9M cumulative performance: %s.Q3", year)
            else:
                logger.debug(
                    "Skipping cumulative performance Q%d %d: "
                    "missing preceding quarters",
                    quarter, year,
                )


def _fill_gaps_with_report(
    annual: list,
    quarterly: list,
    report_fetcher: "DartReportFetcher",
    corp: str,
) -> tuple[list, list]:
    """Fill empty income statements using HTML report parsing fallback.

    Five passes:
    1. Fill annual gaps from HTML reports
    2. Fill quarterly gaps with standalone 3-month data
    3. Fill remaining None fields via cumulative subtraction
    4. Compute Q4 = Annual - (Q1 + Q2 + Q3)
    5. Recalculate quarterly YoY
    """
    # Pass 1: Fill annual gaps
    for i, item in enumerate(annual):
        if _is_empty_is(item):
            year, _ = _parse_quarter_from_period(item.period)
            logger.info("Report fallback: annual %d for %s", year, corp)
            report_item = report_fetcher.get_income_statement(corp, year, quarter=0)
            if not _is_empty_is(report_item):
                report_item.period = item.period
                annual[i] = report_item

    # Pass 2: Fill quarterly gaps with standalone 3-month data
    for i, item in enumerate(quarterly):
        if _is_empty_is(item):
            year, q = _parse_quarter_from_period(item.period)
            if q == 0 or q == 4:
                continue
            logger.info("Report fallback: %d Q%d for %s", year, q, corp)
            report_item = report_fetcher.get_income_statement(corp, year, quarter=q)
            if not _is_empty_is(report_item):
                report_item.period = item.period
                quarterly[i] = report_item

    # Pass 3: Fill remaining None fields via cumulative subtraction
    _fill_quarterly_cumulative(quarterly, report_fetcher, corp)

    # Pass 4: Compute Q4 = Annual - (Q1 + Q2 + Q3) for years where we have all three
    yearly_quarters: dict[int, dict[int, IncomeStatementItem]] = {}
    for item in quarterly:
        if ".Q" in item.period:
            year, q = _parse_quarter_from_period(item.period)
            yearly_quarters.setdefault(year, {})[q] = item

    annual_by_year: dict[int, IncomeStatementItem] = {}
    for item in annual:
        year, _ = _parse_quarter_from_period(item.period)
        annual_by_year[year] = item

    for year, quarters in yearly_quarters.items():
        q4_item = quarters.get(4)
        if q4_item and _is_empty_is(q4_item):
            annual_item = annual_by_year.get(year)
            q1 = quarters.get(1)
            q2 = quarters.get(2)
            q3 = quarters.get(3)
            if (
                annual_item and not _is_empty_is(annual_item)
                and q1 and not _is_empty_is(q1)
                and q2 and not _is_empty_is(q2)
                and q3 and not _is_empty_is(q3)
            ):
                ann_rev = annual_item.revenue or 0
                if ann_rev <= 0:
                    continue
                q123_rev = sum(
                    v for v in [q1.revenue, q2.revenue, q3.revenue] if v is not None
                )
                if q123_rev > ann_rev:
                    logger.warning(
                        "Q1-Q3 sum (%s) exceeds annual (%s) for %d — skipping Q4",
                        q123_rev, ann_rev, year,
                    )
                    continue

                for field in ("revenue", "operating_income", "net_income"):
                    ann_val = getattr(annual_item, field)
                    q1_val = getattr(q1, field)
                    q2_val = getattr(q2, field)
                    q3_val = getattr(q3, field)
                    if all(v is not None for v in [ann_val, q1_val, q2_val, q3_val]):
                        setattr(q4_item, field, ann_val - q1_val - q2_val - q3_val)
                logger.info(
                    "Computed Q4 %d: rev=%s, op=%s, ni=%s",
                    year, q4_item.revenue, q4_item.operating_income, q4_item.net_income,
                )

    # Pass 5: Recalculate quarterly YoY after filling gaps (ascending order)
    quarterly_by_q: dict[int, list] = {}
    for q_item in quarterly:
        if ".Q" in q_item.period:
            q_num = int(q_item.period.split(".Q")[1])
            quarterly_by_q.setdefault(q_num, []).append(q_item)

    for items in quarterly_by_q.values():
        items_asc = sorted(items, key=lambda x: x.period)
        for j in range(1, len(items_asc)):
            curr = items_asc[j]
            prev = items_asc[j - 1]
            if curr.revenue_yoy is None:
                curr.revenue_yoy = _format_yoy(curr.revenue, prev.revenue)
            if curr.operating_income_yoy is None:
                curr.operating_income_yoy = _format_yoy(
                    curr.operating_income, prev.operating_income,
                )
            if curr.net_income_yoy is None:
                curr.net_income_yoy = _format_yoy(curr.net_income, prev.net_income)

    return annual, quarterly


def _is_sparse_rights_issue(data: dict | None) -> bool:
    """Check if rights issue parse result lacks key fields."""
    if not data:
        return True
    key_fields = ("new_shares", "issue_price", "payment_date", "board_decision_date")
    filled = sum(1 for k in key_fields if data.get(k))
    return filled < 2


def _is_sparse_cb_issuance(cb: CBIssuance) -> bool:
    """Check if CB issuance parse result lacks key fields."""
    return cb.face_value is None and cb.conversion_terms is None


def _try_exchange_llm_fallback(
    raw_text: str, api_key: str, model: str,
) -> tuple[str, dict] | None:
    """Try LLM extraction for unrecognized exchange disclosures.

    Attempts performance, contract, then overhang extractors in sequence.
    Returns (category, data) on success, or None.
    """
    from auto_reports.summarizers.exchange_llm_summarizer import (
        extract_exchange_contract,
        extract_exchange_overhang,
        extract_exchange_performance,
    )

    perf = extract_exchange_performance(raw_text, api_key, model)
    if perf and perf.get("income_changes"):
        return ("sales", perf)

    contract = extract_exchange_contract(raw_text, api_key, model)
    if contract and contract.get("type"):
        return ("backlog", contract)

    overhang = extract_exchange_overhang(raw_text, api_key, model)
    if overhang:
        return overhang

    return None


def _try_cb_issuance_llm_fallback(
    raw_text: str, api_key: str, model: str,
) -> CBIssuance | None:
    """Try LLM extraction for CB/BW issuance disclosures."""
    from auto_reports.summarizers.disclosure_llm_summarizer import extract_cb_issuance

    llm_data = extract_cb_issuance(raw_text, api_key, model)
    if not llm_data:
        return None

    # BondType uses aliases ("회차", "종류") - pass raw dict directly
    bond_type = None
    bt_raw = llm_data.get("bond_type")
    if isinstance(bt_raw, dict):
        bond_type = BondType(**bt_raw)

    conv_price = llm_data.get("conversion_price")
    conv_shares = llm_data.get("convertible_shares")
    share_ratio = llm_data.get("share_ratio_pct")
    exercise_period = llm_data.get("exercise_period")

    # ConversionTerms uses aliases - construct with alias keys
    conv_terms = None
    if conv_price or conv_shares:
        share_info = {}
        if conv_shares:
            share_info["주식수"] = conv_shares
        if share_ratio:
            share_info["주식총수 대비 비율(%)"] = share_ratio
        ct_kwargs: dict = {}
        if conv_price is not None:
            ct_kwargs["전환가액 (원/주)"] = conv_price
        if share_info:
            ct_kwargs["전환에_따라_발행할_주식"] = share_info
        if exercise_period:
            ct_kwargs["전환청구기간"] = exercise_period
        conv_terms = ConversionTerms(**ct_kwargs)

    return CBIssuance(
        issuance_type_name=llm_data.get("issuance_type_name", ""),
        bond_type=bond_type,
        face_value=llm_data.get("face_value"),
        conversion_terms=conv_terms,
    )


# ─── Phase 3b fallback helpers: LLM extraction from local PDFs ───

# Regex for 주요사항보고서 PDF filenames in stock directories
_PDF_DISCLOSURE_RE = re.compile(
    r"^(\d{8})_\[[^\]]+\]_\d+_.*주요사항보고서\((.+?)\)\.pdf$"
)

_OVERHANG_PDF_TYPES = {
    "유상증자결정": "RIGHTS_ISSUE",
    "전환사채권발행결정": "CB",

    "신주인수권부사채권발행결정": "BW",
    "주식매수선택권부여결정": "STOCK_OPTION",
}


def _read_pdf_text_for_disclosure(path: Path, max_chars: int = 8000) -> str:
    """Read text from a disclosure PDF file using pdfplumber or PyPDF2."""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                if len(text) > max_chars:
                    break
            return text[:max_chars]
    except ImportError:
        pass
    try:
        import PyPDF2

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                if len(text) > max_chars:
                    break
            return text[:max_chars]
    except Exception:
        logger.warning("Failed to read PDF: %s", path.name)
        return ""


def _convert_cb_llm_to_event(llm_data: dict, default_category: str) -> dict:
    """Convert LLM extract_cb_issuance output to process_event format."""
    bond_type = llm_data.get("bond_type") or {}
    series = bond_type.get("회차", 0) or 0
    kind = bond_type.get("종류", "")
    category = default_category
    if "전환" in kind:
        category = "CB"
    elif "신주인수권" in kind:
        category = "BW"
    exercise_period = llm_data.get("exercise_period") or {}
    return {
        "category": category,
        "series": series,
        "kind": kind,
        "face_value": llm_data.get("face_value") or 0,
        "conversion_price": llm_data.get("conversion_price") or 0,
        "convertible_shares": llm_data.get("convertible_shares") or 0,
        "cv_start": exercise_period.get("시작일", ""),
        "cv_end": exercise_period.get("종료일", ""),
    }


def _convert_ri_llm_to_event(llm_data: dict) -> dict:
    """Convert LLM extract_rights_issue output to process_event format."""
    new_shares = llm_data.get("new_shares") or {}
    common = new_shares.get("보통주식", 0) or 0
    other = new_shares.get("기타주식", 0) or 0
    conversion = llm_data.get("conversion") or {}
    exercise_period = conversion.get("전환청구기간") or {}
    conv_price = conversion.get("전환가액(원/주)") or 0
    conv_shares = conversion.get("전환주식수") or 0
    total_shares = common + other
    # Use 기타주식 only when common==0, to mirror the API filter (nstk_ostk_cnt==0)
    shares_for_match = other if (other > 0 and common == 0) else total_shares
    return {
        "category": "RIGHTS_ISSUE",
        "shares": shares_for_match,
        "new_shares_common": common,
        "new_shares_other": other,
        "issue_price": llm_data.get("issue_price") or 0,
        "conversion_price": conv_price,
        "convertible_shares": conv_shares if conv_shares else total_shares,
        "cv_start": exercise_period.get("시작일", ""),
        "cv_end": exercise_period.get("종료일", ""),
        "payment_date": llm_data.get("payment_date") or "",
        "share_type": llm_data.get("share_type") or "",
    }


def _phase3b_llm_pdf_fallback(
    stock_dir: Path,
    reference_date: str,
    overhang_analyzer: "OverhangAnalyzer",
    api_key: str,
    model: str,
    console: "Console",
    skip_categories: set[str] | None = None,
) -> int:
    """LLM enrichment/fallback: extract overhang data from local 주요사항보고서 PDFs.

    Always runs for RIGHTS_ISSUE (piicDecsn API lacks conversion terms).
    For CB/BW/SO, runs only as fallback when the DS005 API returned no data.
    Categories in *skip_categories* are skipped (already handled by API).

    Returns number of events processed.
    """
    from auto_reports.summarizers.disclosure_llm_summarizer import (
        extract_cb_issuance,
        extract_rights_issue,
        extract_stock_option,
    )

    ref_clean = reference_date.replace("-", "").replace(".", "")
    count = 0

    for pdf_path in sorted(stock_dir.glob("*주요사항보고서*.pdf")):
        m = _PDF_DISCLOSURE_RE.match(pdf_path.name)
        if not m:
            continue
        file_date = m.group(1)
        file_type = m.group(2)

        # Only post-기준일
        if ref_clean and file_date <= ref_clean:
            continue
        # Only overhang-relevant types
        if file_type not in _OVERHANG_PDF_TYPES:
            continue

        category = _OVERHANG_PDF_TYPES[file_type]

        # Skip categories already handled by DS005 API
        if skip_categories and category in skip_categories:
            continue

        console.print(f"    [dim]LLM enrichment: {pdf_path.name}[/dim]")
        text = _read_pdf_text_for_disclosure(pdf_path)
        if not text:
            continue

        if category == "RIGHTS_ISSUE":
            llm_data = extract_rights_issue(text, api_key, model)
            if llm_data:
                event = _convert_ri_llm_to_event(llm_data)
                if event.get("shares", 0) <= 0:
                    logger.warning(
                        "LLM PDF enrichment: RIGHTS_ISSUE new_shares not extracted from %s, skipping",
                        pdf_path.name,
                    )
                    continue
                # Pass file_date for stable key generation (avoids ordinal collision)
                event["pdf_date"] = file_date
                overhang_analyzer.process_event(event)
                count += 1
                logger.info("LLM PDF enrichment: %s from %s", category, pdf_path.name)
        elif category == "STOCK_OPTION":
            llm_data = extract_stock_option(text, api_key, model)
            if llm_data:
                shares = llm_data.get("shares") or 0
                if shares <= 0:
                    logger.warning(
                        "LLM PDF fallback: STOCK_OPTION shares not extracted from %s, skipping",
                        pdf_path.name,
                    )
                    continue
                exercise_period = llm_data.get("exercise_period") or {}
                event = {
                    "category": "STOCK_OPTION",
                    "series": llm_data.get("series") or 0,
                    "shares": shares,
                    "convertible_shares": shares,
                    "exercise_price": llm_data.get("exercise_price") or 0,
                    "conversion_price": llm_data.get("exercise_price") or 0,
                    "cv_start": exercise_period.get("시작일", ""),
                    "cv_end": exercise_period.get("종료일", ""),
                }
                overhang_analyzer.process_event(event)
                count += 1
                logger.info("LLM PDF fallback: %s from %s", category, pdf_path.name)
        else:
            llm_data = extract_cb_issuance(text, api_key, model)
            if llm_data:
                event = _convert_cb_llm_to_event(llm_data, category)
                overhang_analyzer.process_event(event)
                count += 1
                logger.info("LLM PDF fallback: %s from %s", category, pdf_path.name)

    if count:
        console.print(f"  [green]LLM PDF fallback[/green]: {count} issuances extracted")
    return count


def _is_sparse_exchange_result(category: str, result: dict) -> bool:
    """Check if deterministic exchange disclosure result is too sparse to use."""
    if not result:
        return True
    if category == "exercise":
        return not result.get("cb_balance") and not result.get("daily_claims")
    if category == "price_adj":
        return not result.get("adjustments")
    if category == "backlog":
        return not result.get("contract_amount") or not result.get("description")
    if category == "sales":
        return not result.get("income_changes")
    return True


PARSER_MAP = {
    DisclosureType.CONVERT: parse_convert,
    DisclosureType.CONVERT_PRICE_CHANGE: parse_convert_price,
    DisclosureType.CONTRACT: parse_contract,
    DisclosureType.PERFORMANCE: parse_performance,
    DisclosureType.ISSUE: parse_issue,
    DisclosureType.RIGHTS_ISSUE: parse_rights_issue_html,
}


def run_pipeline(
    config_path: str | Path,
    output_dir: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Run the full pipeline: fetch -> classify -> parse -> analyze -> generate.

    Args:
        config_path: Path to company YAML config file.
        output_dir: Override output directory.
        verbose: Enable verbose logging.
        dry_run: Parse and analyze but don't write report.

    Returns:
        Path to generated report file, or None on failure.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load configuration
    config_path = Path(config_path)
    company_config = CompanyYamlConfig.from_yaml(config_path)
    settings = Settings()

    # Set defaults that don't depend on stock_dir
    company_config.ensure_defaults()

    # Auto-fill empty config fields from stock data directory
    stock_dir = company_config.resolve_stock_dir(settings.stocks_base_dir)
    if not stock_dir and settings.output_dir != settings.stocks_base_dir:
        stock_dir = company_config.resolve_stock_dir(settings.output_dir)
    if stock_dir:
        company_config.auto_fill_from_stock_dir(stock_dir)

    company = company_config.company
    report_cfg = company_config.report

    console.print(f"\n[bold]Finance Parser[/bold] - {company.name} ({company.ticker})")
    if stock_dir:
        console.print(f"  Stock data dir: {stock_dir}")
    console.print(f"  Statement type: {report_cfg.statement_type}")
    console.print(f"  Lookback: {report_cfg.years} years, {report_cfg.quarters} quarters\n")

    # Resolve URLs - use project root (cwd) as base for relative paths
    project_root = Path.cwd()
    urls = company_config.disclosures.load_urls(project_root)
    # Filter out non-DART URLs (e.g., Naver Stock links)
    dart_urls = [u for u in urls if "dart.fss.or.kr" in u]
    console.print(f"[dim]Found {len(dart_urls)} DART disclosure URLs[/dim]")

    # ─── Phase 1: Fetch and parse disclosures ───
    parsed_disclosures: dict[DisclosureType, list] = {t: [] for t in DisclosureType}
    errors: list[str] = []

    fetcher = DartHtmlFetcher(
        delay=settings.request_delay,
        timeout=settings.request_timeout,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing disclosures...", total=len(dart_urls))

        for url in dart_urls:
            try:
                title, soup = fetcher.fetch_disclosure(url)
                disc_type = classify_disclosure(title)

                if disc_type == DisclosureType.UNKNOWN:
                    logger.debug("Unknown disclosure type: %s (%s)", title, url)
                    progress.advance(task)
                    continue

                parser_fn = PARSER_MAP.get(disc_type)
                if parser_fn:
                    result = parser_fn(soup)
                    # LLM fallback for sparse deterministic results
                    if settings.openai_api_key:
                        if (
                            disc_type == DisclosureType.RIGHTS_ISSUE
                            and isinstance(result, dict)
                            and _is_sparse_rights_issue(result)
                        ):
                            raw_text = soup.get_text(separator="\n", strip=True)
                            from auto_reports.summarizers.disclosure_llm_summarizer import (
                                extract_rights_issue,
                            )
                            llm_ri = extract_rights_issue(
                                raw_text, settings.openai_api_key, settings.openai_model,
                            )
                            if llm_ri:
                                result = llm_ri
                                logger.info("LLM fallback for sparse rights issue: %s", url)
                    # Set disclosure_date from URL rcpNo for ISSUE disclosures
                    if disc_type == DisclosureType.ISSUE and isinstance(result, CBIssuance):
                        result.disclosure_date = _extract_date_from_url(url)
                    elif disc_type == DisclosureType.RIGHTS_ISSUE and isinstance(result, dict):
                        result["disclosure_date"] = _extract_date_from_url(url)
                    parsed_disclosures[disc_type].append(result)
                    logger.debug("Parsed %s: %s", disc_type.name, url)

            except Exception as e:
                errors.append(f"{url}: {e}")
                logger.warning("Failed to parse %s: %s", url, e)

            progress.advance(task)

    # Print summary
    for dt, items in parsed_disclosures.items():
        if items:
            console.print(f"  [green]{dt.name}[/green]: {len(items)} disclosures parsed")
    if errors:
        console.print(f"  [yellow]Errors: {len(errors)} URLs failed[/yellow]")
        if verbose:
            for err in errors:
                console.print(f"    [dim]{err}[/dim]")

    # ─── Phase 1.5: Fetch and parse exchange disclosures ───
    # Exchange disclosures (from KRX) use a dedicated parser that handles
    # the EUC-KR HTML format. Results are collected by category:
    #   exercise/price_adj → overhang processing in Phase 3
    #   sales → performance integration
    #   backlog → exchange contract list for report
    exchange_overhang_exercises: list[dict] = []
    exchange_overhang_price_adjs: list[dict] = []
    exchange_contracts_raw: list[dict] = []

    try:
        exchange_entries = company_config.disclosures.load_exchange_disclosures(project_root)
    except Exception as e:
        logger.warning("Failed to load exchange disclosures: %s", e)
        exchange_entries = []
    if exchange_entries:
        console.print(f"\n[dim]Parsing {len(exchange_entries)} exchange disclosures...[/dim]")
        for entry in exchange_entries:
            url = entry.get("url", "")
            title = entry.get("title", "")
            if not url:
                continue
            try:
                # Fetch the HTML (DartHtmlFetcher handles EUC-KR)
                if title:
                    _, soup = fetcher.fetch_disclosure(url)
                else:
                    title, soup = fetcher.fetch_disclosure(url)

                # Use the exchange disclosure parser (not DART parsers)
                category, result = parse_exchange_disclosure_from_soup(soup)

                # LLM enhancement: if deterministic failed or returned sparse
                # result, use unified LLM extraction (single call handles all types)
                if settings.openai_api_key and (
                    category == "unknown"
                    or _is_sparse_exchange_result(category, result)
                ):
                    raw_text = soup.get_text(separator="\n", strip=True)
                    from auto_reports.summarizers.exchange_llm_summarizer import (
                        extract_exchange_disclosure_unified,
                    )

                    llm_out = extract_exchange_disclosure_unified(
                        raw_text, title,
                        settings.openai_api_key, settings.openai_model,
                    )
                    if llm_out:
                        category, result = llm_out
                        logger.info(
                            "LLM enhanced exchange %s: %s", category, title or url,
                        )

                if category == "exercise":
                    result["_entry_date"] = entry.get("date", "")
                    exchange_overhang_exercises.append(result)
                    logger.debug("Parsed exchange exercise: %s", title or url)
                elif category == "price_adj":
                    result["_entry_date"] = entry.get("date", "")
                    exchange_overhang_price_adjs.append(result)
                    logger.debug("Parsed exchange price_adj: %s", title or url)
                elif category == "sales":
                    # Convert dict back to Performance model for integration
                    perf = Performance(**result) if result.get("income_changes") else None
                    if perf:
                        parsed_disclosures[DisclosureType.PERFORMANCE].append(perf)
                    logger.debug("Parsed exchange sales: %s", title or url)
                elif category == "backlog":
                    result["_entry_date"] = entry.get("date", "")
                    exchange_contracts_raw.append(result)
                    logger.debug("Parsed exchange backlog: %s", title or url)
                elif category == "unknown":
                    # Last resort: try DART classifier for non-exchange types
                    disc_type = classify_disclosure(title) if title else DisclosureType.UNKNOWN
                    if disc_type != DisclosureType.UNKNOWN:
                        parser_fn = PARSER_MAP.get(disc_type)
                        if parser_fn:
                            dart_result = parser_fn(soup)
                            if disc_type == DisclosureType.ISSUE and isinstance(dart_result, CBIssuance):
                                entry_date = entry.get("date", "")
                                if entry_date:
                                    dart_result.disclosure_date = entry_date.replace("-", ".")
                                else:
                                    dart_result.disclosure_date = _extract_date_from_url(url)
                            elif disc_type == DisclosureType.RIGHTS_ISSUE and isinstance(dart_result, dict):
                                entry_date = entry.get("date", "")
                                if entry_date:
                                    dart_result["disclosure_date"] = entry_date.replace("-", "").replace(".", "")
                                else:
                                    dart_result["disclosure_date"] = _extract_date_from_url(url)
                            parsed_disclosures[disc_type].append(dart_result)
                            logger.debug("Parsed exchange (DART fallback) %s: %s", disc_type.name, url)
            except Exception as e:
                errors.append(f"exchange:{url}: {e}")
                logger.warning("Failed to parse exchange disclosure %s: %s", url, e)

        # Print summary
        counts = {
            "exercise": len(exchange_overhang_exercises),
            "price_adj": len(exchange_overhang_price_adjs),
            "backlog": len(exchange_contracts_raw),
        }
        for cat, cnt in counts.items():
            if cnt:
                console.print(f"  [green]Exchange {cat}[/green]: {cnt} disclosures")
        for dt, items in parsed_disclosures.items():
            if items:
                console.print(f"  [green]{dt.name}[/green]: {len(items)} total disclosures")

    # ─── Phase 2: Fetch financial data from APIs ───
    market_data = None
    balance_sheet = None
    prev_balance_sheet = None
    annual_statements = []
    quarterly_statements = []
    quarterly_with_extra = []
    dart_fetcher = None
    corp_code = company.corp_code

    if settings.dart_api_key:
        console.print("\n[dim]Fetching OpenDART financial data...[/dim]")
        try:
            fs_div = "CFS" if report_cfg.statement_type == "연결" else "OFS"
            dart_fetcher = OpenDartFetcher(settings.dart_api_key, fs_div=fs_div)
            corp_code = company.corp_code
            if not corp_code:
                corp_code = dart_fetcher.resolve_corp_code(company.ticker)

            if corp_code:
                financials = dart_fetcher.get_multi_year_financials(
                    corp_code,
                    years=report_cfg.years,
                )
                annual_statements = list(reversed(financials.get("annual", [])))
                # Keep extra quarters for YoY calculation (2023 quarters
                # needed for 2024 YoY); trim to display count later.
                all_quarterly = list(reversed(financials.get("quarterly", [])))
                quarterly_with_extra = all_quarterly[:report_cfg.quarters + 4]

                # Integrate performance disclosures (매출액또는손익구조 변동)
                # BEFORE gap-filling so annual data is available for Q4 computation.
                _integrate_performance_disclosures(
                    parsed_disclosures, annual_statements, quarterly_with_extra,
                )

                # ── Report HTML fallback for missing API data ──
                report_fetcher = None
                has_gaps = any(
                    _is_empty_is(s) for s in annual_statements + quarterly_with_extra
                )
                if has_gaps:
                    console.print("  [dim]Filling gaps via report HTML parsing...[/dim]")
                    # Use the actual statement type that produced API data
                    # (avoids mismatch when CFS is empty and API fell back to OFS)
                    fs_pref = dart_fetcher.effective_fs_pref
                    report_fetcher = DartReportFetcher(
                        settings.dart_api_key, fs_pref=fs_pref,
                    )
                    annual_statements, quarterly_with_extra = _fill_gaps_with_report(
                        annual_statements,
                        quarterly_with_extra,
                        report_fetcher,
                        corp_code,
                    )

                # Filter out empty statements (years/quarters with no API data)
                annual_statements = [
                    s for s in annual_statements
                    if not _is_empty_is(s)
                ]
                quarterly_with_extra = [
                    s for s in quarterly_with_extra
                    if not _is_empty_is(s)
                ]
                # Trim to display count (extra quarters were for YoY only)
                quarterly_statements = quarterly_with_extra[:report_cfg.quarters]
                # Only display quarters from the latest 2 calendar years
                # (older quarters were kept only for YoY calculation)
                if quarterly_statements:
                    qtr_years = [
                        int(s.period.split(".Q")[0])
                        for s in quarterly_statements if ".Q" in s.period
                    ]
                    if qtr_years:
                        max_qtr_year = max(qtr_years)
                        quarterly_statements = [
                            s for s in quarterly_statements
                            if ".Q" in s.period
                            and int(s.period.split(".Q")[0]) >= max_qtr_year - 1
                        ]

                # Balance sheet: try latest quarterly, fall back to annual
                balance_sheet, prev_balance_sheet = dart_fetcher.get_latest_balance_sheet(corp_code)

                # Balance sheet report HTML fallback
                if balance_sheet.total_assets is None:
                    console.print("  [dim]BS fallback via report HTML...[/dim]")
                    current_year = datetime.now().year
                    if report_fetcher is None:
                        fs_pref = dart_fetcher.effective_fs_pref
                        report_fetcher = DartReportFetcher(
                            settings.dart_api_key, fs_pref=fs_pref,
                        )
                    # Try quarterly HTML fallback too
                    for q in (3, 2, 1):
                        balance_sheet = report_fetcher.get_balance_sheet(
                            corp_code, current_year - 1, quarter=q,
                        )
                        if balance_sheet.total_assets is not None:
                            break
                    if balance_sheet.total_assets is None:
                        balance_sheet = report_fetcher.get_balance_sheet(
                            corp_code, current_year - 1,
                        )
                    if balance_sheet.total_assets is None:
                        balance_sheet = report_fetcher.get_balance_sheet(
                            corp_code, current_year - 2,
                        )
                    if balance_sheet.total_assets is not None and prev_balance_sheet is None:
                        prev_balance_sheet = report_fetcher.get_balance_sheet(
                            corp_code, current_year - 2,
                        )
                        if prev_balance_sheet.total_assets is None:
                            prev_balance_sheet = None

                console.print(f"  [green]Annual data[/green]: {len(annual_statements)} years")
                console.print(f"  [green]Quarterly data[/green]: {len(quarterly_statements)} quarters")
        except Exception as e:
            logger.warning("OpenDART fetch failed: %s", e)
            console.print(f"  [yellow]OpenDART failed: {e}[/yellow]")
    else:
        console.print("\n[yellow]No DART_API_KEY set - skipping financial data fetch[/yellow]")

    # Market data
    console.print("[dim]Fetching market data...[/dim]")
    market_fetcher = None
    try:
        market_fetcher = MarketDataFetcher()
        market_data = market_fetcher.get_market_data(company.ticker)
        if market_data.stock_price:
            console.print(f"  [green]Stock price[/green]: {market_data.stock_price:,}원")
    except Exception as e:
        logger.warning("Market data fetch failed: %s", e)
        console.print(f"  [yellow]Market data failed: {e}[/yellow]")

    # Fetch 1-year price history
    price_history = None
    try:
        if market_fetcher:
            price_history = market_fetcher.get_price_history(company.ticker)
            if price_history:
                console.print(
                    f"  [green]Price history[/green]: {price_history.period_start} ~ "
                    f"{price_history.period_end} ({price_history.return_1y:+.1f}%)"
                )
    except Exception as e:
        logger.warning("Price history fetch failed: %s", e)

    # ─── Phase 2.5: Fetch business sections from report ───
    business_summary = None
    if settings.dart_api_key:
        console.print("\n[dim]Fetching business section from report...[/dim]")
        try:
            biz_fetcher = DartBusinessFetcher(settings.dart_api_key)
            corp_id = company.corp_code or company.ticker
            biz_sections = biz_fetcher.fetch_business_sections(corp_id)
            if biz_sections and settings.openai_api_key:
                console.print("  [dim]Summarizing with LLM...[/dim]")
                from auto_reports.summarizers.openai_summarizer import (
                    summarize_business_sections,
                )

                business_summary = summarize_business_sections(
                    사업개요=biz_sections.사업개요,
                    주요제품=biz_sections.주요제품,
                    원재료=biz_sections.원재료,
                    매출수주=biz_sections.매출수주,
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                )
                business_summary.report_source = f"{biz_sections.report_title} 기준"
                console.print("  [green]Business section summarized[/green]")
            elif biz_sections and not settings.openai_api_key:
                console.print("  [yellow]No OPENAI_API_KEY - skipping summarization[/yellow]")
        except Exception as e:
            logger.warning("Business section fetch/summarize failed: %s", e)
            console.print(f"  [yellow]Business section failed: {e}[/yellow]")

    # ─── Phase 3: Analyze overhang (3 sub-phases) ───
    console.print("\n[dim]Analyzing overhang...[/dim]")

    total_shares = market_data.shares_outstanding if market_data and market_data.shares_outstanding else 0
    overhang_analyzer = OverhangAnalyzer(total_shares=total_shares)
    reference_date = ""  # 기준일 (period end date of latest periodic report)

    # ─── Phase 3a: Baseline from 정기보고서 재무제표 주석 ───
    if dart_fetcher and corp_code:
        console.print("  [dim]Phase 3a: Fetching overhang from 재무제표 주석...[/dim]")
        try:
            notes_instruments, reference_date = dart_fetcher.get_overhang_from_notes(corp_code)
            for inst in notes_instruments:
                overhang_analyzer.process_notes_instrument(inst)
            if notes_instruments:
                console.print(f"  [green]Notes overhang[/green]: {len(notes_instruments)} active instruments")
            if reference_date:
                fmt_ref = f"{reference_date[:4]}.{reference_date[4:6]}.{reference_date[6:8]}"
                console.print(f"  [green]기준일[/green]: {fmt_ref}")
        except Exception as e:
            logger.warning("Overhang notes fetch failed: %s", e)

    if not reference_date and dart_fetcher and corp_code:
        console.print("  [yellow]기준일 could not be determined — Phase 3b/3c skipped[/yellow]")
        logger.warning("기준일 could not be determined — Phase 3b/3c will be skipped or use all data")

    # ─── Phase 3b: Post-기준일 주요사항보고서 (CB/BW/유상증자) ───
    # Step 3a: OpenDART DS005 API first (cvbdIsDecsn, bdwtIsDecsn, piicDecsn)
    new_issuances: list = []
    if dart_fetcher and corp_code and reference_date:
        console.print(f"  [dim]Phase 3b: Fetching post-기준일 주요사항보고서 (기준일={reference_date})...[/dim]")
        try:
            new_issuances = dart_fetcher.get_overhang_issuances(
                corp_code, after_date=reference_date,
            )
            for event in new_issuances:
                cat = event.get("category", "?")
                shares = event.get("shares") or event.get("convertible_shares") or 0
                console.print(f"    [dim]→ {cat}: {shares:,} shares (rcept={event.get('rcept_no', '?')[:14]})[/dim]")
                overhang_analyzer.process_event(event)
            if new_issuances:
                console.print(
                    f"  [green]Post-기준일 issuances[/green]: {len(new_issuances)} "
                    f"(CB/BW/유상증자/주식매수선택권)"
                )
            else:
                console.print("  [yellow]Phase 3b: No post-기준일 issuances found from DS005 APIs[/yellow]")
        except Exception as e:
            console.print(f"  [red]Phase 3b DS005 API failed: {e}[/red]")
            logger.warning("Post-기준일 issuance fetch failed: %s", e)

    # Step 3b enrichment: LLM extraction from local 주요사항보고서 PDFs
    # Always runs for RIGHTS_ISSUE (piicDecsn API lacks conversion terms);
    # for CB/BW/SO, only runs when the API returned no data (fallback).
    if settings.openai_api_key and stock_dir:
        console.print("  [dim]Phase 3b: LLM enrichment from local 주요사항보고서 PDFs...[/dim]")
        _phase3b_llm_pdf_fallback(
            stock_dir, reference_date or "", overhang_analyzer,
            settings.openai_api_key, settings.openai_model, console,
            skip_categories={"CB", "BW", "STOCK_OPTION"} if new_issuances else set(),
        )

    # ─── Phase 3c: Post-기준일 거래소 공시 updates ───
    # Filter exchange disclosures to only include those after 기준일
    post_ref_exercises: list[dict] = []
    post_ref_price_adjs: list[dict] = []

    if reference_date:
        for ex_data in exchange_overhang_exercises:
            entry_date = ex_data.get("_entry_date", "").replace("-", "").replace(".", "")
            if not entry_date:
                logger.warning("Exchange exercise has no _entry_date, skipping: %s", ex_data.get("title", "unknown"))
                continue
            if entry_date > reference_date:
                post_ref_exercises.append(ex_data)
        for pa_data in exchange_overhang_price_adjs:
            entry_date = pa_data.get("_entry_date", "").replace("-", "").replace(".", "")
            if not entry_date:
                logger.warning("Exchange price_adj has no _entry_date, skipping: %s", pa_data.get("title", "unknown"))
                continue
            if entry_date > reference_date:
                post_ref_price_adjs.append(pa_data)
    else:
        # No reference date — use all exchange data (backward compat)
        post_ref_exercises = exchange_overhang_exercises
        post_ref_price_adjs = exchange_overhang_price_adjs

    if post_ref_exercises or post_ref_price_adjs:
        console.print("  [dim]Phase 3c: Applying post-기준일 거래소 공시...[/dim]")

    for ex_data in post_ref_exercises:
        overhang_analyzer.process_exchange_exercise(ex_data)
    for pa_data in post_ref_price_adjs:
        overhang_analyzer.process_exchange_price_adj(pa_data)

    if post_ref_exercises or post_ref_price_adjs:
        console.print(
            f"  [green]Exchange overhang[/green]: "
            f"{len(post_ref_exercises)} exercises, "
            f"{len(post_ref_price_adjs)} price adjustments"
        )

    overhang_items = overhang_analyzer.get_overhang_items()
    console.print(f"  [green]Overhang instruments[/green]: {len(overhang_items)}")

    # ─── Phase 3.1: Fetch supplementary sections (주주, 배당, 종속회사) ───
    shareholder_info = ""
    dividend_info = ""
    subsidiary_info = ""
    if settings.dart_api_key and settings.openai_api_key and corp_code:
        console.print("\n[dim]Fetching supplementary sections (주주/배당/종속회사)...[/dim]")
        try:
            from auto_reports.fetchers.dart_supplementary import DartSupplementaryFetcher
            from auto_reports.summarizers.supplementary_summarizer import (
                extract_dividend_info,
                extract_shareholder_info,
                extract_subsidiary_info,
            )

            supp_fetcher = DartSupplementaryFetcher(settings.dart_api_key)
            supp_sections = supp_fetcher.fetch_supplementary_sections(corp_code)

            llm_model = settings.openai_model

            if supp_sections["shareholder"]:
                shareholder_info = extract_shareholder_info(
                    supp_sections["shareholder"], settings.openai_api_key, llm_model,
                )
                if shareholder_info:
                    console.print(f"  [green]주주 구성[/green]: {shareholder_info}")

            if supp_sections["dividend"]:
                dividend_info = extract_dividend_info(
                    supp_sections["dividend"], settings.openai_api_key, llm_model,
                )
                if dividend_info:
                    console.print(f"  [green]배당[/green]: {dividend_info}")

            if supp_sections["subsidiary"]:
                subsidiary_info = extract_subsidiary_info(
                    supp_sections["subsidiary"], settings.openai_api_key, llm_model,
                )
                if subsidiary_info:
                    console.print(f"  [green]종속 회사[/green]: {subsidiary_info}")
                else:
                    subsidiary_info = "해당없음"
            else:
                subsidiary_info = "해당없음"

            if supp_sections["report_name"]:
                console.print(f"  [dim]소스: {supp_sections['report_name']}[/dim]")

        except Exception as e:
            logger.warning("Supplementary section fetch failed: %s", e)
            console.print(f"  [yellow]Supplementary fetch failed: {e}[/yellow]")

    if not subsidiary_info:
        subsidiary_info = "해당없음"

    # Build financial tables
    annual_rows = build_annual_rows(annual_statements)
    cumulative_row = build_cumulative_annual_row(quarterly_with_extra)
    if cumulative_row:
        annual_rows.insert(0, cumulative_row)
    quarterly_rows = build_quarterly_rows(quarterly_statements)
    bs_rows = build_balance_sheet_rows(balance_sheet, prev_balance_sheet) if balance_sheet else []

    # ─── Compute display strings (needed for Phase 3.5 and Phase 4) ───
    market_cap_str = ""
    if market_data and market_data.stock_price and market_data.shares_outstanding:
        mc_eok = round((market_data.stock_price * market_data.shares_outstanding) / 1_0000_0000)
        market_cap_str = (
            f"{mc_eok:,} 억원 (주가 {market_data.stock_price:,}원 "
            f"× 발행주식수 {market_data.shares_outstanding:,}주, "
            f"{market_data.date} 기준)"
        )

    # Compute Fully-Diluted market cap
    fully_diluted_market_cap_str = ""
    if market_data and market_data.stock_price and market_data.shares_outstanding:
        dilutive_shares = overhang_analyzer.get_total_dilutive_shares()
        fd_total_shares = market_data.shares_outstanding + dilutive_shares
        fd_market_cap = fd_total_shares * market_data.stock_price
        fd_mc_eok = round(fd_market_cap / 1_0000_0000)
        fully_diluted_market_cap_str = (
            f"{fd_mc_eok:,}억원 (희석주식 포함 총 {fd_total_shares:,}주 기준)"
        )

    # Compute PBR = market_cap / equity (from balance sheet)
    pbr_str = ""
    market_cap_won = 0
    if market_data and market_data.stock_price and market_data.shares_outstanding:
        market_cap_won = market_data.stock_price * market_data.shares_outstanding
    if market_cap_won > 0 and balance_sheet and balance_sheet.total_equity and balance_sheet.total_equity > 0:
        pbr = market_cap_won / balance_sheet.total_equity
        mc_eok_val = round(market_cap_won / 1_0000_0000)
        eq_eok_val = round(balance_sheet.total_equity / 1_0000_0000)
        pbr_str = f"{pbr:.2f}배 (시가총액 {mc_eok_val:,} 억원 / 자본총계 {eq_eok_val:,} 억원)"

    # Compute trailing PER = market_cap / most recent full-year net income
    trailing_per_str = ""
    if market_cap_won > 0 and annual_statements:
        for stmt in annual_statements:
            if stmt.net_income is not None:
                mc_eok_val = round(market_cap_won / 1_0000_0000)
                if stmt.net_income > 0:
                    per = market_cap_won / stmt.net_income
                    ni_eok_val = round(stmt.net_income / 1_0000_0000)
                    trailing_per_str = (
                        f"{per:.1f}배 (시가총액 {mc_eok_val:,} 억원 / "
                        f"{stmt.period}년 순이익 {ni_eok_val:,} 억원)"
                    )
                else:
                    trailing_per_str = f"해당없음 ({stmt.period}년 순손실)"
                break

    # Estimated PER from research reports (if available)
    per_str = ""
    analysis_cfg = company_config.analysis
    if settings.openai_api_key and analysis_cfg.reports_dir and market_cap_won > 0:
        reports_dir = Path(analysis_cfg.reports_dir)
        if not reports_dir.is_absolute():
            reports_dir = project_root / reports_dir
        if reports_dir.is_dir():
            console.print("[dim]Extracting earnings estimates from research reports...[/dim]")
            try:
                from auto_reports.summarizers.analysis_summarizer import extract_estimated_earnings

                target_year = datetime.now().year
                estimated_ni = extract_estimated_earnings(
                    reports_dir=reports_dir,
                    company_name=company.name,
                    target_year=target_year,
                    api_key=settings.openai_api_key,
                    model=analysis_cfg.analysis_model or settings.openai_model,
                )
                if estimated_ni and estimated_ni > 0:
                    est_per = market_cap_won / estimated_ni
                    mc_eok_val = round(market_cap_won / 1_0000_0000)
                    ni_eok_val = round(estimated_ni / 1_0000_0000)
                    per_str = (
                        f"{est_per:.1f}배 (시가총액 {mc_eok_val:,} 억원 / "
                        f"{target_year}E 순이익 {ni_eok_val:,} 억원)"
                    )
                    console.print(f"  [green]Estimated PER[/green]: {per_str}")
                else:
                    console.print("  [dim]No earnings estimates found in reports[/dim]")
            except Exception as e:
                logger.warning("Earnings estimate extraction failed: %s", e)
                console.print(f"  [yellow]Estimate extraction failed: {e}[/yellow]")

    bs_period = ""
    bs_prev_period = ""
    if balance_sheet and balance_sheet.period:
        bs_period = f"{balance_sheet.period} {balance_sheet.statement_type} 기준"
    if prev_balance_sheet and prev_balance_sheet.period:
        bs_prev_period = prev_balance_sheet.period

    # ─── Phase 3.5: LLM analysis for sections 5-7 ───
    analysis_result = None
    sections_1_to_4 = ""
    if settings.openai_api_key and analysis_cfg.prompt_file:
        prompt_path = project_root / analysis_cfg.prompt_file
        if prompt_path.exists():
            console.print("\n[dim]Running LLM analysis for sections 5-7...[/dim]")
            try:
                from auto_reports.summarizers.analysis_summarizer import generate_analysis

                # Build a temporary report with sections 1-4 for context
                temp_data = ReportData(
                    company_name=company.name,
                    market_cap_str=market_cap_str or "데이터 없음",
                    latest_pbr_str=pbr_str or "데이터 없음",
                    trailing_per_str=trailing_per_str or "데이터 없음",
                    estimated_per_str=per_str or "데이터 없음",
                    overhang_items=overhang_items,
                    total_shares=total_shares,
                    balance_sheet_period=bs_period or "데이터 없음",
                    balance_sheet_prev_period=bs_prev_period,
                    balance_sheet_rows=bs_rows,
                    annual_rows=annual_rows,
                    quarterly_rows=quarterly_rows,
                    business_model=business_summary.business_model if business_summary else "",
                    major_customers=business_summary.major_customers if business_summary else "",
                    major_suppliers=business_summary.major_suppliers if business_summary else "",
                    revenue_breakdown=business_summary.revenue_breakdown if business_summary else "",
                    order_backlog=business_summary.order_backlog if business_summary else "",
                    business_source=business_summary.report_source if business_summary else "",
                )
                sections_1_to_4 = generate_report(temp_data)

                prompt_template = prompt_path.read_text(encoding="utf-8")
                reports_dir = Path(analysis_cfg.reports_dir) if analysis_cfg.reports_dir else None
                news_file = Path(analysis_cfg.news_file) if analysis_cfg.news_file else None
                # Resolve relative paths against project root
                if reports_dir and not reports_dir.is_absolute():
                    reports_dir = project_root / reports_dir
                if news_file and not news_file.is_absolute():
                    news_file = project_root / news_file

                analysis_model = analysis_cfg.analysis_model or settings.openai_model
                analysis_result = generate_analysis(
                    company_name=company.name,
                    report_sections_1_to_4=sections_1_to_4,
                    prompt_template=prompt_template,
                    api_key=settings.openai_api_key,
                    model=analysis_model,
                    reports_dir=reports_dir,
                    news_file=news_file,
                )
                console.print("  [green]Analysis sections generated[/green]")
            except Exception as e:
                logger.warning("LLM analysis failed: %s", e)
                console.print(f"  [yellow]Analysis failed: {e}[/yellow]")
        else:
            console.print(f"  [yellow]Prompt file not found: {prompt_path}[/yellow]")

    # ─── Phase 3.7: Feedback loop - supplement Value Driver & 경쟁사 비교 ───
    if (
        settings.openai_api_key
        and analysis_result
        and (not analysis_result.value_driver or not analysis_result.competitor_comparison)
    ):
        console.print("[dim]Supplementing Value Driver & 경쟁사 비교...[/dim]")
        try:
            from auto_reports.summarizers.analysis_summarizer import (
                supplement_value_driver_and_competitors,
            )

            reports_dir_fb = Path(analysis_cfg.reports_dir) if analysis_cfg.reports_dir else None
            news_file_fb = Path(analysis_cfg.news_file) if analysis_cfg.news_file else None
            if reports_dir_fb and not reports_dir_fb.is_absolute():
                reports_dir_fb = project_root / reports_dir_fb
            if news_file_fb and not news_file_fb.is_absolute():
                news_file_fb = project_root / news_file_fb

            sup_model = analysis_cfg.analysis_model or settings.openai_model
            vd_sup, cc_sup = supplement_value_driver_and_competitors(
                company_name=company.name,
                report_sections_1_to_4=sections_1_to_4,
                api_key=settings.openai_api_key,
                model=sup_model,
                reports_dir=reports_dir_fb,
                news_file=news_file_fb,
            )
            if not analysis_result.value_driver and vd_sup:
                analysis_result.value_driver = vd_sup
                console.print("  [green]Value Driver supplemented[/green]")
            if not analysis_result.competitor_comparison and cc_sup:
                analysis_result.competitor_comparison = cc_sup
                console.print("  [green]경쟁사 비교 supplemented[/green]")
        except Exception as e:
            logger.warning("Value Driver/경쟁사 비교 supplement failed: %s", e)
            console.print(f"  [yellow]Supplement failed: {e}[/yellow]")

    # ─── Phase 3.8: Feedback loop - generate momentum text ───
    momentum_text = ""
    if settings.openai_api_key and price_history:
        console.print("[dim]Generating momentum text...[/dim]")
        try:
            from auto_reports.summarizers.analysis_summarizer import generate_momentum_text

            # Reuse sections_1_to_4 from Phase 3.5
            sections_ctx = sections_1_to_4

            reports_dir_m = Path(analysis_cfg.reports_dir) if analysis_cfg.reports_dir else None
            news_file_m = Path(analysis_cfg.news_file) if analysis_cfg.news_file else None
            if reports_dir_m and not reports_dir_m.is_absolute():
                reports_dir_m = project_root / reports_dir_m
            if news_file_m and not news_file_m.is_absolute():
                news_file_m = project_root / news_file_m

            momentum_text = generate_momentum_text(
                company_name=company.name,
                report_sections_1_to_4=sections_ctx,
                api_key=settings.openai_api_key,
                model=analysis_cfg.analysis_model or settings.openai_model,
                reports_dir=reports_dir_m,
                news_file=news_file_m,
            )
            if momentum_text:
                console.print("  [green]Momentum text generated[/green]")
        except Exception as e:
            logger.warning("Momentum generation failed: %s", e)

    # ─── Phase 3.9: Generate stock price chart ───
    chart_image = ""
    console.print("[dim]Generating stock chart...[/dim]")
    try:
        from auto_reports.generators.chart import generate_stock_chart

        chart_out_dir = Path(settings.obsidian_attachments) if settings.obsidian_attachments else Path(output_dir or report_cfg.output_dir or "output")
        chart_filename = generate_stock_chart(
            ticker=company.ticker,
            output_dir=chart_out_dir,
        )
        if chart_filename:
            chart_image = chart_filename
            console.print(f"  [green]Chart generated[/green]: {chart_filename}")
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)
        console.print(f"  [yellow]Chart failed: {e}[/yellow]")

    # ─── Phase 3.95: Build exchange contract list for report ───
    # Filter exchange contracts to only include post-기준일 entries
    if reference_date:
        filtered_contracts_raw = []
        for raw in exchange_contracts_raw:
            entry_date = raw.get("_entry_date", "").replace("-", "").replace(".", "")
            if not entry_date:
                logger.warning("Exchange backlog has no _entry_date, skipping")
                continue
            if entry_date > reference_date:
                filtered_contracts_raw.append(raw)
        logger.info(
            "Exchange contracts: %d total, %d after 기준일 (%s)",
            len(exchange_contracts_raw), len(filtered_contracts_raw), reference_date,
        )
    else:
        filtered_contracts_raw = exchange_contracts_raw

    # Unit-to-원 multiplier for normalizing amounts to 억원
    _UNIT_MULTIPLIER = {
        "원": 1,
        "천원": 1_000,
        "백만원": 1_000_000,
        "억원": 1_0000_0000,
    }

    exchange_contracts: list[ExchangeContract] = []
    for raw in filtered_contracts_raw:
        is_cancel = raw.get("type", "") == "단일판매ㆍ공급계약해지"
        amount_raw = raw.get("cancel_amount") if is_cancel else raw.get("contract_amount")
        amount_eok = ""
        if amount_raw and isinstance(amount_raw, (int, float)) and amount_raw > 0:
            # Detect unit from LLM/parser output and convert to 원 first
            unit = raw.get("amount_unit", "원")
            multiplier = _UNIT_MULTIPLIER.get(unit, 1)
            amount_in_won = amount_raw * multiplier
            amount_eok = f"{round(amount_in_won / 1_0000_0000):,}"
        rev_ratio = raw.get("revenue_ratio_pct")
        rev_str = f"{rev_ratio:.1f}%" if rev_ratio else None

        # Build contract period string
        # Deterministic parser stores period as {"start": ..., "end": ...}
        # LLM stores as contract_period_start / contract_period_end
        period_dict = raw.get("period") or {}
        period_start = raw.get("contract_period_start") or period_dict.get("start") or ""
        period_end = raw.get("contract_period_end") or period_dict.get("end") or ""
        contract_period = ""
        if period_start and period_end:
            contract_period = f"{period_start}~{period_end}"
        elif period_start:
            contract_period = f"{period_start}~"
        elif period_end:
            contract_period = f"~{period_end}"

        exchange_contracts.append(ExchangeContract(
            contract_type="해지" if is_cancel else "체결",
            description=raw.get("description") or "",
            detail=raw.get("detail") or "",
            amount_eok=amount_eok,
            counterparty=raw.get("counterparty") or "",
            revenue_ratio_pct=rev_str,
            date=raw.get("cancel_date") if is_cancel else raw.get("contract_date") or "",
            contract_period=contract_period,
        ))

    # ─── Phase 4: Generate report ───
    console.print("\n[dim]Generating report...[/dim]")

    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_data = ReportData(
        frontmatter=ReportFrontmatter(
            created=today,
            updated="",
            tags=company.tags,
        ),
        company_name=company.name,
        market_cap_str=market_cap_str or "데이터 없음",
        latest_pbr_str=pbr_str or "데이터 없음",
        trailing_per_str=trailing_per_str or "데이터 없음",
        estimated_per_str=per_str or "데이터 없음",
        overhang_items=overhang_items,
        total_shares=total_shares,
        fully_diluted_market_cap_str=fully_diluted_market_cap_str,
        shareholder_info=shareholder_info,
        dividend_info=dividend_info,
        subsidiary_info=subsidiary_info,
        # Chart
        chart_image=chart_image,
        # Price history
        price_period=(
            f"{price_history.period_start} ~ {price_history.period_end}"
            if price_history else ""
        ),
        price_start=f"{price_history.start_price:,}원" if price_history else "",
        price_current=f"{price_history.current_price:,}원" if price_history else "",
        price_low=(
            f"{price_history.low_price:,}원 ({price_history.low_date})"
            if price_history else ""
        ),
        price_high=(
            f"{price_history.high_price:,}원 ({price_history.high_date})"
            if price_history else ""
        ),
        price_return_1y=(
            f"{price_history.return_1y:+.1f}%"
            if price_history else ""
        ),
        price_return_label=(
            price_history.return_label
            if price_history else "1년수익률"
        ),
        momentum_text=momentum_text.replace("\n", " ").replace("\r", "").strip(),
        balance_sheet_period=bs_period or "데이터 없음",
        balance_sheet_prev_period=bs_prev_period,
        balance_sheet_rows=bs_rows,
        annual_rows=annual_rows,
        quarterly_rows=quarterly_rows,
        business_model=business_summary.business_model if business_summary else "",
        major_customers=business_summary.major_customers if business_summary else "",
        major_suppliers=business_summary.major_suppliers if business_summary else "",
        revenue_breakdown=business_summary.revenue_breakdown if business_summary else "",
        order_backlog=business_summary.order_backlog if business_summary else "",
        exchange_contracts=exchange_contracts,
        business_source=business_summary.report_source if business_summary else "",
        # Sections 4 additions + 5-7 from analysis
        value_driver=analysis_result.value_driver if analysis_result else "",
        competitor_comparison=analysis_result.competitor_comparison if analysis_result else "",
        investment_ideas=analysis_result.investment_ideas if analysis_result else "",
        risk_structural=analysis_result.risk_structural if analysis_result else "",
        risk_counter=analysis_result.risk_counter if analysis_result else "",
        risk_tail=analysis_result.risk_tail if analysis_result else "",
        conclusion=analysis_result.conclusion if analysis_result else "",
    )

    # ─── Phase 4.5: Generate tags via LLM ───
    if settings.openai_api_key:
        console.print("[dim]Generating tags...[/dim]")
        llm_tags = generate_report_tags(
            company_name=company.name,
            business_model=report_data.business_model,
            revenue_breakdown=report_data.revenue_breakdown,
            investment_ideas=report_data.investment_ideas,
            conclusion=report_data.conclusion,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        if llm_tags:
            # Merge: keep existing manual tags, append new LLM tags (deduplicated)
            existing = set(report_data.frontmatter.tags)
            merged = list(report_data.frontmatter.tags)
            for tag in llm_tags:
                if tag not in existing:
                    merged.append(tag)
                    existing.add(tag)
            report_data.frontmatter.tags = merged
            console.print(f"  [dim]Tags: {', '.join(merged)}[/dim]")

    if dry_run:
        console.print("[yellow]Dry run - not writing report[/yellow]")
        md = generate_report(report_data)
        console.print(md[:500] + "...")
        return None

    # Write report
    out_dir = Path(output_dir or report_cfg.output_dir or "output")
    safe_name = re.sub(r'[/\\:\0]', '_', company.name).replace('..', '_') or "unnamed"
    output_path = out_dir / f"{safe_name}.md"
    result_path = write_report(report_data, output_path)

    console.print(f"\n[bold green]Report generated:[/bold green] {result_path}")

    if errors:
        console.print(f"\n[yellow]Warning: {len(errors)} disclosure URLs failed to parse.[/yellow]")

    return result_path
