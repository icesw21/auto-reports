"""Financial ratio calculations and formatting for report tables."""

from __future__ import annotations

from typing import Optional

from auto_reports.models.financial import BalanceSheet, ConsensusItem, IncomeStatementItem
from auto_reports.models.report import AnnualRow, BalanceSheetRow, QuarterlyRow


def currency_unit(currency: str = "KRW") -> tuple[int, str]:
    """Return (divisor, unit_label) for the given currency.

    KRW → (100_000_000, "억원")  — 억원 단위
    Others → (1_000_000, "백만 {currency}") — 백만 단위
    """
    if currency == "KRW":
        return 100_000_000, "억원"
    return 1_000_000, f"백만 {currency}"


def to_eok(won: int | None, currency: str = "KRW") -> int | None:
    """Convert amount to display unit (억원 for KRW, 백만 for others)."""
    if won is None:
        return None
    divisor, _ = currency_unit(currency)
    return round(won / divisor)


def format_eok(won: Optional[int], currency: str = "KRW") -> str:
    """Format amount as display-unit string with comma separators."""
    if won is None:
        return "-"
    eok = to_eok(won, currency)
    if eok is None:
        return "-"
    return f"{eok:,}"


def calc_yoy_change(current: Optional[int], previous: Optional[int]) -> str:
    """Calculate YoY change and format as string.

    Returns formatted strings like: +276%, -16%, 흑자전환, 적자전환, 적자지속, 적자축소, 적자확대
    """
    if current is None or previous is None:
        return "-"

    # Handle sign transitions
    curr_positive = current > 0
    prev_positive = previous > 0
    curr_negative = current < 0
    prev_negative = previous < 0

    if curr_positive and prev_negative:
        return "흑자전환"
    if curr_negative and prev_positive:
        return "적자전환"
    if curr_negative and prev_negative:
        if abs(current) < abs(previous):
            return "적자축소"
        elif abs(current) > abs(previous):
            return "적자확대"
        else:
            return "적자지속"

    # Both positive (or zero)
    if previous == 0:
        if current > 0:
            return "흑자전환"
        elif current < 0:
            return "적자전환"
        return "-"

    change_pct = ((current - previous) / abs(previous)) * 100
    sign = "+" if change_pct >= 0 else ""
    return f"{sign}{change_pct:.0f}%"


def format_income_cell(won: Optional[int], yoy: str, currency: str = "KRW") -> str:
    """Format an income statement cell like '4,245 (+276%)'."""
    if won is None:
        return "-"
    eok = to_eok(won, currency)
    if eok is None:
        return "-"
    return f"{eok:,} ({yoy})"


def build_balance_sheet_rows(
    bs: BalanceSheet,
    prev_bs: Optional[BalanceSheet] = None,
) -> list[BalanceSheetRow]:
    """Build balance sheet table rows from BalanceSheet models."""
    rows = []
    cur = bs.currency

    def _row(label: str, current: Optional[int], previous: Optional[int], prefix: str = "", note_extra: str = "") -> BalanceSheetRow:
        amount_str = format_eok(current, cur)
        prev_str = format_eok(previous, cur)
        note = ""
        if previous is not None and current is not None and previous != 0:
            change_pct = ((current - previous) / abs(previous)) * 100
            note = f"{'+' if change_pct >= 0 else ''}{change_pct:.0f}%"
        if note_extra:
            note = f"{note} ({note_extra})" if note else note_extra
        item_label = f"({label})" if prefix == "sub" else f"**{label}**" if prefix == "bold" else label
        amount_display = f"**{amount_str}**" if prefix == "bold" else amount_str
        prev_display = f"**{prev_str}**" if prefix == "bold" else prev_str
        return BalanceSheetRow(item=item_label, amount=amount_display, previous_amount=prev_display, note=f"**{note}**" if prefix == "bold" and note else note)

    prev = prev_bs

    # 현금성자산 = 현금및현금성자산 + 단기금융상품
    def _sum_optional(*values: Optional[int]) -> Optional[int]:
        non_none = [v for v in values if v is not None]
        return sum(non_none) if non_none else None

    cash_total = _sum_optional(bs.cash_and_equivalents, bs.short_term_investments)
    prev_cash_total = _sum_optional(prev.cash_and_equivalents, prev.short_term_investments) if prev else None

    # 이자부부채: 유동/비유동 각각 통합 계정이 있으면 우선 사용
    # 통합: 단기차입금및사채 / 유동 차입금  |  장기차입금및사채
    # 개별: 단기차입금 + 유동성장기부채 + 유동성사채  |  장기차입금 + 사채
    # Fallback: borrowings_total (bare "차입금" 합산)
    def _calc_debt(sheet: BalanceSheet) -> Optional[int]:
        if sheet.short_term_debt_and_bonds is not None:
            # 통합 계정 + 별도로 잡힌 유동성장기차입금/유동성사채 합산
            short = _sum_optional(
                sheet.short_term_debt_and_bonds,
                sheet.current_long_term_debt,
                sheet.current_bonds,
            )
        else:
            short = _sum_optional(
                sheet.short_term_borrowings, sheet.current_long_term_debt,
                sheet.current_bonds,
            )
        if sheet.long_term_debt_and_bonds is not None:
            long = sheet.long_term_debt_and_bonds
        else:
            long = _sum_optional(sheet.long_term_borrowings, sheet.bonds)
        result = _sum_optional(short, long)
        # Fallback: use borrowings_total (bare "차입금" 합산) when
        # individual debt fields are all empty
        if result is None and sheet.borrowings_total is not None:
            result = sheet.borrowings_total
        return result

    debt_total = _calc_debt(bs)
    prev_debt_total = _calc_debt(prev) if prev else None

    rows.append(_row("자산총계", bs.total_assets, prev.total_assets if prev else None, "bold"))
    rows.append(_row("현금성자산", cash_total, prev_cash_total, "sub"))
    rows.append(_row("부채총계", bs.total_liabilities, prev.total_liabilities if prev else None, "bold"))
    rows.append(_row("이자부부채", debt_total, prev_debt_total, "sub"))
    rows.append(_row("자본총계", bs.total_equity, prev.total_equity if prev else None, "bold"))

    return rows


def build_annual_rows(
    statements: list[IncomeStatementItem],
) -> list[AnnualRow]:
    """Build annual income statement rows from multi-year data.

    Expects statements sorted newest-first.
    """
    rows = []
    cur = statements[0].currency if statements else "KRW"
    for i, stmt in enumerate(statements):
        # Find previous year for YoY
        prev = statements[i + 1] if i + 1 < len(statements) else None

        rev_yoy = calc_yoy_change(stmt.revenue, prev.revenue if prev else None)
        op_yoy = calc_yoy_change(stmt.operating_income, prev.operating_income if prev else None)
        ni_yoy = calc_yoy_change(stmt.net_income, prev.net_income if prev else None)

        bold = i == 0  # Bold the most recent year
        year_str = f"**{stmt.period}**" if bold else stmt.period

        rows.append(AnnualRow(
            year=year_str,
            revenue=_bold_wrap(format_income_cell(stmt.revenue, rev_yoy, cur), bold),
            operating_income=_bold_wrap(format_income_cell(stmt.operating_income, op_yoy, cur), bold),
            net_income=_bold_wrap(format_income_cell(stmt.net_income, ni_yoy, cur), bold),
        ))

    return rows


def build_cumulative_annual_row(
    quarterly_statements: list[IncomeStatementItem],
) -> AnnualRow | None:
    """Build an annual row from the latest cumulative quarterly data.

    Sums individual quarter figures (Q1..Qn) for the most recent year
    and computes YoY vs the same cumulative period of the prior year.

    Returns None if there is no quarterly data.
    """
    if not quarterly_statements:
        return None

    # Group quarters by year
    yearly: dict[int, dict[int, IncomeStatementItem]] = {}
    for stmt in quarterly_statements:
        if ".Q" not in stmt.period:
            continue
        parts = stmt.period.split(".Q")
        year, q = int(parts[0]), int(parts[1])
        yearly.setdefault(year, {})[q] = stmt

    if not yearly:
        return None

    # Find the latest year and its latest quarter
    latest_year = max(yearly)
    latest_q = max(yearly[latest_year])

    # Don't add cumulative row for Q4 (that's already a full year in annual)
    if latest_q == 4:
        return None

    # Sum Q1..Qn for latest year
    # For newly-listed companies, earlier quarters may have no data;
    # treat contiguous missing quarters from the start as zero.
    def _sum_field(items: dict[int, IncomeStatementItem], max_q: int, field: str) -> int | None:
        values = []
        found_data = False
        for q in range(1, max_q + 1):
            if q not in items:
                if found_data:
                    return None  # gap in the middle → unreliable
                continue
            val = getattr(items[q], field)
            if val is None:
                if found_data:
                    return None  # gap in the middle
                continue
            found_data = True
            values.append(val)
        return sum(values) if values else None

    curr_rev = _sum_field(yearly[latest_year], latest_q, "revenue")
    curr_op = _sum_field(yearly[latest_year], latest_q, "operating_income")
    curr_ni = _sum_field(yearly[latest_year], latest_q, "net_income")

    # Sum Q1..Qn for previous year (for YoY calculation only)
    prev_year = latest_year - 1
    prev_rev = _sum_field(yearly.get(prev_year, {}), latest_q, "revenue") if prev_year in yearly else None
    prev_op = _sum_field(yearly.get(prev_year, {}), latest_q, "operating_income") if prev_year in yearly else None
    prev_ni = _sum_field(yearly.get(prev_year, {}), latest_q, "net_income") if prev_year in yearly else None

    rev_yoy = calc_yoy_change(curr_rev, prev_rev)
    op_yoy = calc_yoy_change(curr_op, prev_op)
    ni_yoy = calc_yoy_change(curr_ni, prev_ni)

    period_label = f"{latest_year}.{latest_q}Q"

    cur = quarterly_statements[0].currency if quarterly_statements else "KRW"
    return AnnualRow(
        year=f"**{period_label}**",
        revenue=_bold_wrap(format_income_cell(curr_rev, rev_yoy, cur), True),
        operating_income=_bold_wrap(format_income_cell(curr_op, op_yoy, cur), True),
        net_income=_bold_wrap(format_income_cell(curr_ni, ni_yoy, cur), True),
    )


def build_consensus_rows(
    consensus_items: list[ConsensusItem],
    annual_statements: list[IncomeStatementItem],
) -> list[AnnualRow]:
    """Build consensus estimate rows with YoY vs previous year.

    Expects consensus_items sorted oldest-first (e.g. 2025E, 2026E, 2027E).
    Uses the most recent actual year as the initial comparison base, then
    chains year-over-year through successive consensus years.

    Returns rows sorted newest-first (e.g. 2027E, 2026E, 2025E) for display.
    """
    if not consensus_items:
        return []

    # Most recent actual year (annual_statements sorted newest-first)
    prev_stmt = annual_statements[0] if annual_statements else None
    cur = prev_stmt.currency if prev_stmt else "KRW"

    rows = []
    for item in consensus_items:
        rev_yoy = calc_yoy_change(item.revenue, prev_stmt.revenue if prev_stmt else None)
        op_yoy = calc_yoy_change(item.operating_income, prev_stmt.operating_income if prev_stmt else None)
        ni_yoy = calc_yoy_change(item.net_income, prev_stmt.net_income if prev_stmt else None)

        rows.append(AnnualRow(
            year=item.period,
            revenue=format_income_cell(item.revenue, rev_yoy, cur),
            operating_income=format_income_cell(item.operating_income, op_yoy, cur),
            net_income=format_income_cell(item.net_income, ni_yoy, cur),
        ))

        # Chain: advance base only for non-None fields
        prev_stmt = IncomeStatementItem(
            period=item.period,
            revenue=item.revenue if item.revenue is not None else (prev_stmt.revenue if prev_stmt else None),
            operating_income=item.operating_income if item.operating_income is not None else (prev_stmt.operating_income if prev_stmt else None),
            net_income=item.net_income if item.net_income is not None else (prev_stmt.net_income if prev_stmt else None),
        )

    # Reverse to newest-first for display (2027E, 2026E, 2025E)
    rows.reverse()
    return rows


def build_quarterly_rows(
    statements: list[IncomeStatementItem],
) -> list[QuarterlyRow]:
    """Build quarterly income statement rows.

    Expects statements sorted newest-first.
    Each statement should already have YoY strings set if available.
    """
    rows = []
    cur = statements[0].currency if statements else "KRW"
    for stmt in statements:
        rev_yoy = stmt.revenue_yoy or "-"
        op_yoy = stmt.operating_income_yoy or "-"
        ni_yoy = stmt.net_income_yoy or "-"

        rows.append(QuarterlyRow(
            quarter=f"**{stmt.period}**",
            revenue=format_income_cell(stmt.revenue, rev_yoy, cur),
            operating_income=format_income_cell(stmt.operating_income, op_yoy, cur),
            net_income=format_income_cell(stmt.net_income, ni_yoy, cur),
        ))

    return rows


def _bold_wrap(text: str, bold: bool) -> str:
    """Wrap text in markdown bold if requested."""
    return f"**{text}**" if bold else text
