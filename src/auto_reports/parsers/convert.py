"""Parser for 전환/교환/신주인수권/주식매수선택권 행사 disclosures."""

from __future__ import annotations

import logging
from typing import Optional

from bs4 import BeautifulSoup

from auto_reports.models.disclosure import (
    BondName,
    CBBalance,
    ConvertExercise,
    DailyClaim,
)
from auto_reports.parsers.base import clean_text, table_to_dict, table_to_rows

logger = logging.getLogger(__name__)


def _find_tables(soup: BeautifulSoup) -> list:
    return soup.find_all("table")


def _parse_daily_claims(rows: list[dict[str, str]]) -> list[DailyClaim]:
    """Convert table rows into DailyClaim objects."""
    claims: list[DailyClaim] = []
    for row in rows:
        # Map flexible column names to model aliases
        claim_date = (
            row.get("청구일자")
            or row.get("청구일")
            or row.get("행사일자")
            or row.get("행사일")
            or ""
        )
        claim_amount = (
            row.get("청구금액")
            or row.get("행사금액")
            or ""
        )
        conversion_price = (
            row.get("전환가액")
            or row.get("행사가액")
            or ""
        )
        shares_issued = (
            row.get("발행한 주식수")
            or row.get("발행주식수")
            or row.get("주식수")
            or ""
        )
        listing_date = (
            row.get("상장일 또는 예정일")
            or row.get("상장예정일")
            or row.get("상장일")
            or ""
        )
        # Bond name sub-fields
        series_raw = row.get("회차") or ""
        kind_raw = row.get("종류") or ""
        bond_name: Optional[BondName] = None
        if series_raw or kind_raw:
            series_int: Optional[int] = None
            try:
                series_int = int(series_raw.strip()) if series_raw.strip() else None
            except ValueError:
                pass
            bond_name = BondName(**{"회차": series_int, "종류": kind_raw or None})

        try:
            claim = DailyClaim(
                **{
                    "청구일자": claim_date or None,
                    "사채의 명칭": bond_name,
                    "청구금액": claim_amount or None,
                    "전환가액": conversion_price or None,
                    "발행한 주식수": shares_issued or None,
                    "상장일 또는 예정일": listing_date or None,
                }
            )
            claims.append(claim)
        except Exception as exc:
            logger.warning("Failed to parse DailyClaim row %s: %s", row, exc)

    return claims


def _parse_cb_balance(rows: list[dict[str, str]]) -> list[CBBalance]:
    """Convert table rows into CBBalance objects."""
    balances: list[CBBalance] = []
    for row in rows:
        series_raw = row.get("회차") or ""
        series_int: Optional[int] = None
        try:
            series_int = int(series_raw.strip()) if series_raw.strip() else None
        except ValueError:
            pass

        total_face = (
            row.get("발행당시 사채의 권면(전자등록)총액")
            or row.get("사채의 권면총액")
            or row.get("권면총액")
            or ""
        )
        currency = row.get("통화단위") or ""
        remaining = (
            row.get("신고일 현재 미전환사채 잔액")
            or row.get("미전환사채 잔액")
            or row.get("잔액")
            or ""
        )
        remaining_currency = row.get("잔액 통화단위") or row.get("통화단위") or ""
        conversion_price = (
            row.get("전환가액(원)")
            or row.get("전환가액")
            or ""
        )
        convertible_shares = (
            row.get("전환가능 주식수")
            or row.get("전환가능주식수")
            or ""
        )

        try:
            balance = CBBalance(
                **{
                    "회차": series_int,
                    "발행당시 사채의 권면(전자등록)총액": total_face or None,
                    "통화단위": currency or None,
                    "신고일 현재 미전환사채 잔액": remaining or None,
                    "잔액 통화단위": remaining_currency or None,
                    "전환가액(원)": conversion_price or None,
                    "전환가능 주식수": convertible_shares or None,
                }
            )
            balances.append(balance)
        except Exception as exc:
            logger.warning("Failed to parse CBBalance row %s: %s", row, exc)

    return balances


def parse_convert(soup: BeautifulSoup) -> ConvertExercise:
    """Parse a 전환/교환/신주인수권 행사 disclosure into a ConvertExercise model.

    Args:
        soup: BeautifulSoup object for the disclosure document.

    Returns:
        ConvertExercise populated with whatever data could be extracted.
    """
    tables = _find_tables(soup)
    summary: dict[str, str] = {}
    daily_claims: list[DailyClaim] = []
    cb_balance: list[CBBalance] = []

    for table in tables:
        rows_raw = table.find_all("tr")
        if not rows_raw:
            continue

        # Heuristic: if the table has 2 columns with a header-like first column,
        # treat it as a key-value summary table.
        kv = table_to_dict(table)
        if kv:
            summary.update(kv)

        # Detect the daily-claims table by looking for 청구일자 or 행사일자 in headers
        header_text = " ".join(
            clean_text(cell.get_text())
            for cell in (rows_raw[0].find_all(["th", "td"]) if rows_raw else [])
        )
        if "청구일자" in header_text or "행사일자" in header_text:
            daily_rows = table_to_rows(table)
            daily_claims.extend(_parse_daily_claims(daily_rows))

        # Detect the CB balance table
        if "미전환사채" in header_text or "전환사채 잔액" in header_text:
            balance_rows = table_to_rows(table)
            cb_balance.extend(_parse_cb_balance(balance_rows))

    # Extract summary fields using alias keys
    cumulative_shares = (
        summary.get("1. 전환청구권 행사주식수 누계 (주) (기 신고된 주식수량 제외)")
        or summary.get("전환청구권 행사주식수 누계")
        or summary.get("행사주식수 누계")
    )
    total_shares = (
        summary.get("-발행주식총수(주)")
        or summary.get("발행주식총수")
        or summary.get("발행주식총수(주)")
    )
    ratio = (
        summary.get("-발행주식총수 대비(%)")
        or summary.get("발행주식총수 대비(%)")
        or summary.get("발행주식총수 대비")
    )

    try:
        return ConvertExercise(
            **{
                "1. 전환청구권 행사주식수 누계 (주) (기 신고된 주식수량 제외)": cumulative_shares,
                "-발행주식총수(주)": total_shares,
                "-발행주식총수 대비(%)": ratio,
                "일별 전환청구내역": daily_claims,
                "전환사채 잔액": cb_balance,
            }
        )
    except Exception as exc:
        logger.error("Failed to build ConvertExercise: %s", exc)
        return ConvertExercise()
