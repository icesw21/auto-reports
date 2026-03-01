"""Parser for 전환가액/신주인수권행사가액/교환가액 조정 disclosures."""

from __future__ import annotations

import logging
from typing import Optional

from bs4 import BeautifulSoup

from auto_reports.models.disclosure import (
    ConvertPriceChange,
    DocumentInfo,
    PriceAdjustment,
    ShareChange,
)
from auto_reports.parsers.base import clean_text, table_to_dict, table_to_rows

logger = logging.getLogger(__name__)


def _parse_document_info(kv: dict[str, str]) -> DocumentInfo:
    company = kv.get("회사명") or kv.get("발행회사") or None
    report_type = kv.get("보고서종류") or kv.get("보고서 종류") or None
    series = kv.get("회차") or None
    disclosure_date = kv.get("공시일") or kv.get("보고일") or None
    try:
        return DocumentInfo(
            **{
                "회사명": company,
                "보고서종류": report_type,
                "회차": series,
                "공시일": disclosure_date,
            }
        )
    except Exception as exc:
        logger.warning("Failed to parse DocumentInfo: %s", exc)
        return DocumentInfo()


def _parse_price_adjustment(rows: list[dict[str, str]]) -> Optional[PriceAdjustment]:
    if not rows:
        return None
    row = rows[0]
    series_raw = row.get("회차") or ""
    series_int: Optional[int] = None
    try:
        series_int = int(series_raw.strip()) if series_raw.strip() else None
    except ValueError:
        pass

    listed = row.get("상장여부") or None
    price_before = (
        row.get("조정전 전환가액 (원)")
        or row.get("조정전 전환가액")
        or row.get("조정전가액")
        or None
    )
    price_after = (
        row.get("조정후 전환가액 (원)")
        or row.get("조정후 전환가액")
        or row.get("조정후가액")
        or None
    )

    try:
        return PriceAdjustment(
            **{
                "회차": series_int,
                "상장여부": listed,
                "조정전 전환가액 (원)": price_before,
                "조정후 전환가액 (원)": price_after,
            }
        )
    except Exception as exc:
        logger.warning("Failed to parse PriceAdjustment: %s", exc)
        return None


def _parse_share_change(rows: list[dict[str, str]]) -> Optional[ShareChange]:
    if not rows:
        return None
    row = rows[0]
    series_raw = row.get("회차") or ""
    series_int: Optional[int] = None
    try:
        series_int = int(series_raw.strip()) if series_raw.strip() else None
    except ValueError:
        pass

    unconverted = (
        row.get("미전환사채의 권면(전자등록)총액")
        or row.get("미전환사채 권면총액")
        or row.get("미전환사채")
        or None
    )
    currency = row.get("통화단위") or None
    shares_before = (
        row.get("조정전 전환가능 주식수 (주)")
        or row.get("조정전 전환가능주식수")
        or None
    )
    shares_after = (
        row.get("조정후 전환가능 주식수 (주)")
        or row.get("조정후 전환가능주식수")
        or None
    )

    try:
        return ShareChange(
            **{
                "회차": series_int,
                "미전환사채의 권면(전자등록)총액": unconverted,
                "통화단위": currency,
                "조정전 전환가능 주식수 (주)": shares_before,
                "조정후 전환가능 주식수 (주)": shares_after,
            }
        )
    except Exception as exc:
        logger.warning("Failed to parse ShareChange: %s", exc)
        return None


def parse_convert_price(soup: BeautifulSoup) -> ConvertPriceChange:
    """Parse a 전환가액 조정 disclosure into a ConvertPriceChange model.

    Args:
        soup: BeautifulSoup object for the disclosure document.

    Returns:
        ConvertPriceChange populated with whatever data could be extracted.
    """
    tables = soup.find_all("table")
    all_kv: dict[str, str] = {}
    adjustment: Optional[PriceAdjustment] = None
    share_change: Optional[ShareChange] = None

    for table in tables:
        rows_raw = table.find_all("tr")
        if not rows_raw:
            continue

        header_text = " ".join(
            clean_text(cell.get_text())
            for cell in rows_raw[0].find_all(["th", "td"])
        )

        # Detect adjustment table (has 조정전/조정후 columns)
        if "조정전" in header_text and "조정후" in header_text:
            rows = table_to_rows(table)
            if adjustment is None:
                adjustment = _parse_price_adjustment(rows)
            continue

        # Detect share-change table (has 전환가능 주식수 columns)
        if "전환가능" in header_text and ("조정전" in header_text or "조정후" in header_text):
            rows = table_to_rows(table)
            if share_change is None:
                share_change = _parse_share_change(rows)
            continue

        # Otherwise accumulate key-value pairs for document info
        kv = table_to_dict(table)
        all_kv.update(kv)

    doc_info = _parse_document_info(all_kv)

    try:
        return ConvertPriceChange(
            **{
                "문서정보": doc_info,
                "1. 조정에 관한 사항": adjustment,
                "2. 전환가능주식수 변동": share_change,
            }
        )
    except Exception as exc:
        logger.error("Failed to build ConvertPriceChange: %s", exc)
        return ConvertPriceChange()
