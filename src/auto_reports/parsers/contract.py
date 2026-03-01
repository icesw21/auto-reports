"""Parser for 단일판매/공급계약체결/해지 disclosures."""

from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from auto_reports.models.disclosure import (
    Contract,
    ContractDetails,
    ContractPeriod,
    SupplyMethod,
)
from auto_reports.parsers.base import clean_text, table_to_dict

logger = logging.getLogger(__name__)


def _detect_contract_type(soup: BeautifulSoup) -> str:
    """Guess the contract type name from the page title or first heading."""
    for tag in ("title", "h1", "h2", "h3"):
        elem = soup.find(tag)
        if elem:
            text = clean_text(elem.get_text())
            if "해지" in text:
                return "단일판매ㆍ공급계약해지"
            if "체결" in text:
                return "단일판매ㆍ공급계약체결"
    return "단일판매ㆍ공급계약체결"


def _parse_contract_details(kv: dict[str, str]) -> Optional[ContractDetails]:
    conditional = kv.get("조건부 계약여부") or None
    confirmed = kv.get("확정 계약금액") or None
    disclosed = kv.get("공시별 계약금액") or None
    total = kv.get("계약금액 합계(원)") or kv.get("계약금액 합계") or None
    revenue = kv.get("최근 매출액(원)") or kv.get("최근 매출액") or None
    ratio = kv.get("매출액 대비(%)") or kv.get("매출액 대비") or None

    if not any([conditional, confirmed, disclosed, total, revenue, ratio]):
        return None
    try:
        return ContractDetails(
            **{
                "조건부 계약여부": conditional,
                "확정 계약금액": confirmed,
                "공시별 계약금액": disclosed,
                "계약금액 합계(원)": total,
                "최근 매출액(원)": revenue,
                "매출액 대비(%)": ratio,
            }
        )
    except Exception as exc:
        logger.warning("Failed to parse ContractDetails: %s", exc)
        return None


def _parse_contract_period(kv: dict[str, str]) -> Optional[ContractPeriod]:
    start = kv.get("시작일") or kv.get("계약 시작일") or None
    end = kv.get("종료일") or kv.get("계약 종료일") or None
    if not (start or end):
        return None
    try:
        return ContractPeriod(**{"시작일": start, "종료일": end})
    except Exception as exc:
        logger.warning("Failed to parse ContractPeriod: %s", exc)
        return None


def _parse_supply_method(kv: dict[str, str]) -> Optional[SupplyMethod]:
    self_prod = kv.get("자체생산") or None
    outsourced = kv.get("외주생산") or None
    other = kv.get("기타") or None
    if not any([self_prod, outsourced, other]):
        return None
    try:
        return SupplyMethod(**{"자체생산": self_prod, "외주생산": outsourced, "기타": other})
    except Exception as exc:
        logger.warning("Failed to parse SupplyMethod: %s", exc)
        return None


def _extract_notes(soup: BeautifulSoup, kv: dict[str, str]) -> list[str]:
    """Extract 기타 투자판단에 참고할 사항 notes as a list of strings."""
    notes: list[str] = []
    # Try the KV dict first
    raw = kv.get("10. 기타 투자판단에 참고할 사항") or kv.get("기타 투자판단에 참고할 사항")
    if raw:
        for line in re.split(r"[\n•·※]+", raw):
            line = clean_text(line)
            if line:
                notes.append(line)
    return notes


def parse_contract(soup: BeautifulSoup) -> Contract:
    """Parse a 단일판매/공급계약 disclosure into a Contract model.

    Args:
        soup: BeautifulSoup object for the disclosure document.

    Returns:
        Contract populated with whatever data could be extracted.
    """
    contract_type_name = _detect_contract_type(soup)
    tables = soup.find_all("table")

    all_kv: dict[str, str] = {}
    for table in tables:
        kv = table_to_dict(table)
        all_kv.update(kv)

    description = (
        all_kv.get("1. 판매ㆍ공급계약 내용")
        or all_kv.get("판매ㆍ공급계약 내용")
        or all_kv.get("계약 내용")
        or None
    )
    details = _parse_contract_details(all_kv)
    counterparty = (
        all_kv.get("3. 계약상대방")
        or all_kv.get("계약상대방")
        or None
    )
    counterparty_revenue = all_kv.get("- 최근 매출액(원)") or None
    counterparty_major_shareholder = all_kv.get("- 주요주주") or None
    counterparty_relationship = all_kv.get("- 회사와의 관계") or None
    counterparty_recent_transaction = all_kv.get("- 회사와 최근 3기간 거래내역 참여여부") or None
    region = (
        all_kv.get("4. 판매ㆍ공급지역국가")
        or all_kv.get("판매ㆍ공급지역국가")
        or all_kv.get("지역국가")
        or None
    )
    period = _parse_contract_period(all_kv)

    # 6. 주요 계약조건 — store as flat dict of any sub-keys found
    conditions: Optional[dict] = None
    cond_keys = [k for k in all_kv if k.startswith("6.") or "계약조건" in k]
    if cond_keys:
        conditions = {k: all_kv[k] for k in cond_keys}

    supply_method = _parse_supply_method(all_kv)
    contract_date = (
        all_kv.get("8. 계약(수주)일자")
        or all_kv.get("계약(수주)일자")
        or all_kv.get("계약일자")
        or None
    )

    # 9. 공시사항 변경처리근거
    change_info: Optional[dict] = None
    change_keys = [k for k in all_kv if k.startswith("9.") or "변경처리" in k]
    if change_keys:
        change_info = {k: all_kv[k] for k in change_keys}

    notes = _extract_notes(soup, all_kv)

    try:
        return Contract(
            contract_type_name=contract_type_name,
            **{
                "1. 판매ㆍ공급계약 내용": description,
                "2. 계약내용": details,
                "3. 계약상대방": counterparty,
                "- 최근 매출액(원)": counterparty_revenue,
                "- 주요주주": counterparty_major_shareholder,
                "- 회사와의 관계": counterparty_relationship,
                "- 회사와 최근 3기간 거래내역 참여여부": counterparty_recent_transaction,
                "4. 판매ㆍ공급지역국가": region,
                "5. 계약기간": period,
                "6. 주요 계약조건": conditions,
                "7. 판매ㆍ공급방법": supply_method,
                "8. 계약(수주)일자": contract_date,
                "9. 공시사항 변경처리근거": change_info,
                "10. 기타 투자판단에 참고할 사항": notes,
            },
        )
    except Exception as exc:
        logger.error("Failed to build Contract: %s", exc)
        return Contract(contract_type_name=contract_type_name)
