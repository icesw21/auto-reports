"""Parser for CB/BW 발행결정 and 유상증자 disclosures."""

from __future__ import annotations

import io
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from auto_reports.models.disclosure import (
    BondType,
    CBIssuance,
    ConversionTerms,
    parse_korean_float,
    parse_korean_number,
)
from auto_reports.parsers.base import clean_text, table_to_dict

logger = logging.getLogger(__name__)


def _detect_issuance_type(soup: BeautifulSoup) -> str:
    for tag in ("title", "h1", "h2", "h3"):
        elem = soup.find(tag)
        if elem:
            text = clean_text(elem.get_text())
            if "신주인수권부사채" in text:
                return "신주인수권부사채권 발행결정"
            if "전환사채" in text:
                return "전환사채권 발행결정"
    return "전환사채권 발행결정"


def _parse_bond_type(kv: dict[str, str]) -> Optional[BondType]:
    series_raw = (
        kv.get("회차")
        or kv.get("사채 회차")
        or ""
    )
    series_int: Optional[int] = None
    try:
        series_int = int(series_raw.strip()) if series_raw.strip() else None
    except ValueError:
        pass
    kind = (
        kv.get("종류")
        or kv.get("사채의 종류")
        or kv.get("1. 사채의 종류")
        or None
    )
    if series_int is None and kind is None:
        return None
    try:
        return BondType(**{"회차": series_int, "종류": kind})
    except Exception as exc:
        logger.warning("Failed to parse BondType: %s", exc)
        return None


def _parse_conversion_terms(kv: dict[str, str]) -> Optional[ConversionTerms]:
    price_raw = (
        kv.get("전환가액 (원/주)")
        or kv.get("전환가액")
        or kv.get("행사가액 (원/주)")
        or kv.get("행사가액")
        or None
    )
    conversion_price = parse_korean_number(price_raw)

    # Share info sub-dict
    share_type = kv.get("종류") or kv.get("전환주식 종류") or None
    share_count_raw = kv.get("주식수") or kv.get("전환주식수") or None
    share_count = parse_korean_number(share_count_raw)
    ratio_raw = kv.get("주식총수 대비 비율(%)") or kv.get("주식총수 대비 비율") or None
    share_info: Optional[dict] = None
    if any([share_type, share_count, ratio_raw]):
        share_info = {}
        if share_type:
            share_info["종류"] = share_type
        if share_count is not None:
            share_info["주식수"] = share_count
        if ratio_raw:
            from auto_reports.models.disclosure import parse_korean_float
            ratio_val = parse_korean_float(ratio_raw)
            if ratio_val is not None:
                share_info["주식총수 대비 비율(%)"] = ratio_val

    # Request period sub-dict
    start = kv.get("시작일") or kv.get("전환청구 시작일") or None
    end = kv.get("종료일") or kv.get("전환청구 종료일") or None
    request_period: Optional[dict] = None
    if start or end:
        request_period = {}
        if start:
            request_period["시작일"] = start
        if end:
            request_period["종료일"] = end

    if conversion_price is None and share_info is None and request_period is None:
        return None

    try:
        return ConversionTerms(
            **{
                "전환가액 (원/주)": conversion_price,
                "전환에_따라_발행할_주식": share_info,
                "전환청구기간": request_period,
            }
        )
    except Exception as exc:
        logger.warning("Failed to parse ConversionTerms: %s", exc)
        return None


def parse_issue(soup: BeautifulSoup) -> CBIssuance:
    """Parse a CB/EB/BW 발행결정 disclosure into a CBIssuance model.

    Args:
        soup: BeautifulSoup object for the disclosure document.

    Returns:
        CBIssuance populated with whatever data could be extracted.
    """
    issuance_type_name = _detect_issuance_type(soup)
    tables = soup.find_all("table")

    all_kv: dict[str, str] = {}
    for table in tables:
        kv = table_to_dict(table)
        all_kv.update(kv)

    bond_type = _parse_bond_type(all_kv)

    face_value_raw = (
        all_kv.get("2. 사채의 권면(전자등록)총액 (원)")
        or all_kv.get("사채의 권면(전자등록)총액 (원)")
        or all_kv.get("사채의 권면총액")
        or None
    )
    face_value = parse_korean_number(face_value_raw)

    conversion_terms = _parse_conversion_terms(all_kv)

    try:
        return CBIssuance(
            issuance_type_name=issuance_type_name,
            **{
                "1. 사채의 종류": bond_type,
                "2. 사채의 권면(전자등록)총액 (원)": face_value,
                "9. 전환에 관한 사항": conversion_terms,
            },
        )
    except Exception as exc:
        logger.error("Failed to build CBIssuance: %s", exc)
        return CBIssuance(issuance_type_name=issuance_type_name)


def _normalize_korean_date(raw: Optional[str]) -> Optional[str]:
    """Normalize Korean date '2025년 01월 03일' → '2025-01-03'."""
    if not raw:
        return None
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw.strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# 유상증자 (Rights Issue) HTML parser (from DART disclosure pages)
# ---------------------------------------------------------------------------


def parse_rights_issue_html(soup: BeautifulSoup) -> dict:
    """Parse a 유상증자결정 HTML disclosure into structured data.

    Extracts 기타주식 share counts, issue price, conversion details
    (전환가액, 전환주식수, 전환청구기간), and funding amounts from
    DART HTML key-value tables.

    Args:
        soup: BeautifulSoup object for the disclosure document.

    Returns:
        Dict with parsed rights issue data.
    """
    tables = soup.find_all("table")
    all_kv: dict[str, str] = {}
    for table in tables:
        kv = table_to_dict(table)
        all_kv.update(kv)

    result: dict = {"type": "유상증자결정"}

    # --- 기타주식 수 ---
    estk_raw = all_kv.get("기타주식 (주)") or all_kv.get("기타주식(주)")
    estk = parse_korean_number(estk_raw)
    ostk_raw = all_kv.get("보통주식 (주)") or all_kv.get("보통주식(주)")
    ostk = parse_korean_number(ostk_raw)
    new_shares: dict = {}
    if ostk:
        new_shares["보통주식"] = ostk
    if estk:
        new_shares["기타주식"] = estk
    result["new_shares"] = new_shares

    # --- 발행가액 (기타주식) ---
    issue_price_raw = all_kv.get("기타주식 (원)") or all_kv.get("기타주식(원)")
    result["issue_price"] = parse_korean_number(issue_price_raw)

    # --- 전환 관련 (uses same field names as CB conversion terms) ---
    conversion: dict = {}

    conv_price_raw = (
        all_kv.get("전환가액 (원/주)")
        or all_kv.get("전환가액")
        or all_kv.get("전환가액(원/주)")
    )
    conv_price = parse_korean_number(conv_price_raw)
    if conv_price:
        conversion["전환가액(원/주)"] = conv_price

    conv_shares_raw = all_kv.get("주식수") or all_kv.get("전환주식수")
    conv_shares = parse_korean_number(conv_shares_raw)
    if conv_shares:
        conversion["전환주식수"] = conv_shares

    ratio_raw = all_kv.get("주식총수 대비 비율(%)") or all_kv.get("주식총수 대비 비율")
    if ratio_raw:
        ratio_val = parse_korean_float(ratio_raw)
        if ratio_val is not None:
            conversion["주식총수대비비율(%)"] = ratio_val

    conv_ratio_raw = all_kv.get("전환비율 (%)") or all_kv.get("전환비율(%)")
    if conv_ratio_raw:
        conv_ratio_val = parse_korean_float(conv_ratio_raw)
        if conv_ratio_val is not None:
            conversion["전환비율(%)"] = conv_ratio_val

    # 전환청구기간
    start = all_kv.get("시작일") or all_kv.get("전환청구 시작일")
    end = all_kv.get("종료일") or all_kv.get("전환청구 종료일")
    if start or end:
        conversion["전환청구기간"] = {
            "시작일": _normalize_korean_date(start),
            "종료일": _normalize_korean_date(end),
        }

    if conversion:
        result["conversion"] = conversion

    # --- 자금조달 목적 (funding amounts) ---
    funding: dict = {}
    for label in ("시설자금", "영업양수자금", "운영자금", "채무상환자금", "기타자금"):
        raw = all_kv.get(f"{label} (원)") or all_kv.get(f"{label}(원)") or all_kv.get(label)
        val = parse_korean_number(raw)
        if val:
            funding[label] = val
    result["funding_purpose"] = funding or None

    # --- 기타주식 내용 (전환우선주 etc.) ---
    share_type = all_kv.get("주식의 내용") or all_kv.get("기타주식의 내용")
    if share_type:
        result["share_type"] = share_type

    return result


# ---------------------------------------------------------------------------
# 유상증자 (Rights Issue) PDF parser
# ---------------------------------------------------------------------------

def _ri_find(text: str, pattern: str, group: int = 1) -> Optional[str]:
    """Search for a regex pattern in text, return the specified group."""
    m = re.search(pattern, text)
    return m.group(group).strip() if m else None


def _ri_find_number(text: str, pattern: str, group: int = 1) -> Optional[int]:
    """Search for a regex pattern and parse the match as a Korean number."""
    raw = _ri_find(text, pattern, group)
    return parse_korean_number(raw)


def _ri_find_float(text: str, pattern: str, group: int = 1) -> Optional[float]:
    """Search for a regex pattern and parse the match as a Korean float."""
    raw = _ri_find(text, pattern, group)
    return parse_korean_float(raw)


def _ri_find_date(text: str, pattern: str, group: int = 1) -> Optional[str]:
    """Search for a date pattern and normalize to YYYY-MM-DD."""
    raw = _ri_find(text, pattern, group)
    if not raw:
        return None
    # Convert "2023년 12월 13일" -> "2023-12-13"
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # Already YYYY-MM-DD
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    return raw


def parse_rights_issue_pdf(pdf_bytes: bytes) -> dict:
    """Parse a 유상증자 (rights issue) PDF into structured data.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        Dict with parsed rights issue data.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber required for PDF parsing: pip install pdfplumber")
        return {"type": "유상증자결정", "error": "pdfplumber not installed"}

    pages_text: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
    except Exception as exc:
        logger.error("Failed to read PDF: %s", exc)
        return {"type": "유상증자결정", "error": str(exc)}

    full_text = "\n".join(pages_text)

    result: dict = {"type": "유상증자결정"}

    # --- 신주의 종류와 수 ---
    new_shares: dict = {}
    m = re.search(r"1\.\s*신주의\s*종류와\s*수\s*\n?.*?보통주식\s*\(주\)\s*([\d,\-]+)", full_text, re.DOTALL)
    if m:
        new_shares["보통주식"] = parse_korean_number(m.group(1))
    m = re.search(r"1\.\s*신주의\s*종류와\s*수\s*\n?.*?기타주식\s*\(주\)\s*([\d,\-]+)", full_text, re.DOTALL)
    if not m:
        m = re.search(r"기타주식\s*\(주\)\s*([\d,]+)", full_text)
    if m:
        new_shares["기타주식"] = parse_korean_number(m.group(1))
    result["new_shares"] = new_shares

    # --- 액면가액 ---
    result["par_value"] = _ri_find_number(
        full_text, r"1주당\s*액면가액\s*\(원\)\s*([\d,]+)"
    )

    # --- 증자전 발행주식총수 ---
    pre_shares: dict = {}
    m = re.search(r"증자전\s*보통주식\s*\(주\)\s*([\d,]+)", full_text)
    if not m:
        m = re.search(r"3\.\s*증자전\s*\n?.*?보통주식\s*\(주\)\s*([\d,]+)", full_text, re.DOTALL)
    if m:
        pre_shares["보통주식"] = parse_korean_number(m.group(1))
    m = re.search(r"발행주식총수\s*\(주\)\s*기타주식\s*\(주\)\s*([\d,]+)", full_text)
    if m:
        pre_shares["기타주식"] = parse_korean_number(m.group(1))
    result["pre_issue_shares"] = pre_shares

    # --- 자금조달의 목적 ---
    funding: dict = {}
    for label in ("시설자금", "영업양수자금", "운영자금", "채무상환자금"):
        m = re.search(rf"{label}\s*\(원\)\s*([\d,\-]+)", full_text)
        if m:
            val = parse_korean_number(m.group(1))
            if val:
                funding[label] = val
    result["funding_purpose"] = funding or None

    # --- 증자방식 ---
    result["issue_method"] = _ri_find(
        full_text, r"5\.\s*증자방식\s+(.+?)(?:\n|$)"
    )

    # --- 기타주식 내용 (전환우선주 etc.) ---
    result["share_type"] = _ri_find(
        full_text, r"주식의\s*내용\s+(.+?)(?:\n|기타)"
    )

    # --- 전환 관련 ---
    conversion: dict = {}
    ratio = _ri_find_float(full_text, r"전환비율\s*\(%\)\s*([\d,.]+)")
    if ratio:
        conversion["전환비율(%)"] = ratio

    conv_price = _ri_find_number(full_text, r"전환가액\s*\(원/주\)\s*([\d,]+)")
    if conv_price:
        conversion["전환가액(원/주)"] = conv_price

    # 전환에 따라 발행할 주식
    m = re.search(r"종류\s+(.+?기명식\s*보통주)", full_text)
    if m:
        conversion["전환주식종류"] = m.group(1).strip()

    conv_shares = _ri_find_number(
        full_text, r"(?:전환에\s*따라|발행할\s*주식)\s*주식수\s*([\d,]+)"
    )
    if not conv_shares:
        # Alternative pattern: "주식수 246,361"
        m = re.search(r"발행할\s*주식\s*주식수\s*([\d,]+)", full_text)
        if not m:
            m = re.search(r"주식수\s+([\d,]+)\s*\n.*?대비\s*비율", full_text, re.DOTALL)
        if m:
            conv_shares = parse_korean_number(m.group(1))
    if conv_shares:
        conversion["전환주식수"] = conv_shares

    ratio_total = _ri_find_float(full_text, r"대비\s*비율\s*\(%?\)?\s*([\d,.]+)")
    if not ratio_total:
        # PDF layout variant: "주식총수\n1.47\n대비 비율(%)"
        m = re.search(r"주식총수\s*\n?\s*([\d,.]+)\s*\n?\s*대비\s*비율", full_text)
        if m:
            ratio_total = parse_korean_float(m.group(1))
    if ratio_total:
        conversion["주식총수대비비율(%)"] = ratio_total

    # 전환청구기간
    conv_start = _ri_find_date(
        full_text, r"전환청구기간\s*\n?.*?시작일\s+(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)"
    )
    if not conv_start:
        conv_start = _ri_find_date(full_text, r"시작일\s+(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)")
    conv_end = _ri_find_date(full_text, r"종료일\s+(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)")
    if conv_start or conv_end:
        conversion["전환청구기간"] = {
            "시작일": conv_start,
            "종료일": conv_end,
        }

    # 전환가액 조정 하한
    min_price = _ri_find_number(
        full_text, r"최저조정한도.*?(\d[\d,]+)\s*원?\s*\n|발행가액의\s*조정.*?최저.*?(\d[\d,]+)"
    )
    if not min_price:
        m = re.search(r"최초\s*전환가액.*?70%\s*이상", full_text)
        if m:
            conversion["전환가액조정"] = "발행 시 전환가액의 70% 이상"

    if conversion:
        result["conversion"] = conversion

    # --- 발행가액 ---
    issue_price = _ri_find_number(
        full_text, r"6\.\s*신주\s*발행가액\s*\n?.*?기타주식\s*\(원\)\s*([\d,]+)"
    )
    if not issue_price:
        issue_price = _ri_find_number(full_text, r"신주\s*발행가액.*?기타주식\s*\(원\)\s*([\d,]+)")
    result["issue_price"] = issue_price

    # --- 기준주가 ---
    ref_price = _ri_find_number(
        full_text, r"7\.\s*기준주가\s*\n?.*?기타주식\s*\(원\)\s*([\d,]+)"
    )
    if not ref_price:
        ref_price = _ri_find_number(full_text, r"기준주가\s*\n?.*?기타주식\s*\(원\)\s*([\d,]+)")
    result["reference_price"] = ref_price

    # --- 할인율 ---
    result["discount_rate_pct"] = _ri_find_float(
        full_text, r"할인율\s*또는\s*할증률?\s*\(%\)\s*([\d,.]+)"
    )
    if result["discount_rate_pct"] is None:
        result["discount_rate_pct"] = _ri_find_float(
            full_text, r"할인율.*?\(%\)\s*([\d,.]+)"
        )

    # --- 납입일 ---
    result["payment_date"] = _ri_find_date(
        full_text, r"9\.\s*납입일\s+(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)"
    )
    if not result["payment_date"]:
        result["payment_date"] = _ri_find_date(
            full_text, r"납입일\s+(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)"
        )

    # --- 배당기산일 ---
    result["dividend_base_date"] = _ri_find_date(
        full_text, r"배당기산일\s+(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)"
    )

    # --- 이사회결의일 ---
    result["board_decision_date"] = _ri_find_date(
        full_text, r"이사회결의일\s*\(결정일\)\s*(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)"
    )

    # --- VWAP pricing detail ---
    pricing: dict = {}
    m = re.search(
        r"1개월간의\s*가중산술평균주가\s*\(A\)\s*[\d,]+\s+[\d,]+\s+([\d,.]+)",
        full_text,
    )
    if m:
        pricing["1개월_vwap"] = parse_korean_float(m.group(1))
    m = re.search(
        r"1주일간의\s*가중산술평균주가\s*\(B\)\s*[\d,]+\s+[\d,]+\s+([\d,.]+)",
        full_text,
    )
    if m:
        pricing["1주일_vwap"] = parse_korean_float(m.group(1))
    m = re.search(
        r"최근일\s*가중산술평균주가\s*\(C\)\s*[\d,]+\s+[\d,]+\s+([\d,.]+)",
        full_text,
    )
    if m:
        pricing["최근일_vwap"] = parse_korean_float(m.group(1))
    m = re.search(r"산술평균주가\s*\(D\)\s*([\d,.]+)", full_text)
    if m:
        pricing["산술평균"] = parse_korean_float(m.group(1))
    m = re.search(r"기준주가\s*:\s*\(C\)와\s*\(D\)\s*중\s*낮은\s*가액\s*([\d,.]+)", full_text)
    if m:
        pricing["기준주가"] = parse_korean_float(m.group(1))
    m = re.search(r"발행가액\s+([\d,]+)\s*\n", full_text)
    if m:
        pricing["발행가액"] = parse_korean_number(m.group(1))

    if pricing:
        result["pricing_detail"] = pricing

    # --- 제3자배정 대상자 ---
    allocations: list[dict] = []
    # Pattern: company name, then shares count
    for m in re.finditer(
        r"([\w\s]+(?:주식회사|㈜).*?)\s*(?:1년간|의무보유).*?(\d[\d,]+)\s*(?:주|$)",
        full_text,
    ):
        name = m.group(1).strip()
        shares = parse_korean_number(m.group(2))
        if shares and name:
            allocations.append({"name": name, "shares": shares})
    if allocations:
        result["third_party_allocations"] = allocations

    # --- 증권신고서 제출대상 여부 ---
    result["securities_registration"] = _ri_find(
        full_text, r"증권신고서\s*제출대상\s*여부\s+(\S+)"
    )

    return result
