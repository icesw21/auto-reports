"""Pydantic models for 5 DART disclosure types."""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field


def parse_korean_number(text: str | int | float | None) -> Optional[int]:
    """Parse Korean-formatted numbers like '1,932,000,000원' or '5,370' to int.

    Handles: commas, 원 suffix, negative values, dash for empty, percentage strings.
    Returns None for unparseable values.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text)
    text = str(text).strip()
    if not text or text == "-":
        return None
    # Remove 원, 주, %, commas, spaces
    cleaned = re.sub(r"[원주%,\s]", "", text)
    if not cleaned or cleaned == "-":
        return None
    try:
        return int(cleaned)
    except ValueError:
        try:
            return int(float(cleaned))
        except ValueError:
            return None


def parse_korean_float(text: str | int | float | None) -> Optional[float]:
    """Parse Korean-formatted numbers to float (for percentages etc.)."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    text = str(text).strip()
    if not text or text == "-":
        return None
    cleaned = re.sub(r"[원주%,\s]", "", text)
    if not cleaned or cleaned == "-":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


# ──────────────────────────────────────────────
# convert.json - 전환/교환/신주인수권/주식매수선택권 행사
# ──────────────────────────────────────────────

class BondName(BaseModel):
    """사채의 명칭."""
    series: Optional[int] = Field(None, alias="회차")
    kind: Optional[str] = Field(None, alias="종류")


class DailyClaim(BaseModel):
    """일별 전환청구내역."""
    claim_date: Optional[str] = Field(None, alias="청구일자")
    bond_name: Optional[BondName] = Field(None, alias="사채의 명칭")
    claim_amount: Optional[str] = Field(None, alias="청구금액")
    conversion_price: Optional[str] = Field(None, alias="전환가액")
    shares_issued: Optional[str] = Field(None, alias="발행한 주식수")
    listing_date: Optional[str] = Field(None, alias="상장일 또는 예정일")

    @property
    def claim_amount_int(self) -> Optional[int]:
        return parse_korean_number(self.claim_amount)

    @property
    def conversion_price_int(self) -> Optional[int]:
        return parse_korean_number(self.conversion_price)

    @property
    def shares_issued_int(self) -> Optional[int]:
        return parse_korean_number(self.shares_issued)


class CBBalance(BaseModel):
    """전환사채 잔액."""
    series: Optional[int] = Field(None, alias="회차")
    total_face_value: Optional[str] = Field(None, alias="발행당시 사채의 권면(전자등록)총액")
    currency: Optional[str] = Field(None, alias="통화단위")
    remaining: Optional[str] = Field(None, alias="신고일 현재 미전환사채 잔액")
    remaining_currency: Optional[str] = Field(None, alias="잔액 통화단위")
    conversion_price: Optional[str] = Field(None, alias="전환가액(원)")
    convertible_shares: Optional[str] = Field(None, alias="전환가능 주식수")

    @property
    def remaining_int(self) -> Optional[int]:
        return parse_korean_number(self.remaining)

    @property
    def conversion_price_int(self) -> Optional[int]:
        return parse_korean_number(self.conversion_price)

    @property
    def convertible_shares_int(self) -> Optional[int]:
        return parse_korean_number(self.convertible_shares)


class ConvertExercise(BaseModel):
    """전환청구권ㆍ신주인수권ㆍ교환청구권행사 공시."""
    cumulative_shares: Optional[str] = Field(
        None, alias="1. 전환청구권 행사주식수 누계 (주) (기 신고된 주식수량 제외)"
    )
    total_shares: Optional[str] = Field(None, alias="-발행주식총수(주)")
    ratio: Optional[str] = Field(None, alias="-발행주식총수 대비(%)")
    daily_claims: list[DailyClaim] = Field(default_factory=list, alias="일별 전환청구내역")
    cb_balance: list[CBBalance] = Field(default_factory=list, alias="전환사채 잔액")

    model_config = {"populate_by_name": True}

    @property
    def cumulative_shares_int(self) -> Optional[int]:
        return parse_korean_number(self.cumulative_shares)

    @property
    def total_shares_int(self) -> Optional[int]:
        return parse_korean_number(self.total_shares)


# ──────────────────────────────────────────────
# convert-price-change.json - 전환가액/행사가액 조정
# ──────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """문서정보."""
    company_name: Optional[str] = Field(None, alias="회사명")
    report_type: Optional[str] = Field(None, alias="보고서종류")
    series: Optional[str] = Field(None, alias="회차")
    disclosure_date: Optional[str] = Field(None, alias="공시일")


class PriceAdjustment(BaseModel):
    """조정에 관한 사항."""
    series: Optional[int] = Field(None, alias="회차")
    listed: Optional[str] = Field(None, alias="상장여부")
    price_before: Optional[str] = Field(None, alias="조정전 전환가액 (원)")
    price_after: Optional[str] = Field(None, alias="조정후 전환가액 (원)")

    @property
    def price_before_int(self) -> Optional[int]:
        return parse_korean_number(self.price_before)

    @property
    def price_after_int(self) -> Optional[int]:
        return parse_korean_number(self.price_after)


class ShareChange(BaseModel):
    """전환가능주식수 변동."""
    series: Optional[int] = Field(None, alias="회차")
    unconverted_amount: Optional[str] = Field(None, alias="미전환사채의 권면(전자등록)총액")
    currency: Optional[str] = Field(None, alias="통화단위")
    shares_before: Optional[str] = Field(None, alias="조정전 전환가능 주식수 (주)")
    shares_after: Optional[str] = Field(None, alias="조정후 전환가능 주식수 (주)")

    @property
    def unconverted_amount_int(self) -> Optional[int]:
        return parse_korean_number(self.unconverted_amount)

    @property
    def shares_before_int(self) -> Optional[int]:
        return parse_korean_number(self.shares_before)

    @property
    def shares_after_int(self) -> Optional[int]:
        return parse_korean_number(self.shares_after)


class ConvertPriceChange(BaseModel):
    """전환가액ㆍ신주인수권행사가액ㆍ교환가액의조정 공시."""
    document_info: Optional[DocumentInfo] = Field(None, alias="문서정보")
    adjustment: Optional[PriceAdjustment] = Field(None, alias="1. 조정에 관한 사항")
    share_change: Optional[ShareChange] = Field(None, alias="2. 전환가능주식수 변동")

    model_config = {"populate_by_name": True}


# ──────────────────────────────────────────────
# contract.json - 단일판매ㆍ공급계약체결/해지
# ──────────────────────────────────────────────

class ContractDetails(BaseModel):
    """계약내용."""
    conditional: Optional[str] = Field(None, alias="조건부 계약여부")
    confirmed_amount: Optional[str] = Field(None, alias="확정 계약금액")
    disclosed_amount: Optional[str] = Field(None, alias="공시별 계약금액")
    total_amount: Optional[str] = Field(None, alias="계약금액 합계(원)")
    recent_revenue: Optional[str] = Field(None, alias="최근 매출액(원)")
    revenue_ratio: Optional[str] = Field(None, alias="매출액 대비(%)")

    @property
    def total_amount_int(self) -> Optional[int]:
        return parse_korean_number(self.total_amount)

    @property
    def recent_revenue_int(self) -> Optional[int]:
        return parse_korean_number(self.recent_revenue)


class ContractPeriod(BaseModel):
    """계약기간."""
    start_date: Optional[str] = Field(None, alias="시작일")
    end_date: Optional[str] = Field(None, alias="종료일")


class SupplyMethod(BaseModel):
    """판매ㆍ공급방법."""
    self_production: Optional[str] = Field(None, alias="자체생산")
    outsourced: Optional[str] = Field(None, alias="외주생산")
    other: Optional[str] = Field(None, alias="기타")


class Contract(BaseModel):
    """단일판매ㆍ공급계약체결/해지 공시. Top-level key varies by type."""
    contract_type_name: str = ""  # e.g. "단일판매ㆍ공급계약체결"
    description: Optional[str] = Field(None, alias="1. 판매ㆍ공급계약 내용")
    details: Optional[ContractDetails] = Field(None, alias="2. 계약내용")
    counterparty: Optional[str] = Field(None, alias="3. 계약상대방")
    counterparty_revenue: Optional[str] = Field(None, alias="- 최근 매출액(원)")
    counterparty_major_shareholder: Optional[str] = Field(None, alias="- 주요주주")
    counterparty_relationship: Optional[str] = Field(None, alias="- 회사와의 관계")
    counterparty_recent_transaction: Optional[str] = Field(
        None, alias="- 회사와 최근 3기간 거래내역 참여여부"
    )
    region: Optional[str] = Field(None, alias="4. 판매ㆍ공급지역국가")
    period: Optional[ContractPeriod] = Field(None, alias="5. 계약기간")
    conditions: Optional[dict] = Field(None, alias="6. 주요 계약조건")
    supply_method: Optional[SupplyMethod] = Field(None, alias="7. 판매ㆍ공급방법")
    contract_date: Optional[str] = Field(None, alias="8. 계약(수주)일자")
    change_info: Optional[dict] = Field(None, alias="9. 공시사항 변경처리근거")
    notes: list[str] = Field(default_factory=list, alias="10. 기타 투자판단에 참고할 사항")

    model_config = {"populate_by_name": True}


# ──────────────────────────────────────────────
# performance.json - 매출액또는손익구조 변동
# ──────────────────────────────────────────────

class PeriodRange(BaseModel):
    """결산기간."""
    start: Optional[str] = Field(None, alias="- 시작일")
    end: Optional[str] = Field(None, alias="- 종료일")


class IncomeItem(BaseModel):
    """매출액/영업이익/당기순이익 항목."""
    current: Optional[str] = Field(None, alias="당해사업연도")
    previous: Optional[str] = Field(None, alias="직전사업연도")
    change: Optional[str] = Field(None, alias="증감금액")
    change_ratio: Optional[str] = Field(None, alias="증감비율(%)")
    turnaround: Optional[str] = Field(None, alias="흑자적자전환여부")

    @property
    def current_int(self) -> Optional[int]:
        return parse_korean_number(self.current)

    @property
    def previous_int(self) -> Optional[int]:
        return parse_korean_number(self.previous)


class FinancialStatus(BaseModel):
    """재무현황."""
    current: Optional[str] = Field(None, alias="당해사업연도")
    previous: Optional[str] = Field(None, alias="직전사업연도")

    @property
    def current_int(self) -> Optional[int]:
        return parse_korean_number(self.current)

    @property
    def previous_int(self) -> Optional[int]:
        return parse_korean_number(self.previous)


class Performance(BaseModel):
    """매출액또는손익구조30%(대규모법인은15%)이상변동 공시."""
    statement_type: Optional[str] = Field(None, alias="1. 재무제표의 종류")
    period: Optional[dict] = Field(None, alias="2. 결산기간")
    income_changes: Optional[dict] = Field(None, alias="3. 매출액 또는 손익구조변동내용(단위: 원)")
    financial_status: Optional[dict] = Field(None, alias="4. 재무현황(단위 : 원)")
    change_reason: Optional[str] = Field(None, alias="5. 매출액 또는 손익구조 변동 주요원인")
    unit: str = "원"  # Detected unit: 원, 천원, 백만원, 억원

    model_config = {"populate_by_name": True}

    @property
    def revenue(self) -> Optional[IncomeItem]:
        if self.income_changes:
            data = self.income_changes.get("- 매출액") or self.income_changes.get("매출액")
            if data:
                return IncomeItem(**data)
        return None

    @property
    def operating_income(self) -> Optional[IncomeItem]:
        if self.income_changes:
            data = self.income_changes.get("- 영업이익") or self.income_changes.get("영업이익")
            if data:
                return IncomeItem(**data)
        return None

    @property
    def net_income(self) -> Optional[IncomeItem]:
        if self.income_changes:
            data = self.income_changes.get("- 당기순이익") or self.income_changes.get("당기순이익")
            if data:
                return IncomeItem(**data)
        return None

    @property
    def assets(self) -> Optional[FinancialStatus]:
        if self.financial_status:
            data = self.financial_status.get("- 자산총계") or self.financial_status.get("자산총계")
            if data:
                return FinancialStatus(**data)
        return None

    @property
    def liabilities(self) -> Optional[FinancialStatus]:
        if self.financial_status:
            data = self.financial_status.get("- 부채총계") or self.financial_status.get("부채총계")
            if data:
                return FinancialStatus(**data)
        return None

    @property
    def equity(self) -> Optional[FinancialStatus]:
        if self.financial_status:
            data = self.financial_status.get("- 자본총계") or self.financial_status.get("자본총계")
            if data:
                return FinancialStatus(**data)
        return None


# ──────────────────────────────────────────────
# issue.json - CB/BW 발행결정 (주요사항보고서)
# ──────────────────────────────────────────────

class BondType(BaseModel):
    """사채 종류."""
    series: Optional[int] = Field(None, alias="회차")
    kind: Optional[str] = Field(None, alias="종류")


class ConversionTerms(BaseModel):
    """전환에 관한 사항."""
    conversion_price: Optional[int] = Field(None, alias="전환가액 (원/주)")
    share_info: Optional[dict] = Field(None, alias="전환에_따라_발행할_주식")
    request_period: Optional[dict] = Field(None, alias="전환청구기간")

    @property
    def share_type(self) -> Optional[str]:
        if self.share_info:
            return self.share_info.get("종류")
        return None

    @property
    def share_count(self) -> Optional[int]:
        if self.share_info:
            return self.share_info.get("주식수")
        return None

    @property
    def share_ratio(self) -> Optional[float]:
        if self.share_info:
            return self.share_info.get("주식총수 대비 비율(%)")
        return None

    @property
    def request_start(self) -> Optional[str]:
        if self.request_period:
            return self.request_period.get("시작일")
        return None

    @property
    def request_end(self) -> Optional[str]:
        if self.request_period:
            return self.request_period.get("종료일")
        return None


class CBIssuance(BaseModel):
    """전환사채/신주인수권부사채 발행결정 공시."""
    issuance_type_name: str = ""  # e.g. "전환사채권 발행결정"
    bond_type: Optional[BondType] = Field(None, alias="1. 사채의 종류")
    face_value: Optional[int] = Field(None, alias="2. 사채의 권면(전자등록)총액 (원)")
    conversion_terms: Optional[ConversionTerms] = Field(None, alias="9. 전환에 관한 사항")
    disclosure_date: str = ""  # e.g. "2024.06.28" - set by pipeline from rcept_dt

    model_config = {"populate_by_name": True}
