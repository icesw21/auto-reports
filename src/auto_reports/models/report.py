"""Pydantic models for report sections."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OverhangItem(BaseModel):
    """Single overhang/dilutive instrument entry."""
    category: str  # e.g. "제1회 전환사채(CB)"
    remaining_amount: Optional[str] = None  # e.g. "877,346주"
    exercise_price: Optional[str] = None  # e.g. "49,591원"
    dilution_ratio: Optional[str] = None  # e.g. "5.44%"
    exercise_period: str = ""  # e.g. "2024.03.15~2029.03.14"


class ExchangeContract(BaseModel):
    """Exchange disclosure contract entry (단일판매ㆍ공급계약체결/해지)."""
    contract_type: str = ""  # "체결" or "해지"
    description: str = ""  # 공급계약 내용
    detail: str = ""  # 계약명
    amount_eok: str = ""  # formatted in 억원
    counterparty: str = ""
    revenue_ratio_pct: Optional[str] = None  # e.g. "21.14%"
    date: str = ""  # 계약/해지일
    contract_period: str = ""  # e.g. "2025-04-23~2025-12-19"


class ReportFrontmatter(BaseModel):
    """YAML frontmatter for Obsidian."""
    created: str = ""
    updated: str = ""
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class BalanceSheetRow(BaseModel):
    """Balance sheet table row."""
    item: str
    amount: str  # formatted in 억원
    previous_amount: str = ""  # previous period formatted in 억원
    note: str = ""


class AnnualRow(BaseModel):
    """Annual income statement row."""
    year: str
    revenue: str  # e.g. "4,245 (+276%)"
    operating_income: str
    net_income: str


class QuarterlyRow(BaseModel):
    """Quarterly income statement row."""
    quarter: str  # e.g. "2025.4Q"
    revenue: str
    operating_income: str
    net_income: str


class ReportData(BaseModel):
    """All data needed to generate a report."""
    frontmatter: ReportFrontmatter = Field(default_factory=ReportFrontmatter)

    # Section 1: 기본사항
    market_cap_str: str = ""  # e.g. "9,296 억원 (주가 96,800원 × 발행주식수 9,602,955주, 2026.02.12 기준)"
    latest_pbr_str: str = ""  # e.g. "7.80배 (시가총액 9,296억원 / 자본총계 1,095억원)"
    trailing_per_str: str = ""  # e.g. "29.1배 (시가총액 9,296억원 / 2025년 순이익 320억원)"
    estimated_per_str: str = ""  # e.g. "9.3배 (시가총액 9,296억원 / 2026E 순이익 1,000억원)"
    overhang_items: list[OverhangItem] = Field(default_factory=list)
    overhang_note: str = ""  # footnote below overhang table
    total_shares: Optional[int] = None
    fully_diluted_market_cap_str: str = ""  # e.g. "1,234억원 (희석주식 포함 총 25,000,000주 기준)"
    shareholder_info: str = ""  # e.g. "최대주주 이종서 등 13.62%"
    dividend_info: str = ""  # e.g. "주당 500원 배당 실시 (현금배당성향 20.5%, 현금배당총액 100억원)"
    subsidiary_info: str = ""  # e.g. "A사, B사" or "해당없음"

    # Section 1: 주가 차트
    chart_image: str = ""  # e.g. "chart_005930_20260228.png"

    # Section 1: 주가 정보
    price_period: str = ""  # e.g. "2025.03.07 ~ 2026.02.26"
    price_start: str = ""  # e.g. "11,100원"
    price_current: str = ""  # e.g. "31,450원"
    price_low: str = ""  # e.g. "7,650원 (2025.04.09)"
    price_high: str = ""  # e.g. "31,450원 (2026.02.26)"
    price_return_1y: str = ""  # e.g. "+183.3%"
    price_return_label: str = "1년수익률"  # "1년수익률" or "상장후수익률"
    momentum_text: str = ""  # 주요 모멘텀 (LLM-generated in feedback loop)

    # Section 2: 재무상태표
    balance_sheet_period: str = ""  # e.g. "2025년 3분기말 연결 기준"
    balance_sheet_prev_period: str = ""  # e.g. "2024년말"
    balance_sheet_rows: list[BalanceSheetRow] = Field(default_factory=list)
    balance_sheet_note: str = ""

    # Section 3: 손익계산서
    annual_rows: list[AnnualRow] = Field(default_factory=list)
    consensus_rows: list[AnnualRow] = Field(default_factory=list)
    consensus_per_str: str = ""
    quarterly_rows: list[QuarterlyRow] = Field(default_factory=list)
    quarterly_note: str = ""

    # Section 4: 사업 모델
    business_model: str = ""
    major_customers: str = ""
    major_suppliers: str = ""
    revenue_breakdown: str = ""
    order_backlog: str = ""
    exchange_contracts: list[ExchangeContract] = Field(default_factory=list)
    exchange_forecasts: list[dict] = Field(default_factory=list)  # 실적전망 공정공시
    business_source: str = ""  # e.g. "사업보고서 (2024.12) 기준"
    value_driver: str = ""  # Value Driver analysis
    competitor_comparison: str = ""  # 경쟁사 비교

    # Section 5: 투자 아이디어
    investment_ideas: str = ""  # LLM-generated investment thesis

    # Section 6: 리스크
    risk_structural: str = ""  # 구조적 위험
    risk_counter: str = ""  # 반론 제기
    risk_tail: str = ""  # Tail Risk

    # Section 7: 결론
    conclusion: str = ""  # LLM-generated conclusion

    # Company info
    company_name: str = ""
