"""Pydantic models for financial statement and market data."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BalanceSheetItem(BaseModel):
    """Single balance sheet line item."""
    label: str
    amount: Optional[int] = None  # in won
    previous_amount: Optional[int] = None  # previous period, in won
    note: str = ""


class BalanceSheet(BaseModel):
    """Balance sheet data for a single period."""
    period: str = ""  # e.g. "2025.3Q", "2024"
    statement_type: str = "연결"  # 연결 or 별도
    currency: str = "KRW"  # e.g. "KRW", "CNY", "USD"
    total_assets: Optional[int] = None
    cash_and_equivalents: Optional[int] = None
    short_term_investments: Optional[int] = None
    total_liabilities: Optional[int] = None
    short_term_borrowings: Optional[int] = None  # 단기차입금
    current_long_term_debt: Optional[int] = None  # 유동성장기부채
    current_bonds: Optional[int] = None  # 유동성사채
    long_term_borrowings: Optional[int] = None  # 장기차입금
    bonds: Optional[int] = None  # 사채
    short_term_debt_and_bonds: Optional[int] = None  # 단기차입금및사채 (통합)
    long_term_debt_and_bonds: Optional[int] = None  # 장기차입금및사채 (통합)
    total_equity: Optional[int] = None


class IncomeStatementItem(BaseModel):
    """Single income statement item with YoY change."""
    period: str = ""  # e.g. "2025", "2025.3Q"
    currency: str = "KRW"  # e.g. "KRW", "CNY", "USD"
    revenue: Optional[int] = None
    operating_income: Optional[int] = None
    net_income: Optional[int] = None
    revenue_yoy: Optional[str] = None  # formatted string like "+276%"
    operating_income_yoy: Optional[str] = None
    net_income_yoy: Optional[str] = None


class ConsensusItem(BaseModel):
    """Naver consensus estimate for a future period (억원 → 원 변환 후 저장)."""
    period: str = ""          # "2026(E)"
    revenue: Optional[int] = None          # 원 단위
    operating_income: Optional[int] = None
    net_income: Optional[int] = None


class MarketData(BaseModel):
    """Market data from Naver Finance API."""
    date: str = ""
    stock_price: Optional[int] = None
    market_cap: Optional[int] = None  # in won
    shares_outstanding: Optional[int] = None
    pbr: Optional[float] = None
    per: Optional[float] = None
    estimated_per: Optional[float] = None
    estimated_eps: Optional[int] = None
