"""Unified configuration: merges opendart env-var getters and finance-parser pydantic settings."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


# ---------------------------------------------------------------------------
# Global Settings (from .env)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Global application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DART API
    dart_api_key: str = ""
    dart_start_date: str = "20200101"
    dart_keywords: str = (
        "사업보고서;분기보고서;반기보고서;"
        "단일판매ㆍ공급계약체결;주요사항보고서;"
        "전환청구권;신주인수권;교환청구권;"
        "전환가액;행사가액;교환가액;"
        "매출액또는손익구조;재무제표기준영업(잠정)실적;"
        "증권신고서;신규시설투자등;감사보고서"
    )

    # FnGuide
    fnguide_id: str = ""
    fnguide_pw: str = ""

    # News
    news_days_back: int = 90

    # OpenAI / LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    @property
    def llm_api_key(self) -> str:
        """Active LLM API key (Gemini preferred over OpenAI)."""
        return self.gemini_api_key or self.openai_api_key

    @property
    def llm_base_url(self) -> str:
        """Base URL for LLM API (empty = OpenAI default)."""
        if self.gemini_api_key:
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return ""

    @property
    def llm_model(self) -> str:
        """Default LLM model (auto-selects Gemini model when using Gemini API)."""
        if self.gemini_api_key and self.openai_model == "gpt-4.1-mini":
            return self.gemini_model
        return self.openai_model

    # Paths
    stocks_json: str = "./stocks.json"
    stocks_base_dir: str = str(Path(tempfile.gettempdir()) / "auto-reports")
    output_dir: str = str(Path(tempfile.gettempdir()) / "auto-reports")
    obsidian_inbox: str = ""
    obsidian_attachments: str = ""

    # Request
    request_delay: float = 1.0
    request_timeout: int = 30

    # --- Helper methods (replace opendart functional getters) ---

    def get_dart_keywords_list(self) -> list[str]:
        """Return DART keywords as a list."""
        return [k.strip() for k in self.dart_keywords.split(";") if k.strip()]

    def get_news_days_back(self) -> int:
        """Return news_days_back clamped to [1, 365]."""
        return max(1, min(self.news_days_back, 365))

    def validate_collector_config(self) -> None:
        """Raise if DART_API_KEY is missing (required for collection)."""
        if not self.dart_api_key:
            raise ConfigurationError(
                "DART_API_KEY is required for collection. "
                "Set it in .env or as an environment variable."
            )

    def validate_fnguide_config(self) -> None:
        """Raise if FnGuide credentials are missing."""
        missing = []
        if not self.fnguide_id:
            missing.append("FNGUIDE_ID")
        if not self.fnguide_pw:
            missing.append("FNGUIDE_PW")
        if missing:
            raise ConfigurationError(
                f"Required environment variables not set: {', '.join(missing)}. "
                "Set them in .env for FnGuide collection."
            )


# ---------------------------------------------------------------------------
# Per-company YAML config models (from finance-parser)
# ---------------------------------------------------------------------------

class CompanyConfig(BaseModel):
    """Per-company configuration loaded from YAML."""
    name: str
    ticker: str
    corp_code: str = ""
    tags: list[str] = Field(default_factory=list)


class ReportConfig(BaseModel):
    """Report generation settings."""
    output_dir: str = ""
    years: int = 5
    quarters: int = 8
    statement_type: str = "연결"


class DisclosureConfig(BaseModel):
    """Disclosure URL configuration."""
    urls_file: str = ""
    urls: list[str] = Field(default_factory=list)
    exchange_disclosures_file: str = ""

    def load_urls(self, base_dir: Path) -> list[str]:
        """Load disclosure URLs from file or inline list."""
        if self.urls:
            return self.urls
        if self.urls_file:
            urls_path = base_dir / self.urls_file
            if urls_path.exists():
                lines = urls_path.read_text(encoding="utf-8").strip().splitlines()
                return [line.strip() for line in lines if line.strip() and line.strip().startswith("http")]
        return []

    def load_exchange_disclosures(self, base_dir: Path) -> list[dict]:
        """Load exchange disclosure entries from JSON file."""
        if not self.exchange_disclosures_file:
            return []
        path = base_dir / self.exchange_disclosures_file
        if not path.exists():
            return []
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        return [e for e in data if isinstance(e, dict) and e.get("url")]


class AnalysisConfig(BaseModel):
    """Analysis and LLM settings for sections 5-7."""
    prompt_file: str = ""
    reports_dir: str = ""
    news_file: str = ""
    analysis_model: str = ""


class CompanyYamlConfig(BaseModel):
    """Full YAML config file structure."""
    company: CompanyConfig
    report: ReportConfig = Field(default_factory=ReportConfig)
    disclosures: DisclosureConfig = Field(default_factory=DisclosureConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CompanyYamlConfig:
        """Load company config from a YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def resolve_stock_dir(self, stocks_base_dir: str) -> Path | None:
        """Return the per-company stock data directory, or None if it doesn't exist."""
        if not stocks_base_dir:
            return None
        stock_dir = Path(stocks_base_dir) / self.company.name
        return stock_dir if stock_dir.is_dir() else None

    def ensure_defaults(self) -> None:
        """Set default values that don't depend on stock_dir existence."""
        if not self.analysis.prompt_file:
            self.analysis.prompt_file = "prompts/analysis_prompt.md"

    def auto_fill_from_stock_dir(self, stock_dir: Path) -> None:
        """Fill empty config fields by auto-detecting files in stock_dir."""
        if not stock_dir or not stock_dir.is_dir():
            return

        # disclosures.urls - auto-detect DART filing URLs from PDF filenames
        if not self.disclosures.urls_file and not self.disclosures.urls:
            dart_urls = _extract_dart_urls_from_pdfs(stock_dir)
            if dart_urls:
                self.disclosures.urls = dart_urls

        # disclosures.urls_file
        if not self.disclosures.urls_file and not self.disclosures.urls:
            candidates = sorted(stock_dir.glob("*거래소공시*list*.txt"))
            if candidates:
                self.disclosures.urls_file = str(candidates[0])

        # disclosures.exchange_disclosures_file
        if not self.disclosures.exchange_disclosures_file:
            candidates = sorted(stock_dir.glob("*exchange_disclosure*.json"))
            if candidates:
                self.disclosures.exchange_disclosures_file = str(candidates[0])

        # analysis.reports_dir
        if not self.analysis.reports_dir:
            reports_sub = stock_dir / "reports"
            if reports_sub.is_dir():
                self.analysis.reports_dir = str(reports_sub)

        # analysis.news_file (pick latest by sorted filename)
        if not self.analysis.news_file:
            candidates = sorted(stock_dir.glob("*news*.json"))
            if candidates:
                self.analysis.news_file = str(candidates[-1])

        # analysis.prompt_file (default set in ensure_defaults)


_PDF_RCPNO_PATTERN = re.compile(
    r"^\d{8}_\[(?:Filing|공시)\]_(\d+)_.*주요사항보고서.*\.pdf$"
)


def _extract_dart_urls_from_pdfs(stock_dir: Path) -> list[str]:
    """Extract DART disclosure URLs from PDF filenames in stock_dir."""
    urls = []
    seen: set[str] = set()
    for pdf in sorted(stock_dir.glob("*.pdf")):
        m = _PDF_RCPNO_PATTERN.match(pdf.name)
        if m:
            rcpno = m.group(1)
            if rcpno not in seen:
                seen.add(rcpno)
                urls.append(
                    f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcpno}"
                )
    return urls
