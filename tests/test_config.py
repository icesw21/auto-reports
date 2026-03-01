"""Tests for CompanyYamlConfig auto-fill from stock directory."""

from auto_reports.config import (
    AnalysisConfig,
    CompanyConfig,
    CompanyYamlConfig,
    DisclosureConfig,
    ReportConfig,
)


def _make_config(**overrides) -> CompanyYamlConfig:
    """Create a minimal CompanyYamlConfig for testing."""
    return CompanyYamlConfig(
        company=CompanyConfig(name="테스트종목", ticker="000000"),
        report=ReportConfig(),
        disclosures=overrides.get("disclosures", DisclosureConfig()),
        analysis=overrides.get("analysis", AnalysisConfig()),
    )


def test_auto_fill_urls_file(tmp_path):
    """Should detect *거래소공시*list*.txt file."""
    (tmp_path / "테스트종목_거래소공시_list.txt").write_text("http://example.com", encoding="utf-8")
    cfg = _make_config()
    cfg.auto_fill_from_stock_dir(tmp_path)
    assert "거래소공시" in cfg.disclosures.urls_file


def test_auto_fill_exchange_disclosure(tmp_path):
    """Should detect *exchange_disclosure*.json file."""
    (tmp_path / "테스트종목_exchange_disclosure.json").write_text("[]", encoding="utf-8")
    cfg = _make_config()
    cfg.auto_fill_from_stock_dir(tmp_path)
    assert "exchange_disclosure" in cfg.disclosures.exchange_disclosures_file


def test_auto_fill_reports_dir(tmp_path):
    """Should detect reports/ subfolder."""
    (tmp_path / "reports").mkdir()
    cfg = _make_config()
    cfg.auto_fill_from_stock_dir(tmp_path)
    assert cfg.analysis.reports_dir.endswith("reports")


def test_auto_fill_news_file(tmp_path):
    """Should detect *news*.json file."""
    (tmp_path / "news.json").write_text("[]", encoding="utf-8")
    cfg = _make_config()
    cfg.auto_fill_from_stock_dir(tmp_path)
    assert "news" in cfg.analysis.news_file


def test_auto_fill_prompt_file_default(tmp_path):
    """Should set default prompt_file via ensure_defaults, independent of stock_dir."""
    cfg = _make_config()
    cfg.ensure_defaults()
    assert cfg.analysis.prompt_file == "prompts/analysis_prompt.md"


def test_auto_fill_does_not_overwrite_explicit(tmp_path):
    """Explicit YAML values should NOT be overwritten."""
    (tmp_path / "테스트종목_거래소공시_list.txt").write_text("http://example.com", encoding="utf-8")
    (tmp_path / "reports").mkdir()

    cfg = _make_config(
        disclosures=DisclosureConfig(urls_file="my_custom_urls.txt"),
        analysis=AnalysisConfig(
            reports_dir="my_custom_reports",
            prompt_file="my_prompt.md",
        ),
    )
    cfg.auto_fill_from_stock_dir(tmp_path)

    assert cfg.disclosures.urls_file == "my_custom_urls.txt"
    assert cfg.analysis.reports_dir == "my_custom_reports"
    assert cfg.analysis.prompt_file == "my_prompt.md"


def test_auto_fill_empty_dir(tmp_path):
    """Empty directory should leave all fields empty."""
    cfg = _make_config()
    cfg.auto_fill_from_stock_dir(tmp_path)
    assert cfg.disclosures.urls_file == ""
    assert cfg.disclosures.exchange_disclosures_file == ""
    assert cfg.analysis.reports_dir == ""
    assert cfg.analysis.news_file == ""
    # prompt_file default is now set by ensure_defaults(), not auto_fill
    assert cfg.analysis.prompt_file == ""


def test_resolve_stock_dir_exists(tmp_path):
    """Should return stock_dir when the company folder exists."""
    (tmp_path / "테스트종목").mkdir()
    cfg = _make_config()
    result = cfg.resolve_stock_dir(str(tmp_path))
    assert result is not None
    assert result.name == "테스트종목"


def test_resolve_stock_dir_missing(tmp_path):
    """Should return None when the company folder doesn't exist."""
    cfg = _make_config()
    result = cfg.resolve_stock_dir(str(tmp_path))
    assert result is None
