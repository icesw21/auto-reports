"""Unified Click CLI for auto-reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="auto-reports")
def cli():
    """Korean stock financial data collector, parser, and report generator."""


# ---------------------------------------------------------------------------
# download (데이터 다운로드만)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--stocks", "stocks_path", default=None, help="Path to stocks.json.")
@click.option("--company", "-c", default=None, help="Download for a single company only (by name).")
@click.option("--skip-fnguide", is_flag=True, help="Skip FnGuide collection (no Selenium).")
@click.option("--skip-news", is_flag=True, help="Skip news collection.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--workers", "-w", default=1, type=click.IntRange(min=1), help="Number of parallel workers (default: 1=sequential). Use --skip-fnguide with -w>1.")
def download(stocks_path: str | None, company: str | None, skip_fnguide: bool, skip_news: bool, verbose: bool, workers: int):
    """Download data only: DART filings, research PDFs, and news.

    \b
    Examples:
      auto-reports download                      # All stocks
      auto-reports download -c 광동제약           # Single company
      auto-reports download --skip-fnguide       # Without FnGuide (no Selenium)
      auto-reports download -w 3                 # 3 parallel workers
    """
    from auto_reports.config import Settings
    from auto_reports.utils.logging import setup_logging
    from auto_reports.orchestrator import run_collect

    settings = Settings()
    if stocks_path:
        settings.stocks_json = stocks_path

    setup_logging(log_dir=settings.output_dir, log_level="DEBUG" if verbose else "INFO")

    try:
        results = run_collect(
            settings,
            skip_fnguide=skip_fnguide,
            skip_news=skip_news,
            company_filter=company,
            max_workers=workers,
        )
        success = sum(1 for v in results.values() if v)
        console.print(f"\n[bold]Download complete: {success}/{len(results)} companies succeeded.[/bold]")
        console.print(f"Data saved to: {settings.output_dir}/")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# analyze (종목 분석만 - init + batch)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--stocks", "stocks_path", default=None, help="Path to stocks.json.")
@click.option("--company", "-c", default=None, help="Analyze a single company only (by name).")
@click.option("--no-copy", is_flag=True, help="Skip copying reports to Obsidian inbox.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--workers", "-w", default=1, type=click.IntRange(min=1), help="Number of parallel workers (default: 1=sequential). Use --skip-fnguide with -w>1.")
def analyze(stocks_path: str | None, company: str | None, no_copy: bool, verbose: bool, workers: int):
    """Analyze stocks and generate reports only (no data download).

    Runs init (generate configs) + batch (generate reports) in sequence.
    Assumes data has already been downloaded via 'auto-reports download'.

    \b
    Examples:
      auto-reports analyze                       # All stocks
      auto-reports analyze -c 광동제약            # Single company
      auto-reports analyze --no-copy             # Skip Obsidian copy
      auto-reports analyze -w 3                  # 3 parallel workers
    """
    from auto_reports.config import Settings
    from auto_reports.orchestrator import run_analyze

    settings = Settings()
    if stocks_path:
        settings.stocks_json = stocks_path

    try:
        run_analyze(
            settings,
            no_copy=no_copy,
            verbose=verbose,
            company_filter=company,
            max_workers=workers,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# collect (alias for download, kept for compatibility)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--stocks", "stocks_path", default=None, help="Path to stocks.json.")
@click.option("--company", "-c", default=None, help="Collect for a single company only (by name).")
@click.option("--skip-fnguide", is_flag=True, help="Skip FnGuide collection (no Selenium).")
@click.option("--skip-news", is_flag=True, help="Skip news collection.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--workers", "-w", default=1, type=click.IntRange(min=1), help="Number of parallel workers (default: 1=sequential). Use --skip-fnguide with -w>1.")
@click.pass_context
def collect(ctx, stocks_path, company, skip_fnguide, skip_news, verbose, workers):
    """Collect DART filings, research PDFs, and news (alias for 'download')."""
    ctx.invoke(download, stocks_path=stocks_path, company=company,
               skip_fnguide=skip_fnguide, skip_news=skip_news, verbose=verbose, workers=workers)


# ---------------------------------------------------------------------------
# init (from finance-parser)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("names", nargs=-1)
@click.option("--all", "init_all", is_flag=True, help="Generate configs for all stocks in stocks.json.")
@click.option("--tags", "-t", multiple=True, help="Tags to add (e.g. -t 2차전지 -t KOSDAQ).")
@click.option("--statement-type", default="연결", help="Financial statement type (연결/별도).")
def init(names: tuple[str, ...], init_all: bool, tags: tuple[str, ...], statement_type: str):
    """Generate YAML config files from stocks.json.

    \b
    Examples:
      auto-reports init 광동제약 보성파워텍
      auto-reports init --all
      auto-reports init 광동제약 -t 2차전지 -t KOSDAQ
    """
    from auto_reports.config import Settings
    from auto_reports.orchestrator import run_init_all

    settings = Settings()

    if init_all:
        run_init_all(settings, tags=tags, statement_type=statement_type)
        return

    if not names:
        console.print("[yellow]Specify stock names or use --all.[/yellow]")
        sys.exit(1)

    # Resolve names/tickers to a {ticker: name} dict
    stocks_json_path = Path(settings.stocks_json)
    if not stocks_json_path.is_file():
        console.print(f"[red]stocks.json not found: {stocks_json_path}[/red]")
        sys.exit(1)

    stocks: dict[str, str] = json.loads(stocks_json_path.read_text(encoding="utf-8"))
    name_to_ticker = {name: ticker for ticker, name in stocks.items()}

    target_stocks: dict[str, str] = {}
    for name in names:
        if name in name_to_ticker:
            target_stocks[name_to_ticker[name]] = name
        elif name in stocks:
            target_stocks[name] = stocks[name]
        else:
            console.print(f"[yellow]'{name}' not found in stocks.json, skipping[/yellow]")

    if not target_stocks:
        sys.exit(1)

    run_init_all(settings, tags=tags, statement_type=statement_type, target_stocks=target_stocks)


# ---------------------------------------------------------------------------
# generate (from finance-parser)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, help="Override output directory.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--dry-run", is_flag=True, help="Parse and analyze without writing report.")
@click.option("--no-copy", is_flag=True, help="Skip writing report to Obsidian inbox.")
@click.option("--force-overwrite", is_flag=True, help="Overwrite existing notes entirely (skip smart merge).")
def generate(config_path: str, output_dir: str | None, verbose: bool, dry_run: bool, no_copy: bool, force_overwrite: bool):
    """Generate a report from a company config YAML file.

    Example: auto-reports generate config/광동제약.yaml
    """
    from auto_reports.config import Settings
    from auto_reports.pipeline import run_pipeline

    settings = Settings()

    # Write directly to Obsidian inbox unless overridden
    effective_output_dir = output_dir
    if not effective_output_dir and settings.obsidian_inbox and not no_copy:
        effective_output_dir = settings.obsidian_inbox

    try:
        result = run_pipeline(
            config_path=config_path,
            output_dir=effective_output_dir,
            verbose=verbose,
            dry_run=dry_run,
            force_overwrite=force_overwrite,
        )
        if result:
            console.print(f"\nDone. Report at: {result}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# batch (from finance-parser)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True))
@click.option("--all", "run_all", is_flag=True, help="Run all YAML configs in config/ directory.")
@click.option("--output-dir", "-o", default=None, help="Override output directory.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--no-copy", is_flag=True, help="Skip copying reports to Obsidian inbox.")
@click.option("--workers", "-w", default=1, type=click.IntRange(min=1), help="Number of parallel workers (default: 1=sequential). Use --skip-fnguide with -w>1.")
@click.option("--force-overwrite", is_flag=True, help="Overwrite existing notes entirely (skip smart merge).")
def batch(config_paths: tuple[str, ...], run_all: bool, output_dir: str | None, verbose: bool, no_copy: bool, workers: int, force_overwrite: bool):
    """Generate reports for multiple companies at once.

    \b
    Examples:
      auto-reports batch --all
      auto-reports batch --all -w 3              # 3 parallel workers
      auto-reports batch --all --no-copy
    """
    from auto_reports.config import Settings
    from auto_reports.orchestrator import run_batch_all

    settings = Settings()

    if run_all:
        run_batch_all(settings, no_copy=no_copy, verbose=verbose, output_dir=output_dir, max_workers=workers, force_overwrite=force_overwrite)
        return

    if not config_paths:
        console.print("[yellow]No config files specified. Use --all or provide paths.[/yellow]")
        sys.exit(1)

    # Individual config paths: use orchestrator with explicit config list
    from auto_reports.pipeline import run_pipeline

    # Write directly to Obsidian inbox unless overridden
    effective_output_dir = output_dir
    if not effective_output_dir and settings.obsidian_inbox and not no_copy:
        effective_output_dir = settings.obsidian_inbox

    configs = [Path(p) for p in config_paths]

    console.print(f"\n[bold]Batch processing {len(configs)} companies[/bold]\n")

    results: list[tuple[str, Path | None, str | None]] = []

    for i, config_path in enumerate(configs, 1):
        name = config_path.stem
        console.print(f"\n{'─' * 60}")
        console.print(f"[bold][{i}/{len(configs)}] {name}[/bold]")
        console.print(f"{'─' * 60}")

        try:
            result = run_pipeline(
                config_path=str(config_path),
                output_dir=effective_output_dir,
                verbose=verbose,
                dry_run=False,
            )
            if result:
                console.print(f"  [green]Report generated[/green]: {result}")
                results.append((name, result, None))
            else:
                results.append((name, None, "No output"))
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()
            results.append((name, None, str(e)))

    # Summary
    console.print(f"\n{'═' * 60}")
    console.print("[bold]Batch Summary[/bold]\n")
    success = sum(1 for _, p, e in results if p and not e)
    failed = sum(1 for _, _, e in results if e)
    console.print(f"  Total: {len(results)}, Success: {success}, Failed: {failed}")
    for name, path, error in results:
        if error:
            console.print(f"  [red]x[/red] {name}: {error}")
        else:
            console.print(f"  [green]v[/green] {name}: {path}")
    console.print()


# ---------------------------------------------------------------------------
# run (NEW orchestrator)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--stocks", "stocks_path", default=None, help="Path to stocks.json.")
@click.option("--company", "-c", default=None, help="Run pipeline for a single company only (by name).")
@click.option("--skip-fnguide", is_flag=True, help="Skip FnGuide during collection.")
@click.option("--skip-news", is_flag=True, help="Skip news during collection.")
@click.option("--no-copy", is_flag=True, help="Skip Obsidian copy after report generation.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--workers", "-w", default=1, type=click.IntRange(min=1), help="Number of parallel workers for collection and report generation (default: 1=sequential).")
def run(stocks_path: str | None, company: str | None, skip_fnguide: bool, skip_news: bool, no_copy: bool, verbose: bool, workers: int):
    """Execute the full pipeline: collect -> init -> batch.

    This is the main unified command that runs the entire workflow.

    \b
    Example:
      auto-reports run
      auto-reports run -c 광동제약
      auto-reports run -w 3                      # 3 parallel workers
    """
    from auto_reports.config import Settings
    from auto_reports.orchestrator import run_full_pipeline

    settings = Settings()
    if stocks_path:
        settings.stocks_json = stocks_path

    try:
        run_full_pipeline(
            settings,
            skip_fnguide=skip_fnguide,
            skip_news=skip_news,
            no_copy=no_copy,
            verbose=verbose,
            company_filter=company,
            max_workers=workers,
        )
    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# parse (debug, from finance-parser)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("url")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def parse(url: str, verbose: bool):
    """Parse a single DART disclosure URL and print the result as JSON.

    Example: auto-reports parse https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260202901050
    """
    from auto_reports.utils.logging import setup_logging
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    from auto_reports.fetchers.dart_html import DartHtmlFetcher
    from auto_reports.parsers.classifier import DisclosureType, classify_disclosure
    from auto_reports.parsers import (
        parse_convert, parse_convert_price, parse_contract,
        parse_performance, parse_issue,
    )

    parser_map = {
        DisclosureType.CONVERT: parse_convert,
        DisclosureType.CONVERT_PRICE_CHANGE: parse_convert_price,
        DisclosureType.CONTRACT: parse_contract,
        DisclosureType.PERFORMANCE: parse_performance,
        DisclosureType.ISSUE: parse_issue,
    }

    try:
        fetcher = DartHtmlFetcher()
        title, soup = fetcher.fetch_disclosure(url)
        disc_type = classify_disclosure(title)

        console.print(f"[bold]Title:[/bold] {title}")
        console.print(f"[bold]Type:[/bold] {disc_type.name}")

        if disc_type == DisclosureType.UNKNOWN:
            console.print("[yellow]Unknown disclosure type - cannot parse[/yellow]")
            return

        parser_fn = parser_map.get(disc_type)
        if parser_fn:
            result = parser_fn(soup)
            output = result.model_dump(by_alias=True, exclude_none=True)
            console.print(json.dumps(output, ensure_ascii=False, indent=2))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# fetch (debug, from finance-parser)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("ticker")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def fetch(ticker: str, verbose: bool):
    """Fetch financial data for a stock ticker (debug command).

    Example: auto-reports fetch 171090
    """
    from auto_reports.utils.logging import setup_logging
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    from auto_reports.config import Settings
    from auto_reports.fetchers.market_data import MarketDataFetcher

    console.print(f"\n[bold]Fetching data for ticker: {ticker}[/bold]\n")

    console.print("[dim]Market data...[/dim]")
    try:
        market_fetcher = MarketDataFetcher()
        md = market_fetcher.get_market_data(ticker)
        console.print(f"  Price: {md.stock_price:,}원" if md.stock_price else "  Price: N/A")
        console.print(f"  Market Cap: {md.market_cap:,}원" if md.market_cap else "  Market Cap: N/A")
        console.print(f"  Shares: {md.shares_outstanding:,}" if md.shares_outstanding else "  Shares: N/A")
    except Exception as e:
        console.print(f"  [yellow]Failed: {e}[/yellow]")

    settings = Settings()
    if settings.dart_api_key:
        console.print("\n[dim]OpenDART financial data...[/dim]")
        try:
            from auto_reports.fetchers.opendart import OpenDartFetcher
            dart_fetcher = OpenDartFetcher(settings.dart_api_key)
            corp_code = dart_fetcher.resolve_corp_code(ticker)
            console.print(f"  Corp code: {corp_code}")

            if corp_code:
                from datetime import datetime
                year = datetime.now().year - 1
                bs = dart_fetcher.get_balance_sheet(corp_code, year)
                console.print(f"  Assets: {bs.total_assets:,}원" if bs.total_assets else "  Assets: N/A")
                console.print(f"  Equity: {bs.total_equity:,}원" if bs.total_equity else "  Equity: N/A")
        except Exception as e:
            console.print(f"  [yellow]Failed: {e}[/yellow]")
    else:
        console.print("\n[yellow]No DART_API_KEY set - skipping OpenDART[/yellow]")


if __name__ == "__main__":
    cli()
