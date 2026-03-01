"""Orchestrator: chains collect -> init -> batch for the unified 'run' command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

from auto_reports.config import Settings
from auto_reports.utils.logging import setup_logging, get_logger

console = Console()


def run_collect(
    settings: Settings,
    skip_fnguide: bool = False,
    skip_news: bool = False,
    skip_naver: bool = True,
    company_filter: str | None = None,
) -> dict[str, bool]:
    """Execute data collection for all (or filtered) stocks.

    Returns dict of {company_name: success_bool}.
    """
    from auto_reports.collectors import DartCollector, FnGuideCollector, NaverCollector, NewsCollector
    from auto_reports.utils.file_utils import load_stock_json

    logger = get_logger("orchestrator.collect")

    settings.validate_collector_config()

    # Load target stocks
    target_dict = load_stock_json(settings.stocks_json)
    if not target_dict:
        console.print("[red]No stocks loaded. Check stocks.json.[/red]")
        return {}

    # Optional single-company filter
    if company_filter:
        filtered = {k: v for k, v in target_dict.items() if v == company_filter}
        if not filtered:
            console.print(f"[yellow]Company '{company_filter}' not found in stocks.json.[/yellow]")
            return {}
        target_dict = filtered

    # Collection parameters
    start_date = settings.dart_start_date
    end_date = datetime.now().strftime("%Y%m%d")
    keywords = settings.get_dart_keywords_list()

    # Initialize collectors
    dart_collector = DartCollector(
        api_key=settings.dart_api_key,
        output_dir=settings.output_dir,
    )

    fnguide_collector = None
    if not skip_fnguide:
        try:
            settings.validate_fnguide_config()
            fnguide_collector = FnGuideCollector(
                user_id=settings.fnguide_id,
                user_pw=settings.fnguide_pw,
                output_dir=settings.output_dir,
            )
        except Exception as e:
            console.print(f"[yellow]FnGuide skipped: {e}[/yellow]")

    news_collector = None
    if not skip_news:
        news_collector = NewsCollector(
            news_days_back=settings.get_news_days_back(),
            output_dir=settings.output_dir,
        )

    naver_collector = None
    if not skip_naver:
        naver_collector = NaverCollector(output_dir=settings.output_dir)

    logger.info("=== Collection started ===")
    logger.info(f"Target companies: {len(target_dict)}")
    logger.info(f"Collection period: {start_date} ~ {end_date}")

    results: dict[str, bool] = {}
    total = len(target_dict)

    for idx, (stock_code, company_name) in enumerate(target_dict.items(), 1):
        logger.info(f"[{idx}/{total}] {company_name} collection started")
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold][{idx}/{total}] {company_name}[/bold]")
        console.print(f"{'=' * 60}")

        try:
            console.print("Collecting DART filings...")
            dart_collector.collect(
                stock_code, company_name,
                start_date=start_date, end_date=end_date, keywords=keywords,
            )

            if fnguide_collector:
                console.print("Collecting FnGuide research...")
                fnguide_collector.collect(stock_code, company_name)

            if naver_collector:
                console.print("Collecting Naver research...")
                naver_collector.collect(
                    stock_code, company_name, start_date=start_date,
                )

            if news_collector:
                console.print("Collecting news articles...")
                news_collector.collect(stock_code, company_name)

            results[company_name] = True
            logger.info(f"[{idx}/{total}] {company_name} collection complete")

        except Exception as e:
            logger.exception(f"Error processing {company_name}")
            console.print(f"[red]Error processing {company_name}: {e}[/red]")
            results[company_name] = False

    logger.info(f"=== Collection complete: {total} companies ===")
    return results


def run_init_all(
    settings: Settings,
    tags: tuple[str, ...] = (),
    statement_type: str = "연결",
    company_filter: str | None = None,
    target_stocks: dict[str, str] | None = None,
) -> int:
    """Generate YAML configs for all (or filtered) stocks. Returns number of configs created.

    Args:
        settings: Global application settings.
        tags: Tags to add to each generated config.
        statement_type: Financial statement type (연결/별도).
        company_filter: Optional single company name to filter to.
        target_stocks: Optional pre-resolved {ticker: name} dict. If provided,
            stocks.json is not loaded and company_filter is ignored.
    """
    import yaml as yaml_lib

    if target_stocks is not None:
        stocks = target_stocks
    else:
        stocks_json_path = Path(settings.stocks_json)
        if not stocks_json_path.is_file():
            console.print(f"[red]stocks.json not found: {stocks_json_path}[/red]")
            return 0

        stocks = json.loads(stocks_json_path.read_text(encoding="utf-8"))

        # Optional single-company filter
        if company_filter:
            stocks = {k: v for k, v in stocks.items() if v == company_filter}
            if not stocks:
                console.print(f"[yellow]Company '{company_filter}' not found in stocks.json.[/yellow]")
                return 0

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    stocks_base = Path(settings.stocks_base_dir)

    created = 0
    for ticker, name in stocks.items():
        config_path = config_dir / f"{name}.yaml"
        stock_dir = stocks_base / name

        # Auto-detect files
        exchange_file = ""
        news_file = ""
        reports_dir = ""

        if stock_dir.is_dir():
            candidates = sorted(stock_dir.glob("*exchange_disclosure*.json"))
            if candidates:
                exchange_file = str(candidates[0]).replace("\\", "/")

            candidates = sorted(stock_dir.glob("*news*.json"))
            if candidates:
                news_file = str(candidates[-1]).replace("\\", "/")

            reports_sub = stock_dir / "reports"
            if reports_sub.is_dir():
                reports_dir = str(reports_sub).replace("\\", "/")

        tag_list = list(tags) if tags else []
        config_data = {
            "company": {"name": name, "ticker": ticker, "tags": tag_list},
            "report": {
                "output_dir": "",
                "years": 5,
                "quarters": 8,
                "statement_type": statement_type,
            },
        }

        disclosures = {}
        if exchange_file:
            disclosures["exchange_disclosures_file"] = exchange_file
        if disclosures:
            config_data["disclosures"] = disclosures

        analysis = {}
        if reports_dir:
            analysis["reports_dir"] = reports_dir
        if news_file:
            analysis["news_file"] = news_file
        if analysis:
            config_data["analysis"] = analysis

        yaml_str = yaml_lib.dump(config_data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        config_path.write_text(yaml_str, encoding="utf-8")

        status = "created"
        if not stock_dir.is_dir():
            status += " (no stock dir)"
        console.print(f"  {status}: {config_path} ({ticker})")
        created += 1

    console.print(f"\n[bold]{created} config(s) generated.[/bold]")
    return created


def run_batch_all(
    settings: Settings,
    no_copy: bool = False,
    verbose: bool = False,
    output_dir: str | None = None,
    company_filter: str | None = None,
) -> list[tuple[str, Path | None, str | None]]:
    """Run batch report generation for all (or filtered) YAML configs. Returns results list."""
    from auto_reports.pipeline import run_pipeline

    # Write directly to Obsidian inbox (no intermediate output/ folder)
    effective_output_dir = output_dir
    if not effective_output_dir and settings.obsidian_inbox and not no_copy:
        effective_output_dir = settings.obsidian_inbox

    config_dir = Path("config").resolve()
    if not config_dir.is_dir():
        console.print(f"[red]config/ directory not found: {config_dir}[/red]")
        return []

    # Load stocks.json to determine which companies to process
    from auto_reports.utils.file_utils import load_stock_json

    target_stocks = load_stock_json(settings.stocks_json)

    # Optional single-company filter
    if company_filter:
        if target_stocks:
            target_stocks = {k: v for k, v in target_stocks.items() if v == company_filter}
        if not target_stocks:
            console.print(f"[yellow]Company '{company_filter}' not found in stocks.json.[/yellow]")
            return []

    if target_stocks:
        # Only process configs for companies listed in stocks.json
        target_names = set(target_stocks.values())
        configs = sorted(
            c for c in config_dir.glob("*.yaml")
            if c.stem in target_names
        )
    else:
        # Fallback: process all configs if stocks.json is missing/empty
        all_configs = sorted(config_dir.glob("*.yaml"))
        configs = [c for c in all_configs if "example" not in c.stem.lower()]

    if not configs:
        console.print("[yellow]No config files found in config/.[/yellow]")
        return []

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

    return results


def run_analyze(
    settings: Settings,
    no_copy: bool = False,
    verbose: bool = False,
    company_filter: str | None = None,
) -> None:
    """Execute analysis pipeline only: init -> batch (no data collection)."""
    setup_logging(log_dir=settings.output_dir, log_level="DEBUG" if verbose else "INFO")
    logger = get_logger("orchestrator.analyze")

    # Step 1: Init
    console.print("\n[bold cyan]Step 1/2: Generating configs...[/bold cyan]\n")
    created = run_init_all(settings, company_filter=company_filter)
    console.print(f"[dim]{created} configs generated.[/dim]")

    # Step 2: Batch
    console.print("\n[bold cyan]Step 2/2: Generating reports...[/bold cyan]\n")
    results = run_batch_all(settings, no_copy=no_copy, verbose=verbose, company_filter=company_filter)

    # Summary
    success_reports = sum(1 for _, p, e in results if p and not e)
    failed = sum(1 for _, _, e in results if e)
    console.print(f"\n{'═' * 60}")
    console.print("[bold green]Analysis complete![/bold green]")
    console.print(f"  Configs: {created}")
    console.print(f"  Reports: {success_reports}/{len(results)} (failed: {failed})")
    if settings.obsidian_inbox and not no_copy:
        console.print(f"  Output: {settings.obsidian_inbox}")
    console.print(f"{'═' * 60}\n")

    logger.info("Analysis pipeline complete")


def run_full_pipeline(
    settings: Settings,
    skip_fnguide: bool = False,
    skip_news: bool = False,
    skip_naver: bool = True,
    no_copy: bool = False,
    verbose: bool = False,
    company_filter: str | None = None,
) -> None:
    """Execute the full pipeline: collect -> init -> batch."""
    setup_logging(log_dir=settings.output_dir, log_level="DEBUG" if verbose else "INFO")
    logger = get_logger("orchestrator")

    # Step 1: Collect
    console.print("\n[bold cyan]Step 1/3: Collecting data...[/bold cyan]\n")
    collect_results = run_collect(
        settings,
        skip_fnguide=skip_fnguide,
        skip_news=skip_news,
        skip_naver=skip_naver,
        company_filter=company_filter,
    )
    success_count = sum(1 for v in collect_results.values() if v)
    console.print(f"\n[dim]Collection done: {success_count}/{len(collect_results)} companies succeeded.[/dim]")

    # Step 2: Init
    console.print("\n[bold cyan]Step 2/3: Generating configs...[/bold cyan]\n")
    created = run_init_all(settings, company_filter=company_filter)
    console.print(f"[dim]{created} configs generated.[/dim]")

    # Step 3: Batch
    console.print("\n[bold cyan]Step 3/3: Generating reports...[/bold cyan]\n")
    results = run_batch_all(settings, no_copy=no_copy, verbose=verbose, company_filter=company_filter)

    # Final summary
    console.print(f"\n{'═' * 60}")
    console.print("[bold green]Pipeline complete![/bold green]")
    console.print(f"  Collected: {success_count}/{len(collect_results)} companies")
    console.print(f"  Configs: {created}")
    success_reports = sum(1 for _, p, e in results if p and not e)
    console.print(f"  Reports: {success_reports}/{len(results)}")
    if settings.obsidian_inbox and not no_copy:
        console.print(f"  Output: {settings.obsidian_inbox}")
    console.print(f"{'═' * 60}\n")

    logger.info("Full pipeline complete")
