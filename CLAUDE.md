# auto-reports

Korean stock financial data collector, parser, and investment report generator.

Merges [opendart](../opendart) (data collection) and [finance-parser](../finance-parser) (parsing + report generation) into a unified workflow.

## Quick Start

```bash
pip install -e .
cp .env.example .env  # Fill in API keys
auto-reports run       # Full pipeline: collect -> init -> batch
```

## Commands

| Command | Description |
|---------|-------------|
| `auto-reports run` | Execute full pipeline (collect -> init -> batch) |
| `auto-reports download` | Data download only (DART, FnGuide, Naver, News) |
| `auto-reports analyze` | Analysis only (init configs -> batch reports) |
| `auto-reports collect` | Alias for `download` |
| `auto-reports init --all` | Generate YAML configs from stocks.json |
| `auto-reports batch --all` | Generate reports and send to Obsidian |
| `auto-reports generate <yaml>` | Generate single company report |
| `auto-reports parse <url>` | Parse a DART disclosure URL (debug) |
| `auto-reports fetch <ticker>` | Fetch financial data for ticker (debug) |

### Common Options

- `--company <name>` — Filter to a single company (available on `download`, `analyze`, `run`)
- `--skip-fnguide` — Skip FnGuide collection (available on `download`, `run`)
- `--skip-news` — Skip news collection (available on `download`, `run`)
- `--no-copy` — Don't copy reports to Obsidian (available on `analyze`, `run`, `batch`)

## Project Structure

```
src/auto_reports/
  cli.py              # Click CLI entry point
  config.py           # Unified Settings (pydantic) + YAML config models
  orchestrator.py     # Chains collect -> init -> batch
  pipeline.py         # Report generation pipeline (from finance-parser)
  collectors/         # Data collectors (from opendart)
  fetchers/           # Financial data fetchers (from finance-parser)
  parsers/            # Disclosure parsers (from finance-parser)
  analyzers/          # Financial analysis (from finance-parser)
  summarizers/        # LLM summarization (from finance-parser)
  generators/         # Report generation with Jinja2 (from finance-parser)
  models/             # Pydantic data models (from finance-parser)
  utils/              # Shared utilities (logging, retry, file ops)
```

## Configuration

All settings via `.env` file. Key variables:
- `DART_API_KEY` - Required for DART API access
- `FNGUIDE_ID` / `FNGUIDE_PW` - FnGuide credentials (optional, use --skip-fnguide)
- `OPENAI_API_KEY` - For LLM-powered analysis sections
- `STOCKS_JSON` - Path to target stocks list (default: ./stocks.json)
- `OUTPUT_DIR` - Data output directory (default: ./stocks)
- `OBSIDIAN_INBOX` - Obsidian vault inbox for report delivery

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Build & Lint

```bash
ruff check src/ tests/
```
