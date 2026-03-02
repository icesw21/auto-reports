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

### run — 전체 파이프라인 (collect -> init -> batch)

```bash
auto-reports run                          # 전체 실행
auto-reports run -c 광동제약               # 단일 종목
auto-reports run -w 3                     # 병렬 3 workers
auto-reports run --skip-fnguide --skip-news --no-copy
```

| Option | Description |
|--------|-------------|
| `-c, --company <name>` | 단일 종목 필터 |
| `-w, --workers <N>` | 병렬 worker 수 (기본: 1=순차) |
| `--skip-fnguide` | FnGuide 수집 스킵 (Selenium 불필요) |
| `--skip-news` | 뉴스 수집 스킵 |
| `--no-copy` | Obsidian 복사 스킵 |
| `--stocks <path>` | stocks.json 경로 지정 |
| `-v, --verbose` | 디버그 로깅 |

### download — 데이터 수집만

```bash
auto-reports download                     # 전체 종목
auto-reports download -c 광동제약 -w 3     # 단일 종목, 병렬
auto-reports download --skip-fnguide      # FnGuide 제외
```

| Option | Description |
|--------|-------------|
| `-c, --company <name>` | 단일 종목 필터 |
| `-w, --workers <N>` | 병렬 worker 수 (기본: 1=순차) |
| `--skip-fnguide` | FnGuide 수집 스킵 |
| `--skip-news` | 뉴스 수집 스킵 |
| `--stocks <path>` | stocks.json 경로 지정 |
| `-v, --verbose` | 디버그 로깅 |

`collect`는 `download`의 alias.

### analyze — 분석 + 리포트 생성만 (다운로드 없이)

```bash
auto-reports analyze                      # 전체 종목
auto-reports analyze -c 광동제약 -w 3      # 단일 종목, 병렬
auto-reports analyze --no-copy            # Obsidian 복사 스킵
```

| Option | Description |
|--------|-------------|
| `-c, --company <name>` | 단일 종목 필터 |
| `-w, --workers <N>` | 병렬 worker 수 (기본: 1=순차) |
| `--no-copy` | Obsidian 복사 스킵 |
| `--stocks <path>` | stocks.json 경로 지정 |
| `-v, --verbose` | 디버그 로깅 |

### init — YAML 설정 파일 생성

```bash
auto-reports init --all                   # 전체 종목
auto-reports init 광동제약 보성파워텍        # 지정 종목
auto-reports init 광동제약 -t 2차전지       # 태그 추가
```

| Option | Description |
|--------|-------------|
| `--all` | stocks.json 전체 종목 |
| `-t, --tags <tag>` | 태그 추가 (여러 번 가능) |
| `--statement-type` | 재무제표 유형 (연결/별도, 기본: 연결) |

### batch — 복수 종목 리포트 일괄 생성

```bash
auto-reports batch --all                  # 전체 종목
auto-reports batch --all -w 3             # 병렬
auto-reports batch --all --no-copy        # Obsidian 복사 스킵
auto-reports batch config/광동제약.yaml    # 지정 YAML
```

| Option | Description |
|--------|-------------|
| `--all` | config/ 디렉터리 내 전체 YAML |
| `-w, --workers <N>` | 병렬 worker 수 (기본: 1=순차) |
| `-o, --output-dir <dir>` | 출력 디렉터리 지정 |
| `--no-copy` | Obsidian 복사 스킵 |
| `-v, --verbose` | 디버그 로깅 |

### generate — 단일 종목 리포트 생성

```bash
auto-reports generate config/광동제약.yaml
auto-reports generate config/광동제약.yaml -o output/ --no-copy
```

| Option | Description |
|--------|-------------|
| `-o, --output-dir <dir>` | 출력 디렉터리 지정 |
| `--no-copy` | Obsidian 복사 스킵 |
| `--dry-run` | 파싱/분석만 (파일 미생성) |
| `-v, --verbose` | 디버그 로깅 |

### parse — DART 공시 URL 파싱 (디버그)

```bash
auto-reports parse https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260202901050
```

### fetch — 종목 재무 데이터 조회 (디버그)

```bash
auto-reports fetch 171090
```

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
