# Database Guidelines

> Storage conventions for this file-based quant research project.

---

## Overview

This project does not use a database, ORM, migrations, or transactions. Treat the local filesystem as the persistence layer:

- YAML for configuration.
- CSV for raw Tushare rows, manual holdings, execution templates, and human-readable tabular outputs.
- Parquet for large structured data caches: price panels, factor caches, daily_basic, fundamentals.
- JSON for run status, quality gates, metadata, and machine-readable reports.
- Markdown for human-facing reports.

Because these files drive trading decisions, storage changes must be explicit, tested, and documented.

---

## Canonical Stores

| Store | Path | Format | Owner |
| --- | --- | --- | --- |
| Default config | `config/settings.yaml` | YAML | `src/config_loader.py` |
| Local private config | `config/settings.local.yaml` | YAML, ignored | developer |
| Raw daily stock data | `data/raw/<TS_CODE>.csv` | CSV, UTF-8 BOM when written by project | `src/data_fetcher.py` |
| Mainboard universe | `data/raw/mainboard_a_stocks.csv` | CSV | `src/data_fetcher.py` |
| ST calendar | `data/raw/st_calendar.csv` | CSV | `src/data_fetcher.py` |
| HS300 constituents | `data/raw/hs300_constituents.csv` | CSV | `src/data_fetcher.py` |
| Qlib provider | `data/qlib_data/` | Qlib text/bin/parquet | `src/data_converter.py` |
| Price panels | `data/prices/*.parquet` | Parquet | `src/data_converter.py` |
| Factor cache | `data/factors/alpha158.parquet` | Parquet | `src/factor_calculator.py` |
| Factor cache metadata | `data/factors/alpha158.parquet.meta.json` | JSON | `src/factor_calculator.py` |
| Daily basic cache | `data/factors/daily_basic.parquet` | Parquet | `src/data_fetcher.py` |
| Fundamentals | `data/fundamentals/*.parquet` | Parquet | `src/fundamental_data.py` |
| Run outputs | `outputs/*` | CSV/JSON/Markdown/log | scripts and report writers |

Do not introduce a database dependency unless the project explicitly chooses to move away from local-file workflows.

---

## Read And Write Patterns

### Resolve paths through config

Use `resolve_path` for any configured path:

```python
from src.config_loader import resolve_path

price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
prices = pd.read_parquet(price_path)
```

This keeps scripts stable when called from `.bat` files, terminals, or tests.

### Create parent directories before writing

Most writers call `mkdir(parents=True, exist_ok=True)` immediately before writing:

```python
output_dir = resolve_path(out_dir)
output_dir.mkdir(parents=True, exist_ok=True)
orders.to_csv(path, index=False, encoding="utf-8-sig")
```

Examples: `src/manual_orders.py`, `src/data_health.py`, `src/fundamental_data.py`, `scripts/run_optimize.py`.

### Use UTF-8 JSON/Markdown and UTF-8 BOM CSV for user-facing CSV

- JSON/Markdown: `encoding="utf-8"`, `ensure_ascii=False` when Chinese report text or symbols may appear.
- CSV produced for users or Excel: `encoding="utf-8-sig"`.
- Parquet caches generally use pandas defaults.

Examples:

```python
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
```

### Keep candidate outputs separate from official holdings

When quality gates fail, scripts write candidate artifacts and do not overwrite official holdings:

- `outputs/candidate_signal_YYYY-MM-DD.csv`
- `outputs/candidate_holdings_YYYY-MM-DD.csv`
- `outputs/manual_orders_candidate_YYYY-MM-DD.csv`

Promotion is explicit through `scripts/run_auto_signal.py --promote-candidate`.

### Preserve resumability

Long-running data refreshes write progress and status files:

- `outputs/data_update_progress.json`
- `outputs/auto_run_status.json`
- `outputs/auto_signal_job.json`

When adding long-running work, write a status artifact early and update it per stage, as `_stage()` does in `scripts/run_auto_signal.py`.

---

## Schema Contracts

### Raw daily CSV

Normalized raw daily rows use Tushare-style columns:

```text
ts_code, trade_date, open, high, low, close, vol, amount, adj_factor
```

`normalize_daily_frame` accepts some aliases, but stored output should be normalized and sorted by `ts_code`, `trade_date`.

### Price panels

`data/prices/close*.parquet` is a close-price panel with dates as index and instruments as columns.

`data/prices/ohlcv*.parquet` uses MultiIndex columns:

```text
level 0: field       open/high/low/close/volume/amount/vwap
level 1: instrument  lowercase storage code when generated, normalized to uppercase in logic
```

### Factor and score panels

Factor caches and score panels use a two-level index:

```text
datetime, instrument
```

Tests assert this contract across `src/backtest.py`, `src/factor_ic.py`, `src/scoring.py`, and `src/signal_generator.py`.

### Config

`DEFAULT_CONFIG` in `src/config_loader.py` is the schema baseline. `config/settings.yaml` and `config/settings.local.yaml` deep-merge onto it. New config keys should be added to `DEFAULT_CONFIG`, validated where risky, and tested in `tests/test_config_loader.py`.

---

## Migrations

There are no database migrations. A storage-contract change still needs a migration-style checklist:

1. Update `DEFAULT_CONFIG` and `config/settings.yaml` if new paths or settings are required.
2. Keep backward-compatible readers when possible, especially for CSV column aliases such as `instrument`, `ticker`, `ts_code`, and `con_code`.
3. Add tests using `TemporaryDirectory` with old and new file shapes.
4. Update `README.md` artifact tables and `.bat` docs when user-visible files change.
5. Do not rewrite generated market caches in the repo; tests should use temporary files or `tests/fixtures/data_snapshot/`.

---

## Common Mistakes

### Mistake: writing generated data into version-controlled paths

Do not commit real market caches, account holdings, Tushare URLs/tokens, reports, or `outputs/`. `.gitignore` intentionally excludes:

```text
config/settings.local.yaml
config/account.yaml
config/current_holdings.csv
data/raw/*
data/qlib_data/*
data/factors/*
data/fundamentals/*
data/prices/*
outputs/*
```

### Mistake: accepting malformed market data silently

Normalize and validate before caching. `normalize_daily_frame` rejects missing required columns and invalid OHLCV values. `data_converter._apply_adjustment` rejects missing or non-positive `adj_factor`.

### Mistake: treating a field table as a close-price panel

Plain DataFrame price inputs must be close-price panels. If columns look like `open/high/low/close`, use a MultiIndex field/instrument panel instead.

### Mistake: overwriting official state before gates pass

Use candidate files until parameter quality, backtest quality, data health, and governance gates allow official outputs.
