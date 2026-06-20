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
| Multi-index constituents | `data/raw/index_constituents.csv` | CSV | `scripts/run_build_universe.py` |
| Historical stock universe | `data/raw/historical_universe.csv` | CSV | `src/universe_builder.py` |
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

### Scenario: Raw Daily Suspension And Adj-Factor Metadata Gates

#### 1. Scope / Trigger

- Trigger: raw Tushare daily CSVs and adj-factor metadata feed data health/governance gates for the auto-signal workflow.
- Storage owners: `src/data_fetcher_frames.py`, `src/adj_factor_metadata.py`, and `src/data_governance.py`.

#### 2. Signatures

- Raw stock file: `data/raw/<TS_CODE>.csv` for `.SH`, `.SZ`, and `.BJ` A-share symbols, for example `data/raw/600717.SH.csv`.
- Raw index file: `data/raw/000300.SH.csv` or `data/raw/000905.SH.csv`.
- Metadata builder: `build_adj_factor_metadata(raw_dir=..., output_path=...)`.

#### 3. Contracts

- Stock daily rows must store normalized `ts_code`, `trade_date`, OHLC, `vol`, `amount`, and `adj_factor`.
- A suspended stock row with `close > 0`, `open/high/low == 0`, and `vol/amount == 0` is valid after normalization by filling `open/high/low` from `close`; this preserves the trading-day close without fabricating turnover.
- Index raw files do not carry `adj_factor` and must not be counted as missing adj-factor metadata.
- Adj-factor metadata readiness is based on raw stock CSVs that are expected to carry `adj_factor`, not on index benchmark CSVs.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Stock row has non-positive OHLC after suspension normalization | `ValueError` naming invalid OHLCV rows |
| Stock row has negative `vol` or `amount` | `ValueError` naming invalid OHLCV rows |
| Stock CSV lacks `adj_factor` | Adj-factor metadata records the stock as an issue |
| Index CSV lacks `adj_factor` | Skip it for adj-factor metadata readiness |

#### 5. Good/Base/Bad Cases

- Good: `600717.SH` suspended row with `close=1.23`, zero `open/high/low`, and zero `vol/amount` stores OHLC as `1.23`.
- Base: normal active stock row keeps source OHLCV and `adj_factor`.
- Bad: `000300.SH` missing `adj_factor` fails governance readiness; index files are not stock adjustment inputs and should be excluded.

#### 6. Tests Required

- `tests/test_data_fetcher.py` must assert suspended stock rows are normalized to close-filled OHLC.
- `tests/test_data_governance.py` must assert index raw files are excluded from adj-factor metadata coverage and governance hard issues.

#### 7. Wrong vs Correct

##### Wrong

```python
for path in raw_dir.glob("*.csv"):
    require_adj_factor(path)
```

##### Correct

```python
for path in raw_dir.glob("*.csv"):
    if is_adj_factor_stock_csv(path):
        require_adj_factor(path)
```

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

### Historical universe snapshots

`data/raw/index_constituents.csv` stores normalized Tushare `index_weight` rows for every index needed by the configured universe builder:

```text
index_code, con_code, trade_date, weight
```

`data/raw/historical_universe.csv` stores point-in-time stock-pool snapshots generated from those rows:

```text
trade_date, instrument, sources, index_codes, source_count, weight,
hs300_weight, csi500_weight, csi1000_weight, csi1000_rank
```

When `universe_builder.enabled` is true, backtest and signal score panels must be filtered source by source: for each source label such as `hs300`, `csi500`, and `csi1000`, pick that source's latest snapshot whose `trade_date <= score_date`, then union the members. Never use a later snapshot to fill an earlier score date, and never use one source's newer snapshot date to discard another source's still-valid prior snapshot.

## Scenario: Historical Universe Builder

### 1. Scope / Trigger

- Trigger: the project now has a cross-layer stock-pool contract spanning Tushare `index_weight`, local CSV cache, score filtering, backtest, signal generation, README, and Windows batch entrypoints.

### 2. Signatures

- Command: `.\.venv\Scripts\python.exe scripts\run_build_universe.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD|auto] [--core-index-codes CSV] [--satellite-index-code CODE] [--satellite-top-n N] [--index-constituents-file PATH] [--out-file PATH] [--skip-fetch] [--max-index-windows N] [--index-window-days N] [--skip-index-errors]`.
- Batch wrapper: `14_构建历史股票池.bat`.
- API: `src.universe_builder.build_historical_universe(index_constituents, core_index_codes, satellite_index_code, satellite_top_n) -> pd.DataFrame`.
- API: `src.universe_builder.filter_scores_by_historical_universe(scores, universe) -> pd.Series`.

### 3. Contracts

- Input constituent rows use normalized `index_code`, `con_code`, `trade_date`, and `weight` columns.
- Default core indices are `000300.SH` and `000905.SH`, both kept in full. Known equivalent fallback codes such as `399300.SZ` for `hs300` are accepted as the same source during build and filtering.
- Default satellite index is `000852.SH`, ranked by descending `weight`, with the top `300` rows kept per `trade_date`.
- Official historical-universe builds fail on Tushare/index-window errors by default. `--skip-index-errors` is only for smoke tests or partial diagnostics.
- Output `instrument` values are uppercase Tushare symbols.
- Output `sources` and `index_codes` are pipe-separated labels for symbols that appear in multiple selected indices.
- Config keys live under `universe_builder` and must be present in `DEFAULT_CONFIG` plus `_CONFIG_VALIDATORS`; `require_file` defaults to `true` so enabled filtering does not silently fall back to the unfiltered score panel.
- Data governance must check `historical_universe.csv` by source label when `universe_builder.enabled=true`. Required default sources are `hs300`, `csi500`, and `csi1000`; each source must have monthly snapshots across the point-in-time factor window unless `min_historical_universe_source_month_coverage` is explicitly relaxed.
- For the terminal partial month only, a source's previous-month snapshot may be carried forward when the previous month is present and the current source has not published a new snapshot yet. Missing middle months still fail coverage.
- Raw daily conversion must discover market files through `src.common.is_stock_csv(path)`. Metadata CSVs such as `index_constituents.csv`, `historical_universe.csv`, `mainboard_a_stocks.csv`, `st_calendar.csv`, and `failed_fetches.csv` must never be passed to `normalize_daily_frame()`.

### 4. Validation & Error Matrix

- Missing `index_constituents_file` when building from file -> `FileNotFoundError` telling the user to run `scripts/run_build_universe.py`.
- Missing historical universe file while filtering and `require_file=true` -> `FileNotFoundError`.
- Missing historical universe file while filtering and `require_file=false` -> warning and unchanged scores.
- Historical universe input missing `trade_date` or `instrument`/`con_code`/`ts_code` -> `ValueError` naming missing columns.
- Enabled historical-universe governance with a missing file -> issue `historical_universe_file_missing` and a repair action pointing to `scripts/run_build_universe.py`.
- Enabled historical-universe governance with missing `trade_date`, instrument alias, or `sources` column -> issue `historical_universe_missing_columns:<columns>`.
- Enabled historical-universe governance with insufficient source-month coverage -> issue `historical_universe_source_month_coverage_below_required:<source>:<observed>/<expected><threshold>` where the final separator is the literal `<` comparison operator.
- Score input without a `datetime`/`instrument` MultiIndex -> `ValueError("scores must use MultiIndex: datetime/instrument.")`.

### 5. Good/Base/Bad Cases

- Good: `000300.SH` or fallback `399300.SZ` + `000905.SH` + top 300 of `000852.SH` produces one row per `(trade_date, instrument)` with source labels and satellite ranks.
- Base: `--skip-fetch` rebuilds `historical_universe.csv` from an existing cached `index_constituents.csv` without network access.
- Bad: a score date earlier than the first universe snapshot gets no allowed members rather than using a future snapshot.
- Bad: when `csi1000` has a newer snapshot than `hs300`/`csi500`, filtering still carries forward the prior `hs300` and `csi500` snapshots instead of shrinking the pool to `csi1000` only.

### 6. Tests Required

- Unit test that core members are kept in full and satellite members are limited by per-date weight rank.
- Unit test that `399300.SZ` fallback rows are retained as `hs300`.
- Unit test that score filtering uses the latest prior snapshot per source and does not leak future membership.
- Unit test that asynchronous source snapshots are carried forward independently before unioning members.
- Governance test that a terminal partial month can be covered by the previous source snapshot, while non-terminal gaps still fail.
- Config test that `DEFAULT_CONFIG["universe_builder"]` contains the default paths, index codes, and top-N.
- Governance test that enabled historical universe coverage is checked independently for `hs300`, `csi500`, and `csi1000` and creates a `historical_universe` repair action on gaps.
- Script/docs test that the batch entrypoint is UTF-8, CRLF, and documented in README.

### 7. Wrong vs Correct

#### Wrong

```python
latest = universe["trade_date"].max()
allowed = universe[universe["trade_date"] == latest]
scores = scores[scores.index.get_level_values("instrument").isin(allowed["instrument"])]
```

This can both leak the newest known membership into every historical score date and discard sources whose latest valid snapshot is older than another source's latest date.

#### Correct

```python
scores = filter_scores_by_historical_universe(scores, universe)
```

The helper picks each source's latest snapshot with `trade_date <= score_date`, then unions those members.

## Scenario: Configured End Date Governance Window

### 1. Scope / Trigger

- Trigger: fixed-date research or auto-signal runs may use factor caches whose metadata extends beyond the requested `data.end_date`.
- Owner: `src.data_governance.build_data_governance_report`.

### 2. Signatures

- Config input: `data.end_date`, already resolved by `scripts/run_auto_signal.py` before governance runs.
- Metadata input: `<factor_cache>.meta.json` with `start_date` and `end_date`.

### 3. Contracts

- Governance coverage checks use `min(data.end_date, factor_cache_meta.end_date)` as the required point-in-time end when `data.end_date` is a concrete date.
- If `data.end_date` is absent or cannot be parsed, governance falls back to `factor_cache_meta.end_date`.
- The report still records the raw `factor_cache_meta_end_date`; only required coverage windows, freshness warnings, and repair-action end dates are capped.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| `data.end_date="2026-05-27"` and factor metadata ends `2026-06-12` | Require daily-basic/index/historical-universe coverage only through `2026-05-27`. |
| `data.end_date="auto"` and factor metadata ends `2026-06-12` | Use factor metadata end after the caller resolves `auto`, or fall back to metadata if unresolved. |
| Factor metadata is missing | Keep existing `factor_cache_meta_missing` warning and avoid inventing an end date. |

### 5. Good/Base/Bad Cases

- Good: A fixed May backtest with June factor-cache metadata passes governance when all May evidence is present.
- Base: A normal latest run still requires coverage through the resolved latest target date.
- Bad: Blocking a fixed May run because csi500/csi1000 June index snapshots have not been published yet.

### 6. Tests Required

- `tests/test_data_governance.py` must assert configured `data.end_date` caps expected daily-basic dates and historical-universe source months while preserving the raw factor metadata end date in the report.

### 7. Wrong vs Correct

#### Wrong

```python
expected_index_months = _month_range_texts(point_in_time_start, factor_meta_end_date)
```

#### Correct

```python
point_in_time_end = _point_in_time_end_date(data_cfg.get("end_date"), factor_meta_end_date)
expected_index_months = _month_range_texts(point_in_time_start, point_in_time_end)
```

### Scenario: Backtest Selection Schedule

#### 1. Scope / Trigger

- Trigger: research workflows may route among score sources whose target holding count and turnover policy differ by signal date.
- Owner: `src/backtest.py` consumes the schedule; research scripts such as `scripts/run_annual_state_router_backtest.py` may produce it.

#### 2. Signatures

- Backtest config key: `selection_schedule`.
- Shape: mapping of signal-date string/Timestamp to a mapping with optional integer fields:
  - `top_n`
  - `max_turnover`
  - `rank_buffer`

#### 3. Contracts

- Signal dates are normalized with `pd.Timestamp(...).normalize()`.
- The schedule is keyed by signal date, not trade date, because `run_backtest()` already maps each signal date to the next executable price date.
- When a signal date has no schedule entry, the scalar `top_n`, `max_turnover`, and `rank_buffer` config values remain in force.
- Calendar/regime routers must decide the route by the execution trading date when a signal executes in a different calendar year than the signal date.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| `selection_schedule` is missing | Use scalar selection config only |
| `selection_schedule` is not a mapping | `ValueError("selection_schedule must be a mapping of date to selection settings.")` |
| A schedule value is not a mapping | `ValueError("selection_schedule values must be mappings.")` |
| `top_n < 1` or `max_turnover < 1` | `ValueError("selection_schedule.<field> must be >= 1.")` |
| `rank_buffer < 0` | `ValueError("selection_schedule.rank_buffer must be >= 0.")` |

#### 5. Good/Base/Bad Cases

- Good: `{"2024-12-31": {"top_n": 5, "max_turnover": 5, "rank_buffer": 10}}` where the `2024-12-31` signal trades on `2025-01-02` and uses the route selected for 2025.
- Base: no schedule, so every rebalance uses fixed scalar config values.
- Bad: deciding an annual route from `signal_date.year`; a year-end signal can execute in the next year and would lag the route by one rebalance.

#### 6. Tests Required

- `tests/test_backtest.py` must assert scheduled selection values change realized holdings on the matching signal date.
- Router tests must assert signal dates map to the next price date before choosing the route year.

#### 7. Wrong vs Correct

##### Wrong

```python
route_year = signal_date.year
```

##### Correct

```python
trade_date = next_price_date_after(signal_date)
route_year = trade_date.year
```

### Scenario: Annual State Router Research Grid

#### 1. Scope / Trigger

- Trigger: research workflows need resumable formal score/trade grid searches over annual market-state routes.
- Owners: `scripts/run_annual_state_router_backtest.py` defines score sources and route contracts; `scripts/run_annual_state_router_grid.py` caches sources, appends grid rows, and writes hit artifacts.

#### 2. Signatures

- Single formal run:
  - `scripts/run_annual_state_router_backtest.py --include-expanded-sources --moderate-positive-source <source> --moderate-low-source <source> --moderate-low-exposure <float> --turnover-boost-reasons <comma-list>`
- Resumable grid:
  - `scripts/run_annual_state_router_grid.py --cache-dir outputs/router_score_cache --output outputs/<name>.csv --write-hit-prefix outputs/<prefix> [--max-industry-weights none,0.35]`
- Expanded source names currently include `roc60`, `db_total`, `beta20`, and `rsqr20`.

#### 3. Contracts

- Score caches are parquet files under `outputs/router_score_cache/<source>_<end-date>.parquet` with a `score` column and a two-level date/instrument index.
- Grid output is append-only CSV keyed by the stable `combo_key`; reruns skip keys already present in the output CSV.
- `turnover_boost_reason_sets` uses semicolon-separated reason sets; reasons inside one set use `+`, for example `low_vol_moderate_uptrend+moderate_positive_roc60`.
- `moderate_low_exposure` multiplies route exposure only when `moderate_low_source` is selected for a `default_beta` route whose `ret252` is in `[moderate_low_ret252_min, moderate_low_ret252_max)`.
- `max_industry_weights` enumerates research-only `strategy.max_industry_weight` overrides. `none` preserves the configured/default behavior; numeric values are passed through `RiskPolicy` and existing selection constraints.
- A full-gate hit writes `_metrics.json`, `_years.csv`, `_year_routes.csv`, `_score_routes.csv`, `_holdings.csv`, `_trades.csv`, and `_equity.csv` beside the requested hit prefix.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Score cache exists and `--force-rebuild-cache` is absent | Load the cached parquet source. |
| Score cache is missing | Build the source from configured factor/selector inputs and write the parquet cache. |
| `turnover_boost_reasons=none` | Do not boost route turnover; collapse boost max/rank combinations to a single grid row. |
| `moderate_low_source=none` | Do not route the low ret252 band; collapse `moderate_low_exposure` to `1.0`. |
| `max_industry_weights=none,0.35` | Run the same grid combo with and without a 35% industry cap overlay. |
| Routed source is missing from score sources | Raise `ValueError("Routed source is not in score sources: ...")`. |
| Hit satisfies annual return, drawdown, turnover, cost, and yearly gates | Write hit artifacts and stop the grid early. |

#### 5. Good/Base/Bad Cases

- Good: `moderate_low_source=beta20`, `moderate_low_exposure=0.4`, and `turnover_mode=rank10` produce a formal hit whose cost gate still passes.
- Base: no expanded low-band source; route remains `default_beta`, scalar exposure remains `1.0`.
- Bad: proving yearly returns from standalone candidate equity files only; path-dependent annual routing must be reconstructed through score panels, holdings, trades, costs, and gates.

#### 6. Tests Required

- `tests/test_run_annual_state_router_backtest.py` must assert expanded source definitions and route exposure scaling.
- `tests/test_run_annual_state_router_grid.py` must assert reason-set parsing and `moderate_low_exposure` enumeration/collapse.
- `tests/test_run_annual_state_router_grid.py` must assert `max_industry_weights` enumeration.
- Backtest tests must continue to assert `selection_schedule` values affect realized holdings on matching signal dates.

#### 7. Wrong vs Correct

##### Wrong

```powershell
python scripts/run_annual_state_router_probe.py --source beta20=...equity.csv
```

Using candidate equity curves can overstate a switcher's yearly result because each source has its own prior holdings, cash path, and costs.

##### Correct

```powershell
python scripts/run_annual_state_router_grid.py --moderate-low-source beta20 --moderate-low-exposure 0.4 --write-hit-prefix outputs/router_hit
```

The grid reconstructs routed scores, scheduled holdings, executed trades, exposure, turnover, costs, and yearly gates in one formal path.

### Scenario: Annual State Router Auto-Signal Mode

#### 1. Scope / Trigger

- Trigger: the main `scripts/run_auto_signal.py` workflow can use a formally validated annual-state score router instead of the legacy walk-forward optimizer strategy family.
- Owners: `config/settings.yaml`, `src/config_loader.py`, `scripts/run_auto_signal.py`, `src/signal_generator.py`, and `scripts/run_annual_state_router_backtest.py`.

#### 2. Signatures

- Config section: `annual_state_router`.
- Required mode fields:
  - `enabled`
  - `factor_file`, `industry_factor_file`, `selector_file`
  - `source_factor_files`
  - `turnover_mode`
  - `full_turnover_on_route_change`
  - `use_defensive_timing`
  - route thresholds/exposures such as `missing_ret252_exposure`, `moderate_low_source`, and `moderate_low_exposure`
  - evidence files `evidence_metrics_file` and `evidence_years_file`
- Auto outputs:
  - `outputs/auto_annual_state_router_score_routes.csv`
  - `outputs/auto_annual_state_router_year_routes.csv`
  - `outputs/auto_parameter_quality.json`
  - `outputs/auto_backtest_quality.json`
  - `outputs/auto_run_status.json`

#### 3. Contracts

- When `annual_state_router.enabled=true`, `run_auto_signal.py` skips the legacy optimizer with `selected_params_status=annual_state_router`.
- Parameter quality is built from formal router evidence and must fail if the evidence file is missing, the full gate is not met, or the configured combo differs from the evidence combo.
- Historical backtest must rebuild routed score panels and call `run_backtest()` with the routed scores and `routed_backtest_config()` output. Evidence files alone are not enough to mark the run executable.
- Signal generation must use the same routed score panel and the latest route's `top_n`, `max_turnover`, and `rank_buffer`; it must not fall back to `build_latest_strategy_scores()` for the official/candidate signal.
- The auto-signal report/status must record `strategy_mode=annual_state_router` and include the router route files for diagnostics.
- Official outputs are still gated by data health, point-in-time governance, parameter quality, backtest quality, and account checks.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| `annual_state_router.enabled=false` | Use the legacy optimizer/current-strategy path. |
| Evidence metrics file is missing | Parameter quality is unacceptable with `annual_state_router_evidence_metrics_file_missing` or `_not_found`. |
| Evidence combo differs from configured router fields | Parameter quality is unacceptable with `annual_state_router_evidence_combo_mismatch:<field>`. |
| Evidence passes but rebuilt auto backtest fails | Candidate outputs only; `block_reasons` records the backtest quality failures. |
| Router score panel is empty or route missing for the signal date | Raise `ValueError` naming the router score/route contract. |
| Data/governance/account gates fail | Candidate outputs only, even if router evidence and backtest pass. |

#### 5. Good/Base/Bad Cases

- Good: `turnover_mode=rank10`, `full_turnover_on_route_change=true`, and `use_defensive_timing=true` reproduce the formal router's score/trade path; the rebuilt backtest passes and official outputs are written.
- Base: router disabled, so auto-signal runs the existing optimization/backtest/signal flow.
- Bad: reading `outputs/*_hit_metrics.json` and marking the run executable without rebuilding routed scores, holdings, trades, and quality gates in the current workflow.

#### 6. Tests Required

- `tests/test_run_auto_signal.py` must assert annual-router evidence becomes parameter quality only when the combo matches.
- `tests/test_run_auto_signal.py` must assert `_run_backtest_stage()` uses routed scores instead of `build_strategy_scores()` when the router is enabled.
- `tests/test_signal_generator.py` must assert `generate_signal(scores=...)` uses precomputed score panels and falls back to the latest score date on or before the requested date.
- `tests/test_config_loader.py` must assert current settings load without warnings and validate the annual-router config keys.

#### 7. Wrong vs Correct

##### Wrong

```python
parameter_quality = formal_hit_metrics
signal_df, holdings = generate_signal("latest", config=selected_config, factors=factors)
```

This uses formal evidence for the gate but generates the actual signal from a different strategy family.

##### Correct

```python
routed = run_annual_state_score_router(...)
bt_config = routed_backtest_config(...)
result = run_backtest(routed.scores, prices, start_date, end_date, bt_config)
signal_df, holdings = generate_signal("latest", config=route_config, scores=routed.scores)
```

The current backtest and current signal both consume the same routed score panel and route schedule.

### Scenario: Quant Backtest Diagnostic Report

#### 1. Scope / Trigger

- Trigger: users need to diagnose unprofitable backtests before optimizing strategy parameters.
- Owners: `src.quant_diagnostics` builds artifact-level checks; `scripts/run_quant_diagnostics.py` writes the report.

#### 2. Signatures

- Command: `.\.venv\Scripts\python.exe scripts\run_quant_diagnostics.py [--artifact-dir outputs] [--compare-dir outputs_rerun] [--out-dir outputs] [--tolerance 1e-6]`.
- API: `build_quant_diagnostic_report(artifact_dir="outputs", compare_dir=None, tolerance=1e-6) -> dict`.
- API: `write_quant_diagnostic_report(report, out_dir="outputs") -> dict[str, str]`.
- Outputs: `outputs/quant_diagnostic_report.json` and `outputs/quant_diagnostic_report.md`.

#### 3. Contracts

- The report is an aggregator over existing artifacts; it must not rerun backtests, optimization, data downloads, or signal promotion.
- Checks are grouped into `backtest_engine`, `data`, `factor`, `portfolio`, and `optimization` layers.
- Every check has `layer`, `name`, `status`, `summary`, `evidence`, and `caveats`.
- Valid check statuses are `pass`, `warn`, and `fail`.
- Missing optional evidence is a `warn`; failed accounting/data-governance invariants are `fail`.
- `optimization_ready` is true only when the backtest engine, data, factor, and portfolio layers all pass.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Required backtest artifacts are missing | `required_artifacts` fails and optimization is blocked |
| `--compare-dir` is absent | Reproducibility check warns with a caveat |
| Comparable artifacts differ | Reproducibility check fails |
| Trade-cost metric differs from cost columns | Trade-cost invariant fails |
| Holdings value exceeds equity | Cash/equity invariant fails |
| Data health/governance reports contain blocking issues | Data layer fails |
| IC or group-return artifacts are missing | Factor layer warns and optimization is blocked |

#### 5. Good/Base/Bad Cases

- Good: two artifact directories match exactly, accounting invariants pass, data/governance reports are ready, and factor/portfolio evidence exists.
- Base: one artifact directory is available; reproducibility warns, but accounting/data/factor/portfolio evidence is still reported.
- Bad: optimizing parameters while data health or point-in-time governance still fails.

#### 6. Tests Required

- Unit test complete artifact directories that pass every layer.
- Unit test missing artifacts and caveats.
- Unit test trade-cost and holding roll-forward invariant failures.
- Unit test reproducibility mismatch.
- Unit test JSON and Markdown report writers.

#### 7. Wrong vs Correct

##### Wrong

```powershell
.\.venv\Scripts\python.exe scripts\run_optimize.py --full-grid
```

Starting optimization while data or accounting diagnostics are still unresolved can overfit around invalid evidence.

##### Correct

```powershell
.\.venv\Scripts\python.exe scripts\run_quant_diagnostics.py --artifact-dir outputs --compare-dir outputs_rerun
```

Run the ordered diagnostic gate first, then optimize only after earlier layers pass.

### Scenario: Factor Diagnostic Evidence Tables

#### 1. Scope / Trigger

- Trigger: the five-layer quant diagnostic report requires factor IC, yearly stability, and quantile spread evidence before optimization is considered.
- Owners: `src.factor_diagnostics` builds tables; `scripts/run_factor_diagnostics.py` loads configured factor/price artifacts and writes outputs.

#### 2. Signatures

- Command: `.\.venv\Scripts\python.exe scripts\run_factor_diagnostics.py [--factor-groups momentum] [--out-dir outputs] [--horizon 1] [--method spearman] [--min-obs 20] [--quantiles 5]`.
- API: `build_factor_diagnostics(factor_df, price_df, horizon=1, method="spearman", min_obs=20, quantiles=5) -> dict[str, DataFrame]`.
- API: `write_factor_diagnostics(tables, out_dir="outputs") -> dict[str, str]`.
- Outputs: `outputs/factor_daily_ic.csv`, `outputs/factor_ic_summary.csv`, `outputs/factor_ic_yearly.csv`, and `outputs/factor_group_returns.csv`.

#### 3. Contracts

- Default command scope follows the configured strategy factor group to avoid scanning every cached factor unless `--factor-groups all` is explicit.
- Factor input must use the standard `datetime`/`instrument` MultiIndex contract.
- Price input must use a supported close-price panel or OHLCV field/instrument panel.
- `factor_ic_summary.csv` contains `mean_ic`, `std_ic`, `ic_ir`, `positive_ratio`, and `count`.
- `factor_ic_yearly.csv` groups IC summary by calendar year and factor.
- `factor_group_returns.csv` stores per-date factor quantile forward returns and `top_minus_bottom` spreads.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Factor frame lacks a MultiIndex | Raise `ValueError("factor_df must use MultiIndex: datetime/instrument.")` |
| Price file is missing | `scripts/run_factor_diagnostics.py` raises `FileNotFoundError` telling the user to run conversion first |
| A factor/date has too few observations | Skip that date/factor quantile row instead of fabricating evidence |
| `--factor-groups all` is used | Load all cached factor columns intentionally |

#### 5. Good/Base/Bad Cases

- Good: configured factor group produces non-empty IC, yearly IC, and group-return tables.
- Base: a sparse factor skips some group-return rows but still writes the available evidence.
- Bad: treating a missing group-return file as proof the factor has no spread.

#### 6. Tests Required

- Unit test that synthetic factor/price panels produce IC summary, yearly IC, and group-return spread columns.
- Writer test that all four CSV outputs are persisted.
- Quant-diagnostic test that factor layer passes when the three required evidence files exist.

#### 7. Wrong vs Correct

##### Wrong

```powershell
.\.venv\Scripts\python.exe scripts\run_quant_diagnostics.py
```

Running the final diagnostic before factor evidence exists leaves the factor layer in `warn`.

##### Correct

```powershell
.\.venv\Scripts\python.exe scripts\run_factor_diagnostics.py --factor-groups momentum
.\.venv\Scripts\python.exe scripts\run_quant_diagnostics.py
```

Generate the missing factor evidence first, then re-run the five-layer gate.

### Scenario: Post-Diagnostic Optimization Review

#### 1. Scope / Trigger

- Trigger: the first four diagnostic layers passed and users need an evidence-backed optimization decision.
- Owners: `src.optimization_review` reads existing auto-run diagnostics; `scripts/run_optimization_review.py` writes the review.

#### 2. Signatures

- Command: `.\.venv\Scripts\python.exe scripts\run_optimization_review.py [--artifact-dir outputs] [--out-dir outputs]`.
- API: `build_optimization_review(artifact_dir="outputs") -> dict`.
- API: `write_optimization_review(report, out_dir="outputs") -> dict[str, str]`.
- Outputs: `outputs/optimization_review.json` and `outputs/optimization_review.md`.

#### 3. Contracts

- The review must not optimize parameters, rerun backtests, or promote signals.
- `status` is `ready` only when `quant_diagnostic_report.json` has `optimization_ready=true` and the auto-signal/backtest quality artifacts are executable/acceptable.
- Performance comparison reads baseline `backtest_metrics.json` and optimized `auto_backtest_metrics.json`.
- Style recognition reads annual-state-router route CSVs and summarizes source/reason/exposure distribution.
- Risk flags are warnings, not automatic blockers, for low position count, industry concentration, and market-cap bucket concentration.
- Trading flags identify annual trade-cost or rebalance-trim cost pressure and should prevent further turnover loosening.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Quant diagnostic gate is not ready | `status=review`; recommendations must not imply optimization can proceed |
| Auto signal is not executable or backtest quality is unacceptable | `status=review` |
| Baseline/optimized metrics are missing | Performance deltas are `null` rather than fabricated |
| Latest holdings are concentrated | Add risk flags, but keep the review artifact writable |
| Annual trade-cost ratio exceeds target | Add a trading flag and recommend against more turnover |

#### 5. Tests Required

- Unit test baseline versus optimized performance deltas.
- Unit test annual router source/reason summaries.
- Unit test risk flags for low position count, industry concentration, and small-cap concentration.
- Unit test trading flags for high annual trade-cost ratios.
- Writer test that JSON and Markdown review artifacts are persisted.

### Scenario: Evidence-Backed Optimization Plan

#### 1. Scope / Trigger

- Trigger: diagnostics and optimization review are ready, and the next step is to optimize style routing, risk exposure, and trading constraints from trusted evidence.
- Owners: `src.evidence_optimizer` builds the plan; `scripts/run_evidence_optimizer.py` writes it.

#### 2. Signatures

- Command: `.\.venv\Scripts\python.exe scripts\run_evidence_optimizer.py [--artifact-dir outputs] [--out-dir outputs] [--grid-glob *router_grid*.csv]`.
- API: `build_evidence_optimization_plan(artifact_dir="outputs", grid_glob="*router_grid*.csv", max_industry_weight_target=0.35, annual_trade_cost_ratio_target=0.20) -> dict`.
- API: `write_evidence_optimization_plan(report, out_dir="outputs") -> dict[str, str]`.
- Outputs: `outputs/evidence_optimization_plan.json` and `outputs/evidence_optimization_plan.md`.

#### 3. Contracts

- The plan reads existing diagnostics, optimization review, selected auto params, and router-grid CSV evidence; it must not mutate `config/settings.yaml` or promote signals.
- A style candidate is eligible only when the grid row has `full_goal=true` and `annual_trade_cost_ratio <= annual_trade_cost_ratio_target`.
- Risk flags become research overlays: high industry concentration maps to `max_industry_weight_target`, low latest positions maps to a target minimum position count, and small-cap concentration maps to a small-cap reduction action.
- Trading flags keep turnover from increasing and preserve the selected full-goal candidate's turnover mode/boost settings for follow-up grid commands.
- The generated next command must include `--max-industry-weights` when an industry cap overlay is recommended.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Optimization review is not ready | `status=review` with a caveat |
| No router-grid evidence has required columns | `status=review` with a caveat |
| Full-goal candidates exist but exceed the cost target | `status=review`; do not fabricate a candidate |
| High industry concentration is flagged | Add a `max_industry_weight` research overlay |
| Candidate is selected | Emit JSON/Markdown and a resumable router-grid command |

#### 5. Tests Required

- Unit test full-goal/cost-eligible candidate selection.
- Unit test risk flags become max-industry, min-position, and small-cap actions.
- Unit test high-cost trading flags produce `do_not_increase_turnover`.
- Writer test that JSON and Markdown plan artifacts are persisted.

### Scenario: Backtest Exposure Schedule Composition

#### 1. Scope / Trigger

- Trigger: routers, regime overlays, and defensive timing may all want to scale portfolio exposure in the same backtest.
- Owners: `src/market_regime.py` produces defensive timing schedules; `src/backtest.py` consumes the final `exposure_schedule`.

#### 2. Signatures

- Backtest config key: `exposure_schedule`.
- Defensive timing adapter: `apply_defensive_timing_to_backtest_config(bt_config, price_df, config)`.

#### 3. Contracts

- If `bt_config` already has an `exposure_schedule`, defensive timing must multiply its schedule by the existing schedule instead of replacing it.
- Existing exposure schedule dates are normalized to trading dates, sorted, de-duplicated with the latest value, and forward-filled onto the defensive timing date index.
- Missing existing schedule values default to `1.0`, so defensive timing alone preserves its prior behavior.
- The composed schedule remains named `exposure_scale` and is consumed by the normal backtest exposure machinery.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Existing `exposure_schedule` is missing or empty | Use defensive timing schedule unchanged |
| Existing schedule has invalid dates or non-numeric values | Drop invalid entries; if none remain, use defensive timing unchanged |
| Existing route exposure is `0.5` and defensive bear exposure is `0.4` | Final composed exposure is `0.2` |

#### 5. Tests Required

- `tests/test_market_regime.py` must assert existing route exposure and defensive timing exposure are multiplied.

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
