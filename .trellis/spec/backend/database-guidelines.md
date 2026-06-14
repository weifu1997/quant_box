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

### Scenario: Raw Daily Suspension And Adj-Factor Metadata Gates

#### 1. Scope / Trigger

- Trigger: raw Tushare daily CSVs and adj-factor metadata feed data health/governance gates for the auto-signal workflow.
- Storage owners: `src/data_fetcher_frames.py`, `src/adj_factor_metadata.py`, and `src/data_governance.py`.

#### 2. Signatures

- Raw stock file: `data/raw/<TS_CODE>.csv`, for example `data/raw/600717.SH.csv`.
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
  - `scripts/run_annual_state_router_grid.py --cache-dir outputs/router_score_cache --output outputs/<name>.csv --write-hit-prefix outputs/<prefix>`
- Expanded source names currently include `roc60`, `db_total`, `beta20`, and `rsqr20`.

#### 3. Contracts

- Score caches are parquet files under `outputs/router_score_cache/<source>_<end-date>.parquet` with a `score` column and a two-level date/instrument index.
- Grid output is append-only CSV keyed by the stable `combo_key`; reruns skip keys already present in the output CSV.
- `turnover_boost_reason_sets` uses semicolon-separated reason sets; reasons inside one set use `+`, for example `low_vol_moderate_uptrend+moderate_positive_roc60`.
- `moderate_low_exposure` multiplies route exposure only when `moderate_low_source` is selected for a `default_beta` route whose `ret252` is in `[moderate_low_ret252_min, moderate_low_ret252_max)`.
- A full-gate hit writes `_metrics.json`, `_years.csv`, `_year_routes.csv`, `_score_routes.csv`, `_holdings.csv`, `_trades.csv`, and `_equity.csv` beside the requested hit prefix.

#### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Score cache exists and `--force-rebuild-cache` is absent | Load the cached parquet source. |
| Score cache is missing | Build the source from configured factor/selector inputs and write the parquet cache. |
| `turnover_boost_reasons=none` | Do not boost route turnover; collapse boost max/rank combinations to a single grid row. |
| `moderate_low_source=none` | Do not route the low ret252 band; collapse `moderate_low_exposure` to `1.0`. |
| Routed source is missing from score sources | Raise `ValueError("Routed source is not in score sources: ...")`. |
| Hit satisfies annual return, drawdown, turnover, cost, and yearly gates | Write hit artifacts and stop the grid early. |

#### 5. Good/Base/Bad Cases

- Good: `moderate_low_source=beta20`, `moderate_low_exposure=0.4`, and `turnover_mode=rank10` produce a formal hit whose cost gate still passes.
- Base: no expanded low-band source; route remains `default_beta`, scalar exposure remains `1.0`.
- Bad: proving yearly returns from standalone candidate equity files only; path-dependent annual routing must be reconstructed through score panels, holdings, trades, costs, and gates.

#### 6. Tests Required

- `tests/test_run_annual_state_router_backtest.py` must assert expanded source definitions and route exposure scaling.
- `tests/test_run_annual_state_router_grid.py` must assert reason-set parsing and `moderate_low_exposure` enumeration/collapse.
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
