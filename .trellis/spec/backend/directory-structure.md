# Directory Structure

> Backend organization rules for this local A-share quant research project.

---

## Overview

`quant_box` is a single-repo Python project. There is no web API layer and no package split. Core reusable logic lives in `src/`, command-line entry points live in `scripts/`, Windows one-click wrappers live as numbered `.bat` files in the repo root, and tests mirror behavior under `tests/`.

The project is a local data pipeline for manual trading decisions. Most changes should preserve this separation:

- `src/` contains importable business logic, data normalization, scoring, risk, backtest, reporting, and file writers.
- `scripts/` contains thin orchestration and CLI parsing around `src/` modules.
- `config/` contains committed defaults and examples only.
- `data/` and `outputs/` contain local generated caches and reports; they are ignored except `.gitkeep` files and committed test fixtures.
- `tests/fixtures/data_snapshot/` is the deterministic committed market-data slice used by tests.

---

## Directory Layout

```text
quant_box/
  src/
    config_loader.py          config merge, validation, and repo-root path resolution
    common.py                 shared normalization helpers
    tushare_client.py         HTTP proxy client and Tushare response parsing
    data_fetcher*.py          Tushare update, raw CSV, point-in-time caches
    data_converter.py         raw CSV -> Qlib provider + local price panels
    factor_*.py               Alpha158, IC, rolling IC, factor cache helpers
    scoring.py                score construction and factor selection logic
    strategy.py               selection and rebalance logic
    risk_policy.py            central adapter for selection/execution risk controls
    backtest*.py              backtest engine, costs, circuit breaker, exposure
    *_data.py                 fundamental, governance, health, diagnostics modules
    signal_generator.py       signal and holdings file generation
    manual_orders.py          manual order and fill-feedback workflow
    reporting.py              Markdown/JSON report rendering and archive copying
  scripts/
    _shared.py                helpers reused by multiple script entry points
    run_*.py                  CLI orchestration scripts
    check_tushare_config.py   safe local config check
    benchmark_scoring.py      lightweight performance probe
  tests/
    test_<module>.py          unit/regression tests
    fixtures/real_data.py     real-data fixture loader
    fixtures/data_snapshot/   committed deterministic parquet snapshot
  config/
    settings.yaml             committed defaults
    settings.local.yaml       local secrets/overrides, ignored
    account.example.yaml      committed example only
  data/
    raw/                      generated raw CSV caches, ignored
    qlib_data/                generated Qlib provider data, ignored
    factors/                  generated factor/fundamental parquet caches, ignored
    prices/                   generated price panels, ignored
  outputs/                    generated reports, logs, signals, ignored
```

---

## Module Boundaries

### Reusable logic belongs in `src/`

Put data validation, DataFrame transformations, score construction, risk logic, and report-building functions in `src/`. Keep functions importable and testable without invoking the command line.

Examples:

- `src/common.py` owns normalization primitives such as `normalize_instrument`, `normalize_datetime_index`, and `close_price_frame`.
- `src/data_fetcher_frames.py` owns normalization of Tushare frames before storage.
- `src/risk_policy.py` is the single adapter that reads configured risk controls and applies them to selection/backtest paths.

### CLI orchestration belongs in `scripts/`

Every long-running user workflow should have a `scripts/run_*.py` entry point with this shape:

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
```

Examples: `scripts/run_auto_signal.py`, `scripts/run_backtest.py`, `scripts/run_update_data.py`, `scripts/run_convert_data.py`.

Use `scripts/_shared.py` for helpers that are needed by multiple scripts, such as dated output paths, yearly summaries, requested factor column resolution, and parquet subsets.

### Root `.bat` files are user shortcuts

The numbered `.bat` files are part of the product surface for Windows users. When a script behavior changes in a user-visible way, update the matching `.bat`, `README.md`, and `tests/test_scripts_docs.py`.

---

## Data And Path Conventions

Use `src.config_loader.resolve_path()` for project-relative paths. Do not manually join `Path.cwd()` with configured paths; scripts may be launched from wrappers or terminals.

Key path contracts:

- Config defaults live in `config/settings.yaml`.
- Local private overrides live in `config/settings.local.yaml` and environment variables.
- Raw daily stock files are `data/raw/<TS_CODE>.csv`, for example `data/raw/000001.SZ.csv`.
- Qlib features are generated under `data/qlib_data/features/<lowercase-code>/day.parquet`.
- Price panels are generated under `data/prices/`, especially `ohlcv_adjusted.parquet`.
- Factor caches live under `data/factors/`, especially `alpha158.parquet` and `rolling_ic_weights.parquet`.
- Automatic workflow outputs live under `outputs/`, with logs under `outputs/logs/`.

---

## DataFrame Contracts

Most backend modules communicate with pandas objects. Preserve these contracts:

- Factor and score data uses a two-level MultiIndex named `datetime`/`instrument` or `date`/`instrument`.
- Price panels with multiple fields use MultiIndex columns named `field`/`instrument`.
- Plain price DataFrames are close-price panels with instrument columns only.
- Instruments are normalized to uppercase Tushare symbols such as `000001.SZ` for in-memory logic.
- Qlib feature directories and generated instrument text files use lowercase codes where Qlib expects them.
- Dates are normalized with `pd.Timestamp(...).normalize()` or `normalize_datetime_index`.

Example from `src/signal_generator.py`:

```python
if scores.empty or not isinstance(scores.index, pd.MultiIndex):
    raise ValueError("scores must use MultiIndex: datetime/instrument.")
```

Example from `src/common.py`:

```python
if looks_like_field_table(price_df.columns):
    raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
```

---

## Naming Conventions

- Public module functions use descriptive verbs: `load_or_compute_factors`, `generate_signal`, `run_backtest`, `write_daily_signal_report`.
- Internal helpers are prefixed with `_`: `_latest_daily_scores`, `_write_status`, `_configure_run_logging`.
- Dataclasses are used for result bundles and small immutable adapters: `OptimizationStageResult`, `DataPreparationStageResult`, `RiskPolicy`.
- Test files are named `tests/test_<module>.py`; test classes usually use `unittest.TestCase`.
- Script files use `run_<workflow>.py` when they execute a user workflow, and concise imperative names for checks/exporters.

---

## Adding New Backend Work

1. Put reusable behavior in `src/`, not in a script body.
2. Add a `scripts/run_*.py` only when users need a command-line workflow.
3. Resolve configured paths with `resolve_path`.
4. Keep generated files under `data/` or `outputs/`, not beside source files.
5. Add focused tests under `tests/`, using `TemporaryDirectory` and patching `load_config`/`resolve_path` when filesystem isolation matters.
6. Update `README.md`, `.bat` wrappers, and `tests/test_scripts_docs.py` for user-visible workflow changes.

---

## Good Examples

- `src/config_loader.py` centralizes defaults, local overrides, env expansion, validation, and path resolution.
- `src/data_converter.py` shows the raw CSV -> Qlib -> local price-panel boundary.
- `src/risk_policy.py` prevents risk settings from being copied across signal and backtest paths.
- `scripts/run_auto_signal.py` breaks a long workflow into stage result dataclasses and writes resumable status artifacts.
- `tests/fixtures/real_data.py` provides deterministic real-data fixtures without depending on private local caches.
