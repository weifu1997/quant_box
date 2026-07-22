# Directory Structure

> Backend organization rules for this local A-share quant research project.

---

## Overview

`quant_box` is a single-repo Python project with a small local web API for controlled dashboard review and no package split. Core reusable logic lives in `src/`, command-line entry points live in `scripts/`, the React/Vite dashboard lives in `web/`, Windows one-click wrappers live as numbered `.bat` files in the repo root, and tests mirror behavior under `tests/`.

The project is a local data pipeline for manual trading decisions. Most changes should preserve this separation:

- `src/` contains importable business logic, data normalization, scoring, risk, backtest, reporting, and file writers.
- `scripts/` contains thin orchestration and CLI parsing around `src/` modules.
- `web/` contains the local React/Vite dashboard frontend; generated `web/node_modules/` and `web/dist/` stay ignored.
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
    annual_router.py          annual market-state routing and route-derived schedules
    auto_signal/              importable auto-signal stages, models, status, router integration
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
  web/
    package.json              React/Vite dashboard frontend
    src/                      dashboard components, API client, styles
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

Annual-router entrypoints follow the same boundary: `src/annual_router.py` owns the engine contract, market-state decisions, routed score panels, turnover transforms, and route-derived schedules. The backtest, grid, and probe scripts own CLI parsing, file loading, score-source construction, grid iteration, and report writing. Existing script modules may re-export engine names for compatibility, but production callers should import reusable behavior from `src.annual_router`.

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

## Scenario: Importable Auto-Signal Stage Pipeline

### 1. Scope / Trigger

- Trigger: the automatic signal workflow needs reusable, independently testable stages without importing CLI implementation code.
- Owners: `src/auto_signal/` owns stage behavior and run contracts; `scripts/run_auto_signal.py` owns argument parsing, dependency wiring, linear orchestration, top-level failure handling, and compatibility exports.

### 2. Signatures

- Stage entrypoints:
  - `run_data_preparation_stage(...) -> DataPreparationStageResult`
  - `run_optimization_stage(...) -> OptimizationStageResult`
  - `run_backtest_stage(...) -> BacktestStageResult`
  - `run_signal_stage(...) -> SignalStageResult`
  - `write_auto_report_stage(...) -> ReportStageResult`
- Shared status primitive: `src.auto_signal.status.stage(status, out_dir, name, state, message="")`.
- CLI remains `scripts/run_auto_signal.py` with the existing flags and defaults.

### 3. Contracts

- Stage functions receive parsed arguments and typed result bundles; they never parse `sys.argv`.
- Production modules under `src/auto_signal/` must not import `scripts.run_auto_signal`.
- Script compatibility wrappers construct typed service bundles from script-level collaborators. This preserves established patch/injection seams while keeping production implementations independent from the CLI module.
- Backtest and signal stages share the same `AnnualStateRouterRuntime`; the signal stage must not rebuild a second routed score panel.
- `stage()` remains append-and-write: every transition appends one row and immediately rewrites `outputs/auto_run_status.json`.
- Candidate/official output decisions remain owned by the signal stage and use all data, governance, parameter, backtest, and account gates.
- Report/archive stages consume prior result bundles; they do not recompute gates or promote candidate outputs.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Stage input is invalid or a required artifact is missing | Preserve the existing precise `ValueError` / `FileNotFoundError` / `RuntimeError` contract |
| Optimization exceeds its time budget | Persist partial validation artifacts, record `optimizer_timeout`, append a timeout stage, then re-raise `OptimizationTimeoutError` |
| Annual router is enabled | Optimization is skipped; backtest builds one router runtime and signal reuses it |
| A quality or account gate fails | Signal stage writes candidate artifacts and non-empty `block_reasons`; official state is not overwritten |
| A stage raises unexpectedly | CLI orchestration records final `status=failed`, `last_error`, and a failed `run` stage before re-raising |

### 5. Good/Base/Bad Cases

- Good: a test injects a fake `run_backtest` through `BacktestStageServices` and verifies routed scores and schedules without importing the CLI parser.
- Base: `scripts/run_auto_signal.py` wires default services, calls stages in order, and produces byte/schema-compatible artifacts.
- Bad: a production stage imports `scripts.run_auto_signal` to reach a helper, creating a circular CLI dependency.
- Bad: signal/report stages independently recalculate quality gates and disagree about executable status.

### 6. Tests Required

- `tests/test_auto_signal_stages.py` must assert every stage entrypoint is importable and no `src.auto_signal` module imports the CLI script.
- `tests/test_run_auto_signal.py` must continue to assert CLI flags, status transitions, annual-router behavior, candidate/official boundaries, timeout artifacts, and reports.
- Full tests must cover both annual-router and legacy optimizer paths after any stage-boundary change.

### 7. Wrong vs Correct

#### Wrong

```python
# src/auto_signal/backtest_stage.py
from scripts.run_auto_signal import _build_annual_state_router_runtime
```

#### Correct

```python
services = BacktestStageServices(
    annual_state_router_enabled=_annual_state_router_enabled,
    build_annual_state_router_runtime=_build_annual_state_router_runtime,
    run_backtest=run_backtest,
)
result = run_backtest_stage(..., services=services)
```

The CLI owns wiring; the importable stage owns behavior.

---

## Scenario: Local Web Dashboard

### 1. Scope / Trigger

- Trigger: the project exposes local auto-signal review artifacts and a narrow set of controlled repair/rerun actions through a local web dashboard.
- Owners: `src/dashboard.py` builds the view model, `src/dashboard_control.py` owns whitelisted background actions, `src/dashboard_api.py` exposes FastAPI routes, `scripts/run_dashboard.py` starts the backend, and `web/` owns the React/Vite UI.

### 2. Signatures

- Command: `.\.venv\Scripts\python.exe scripts\run_dashboard.py [--host 127.0.0.1] [--port 8000] [--reload]`.
- Batch wrapper: `15_启动Web仪表盘.bat` starts the FastAPI backend and React/Vite frontend, waits for `http://127.0.0.1:8000/api/health` and `http://127.0.0.1:5173`, then opens the dashboard URL.
- API: `GET /api/health -> {"status": "ok"}`.
- API: `GET /api/dashboard/latest -> DashboardSnapshot`.
- API: `GET /api/dashboard/precheck -> DashboardPrecheck`.
- API: `GET /api/dashboard/jobs -> {"jobs": DashboardJob[], "active_job": DashboardJob|null}`.
- API: `POST /api/dashboard/jobs` with JSON `{"action": "repair_point_in_time"}` or `{"action": "run_auto_signal", "mode": "candidate"|"normal"} -> {"job": DashboardJob}`.
- API: `POST /api/dashboard/jobs/{job_id}/stop -> {"job": DashboardJob}` stops a running dashboard job and its child process tree.
- API: `GET /api/dashboard/artifacts/{artifact_id} -> FileResponse` for downloadable artifacts under the configured output directory.
- Frontend dev command: `cd web && npm run dev`, with Vite proxying `/api` to `http://127.0.0.1:8000`.

### 3. Contracts

- The dashboard may start only whitelisted local actions:
  - `repair_point_in_time`: runs `scripts/run_update_point_in_time_data.py` for the `daily_basic` repair window and includes `--skip-index-constituents --skip-st-calendar`.
  - `run_auto_signal` with `mode="candidate"`: runs `scripts/run_auto_signal.py --no-archive --candidate-only`.
  - `run_auto_signal` with `mode="normal"`: runs `scripts/run_auto_signal.py --no-archive` and must not add `--candidate-only` or `--force-official`.
- Dashboard actions must not expose arbitrary command execution, edit configs, promote candidate signals, apply fills, or directly update holdings.
- Only one dashboard background job may run at a time. The persisted running-job check, initial status write, process spawn, and in-memory process registration must be serialized as one start reservation so concurrent POST requests cannot both pass the check. Job status JSON lives under `outputs/dashboard_jobs/`; logs live under `outputs/logs/dashboard_job_*.log`.
- Dashboard job status writes must use a same-directory temporary file plus atomic replacement. Terminal updates must re-read the latest persisted status before deciding between `succeeded` / `failed` and `cancelled`, so a concurrent stop request cannot be overwritten by a stale in-memory `running` snapshot.
- Dashboard job stop is a controlled action, not arbitrary process management. It may only target a recorded dashboard job whose status is `running` or `stopping`; successful user stops end as `status="cancelled"`. After a service restart, a live PID counts as the recorded job only when its command line can be read and matched to the stored command. Missing command-line evidence must fail closed: mark the job `stale` and never terminate that PID.
- The backend reads only the latest/current artifacts from `outputs.dir`, especially `auto_signal_report.json`, `auto_run_status.json`, `daily_signal_report.md`, and CSV/JSON paths referenced by the latest report.
- Missing or malformed artifacts become explicit dashboard statuses (`missing` / `error`) instead of uncaught exceptions.
- `DashboardSnapshot` must keep the frontend decoupled from large raw report JSON by returning compact sections: `readiness`, `latest_run`, `gates`, `block_reasons`, `blocker_actions`, `quality_warnings`, `signal_summary`, `orders`, `artifacts`, and `report`.
- `DashboardSnapshot.orders.rows` may enrich manual-order CSV rows with `name` from the configured stock universe file. The backend owns this lookup; the frontend must not read `data/raw/mainboard_a_stocks.csv` directly.
- `blocker_actions` is the structured repair center contract. It maps each current blocker or stale-report freshness note to a user-facing title/detail, severity, normalized issue id, an optional whitelisted dashboard job action, and an optional bounded `report_artifact`. The frontend must not derive shell commands or mark blockers fixed locally.
- Parameter/backtest quality blockers must preserve the configured gate and candidate-only outcome. Their detail should identify the actual failing evidence; when the authoritative quality JSON exists, `report_artifact.downloadable=true` and the UI renders a real `/api/dashboard/artifacts/{id}` link. When it is absent, the UI says `报告不可用` and must not render a fake clickable `查看报告` label.
- `DashboardPrecheck` is a read-only pre-run evidence model with `status`, `summary`, `can_run_normal`, `target_date_resolution`, and compact `items`. It must not download data, recompute factors, write signals, promote candidate artifacts, or mutate account/holding files.
- Precheck items cover target trading date, data health evidence, point-in-time governance evidence, factor freshness, and account/current holdings. Missing artifacts become `status="missing"` or `warn`, not fabricated passes. Failed daily-basic governance may expose only the existing `repair_point_in_time` job action; factor/data blockers may expose only the whitelisted `run_auto_signal` action.
- If `data_governance_report.json` is newer than `auto_signal_report.json`, the dashboard governance gate must use the standalone governance report instead of the stale governance snapshot embedded in the auto-signal report. Resolved stale governance block reasons should be filtered from dashboard `block_reasons` / `quality_warnings`, and `freshness_notes` should tell the frontend that the auto-signal report still needs a rerun to refresh the final verdict.
- `DashboardJob` must keep the frontend decoupled from process internals by returning compact fields: `id`, `action`, `mode`, `label`, `status`, `message`, `command`, `started_at`, `completed_at`, `return_code`, `log_path`, `log_tail`, and `progress`.
- `DashboardJob.progress` contains `summary`, `percent`, `active_step`, and compact step rows. Auto-signal progress is derived from `outputs/auto_run_status.json` only when that status belongs to the current job; daily-basic repair progress is inferred from the controlled job log tail.
- Artifact download routes must be constrained to files inside the resolved output directory.
- Frontend source belongs under `web/src/`; generated `web/node_modules/` and `web/dist/` stay ignored.

## Scenario: Dashboard Trade Execution And Holdings Update

### 1. Scope / Trigger

- Trigger: the local Web dashboard records real broker fills from the latest official execution template and applies them to `config/current_holdings.csv`.
- Owners: `src/dashboard_execution.py` owns template selection, editable-field merging, validation, atomic writes, and audit output; `src/dashboard_api.py` exposes the API; `web/src/ExecutionWorkspace.tsx` owns user input and explicit confirmation.

### 2. Signatures

- API: `GET /api/dashboard/execution -> ExecutionWorkspaceData`.
- API: `POST /api/dashboard/execution/preview` with `{"source_id": str, "rows": ExecutionFillRow[]} -> ExecutionPreview`.
- API: `POST /api/dashboard/execution/apply` with `{"source_id": str, "rows": ExecutionFillRow[], "confirm": true} -> ExecutionApplyResult`.
- Source template: latest `outputs/fill_feedback/fill_feedback_YYYY-MM-DD.csv` or configured `manual_orders.fill_feedback_dir` equivalent.
- Holdings output: configured `account.current_holdings_file`, normally `config/current_holdings.csv`.
- Audit output: `outputs/fill_apply_audit_<signal-date>.json`.

### 3. Contracts

- Only official templates named `fill_feedback_*.csv` are eligible. `fill_feedback_candidate_*.csv` must never be returned as an applicable template.
- The backend resolves the latest template and returns its filename as `source_id`. Preview/apply requests must send that exact id; stale or user-selected paths are rejected.
- Every original row must be submitted exactly once with its backend-provided `row_id`.
- Web-editable fields are limited to `fill_status`, `actual_trade_date`, `executed_shares`, `executed_price`, costs, broker id, slippage note, and fill note.
- `instrument`, `side`, `planned_order_shares`, signal date, and other plan fields always come from the stored template. Payload attempts to change them are ignored.
- Preview reuses `validate_fill_feedback` and `apply_fill_feedback` and never writes files.
- Apply requires `confirm=true`, repeats validation, atomically replaces the fill CSV and holdings CSV, and writes a JSON audit.
- The existing validation rules remain authoritative: `PENDING` is invalid for application, FILLED/PARTIAL require positive executed shares, execution cannot exceed planned size, and sells cannot make holdings negative.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| No official fill template exists | GET returns `status="missing"`; preview/apply returns 404 |
| Only candidate template exists | Treat as no official template; never allow holdings update |
| `source_id` differs from current latest template | 400 asking the user to refresh the execution workspace |
| Rows are missing, duplicated, or have invalid ids | 400; no files are written |
| Payload changes instrument/side/planned shares | Ignore those values and preserve stored template fields |
| Any row remains `PENDING` | Preview returns `valid=false`; apply returns 400 |
| Executed shares exceed planned shares | Preview invalid; apply rejected |
| Sell would make a position negative | Preview invalid; apply rejected |
| `confirm` is not exactly `true` | Apply returns 400; holdings remain unchanged |
| Validation succeeds | Save fill feedback, update holdings, and write audit JSON |

### 5. Good/Base/Bad Cases

- Good: an official BUY row is marked FILLED for its planned quantity, preview shows the increased position, explicit confirmation updates holdings, and an audit file is written.
- Base: no official signal exists yet; the UI explains that there is no applicable execution template.
- Good: CANCELLED or SKIPPED rows do not change holdings and do not require executed shares.
- Bad: a browser request changes a SELL row into BUY or increases planned shares; backend merging must ignore both attempted changes.
- Bad: a candidate fill template is applied to official holdings.

### 6. Tests Required

- `tests/test_dashboard_execution.py` must prove candidate templates are excluded.
- Tests must prove immutable plan fields survive malicious payload overrides.
- Tests must prove apply requires explicit confirmation and writes holdings plus audit only after validation.
- Tests must prove a stale `source_id` is rejected.
- API tests must assert validation errors map to HTTP 400 and missing templates map to HTTP 404.
- Existing `tests/test_run_apply_fills.py` and `src.manual_orders.validate_fill_feedback` tests remain the source of truth for fill validation semantics.

### 7. Wrong vs Correct

#### Wrong

```python
fills = pd.DataFrame(payload["rows"])
save_updated_holdings(apply_fill_feedback(current, fills), config)
```

This trusts browser-supplied instruments, sides, and planned quantities and can apply a candidate or stale template.

#### Correct

```python
original = pd.read_csv(latest_official_fill_path)
fills = merge_only_editable_fields(original, payload["rows"])
issues = validate_fill_feedback(current, fills)
if issues or payload.get("confirm") is not True:
    reject_without_writes()
```

The stored official template owns the order plan; the browser supplies execution results only.

## Scenario: Dashboard Account And Holdings Management

### 1. Scope / Trigger

- Trigger: Web users create or update the account inputs required for official manual orders without editing YAML/CSV by hand.
- Owners: `src/dashboard_account.py` owns the field whitelist, validation, backup, and atomic file replacement; `src/dashboard_api.py` exposes the API; `web/src/AccountWorkspace.tsx` owns the form.

### 2. Signatures

- API: `GET /api/dashboard/account -> AccountWorkspaceData`.
- API: `POST /api/dashboard/account/preview` with `{"account": AccountFormData, "holdings": AccountHoldingRow[]} -> AccountPreview`.
- API: `POST /api/dashboard/account/apply` with the preview payload plus `"confirm": true -> AccountApplyResult`.
- Account output: configured `account.file`, normally `config/account.yaml`.
- Holdings output: configured `account.current_holdings_file`, normally `config/current_holdings.csv`.
- Backups: `outputs/account_backups/<timestamp>/account.yaml` and/or `current_holdings.csv` when prior files exist.

### 3. Contracts

- The API exposes and accepts only `total_asset`, `cash`, `max_position_pct`, `lot_size`, `star_market_lot_size`, and holdings rows with `instrument`/`shares`.
- Tushare credentials, strategy settings, paths, and all unrelated local config must never be included in the response or accepted from the form.
- Account numbers must be finite. Total asset must be positive; cash cannot be negative; max position is empty or within `[0, 1]`; lot sizes are positive integers.
- Holdings instruments are normalized to uppercase Tushare codes. Existing `validate_account_inputs` / `validate_current_holdings` remain authoritative for duplicates, invalid symbols, negative/fractional shares, and lot multiples.
- Preview performs validation only and never writes files.
- Apply requires `confirm=true`, repeats validation, backs up existing files, and atomically replaces the account YAML and holdings CSV.
- Empty holdings are valid and represent an empty portfolio; the holdings file is still written so downstream code knows the state was explicitly provided.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Account or holdings payload has the wrong shape | HTTP 400; no writes |
| Numeric field is NaN or infinite | HTTP 400 naming the field |
| Total asset is zero/non-positive | Preview invalid; apply rejected |
| Duplicate or malformed instrument | Preview invalid; apply rejected |
| Shares are negative, fractional, or not a lot multiple | Preview invalid; apply rejected |
| `confirm` is not exactly `true` | Apply rejected; files unchanged |
| Existing account/holdings files exist | Copy them to a timestamped backup directory before replacement |
| Valid empty holdings list | Save account and an empty `instrument,shares` CSV |

### 5. Good/Base/Bad Cases

- Good: a valid account and normalized holdings preview successfully, then explicit confirmation writes both files and preserves backups.
- Base: a new installation has no account or holdings file; the form starts from configured defaults and creates both files.
- Good: an empty holdings list records an intentional empty portfolio.
- Bad: the account endpoint returns a Tushare token or accepts a user-provided output path.
- Bad: browser-only input checks are trusted without repeating backend validation.

### 6. Tests Required

- `tests/test_dashboard_account.py` must assert normalization and valid preview output.
- Tests must assert duplicate/invalid-lot holdings are rejected.
- Tests must assert NaN/infinite values are rejected.
- Tests must assert apply requires confirmation and creates backups before writing.
- Tests must assert the workspace response does not expose Tushare or unrelated config.
- API tests must assert invalid payloads map to HTTP 400.

### 7. Wrong vs Correct

#### Wrong

```python
config = load_config()
config.update(payload)
yaml.safe_dump(config, account_path.open("w"))
```

This exposes and overwrites unrelated settings, may leak secrets, and skips holdings validation and backups.

#### Correct

```python
account = parse_whitelisted_account_fields(payload["account"])
holdings = normalize_holdings(payload["holdings"])
issues = validate_account_inputs(account, holdings, config)
if issues or payload.get("confirm") is not True:
    reject_without_writes()
backup_existing_files()
atomic_write_account_and_holdings()
```

The Web form controls only its owned contract and the backend remains the final authority.

## Scenario: Parameterized Dashboard Workflows

### 1. Scope / Trigger

- Trigger: Web users need bounded control over dates, batch sizes, cache behavior, optimization size, and advanced research probes without receiving arbitrary command execution.
- Owner: `src/dashboard_control.py` owns `WORKFLOW_CATALOG`, public parameter schemas, validation, and command construction; `web/src/OperationsWorkspace.tsx` renders the public schema.

### 2. Signatures

- Catalog API: `GET /api/dashboard/workflows -> {"workflows": DashboardWorkflow[]}`.
- Start API: `POST /api/dashboard/jobs` with `{"action": str, "parameters": {name: value}}`.
- Internal catalog parameter fields: `name`, `label`, `type`, `flag`, `default`, optional `min`, `max`, `optional`, `pattern`, `help`.
- Public parameter fields omit `flag` and `pattern`; the browser never receives CLI construction details.

### 3. Contracts

- Every action and parameter name is declared in `WORKFLOW_CATALOG`. Unknown actions or parameter keys are rejected.
- Supported parameter types are boolean flags, integers, finite numbers, ISO dates, and bounded text formats.
- Boolean parameters must be JSON booleans; strings such as `"true"` are invalid.
- Integer/number ranges are enforced by the backend even when the browser input has HTML min/max attributes.
- Date values use `YYYY-MM-DD`. Optional blank dates are omitted so the script uses project config defaults.
- Text values require an explicit backend pattern when their grammar matters, such as comma-separated drift thresholds.
- File inputs, output paths, executable paths, shell fragments, and arbitrary symbol lists are not Web parameters. They remain owned by project config or purpose-built APIs.
- Commands are passed to `subprocess.Popen` as argument arrays with `shell=False`; no joined shell string is built.
- Expensive defaults are bounded. For example, annual-router grid runs default to 20 combinations rather than the CLI's unlimited `0` behavior.
- Windows and Ubuntu use the same validated argument list and `sys.executable`.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| `parameters` is not an object | HTTP 400 |
| Unknown parameter key | HTTP 400 naming the unsupported keys |
| Boolean is submitted as a string | HTTP 400 |
| Integer/number is outside min/max | HTTP 400 naming the violated bound |
| Number is NaN/infinite | HTTP 400 |
| Date is not `YYYY-MM-DD` | HTTP 400 |
| Text violates its declared grammar | HTTP 400 |
| Optional value is blank | Omit its CLI flag and use config/script default |
| Valid parameter set | Append the fixed flag/value arguments to the fixed script command |

### 5. Good/Base/Bad Cases

- Good: market update submits `chunk_size=500`, `sleep_seconds=0.5`, and a valid end date; the backend produces separate fixed arguments.
- Base: no parameters are submitted; safe catalog defaults are used.
- Good: an advanced annual-router grid defaults to 20 combinations and reuses its score cache.
- Bad: the browser submits `output=../../outside.csv` or an undeclared `--force-official` equivalent.
- Bad: a text threshold contains `;whoami`; backend format validation rejects it even though subprocess does not invoke a shell.

### 6. Tests Required

- `tests/test_dashboard_control.py` must assert public schemas omit internal flags and scripts.
- Tests must assert typed parameters produce the intended CLI argument list.
- Tests must assert unknown keys, wrong types, non-finite numbers, range violations, invalid dates, and text injection formats are rejected.
- Tests must assert bounded advanced defaults, especially maximum combination limits.
- Playwright tests must edit a generated parameter field and assert the expected JSON `parameters` payload reaches the start API.

### 7. Wrong vs Correct

#### Wrong

```python
command = payload["command"].split()
subprocess.Popen(command)
```

This delegates executable, path, flag, and safety decisions to the browser.

#### Correct

```python
workflow = catalog[action]
parameters = validate_declared_schema(workflow, payload.get("parameters", {}))
command = [sys.executable, project_script(workflow), *workflow.base_args, *parameters]
subprocess.Popen(command, shell=False)
```

The catalog owns the executable contract; the browser supplies bounded values only.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| `auto_signal_report.json` is missing | `readiness.status="missing"` and UI shows an empty state |
| `auto_signal_report.json` is malformed | `readiness.status="error"` and `errors[]` records the JSON read failure |
| Precheck cannot resolve target date | The target-date precheck item is `fail` with an actionable error summary |
| Precheck evidence artifact is missing | The related precheck item is `missing`; the frontend renders an unknown state |
| Manual-order CSV is missing | `orders.exists=false`; the UI renders a non-crashing empty state |
| Stock universe file is missing or lacks a usable name column | Manual-order rows remain valid; `name` enrichment is omitted or blank |
| A gate artifact is missing | Gate status is `missing`, not `pass` |
| `data_governance_report.json` is newer and fixes an issue embedded in `auto_signal_report.json` | Governance gate reflects the newer report, stale governance reasons are filtered from dashboard blockers/warnings, and `freshness_notes` asks the UI to rerun auto signal |
| Artifact id is unknown or outside `outputs.dir` | `GET /api/dashboard/artifacts/{artifact_id}` returns 404 |
| A quality blocker has a downloadable quality artifact | Blocker center renders a real report link through the bounded artifact route |
| A quality blocker has no quality artifact | Blocker center renders `报告不可用`; no broken link or inert `查看报告` control |
| Dashboard job action is unknown | `POST /api/dashboard/jobs` returns 400 |
| Dashboard job mode is not `candidate` or `normal` | `POST /api/dashboard/jobs` returns 400 |
| A dashboard job is already running | `POST /api/dashboard/jobs` returns 409 |
| Two dashboard job start requests arrive concurrently | Exactly one process is spawned; the other request returns 409 after observing the reserved running job |
| A dashboard job is stopped by the user | The process tree is terminated, the job becomes `cancelled`, and logs remain visible |
| A dashboard job stop targets a completed job | `POST /api/dashboard/jobs/{job_id}/stop` returns 409 |
| A prior `running` job has no live process after service restart | The job is marked `stale`, the UI unlocks controls, and the log remains visible |
| A prior job PID is live but its command line is unavailable or does not match | Mark the job `stale`; do not send a termination signal to the unverified process |
| Process exit races with a persisted `stopping` update | Re-read the status file and finalize the job as `cancelled`, never overwrite it as `succeeded` or `failed` from an older snapshot |
| Vite dev dependencies have known moderate-or-higher advisories | Upgrade the frontend toolchain or document why the advisory is not applicable before finishing |

### 5. Good/Base/Bad Cases

- Good: latest report exists, manual orders exist, and dashboard shows a readiness verdict, gate cards, blockers, order preview, and artifact links.
- Good: manual-order rows include `instrument` and backend-enriched `name`, so the UI can show stock code and stock name side by side.
- Good: dashboard repair/rerun buttons start a whitelisted job, show a live log tail, and refresh the latest report when the job completes.
- Good: dashboard shows structured job progress and blocker-specific repair actions instead of requiring users to interpret raw log text.
- Good: a yearly backtest blocker states the failing year, observed annualized return, and threshold, while its `查看报告` link downloads the authoritative backtest-quality JSON.
- Good: before starting auto-signal, dashboard shows a read-only precheck for target date, data health, point-in-time governance, factor freshness, and account/holdings.
- Good: while a dashboard job is running, the UI shows a stop button that calls the backend stop route and then displays `cancelled`.
- Good: after dashboard-triggered `daily_basic` repair, the governance gate stops showing the stale embedded `daily_basic_date_coverage_below_required` issue and instead shows a rerun-needed freshness note.
- Base: no latest report exists yet; dashboard still starts and tells the user the latest report is missing.
- Bad: frontend marks a data issue fixed without starting the backend repair command.
- Bad: precheck downloads market data, recomputes factors, writes official/candidate signal artifacts, or edits holdings.
- Bad: dashboard continues to show `daily_basic_date_coverage_below_required` from an older auto-signal report after a newer governance report proves the gap is fixed.
- Bad: frontend reads raw files directly from the browser or the backend exposes an arbitrary file path download endpoint.
- Bad: a static `<span>` is labeled `查看报告` even though it cannot open any report.
- Bad: normal output mode uses `--force-official` or bypasses auto-signal gates.

### 6. Tests Required

- Unit test dashboard view model with present latest report and manual-order CSV.
- Unit test missing `auto_signal_report.json` returns `readiness.status="missing"`.
- Unit test malformed report JSON returns `readiness.status="error"`.
- Unit test dashboard job command building for `repair_point_in_time`, candidate rerun, normal rerun, invalid mode, and unknown action.
- Unit test dashboard blocker action mapping for `daily_basic` repair, candidate-only rerun, and stale-report rerun notes.
- Unit and Playwright tests must cover available and missing quality-report artifacts, exact yearly-return translation, and a real report navigation/download interaction.
- Unit test dashboard precheck pass/fail/missing behavior, including daily-basic repair action and factor rerun action mapping.
- API test that `GET /api/dashboard/precheck` returns the precheck payload.
- Unit test dashboard job progress from auto-signal status files and daily-basic repair logs.
- Unit test that a newer standalone `data_governance_report.json` supersedes stale embedded auto-report governance, filters resolved stale block reasons, and emits `freshness_notes`.
- API test that `GET /api/dashboard/jobs` reports a running active job.
- API test that invalid dashboard jobs return 400 and already-running jobs return 409 when covered.
- API/control test that a running dashboard job can be stopped and is reported as `cancelled`.
- Control tests must prove concurrent starts spawn one process, PID matching rejects missing command-line evidence, job JSON uses atomic replacement, and finalization preserves a newer persisted `stopping` state.
- Script/docs test asserts `15_启动Web仪表盘.bat` is documented and starts the documented backend and frontend commands.
- Run `npm run build` for React/Vite type-check and production build validation.
- Run `npm audit --audit-level=moderate` after adding or changing frontend dependencies.

### 7. Wrong vs Correct

#### Wrong

```python
@app.get("/files/{path:path}")
def read_file(path: str):
    return FileResponse(path)
```

This turns the local dashboard into arbitrary filesystem access.

#### Correct

```python
@app.get("/api/dashboard/artifacts/{artifact_id}")
def dashboard_artifact(artifact_id: str):
    path = resolve_dashboard_artifact(artifact_id)
    if path is None:
        raise HTTPException(status_code=404)
    return FileResponse(path)
```

The resolver maps known artifact ids from the latest dashboard snapshot and constrains them to `outputs.dir`.

## Scenario: Dashboard Factor Freshness Precheck

### 1. Scope / Trigger

- Trigger: the Web precheck summarizes factor freshness when a small number of suspended or no-new-data stocks do not have a row on the target trading date.
- Owners: `src.data_health.build_data_health_report` owns the authoritative coverage gate; `src.dashboard._precheck_factor_freshness` presents that result and may add current data-update confirmation context.

### 2. Signatures

- Evidence: `outputs/data_health_report.json` fields `issues`, `factor_latest_target_coverage`, `min_factor_coverage`, `factor_latest_target_symbols`, `target_symbols`, and `factor_latest_date`.
- Optional confirmation evidence: `outputs/data_update_progress.json` fields `status`, `target_end_date`, `confirmed_no_new_data_symbols`, and `remaining_unconfirmed_symbols`.
- API surface: the `factor_freshness` item inside `GET /api/dashboard/precheck`.

### 3. Contracts

- `factor_*` issues emitted by data health are authoritative failures; the dashboard must not reinterpret them as passes.
- The current configured `quality.min_factor_coverage` is the active precheck threshold. The threshold stored in an older data-health report remains diagnostic evidence only, so a config increase takes effect before the next auto-signal report is regenerated.
- When `factor_latest_target_coverage >= min_factor_coverage` and there are no factor issues, the factor precheck passes even if the minimum per-symbol `factor_latest_date` is earlier than the target date.
- `factor_latest_date` is diagnostic evidence: it is the minimum latest date across target symbols and may legitimately be old for a suspended stock. It is not a second global freshness gate when coverage evidence exists.
- Data-update confirmation counts may be used only when `target_end_date` matches the current target. A completed progress record with `remaining_unconfirmed_symbols=0` may label the uncovered remainder as confirmed suspended/no-new-data symbols.
- A current progress record with `remaining_unconfirmed_symbols > 0` produces a warning and the existing bounded `run_auto_signal` action.
- Metadata date comparison remains the fallback only when a data-health report with coverage fields is unavailable.
- This precheck never changes market data, factor caches, coverage thresholds, or backtest-quality decisions.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Any `factor_*` issue exists | `status=fail`; preserve issues and expose the normal rerun action |
| Coverage is present but below `min_factor_coverage` | `status=fail` even if an inconsistent report omitted the issue string |
| Coverage passes and current progress reports unconfirmed symbols | `status=warn` with `factor_symbols_unconfirmed:<count>` |
| Coverage passes, no factor issues, no unconfirmed symbols | `status=pass`; do not emit `artifact_before_target` from the minimum symbol date |
| Coverage passes but confirmation progress is absent/stale | `status=pass`; describe uncovered rows without claiming they were confirmed |
| Health evidence is absent but governance factor metadata is current | Use the existing metadata-only pass path |

### 5. Good/Base/Bad Cases

- Good: 2705/2708 symbols cover 2026-07-10 (99.89%) against a 99% threshold, three are confirmed with no new data, and the minimum symbol date is 2026-06-29 -> pass with an honest confirmation summary.
- Base: all target symbols have a target-date factor row -> pass with 100% coverage.
- Bad: the dashboard compares `min(latest_date_by_symbol)` directly with the target and warns forever for a legitimately suspended stock.
- Bad: the dashboard lowers the configured threshold or fabricates target-date factor rows to remove the warning.

### 6. Tests Required

- `tests/test_dashboard.py` must cover threshold-passing partial latest-date coverage with a confirmed-no-new-data remainder.
- Tests must assert the pass summary reports covered/target counts, observed percentage, threshold, and confirmed remainder.
- Tests must assert explicit factor issues remain failures and inconsistent below-threshold numeric evidence cannot pass.
- Tests must assert current unconfirmed progress produces a warning plus the controlled rerun action.
- Real API verification must confirm the current 2705/2708 evidence no longer emits `artifact_before_target:2026-06-29<2026-07-10`.

### 7. Wrong vs Correct

#### Wrong

```python
if factor_latest_date < target_date:
    return warn("artifact_before_target")
```

This converts one suspended stock's last row into a false global freshness warning after data health already accepted the configured coverage.

#### Correct

```python
if factor_issues:
    return fail(factor_issues)
if factor_latest_target_coverage < min_factor_coverage:
    return fail("factor_latest_coverage_below_threshold")
return pass_with_coverage_and_confirmation_context()
```

The Web layer presents the authoritative coverage contract instead of inventing a conflicting gate.

## Scenario: Dashboard Stock Detail And Live-Quote Fallback

### 1. Scope / Trigger

- Trigger: a Web user clicks a stock name or code in the manual-order table and needs a compact current-price modal without granting the browser general Tushare or filesystem access.
- Owners: `src/dashboard_stock.py` owns validation, the fixed quote call, and local fallback; `src/dashboard_api.py` exposes the read-only route; `web/src/StockDetailWorkspace.tsx` renders freshness and refresh behavior.

### 2. Signatures

- API: `GET /api/dashboard/stocks/{instrument} -> StockDetail`.
- Instrument grammar: exactly six digits plus `.SH`, `.SZ`, or `.BJ`; backend normalization uppercases the value before validation.
- Live source: `TushareHttpClient.call("rt_k", params={"ts_code": instrument}, fields=<fixed quote fields>)`.
- Fallback source: configured `data.raw_dir/<TS_CODE>.csv`, normally `data/raw/<TS_CODE>.csv`.
- Response fields: `instrument`, `name`, `status`, `is_live`, `source`, `price`, `change`, `change_pct`, `pre_close`, `open`, `high`, `low`, `volume`, `amount`, `market_date`, `retrieved_at`, and `message`.

### 3. Contracts

- The browser supplies only the stock instrument from a dashboard-owned manual-order row. It cannot select the Tushare API name, fields, proxy URL, token, or fallback path.
- A successful `rt_k` response uses `status="live"`, `is_live=true`, `source="tushare_rt_k"`, and records the retrieval timestamp normalized to configured `data.timezone` (default `Asia/Shanghai`), independent of the server operating-system timezone. Because `rt_k` can return the latest close outside trading hours, the response/UI must disclose that non-trading periods may show the most recent closing quote.
- The configured `rt_k` response does not provide an authoritative quote date/time. Live responses therefore keep `market_date=null`; the UI displays `接口未提供` and must not infer `当前交易时段` or derive a market date from `retrieved_at`.
- A remote error, empty response, or unusable live close automatically reads the latest valid positive close from the stock's local raw daily CSV.
- Fallback responses use `status="fallback"`, `is_live=false`, `source="local_daily"`, include the effective `market_date`, and explicitly say the value is non-live.
- The fallback path is derived only after strict instrument validation and must remain directly under the resolved `data.raw_dir`.
- Stock-name enrichment is shared through `load_instrument_name_map`; dashboard orders and stock details must not implement separate universe-file readers.
- The frontend sends `cache: "no-store"` and the refresh button repeats only the same bounded GET request.
- The frontend keeps the manual-order dashboard mounted behind an accessible modal dialog. The modal supports a visible close button, Escape-key dismissal, backdrop-click dismissal, focus containment/restoration, and body scroll locking while open.
- Desktop uses a centered compact dialog; narrow mobile layouts may use a bottom sheet, but neither layout may introduce page-level horizontal overflow.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Instrument does not match `NNNNNN.SH/SZ/BJ` | HTTP 400; do not call Tushare or read a local file |
| `rt_k` returns a positive close | HTTP 200 live response with quote metrics and retrieval time |
| `rt_k` returns no quote date | Keep `market_date=null`; display `接口未提供` plus retrieval time and the non-trading-period disclosure |
| `rt_k` fails, is empty, or has no usable close; local daily row exists | HTTP 200 fallback response with local market date and non-live label |
| Live quote fails and local file is missing/malformed/has no positive close | HTTP 503 with no fabricated price |
| Local CSV lacks `trade_date` or `close` | Treat fallback as unavailable; HTTP 503 |
| Stock name is missing | Keep the normalized instrument usable; name may be blank |

### 5. Good/Base/Bad Cases

- Good: clicking `000001.SZ` or its displayed name opens the same compact modal over the dashboard, `rt_k` returns a quote, and refresh produces a new GET request without reloading the dashboard.
- Good: a weekend live response shows `行情日期: 接口未提供` and a separate retrieval timestamp; it does not claim the quote belongs to the current trading session.
- Base: the Tushare proxy is offline on a weekend; the page shows the latest local daily close, its market date, and `本地收盘价 · 非实时`.
- Bad: the browser submits `api_name=realtime_quote`, a proxy URL, or `../../config/settings.local.yaml`; no API contract accepts these values.
- Bad: an old local close is displayed under a live/real-time badge or without an effective date.
- Bad: `market_date=null` is rendered as `当前交易时段`; retrieval time describes when the Web request ran, not when the market formed the quote.

### 6. Tests Required

- `tests/test_dashboard_stock.py` must assert the exact fixed `rt_k` call and live response metrics.
- Live quote tests must assert `market_date is None` and the disclosure says the source did not provide a quote date; Playwright must render `接口未提供`.
- Backend tests must assert a Tushare error falls back to the latest valid local daily row and calculates change percentage from previous close.
- Backend/API tests must assert invalid instruments map to HTTP 400 and total quote unavailability maps to HTTP 503.
- Playwright must click both the stock name and stock code, assert the dashboard remains mounted, verify close-button/Escape/backdrop dismissal, assert refresh issues another quote request, assert fallback labeling/date, and check the modal has no page-level horizontal overflow at 390 px.
- Production verification must build `web/dist`, call the real local API, click a real manual-order link, and confirm the refresh request completes.

### 7. Wrong vs Correct

#### Wrong

```python
@app.get("/api/quote")
def quote(api_name: str, symbol: str, fallback_path: str):
    return TushareHttpClient.from_config().call(api_name, {"ts_code": symbol})
```

This gives the browser control over the remote interface and suggests it may also choose a filesystem path.

#### Correct

```python
@app.get("/api/dashboard/stocks/{instrument}")
def dashboard_stock(instrument: str):
    return build_stock_detail(instrument)
```

`build_stock_detail()` validates one bounded symbol, owns the fixed `rt_k` request, and labels a constrained local fallback honestly.

For quote time semantics, preserve source evidence:

```python
return {
    "market_date": None,
    "retrieved_at": retrieved_at,
    "message": "...接口未提供行情日期...",
}
```

`retrieved_at` is the Web retrieval time. It must never be substituted for an absent source market date.

---

## Good Examples

- `src/config_loader.py` centralizes defaults, local overrides, env expansion, validation, and path resolution.
- `src/data_converter.py` shows the raw CSV -> Qlib -> local price-panel boundary.
- `src/risk_policy.py` prevents risk settings from being copied across signal and backtest paths.
- `scripts/run_auto_signal.py` breaks a long workflow into stage result dataclasses and writes resumable status artifacts.
- `tests/fixtures/real_data.py` provides deterministic real-data fixtures without depending on private local caches.

## Scenario: Full-Market Factor And Web Pipeline Performance

### 1. Scope / Trigger

- Trigger: Alpha158 or the automatic Web workflow operates on the full local A-share history, where avoidable copies or stale-cache mistakes can add multiple GiB or several minutes.
- Owners: `src/factor_calculator.py`, `scripts/run_auto_signal.py`, and annual-router source builders.

### 2. Signatures

- Config: `qlib.kernels` is a positive integer; Windows defaults to `4` and Ubuntu may override it locally.
- Factor API: `load_or_compute_factors(start_date, end_date, cache_file, force=False, columns=None)`.
- Automatic command: `scripts/run_auto_signal.py [--skip-convert] [--force-factor]`.
- Conversion skip status message: `cache_current_no_raw_changes`.

### 3. Contracts

- Alpha158 feature-only computation passes `learn_processors=[]`; Qlib's label learning processors must not copy the full feature frame when labels are unused.
- Infinite values are inspected by column. Do not call full-frame `replace([inf, -inf], nan)` on the full Alpha158 matrix.
- A factor cache whose start date equals the first available price date is valid even when configured history starts earlier than local prices.
- Conversion may be skipped only when the market update completed with `written_symbols=0`, the configured price panel covers the target date, and Qlib calendar/instrument outputs exist and cover the target date.
- Annual routing uses canonical month-end dates derived from prices. Each source uses its latest score on or before that date; future scores are never allowed.
- Annual-router source construction first previews yearly routes, builds only reachable sources, and limits expensive quality scoring to the dates that can use it.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| `qlib.kernels < 1` | Config validation fails |
| Raw update wrote one or more files | Conversion runs |
| Price/Qlib output missing or behind target | Conversion runs |
| No raw changes and all outputs cover target | Conversion is skipped with `cache_current_no_raw_changes` |
| Cache begins at first available price date | Reuse cache even if configured history begins earlier |
| Routed source has no score on or before signal date | Fail with `No scores for source=...` |
| Source has a mid-month last observation | Use it only for a later canonical month-end, never create an extra signal date |

### 5. Good/Base/Bad Cases

- Good: a repeated Web run with current raw data skips conversion and reuses the 2015-start factor cache in seconds.
- Good: full Alpha158 computation finishes without the default Qlib learn-frame copy or a full-frame infinity mask.
- Base: changed raw data runs conversion and refreshes factor evidence normally.
- Bad: comparing configured `history_start_date=2012` directly to a cache whose price history legitimately starts in 2015.
- Bad: unioning source-specific partial-month dates into the annual rebalance calendar.

### 6. Tests Required

- `tests/test_factor_calculator.py` must cover bounded kernels, first-price-date cache reuse, and column-wise infinity cleanup.
- `tests/test_run_auto_signal.py` must cover conversion reuse prerequisites and fresh annual-router primary factor files.
- Annual-router tests must prove canonical signal dates use the latest prior source score.
- A real Web-click candidate run must record stage timing and prove that candidate/official output gates remain intact.

### 7. Wrong vs Correct

#### Wrong

```python
factors = factors.replace([np.inf, -np.inf], np.nan)
dates = routed_signal_dates(score_sources)
```

This creates a full-frame mask and lets stale partial-month source dates redefine the rebalance calendar.

#### Correct

```python
factors = _replace_infinite_factors_in_place(factors)
dates = month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
daily = latest_score_on_or_before(score_sources[source], date)
```

Memory stays bounded and the routing calendar remains point-in-time correct.

## Scenario: Cross-Platform Environment Sync And CI

### 1. Scope / Trigger

- Trigger: a checkout is cloned or pulled on Windows/Ubuntu, frontend/Python lock files change, or production `web/dist` must be rebuilt before service restart.
- Owner: `scripts/dev_env.py` owns environment synchronization, read-only diagnostics, dependency stamps, and frontend build fingerprints. Root batch files and `scripts/start_dashboard.sh` are adapters only.

### 2. Signatures

- Development sync: `python scripts/dev_env.py sync [--force] [--build-web] [--with-playwright]`.
- Read-only diagnostics: `<venv-python> scripts/dev_env.py doctor --strict [--backend-only|--frontend-only|--runtime-only] [--require-web-dist]`.
- Frontend post-build stamp: `python scripts/dev_env.py stamp-web-build` (normally called by `npm run build`).
- Windows entry: `00_安装依赖环境.bat`; local dashboard entry: `15_启动Web仪表盘.bat`.
- Ubuntu runtime entry: `bash scripts/start_dashboard.sh`.

### 3. Contracts

- Python runtime is 3.11. Sync installs `requirements-lock.txt`; it does not silently switch to `requirements.txt`.
- Frontend install uses `npm ci` and the committed `web/package-lock.json`. Supported Node versions follow Vite 8 (`^20.19.0 || >=22.12.0`); CI uses Node 22.12.
- Successful sync records ignored SHA-256 stamps under `.venv/` and `web/node_modules/`. Stamps are written only after the related command succeeds.
- `npm run build` records an ignored fingerprint in `web/dist/` covering package files, Vite/TypeScript config, HTML, and every `web/src` file.
- `doctor` never installs packages, builds assets, reads secrets, downloads data, or changes account/holding state.
- Ubuntu service startup uses `doctor --strict --runtime-only`; it requires Python runtime dependencies and a current `web/dist` but does not require Node/npm at runtime.
- GitHub Actions must cover the full Python suite on Windows and Ubuntu, frontend build on Node 22, and Playwright Chromium E2E.

### 4. Validation & Error Matrix

| Condition | Behavior |
| --- | --- |
| Python is not 3.11 | Sync/doctor fails and names the expected version |
| Locked direct package is missing or mismatched | Doctor fails and names the package; sync reinstalls the lock |
| Python/npm lock stamp is missing | Doctor warns; sync establishes the stamp |
| Lock stamp is stale or npm dependency tree is incomplete | Doctor fails; sync reinstalls dependencies |
| Node is outside the Vite-supported range | Sync/doctor fails and recommends Node 22 LTS |
| `web/dist/index.html` is missing | Runtime doctor fails with the `sync --build-web` repair command |
| Frontend inputs changed after the last build | Runtime doctor reports a stale build and refuses production startup |
| Optional Playwright download times out | Sync exits non-zero without marking the browser step complete; dependency/build stamps already completed remain valid |

### 5. Good/Base/Bad Cases

- Good: after `git pull`, sync sees a changed npm lock, runs `npm ci`, builds current assets, and runtime doctor passes before systemd restart.
- Base: no lock/input changed; sync skips Python/npm reinstall in seconds and doctor passes.
- Good: a production host removes Node after deployment; runtime doctor and FastAPI startup still work with the already-built current assets.
- Bad: systemd startup performs `pip install`, `npm ci`, or `vite build` and fails during a network outage.
- Bad: `web/dist` exists from an older commit and is served without comparing source fingerprints.

### 6. Tests Required

- `tests/test_dev_env.py` must cover requirement parsing/name normalization, version ranges, Windows/POSIX venv paths, input hashing, stamp states, and stale/missing build decisions in temporary directories.
- `tests/test_scripts_docs.py` must assert Windows uses shared sync, Ubuntu start is validation-only, and CI contains Windows/Ubuntu/backend/frontend/browser gates.
- Final verification must run strict doctor, full Python regression, `npm run build`, Playwright, shell syntax, batch CRLF, and `git diff --check`.

### 7. Wrong vs Correct

#### Wrong

```bash
if [ ! -f web/dist/index.html ]; then npm ci && npm run build; fi
exec .venv/bin/python scripts/run_dashboard.py
```

Existence does not prove freshness, and service startup now depends on package registries.

#### Correct

```bash
python3.11 scripts/dev_env.py sync --build-web  # deploy phase
.venv/bin/python scripts/dev_env.py doctor --strict --runtime-only
bash scripts/start_dashboard.sh                 # runtime phase
```

Build and runtime are separate, and source fingerprints prevent stale frontend deployment.
