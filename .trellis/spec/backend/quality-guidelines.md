# Quality Guidelines

> Code quality, testing, and review standards for backend changes.

---

## Overview

`quant_box` is a local quant research pipeline used for manual trading decisions. Quality work here means preserving data contracts, avoiding fake market assumptions, keeping generated/private data out of git, and validating behavior with focused tests.

Use the project virtual environment for tests and scripts:

```powershell
.\.venv\Scripts\python.exe
```

Do not rely on the system `python`; it may not have the pinned pandas/numpy stack.

---

## Required Patterns

### Use project config and path helpers

- Load config through `load_config()`.
- Resolve project-relative paths through `resolve_path()`.
- Add new config defaults to `DEFAULT_CONFIG`.
- Validate risky config ranges in `validate_config()`.
- Test config changes in `tests/test_config_loader.py`.

### Preserve pandas contracts

Before adding logic around factors, scores, or prices, confirm the shape:

- factors/scores: MultiIndex `datetime`/`instrument`,
- OHLCV prices: MultiIndex columns `field`/`instrument`,
- close prices: plain DataFrame with instrument columns,
- dates normalized to session dates,
- instruments normalized to uppercase Tushare symbols in logic.

Use helpers from `src/common.py` instead of copying normalization code.

### Keep business logic importable

Scripts orchestrate; `src/` modules implement. If a script grows reusable behavior, extract it into `src/` or `scripts/_shared.py` and test it directly.

### Keep official outputs gated

Quality gates protect official state:

- Candidate signals and orders are written when gates fail.
- `outputs/latest_holdings.csv` should only be overwritten through official save/promote paths.
- Auto-signal reports must include block reasons and warnings.

### Maintain user-facing docs with workflow changes

If a command, `.bat` file, artifact path, or risk-control workflow changes, update:

- `README.md`,
- relevant root `.bat` file,
- `CHANGELOG.md` for user-visible workflow/risk changes,
- `tests/test_scripts_docs.py` when docs assert the contract.

---

## Forbidden Patterns

### Do not commit private/generated data

Never commit real market caches, account holdings, Tushare URLs/tokens, reports, or `outputs/`. Keep local secrets in `config/settings.local.yaml`, `config/account.yaml`, `config/current_holdings.csv`, or environment variables.

### Do not use fake market data as a replacement for real fixture coverage

For behavior tied to price/factor reality, prefer `tests/fixtures/real_data.py` and `tests/fixtures/data_snapshot/`. Synthetic frames are fine for small shape/unit tests, but they must not replace regression coverage for market-data-sensitive logic.

### Do not silently coerce invalid market data

Invalid OHLCV, missing required fields, malformed MultiIndex, or broken config should raise a specific error. Do not hide these behind empty outputs unless the file/input is explicitly optional.

### Do not duplicate risk/config logic across workflows

Use central adapters:

- `RiskPolicy` for configured selection/execution risk controls.
- `scripts/_shared.py` for factor column selection and yearly quality summaries.
- `src.common` for instrument/date/price normalization.

Search before adding a new helper or constant.

### Do not force expensive work by default

Default workflows favor cached factors and bounded optimization. Preserve options such as `--optimize-timeout-seconds`, `--max-optimize-combinations`, `--skip-*`, and `--force-*` semantics.

---

## Testing Requirements

Run focused tests for the changed surface. Use the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest <tests> -q
```

Baseline regression set from project docs:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_common.py tests/test_data_fetcher.py tests/test_strategy.py tests/test_scoring.py tests/test_backtest.py tests/test_signal_generator.py tests/test_selection_risk.py tests/test_selection_constraints.py tests/test_risk_policy.py tests/test_monitoring.py tests/test_optimizer.py tests/test_run_auto_signal.py tests/test_config_loader.py::ConfigLoaderTests::test_default_quality_includes_full_backtest_return_and_drawdown_gates -q
```

For research scripts, also run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_run_selector_weight_backtest.py tests/test_run_risk_refine.py tests/test_run_regime_blend_probe.py tests/test_run_quality_selector_gate_backtest.py tests/test_run_ml_experiments.py tests/test_run_goal_formal_candidates.py tests/test_run_fundamental_quality_backtest.py -q
```

For optimizer/auto-signal/config changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_optimizer.py tests/test_run_auto_signal.py tests/test_config_loader.py -q
```

For docs-only Trellis spec changes, running a targeted text check or `git diff --check` is sufficient unless code behavior changed.

---

## Test Style

The test suite mostly uses `unittest.TestCase` with pandas assertions and temporary filesystem isolation.

Preferred patterns:

- Use `TemporaryDirectory()` for generated files.
- Patch `load_config()` and `resolve_path()` for isolated config/path tests.
- Use fake Tushare clients with recorded calls instead of network access.
- Use `assertRaisesRegex` for contract errors.
- Use `assertLogs` for warning contracts.
- Use `require_real_market_data()` for deterministic real-data slices.

Example:

```python
with TemporaryDirectory() as tmp:
    root = Path(tmp)
    with patch("src.data_converter.load_config", return_value=config), patch(
        "src.data_converter.resolve_path",
        side_effect=lambda value: Path(value),
    ):
        result = convert_to_qlib_format(raw_dir=root / "raw", qlib_dir=root / "qlib")
```

---

## Code Review Checklist

- Does the change preserve the documented DataFrame index/column contracts?
- Are paths resolved through `resolve_path`?
- Are generated/private files still under ignored paths?
- Are errors specific and actionable?
- Are logs/status artifacts enough to debug a failed long run?
- Does the change avoid leaking tokens or account details?
- Is the official-vs-candidate output boundary preserved?
- Are tests focused on the changed behavior and failure mode?
- Did user-visible workflow changes update README/batch docs/CHANGELOG?
- Did dependency changes update `requirements.txt` and `requirements-lock.txt` deliberately?

---

## Performance And Long-Run Safety

This project can operate on many symbols and years of history. Avoid unnecessary full-frame reads and expensive recomputation:

- Read parquet subsets with `columns=` when possible, as in `scripts/_shared.py`.
- Reuse factor caches by default; use `force` flags only when requested.
- Keep optimization grids bounded and expose timeout/max-combination settings.
- Write progress files for resumable workflows.
- Prefer vectorized pandas/numpy operations over row-wise loops on large panels.
