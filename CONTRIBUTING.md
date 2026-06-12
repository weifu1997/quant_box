# Contributing

## Local Setup

Use the project virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

If the lock file needs to be refreshed, update `requirements.txt`, reinstall the environment, then pin the direct dependencies in `requirements-lock.txt`.

## Validation

Run the focused regression set before submitting changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_common.py tests/test_data_fetcher.py tests/test_strategy.py tests/test_scoring.py tests/test_backtest.py tests/test_signal_generator.py tests/test_selection_risk.py tests/test_selection_constraints.py tests/test_risk_policy.py tests/test_monitoring.py tests/test_optimizer.py tests/test_run_auto_signal.py tests/test_config_loader.py::ConfigLoaderTests::test_default_quality_includes_full_backtest_return_and_drawdown_gates -q
```

For code touching the research scripts, also run the matching script tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_run_selector_weight_backtest.py tests/test_run_risk_refine.py tests/test_run_regime_blend_probe.py tests/test_run_quality_selector_gate_backtest.py tests/test_run_ml_experiments.py tests/test_run_goal_formal_candidates.py tests/test_run_fundamental_quality_backtest.py -q
```

## Data And Secrets

Do not commit real market caches, account holdings, Tushare URLs, tokens, reports, or files under `outputs/`. Keep personal settings in `config/settings.local.yaml` or environment variables.

## Change Style

Prefer small, behavior-preserving changes with focused tests. Keep compatibility wrappers when migrating old script entry points, and update `CHANGELOG.md` for user-visible workflow or risk-control changes.
