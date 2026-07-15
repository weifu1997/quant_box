# Contributing

## Local Setup

Use the shared cross-platform synchronizer on Windows or Ubuntu:

```powershell
python scripts/dev_env.py sync --build-web
.\.venv\Scripts\python.exe scripts/dev_env.py doctor --strict --require-web-dist
```

On Ubuntu, use `python3.11` for the first command and `.venv/bin/python` for doctor. Sync installs `requirements-lock.txt` and uses `npm ci`; it is safe to repeat after every pull. If the lock file needs to be refreshed, update `requirements.txt`, reinstall the environment, then pin the direct dependencies in `requirements-lock.txt`.

## Validation

Run the focused regression set before submitting changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_common.py tests/test_data_fetcher.py tests/test_strategy.py tests/test_scoring.py tests/test_backtest.py tests/test_signal_generator.py tests/test_selection_risk.py tests/test_selection_constraints.py tests/test_risk_policy.py tests/test_monitoring.py tests/test_optimizer.py tests/test_run_auto_signal.py tests/test_config_loader.py::ConfigLoaderTests::test_default_quality_includes_full_backtest_return_and_drawdown_gates -q
```

For code touching the research scripts, also run the matching script tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_run_selector_weight_backtest.py tests/test_run_risk_refine.py tests/test_run_regime_blend_probe.py tests/test_run_quality_selector_gate_backtest.py tests/test_run_ml_experiments.py tests/test_run_goal_formal_candidates.py tests/test_run_fundamental_quality_backtest.py -q
```

Frontend changes must also pass:

```powershell
cd web
npm run build
npm run test:e2e
```

## Data And Secrets

Do not commit real market caches, account holdings, Tushare URLs, tokens, reports, or files under `outputs/`. Keep personal settings in `config/settings.local.yaml` or environment variables.

## Change Style

Prefer small, behavior-preserving changes with focused tests. Keep compatibility wrappers when migrating old script entry points, and update `CHANGELOG.md` for user-visible workflow or risk-control changes.
