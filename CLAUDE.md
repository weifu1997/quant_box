# quant_box Assistant Context

## Runtime

- Use the project virtual environment for tests and scripts:
  `.\.venv\Scripts\python.exe`
- Do not use the system default `python` for project tests; it may not include `pandas` and `numpy`.

## Validation Commands

- Install pinned direct dependencies:
  `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`
- Focused baseline used for the verified development plan:
  `.\.venv\Scripts\python.exe -m pytest tests/test_data_fetcher.py tests/test_strategy.py tests/test_backtest.py tests/test_signal_generator.py tests/test_selection_risk.py tests/test_selection_constraints.py -q`
- CI regression set:
  `.\.venv\Scripts\python.exe -m pytest tests/test_common.py tests/test_data_fetcher.py tests/test_strategy.py tests/test_scoring.py tests/test_backtest.py tests/test_signal_generator.py tests/test_selection_risk.py tests/test_selection_constraints.py tests/test_risk_policy.py tests/test_monitoring.py tests/test_optimizer.py tests/test_run_auto_signal.py tests/test_config_loader.py::ConfigLoaderTests::test_default_quality_includes_full_backtest_return_and_drawdown_gates -q`
- Optimizer and auto-signal checks:
  `.\.venv\Scripts\python.exe -m pytest tests/test_optimizer.py tests/test_run_auto_signal.py tests/test_config_loader.py -q`
- Lightweight scoring benchmark:
  `.\.venv\Scripts\python.exe scripts\benchmark_scoring.py --days 60 --instruments 300 --factors 8 --repeats 3`

## Notes

- Keep generated market data, account files, holdings, and `outputs/` out of version control.
- `scripts/run_auto_signal.py` supports `--optimize-timeout-seconds` and `--max-optimize-combinations` to prevent long optimization runs from failing without diagnostics.
- `scripts/export_auto_status_metrics.py` converts `outputs/auto_run_status.json` to Prometheus text metrics.
- `requirements-lock.txt` pins the direct dependencies used by CI; refresh it deliberately after dependency upgrades.
