# Changelog

## Unreleased

- Added controlled Web dashboard actions for repairing `daily_basic` point-in-time gaps and rerunning auto signal in candidate or normal gated output mode.
- Added live dashboard job status and log-tail polling for long-running local repair/signal tasks.
- Added a local FastAPI + React/Vite dashboard for latest auto-signal manual review.
- Added a Windows one-click launcher for the local Web dashboard backend and frontend.
- Added `run_auto_signal.py --candidate-only` for safe validation runs that never promote or overwrite official signal artifacts.
- Made historical-universe builds fail on index fetch errors by default, required enabled historical-universe files by default, and added source-level governance coverage checks.
- Added optimizer timeout controls with partial validation outputs for automatic signal runs.
- Added stricter daily OHLCV validation before normalized market data enters factor and backtest pipelines.
- Split the automatic optimization stage out of `run_auto_signal.py`.
- Added scoring benchmark support and normalized factor-frame caching.
- Added common date and instrument normalization helpers.
- Added bounded weak-reference price-field caches in backtest and selection-risk paths.
- Added `RiskPolicy` as the centralized entry point for selection risk, industry caps, stop/take-profit, slippage, and capacity configuration.
- Added `requirements-lock.txt`, GitHub Actions CI, and contributor guidance.
- Added Prometheus textfile export for automatic signal run status metrics.
