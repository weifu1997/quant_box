# Changelog

## Unreleased

- Added optimizer timeout controls with partial validation outputs for automatic signal runs.
- Added stricter daily OHLCV validation before normalized market data enters factor and backtest pipelines.
- Split the automatic optimization stage out of `run_auto_signal.py`.
- Added scoring benchmark support and normalized factor-frame caching.
- Added common date and instrument normalization helpers.
- Added bounded weak-reference price-field caches in backtest and selection-risk paths.
- Added `RiskPolicy` as the centralized entry point for selection risk, industry caps, stop/take-profit, slippage, and capacity configuration.
- Added `requirements-lock.txt`, GitHub Actions CI, and contributor guidance.
- Added Prometheus textfile export for automatic signal run status metrics.
