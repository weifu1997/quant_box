# Changelog

## Unreleased

- Made annual-state-router research reproducible with a canonical month-end signal calendar, latest-on-or-before score lookup, engine/source-fingerprinted caches, and per-candidate detail evidence.
- Added engine and score-source provenance to router evidence and made auto-signal reject stale or incompatible formal evidence instead of promoting it under changed backtest semantics.
- Corrected yearly diagnostic trade-cost ratios to use each yearly segment's starting equity rather than the process-wide initial capital.
- Fixed the dashboard quality-blocker report entry so it downloads the bounded authoritative quality artifact, added clear current-year gate evidence, and corrected live quote dates when `rt_k` provides no market date.
- Raised the latest-target factor coverage quality threshold from 95% to 99% and aligned the Web precheck with that authoritative gate.
- Added an in-app stock detail view from manual-order names/codes with refreshable Tushare `rt_k` prices and explicitly labeled local daily-close fallback.
- Reduced a current-cache Web auto-signal run to about six minutes by skipping unchanged conversion outputs and building annual-router sources only for reachable states/dates.
- Prevented full-market Alpha158 memory failures by bounding Qlib workers, disabling unused learn processors, and avoiding full-frame infinity replacement.
- Fixed factor-cache reuse when configured history predates the first available local price, avoiding unnecessary multi-gigabyte recomputation.
- Added Web repair actions and structured progress for full point-in-time data and historical-universe workflows.
- Redacted the private Tushare proxy endpoint from Web-visible configuration-check logs.
- Fixed real-data mobile overflow across the dashboard, operations center, and account workspace.
- Added Web account and current-holdings management with backend validation, explicit confirmation, backups, and atomic file replacement.
- Added Playwright browser coverage for navigation, workflow actions, execution/account forms, and mobile layout.
- Added backend-validated workflow parameter schemas and bounded advanced research actions for the Web operations center.
- Added a cross-platform Web operations center for core data, factor, research, backtest, diagnostic, and candidate-signal workflows.
- Added controlled Web fill-feedback editing and validated current-holdings updates with audit artifacts.
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
