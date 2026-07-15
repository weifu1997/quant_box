# Root Cause and Repair Review

## Runtime Evidence

* `outputs/auto_signal_report.json` contains exactly one blocker: `backtest:backtest_yearly_annual_return_below_threshold:2026=-0.0719<0.2000`.
* `outputs/auto_backtest_quality.json` shows the full-history annual return is 24.73% and max drawdown is -17.69%, both passing their 20% / -20% gates. The failing sub-gate is the 2026 yearly segment.
* `outputs/auto_backtest_yearly_breakdown.csv` covers 2026-01-05 through 2026-07-10 (124 trading days). Its `annual_return` is calculated by the shared backtest metric function and is annualized, not an unannualized YTD return.
* The dashboard blocker center renders every blocker without a background-job action as a static `<span>` labeled `查看报告`; it never receives or resolves a report artifact. The label therefore promises an interaction that does not exist.
* `auto_backtest_quality.json` is already a bounded dashboard artifact and is downloadable through `/api/dashboard/artifacts/backtest_quality`.
* Tushare `rt_k` currently returns price/OHLCV fields but no quote date or time, even when `date` and `time` are requested explicitly.
* The stock modal maps `market_date=null` on a live quote to `当前交易时段`. On weekends or outside trading hours this is false; the same quote may be the most recent close.

## Repair Options Reviewed

### A. Lower or bypass the quality gate

Rejected. It would turn a real current-year performance failure into an official signal without stronger strategy evidence and violate the official/candidate boundary.

### B. Exclude the incomplete current year automatically

Rejected for this repair. The metric is annualized and the strategy is currently losing during the live year. Silently excluding it would remove the most relevant current-regime evidence. A different policy would require an explicit product/risk decision and separate validation.

### C. Preserve the gate and repair evidence/navigation semantics

Selected.

* Translate the blocker into a clear Chinese statement with year, actual annualized return, and required threshold.
* Enrich the blocker card with the authoritative parameter/backtest quality artifact and render a real link only when the backend says it is downloadable.
* Show `报告不可用` when the expected artifact is missing instead of a fake clickable label.
* Add truthful context that full-history metrics pass while the 2026 yearly segment fails.
* For live quotes with no source date, show `接口未提供` and rely on retrieval time plus the existing non-trading-period disclosure. Keep the real `trade_date` for local fallback quotes.

## Risk Review

* Security: artifact links continue through the existing id-based, output-directory-constrained endpoint.
* Data truthfulness: no price date, gate result, or strategy evidence is fabricated.
* Stale artifacts: the blocker receives the artifact object from the same snapshot build, so missing/non-downloadable state is explicit.
* Regression: existing job actions remain unchanged; report navigation is additive and separately tested.
* Official output: no gate configuration or promotion path changes.
