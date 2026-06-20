# Quant Backtest Diagnostics

## Goal

Create an evidence-first diagnostic workflow that helps explain why a quant backtest is unprofitable without prematurely optimizing strategy parameters. The workflow must follow five layers: backtest engine, data, factors, portfolio attribution, and optimization readiness.

## Requirements

* Add a runnable diagnostic entry point that reads existing backtest and auto-run artifacts, then writes a JSON and Markdown report under `outputs/`.
* The report must organize findings into five sections:
  * Backtest engine: reproducibility artifact comparison and accounting invariant checks for equity, holdings, trades, and costs.
  * Data: data health, point-in-time governance, adjustment metadata, suspensions/limit-rule readiness, and survivorship-bias readiness.
  * Factors: IC summary availability, yearly stability, quantile/group-return availability, and industry/market-cap exposure evidence.
  * Portfolio: turnover, concentration, cost drag, drawdown source, failed years, and attribution evidence.
  * Optimization readiness: explicitly block optimization readiness until engine, data, factor, and portfolio diagnostics have sufficient evidence.
* Reuse existing modules and artifacts first: `scripts/run_backtest.py`, `scripts/run_goal_audit.py`, `src/data_health.py`, `src/data_governance.py`, `src/factor_ic.py`, `src/research_diagnostics.py`, `src/failure_analysis.py`, and `src/reporting.py`.
* Keep the implementation focused. Do not change strategy behavior, optimizer behavior, official signal promotion rules, or real market data.
* Add focused tests for diagnostic classification, missing-artifact caveats, invariant checks, and optimization-readiness gating.

## Acceptance Criteria

* [ ] Same-input reproducibility can be checked by comparing two artifact directories or prefixes.
* [ ] Cash/holdings/equity and trade-cost checks produce explicit pass/warn/fail results instead of silently assuming correctness.
* [ ] Data diagnostics explain whether future-function, survivorship, adjustment, suspension, limit-up/down, and ST-calendar evidence is present or missing.
* [ ] Factor diagnostics explain whether IC, yearly stability, group returns, and industry/market-cap exposure evidence is present or missing.
* [ ] Portfolio diagnostics explain whether loss likely comes from selection, timing/regime, costs, turnover, concentration, drawdowns, or failed years when supporting artifacts exist.
* [ ] Optimization readiness remains false until the first four layers have no failed required checks.
* [ ] Focused pytest coverage passes.

## Definition of Done

* Tests added or updated for the new diagnostic workflow.
* Focused validation commands pass.
* Full test suite is attempted; any unrelated failure is reported.
* No commits or pushes are made.

## Technical Approach

Implement a small module for artifact-based diagnostics plus a CLI wrapper. The module should parse existing CSV/JSON outputs and render compact JSON/Markdown reports. Prefer pure functions over side effects so tests can build small synthetic artifact directories.

## Decision (ADR-lite)

Context: The project already emits many backtest, data, factor, research, and failure-analysis artifacts, but no single report enforces the ordered diagnostic path.

Decision: Add a diagnostic aggregator instead of rebuilding backtests or duplicating data/factor calculations. The aggregator treats missing evidence as caveats and failed invariants as blockers.

Consequences: The first version is fast and low-risk because it operates on existing artifacts. Deeper factor group-return calculations or direct rerun orchestration can be added later if the report identifies evidence gaps.

## Out of Scope

* No strategy parameter optimization.
* No dynamic routing or style detector changes.
* No official/candidate signal promotion changes.
* No real-data repair or market-data downloads.
* No commits or pushes.

## Technical Notes

* Existing annual goal audit: `scripts/run_goal_audit.py`.
* Existing run artifacts: `outputs/backtest_equity.csv`, `outputs/backtest_holdings.csv`, `outputs/backtest_trades.csv`, `outputs/backtest_metrics.json`, `outputs/backtest_yearly.csv`, `outputs/backtest_run_summary.json`.
* Existing data artifacts: `outputs/data_health_report.json`, `outputs/data_governance_report.json`.
* Existing research/failure artifacts: `outputs/auto_research_diagnostics.json`, `outputs/auto_failure_analysis.json`, `outputs/auto_drawdown_summary.json`, attribution CSVs.
