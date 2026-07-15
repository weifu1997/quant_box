# Diagnose 2026 Strategy Loss and Validate Controlled Candidates

## Goal

Explain the current strategy's 2026 loss from route, holdings, industry, cost, and risk-exit evidence; test a small hypothesis-driven candidate set; and promote a candidate only if it passes every formal full-history, yearly, drawdown, cost, and out-of-sample gate before rerunning the official auto-signal workflow and Web review.

## Known Baseline

* Current official strategy mode: `annual_state_router`.
* Full-history annual return: about 24.73%.
* Full-history max drawdown: about -17.69%.
* Current blocker: `backtest_yearly_annual_return_below_threshold:2026=-0.0719<0.2000`.
* Current 2026 segment covers 2026-01-05 through 2026-07-10 and has 124 trading days.
* Candidate artifacts must not overwrite official signals, holdings, or evidence.

## Requirements

* Establish an immutable baseline from the current config and output artifacts.
* Attribute 2026 performance to annual-state routes, instruments/holdings, industries, trading costs, and risk-exit behavior.
* Generate only a small controlled candidate set whose changes are justified by the diagnosis.
* Run candidates through the same path-dependent score, holdings, trade, cost, and risk logic as the formal strategy.
* Validate full-history annual return, yearly annual returns, max drawdown, yearly drawdown, turnover/cost, validation evidence, and out-of-sample behavior.
* Keep all exploratory outputs candidate/research-only.
* Update formal evidence/config only if one candidate passes every current gate without threshold relaxation.
* After promotion, rerun automatic signal normally and verify official output in the Web dashboard.
* If no candidate passes, preserve the current formal strategy and produce an evidence-backed failure report instead of forcing promotion.

## Acceptance Criteria

* [x] Baseline metrics and configured thresholds are recorded.
* [x] 2026 route-by-route contribution and exposure are explained.
* [x] Top positive/negative instruments and industries are identified.
* [x] 2026 costs, turnover, blocked trades, and risk exits are quantified.
* [x] Candidate set is bounded and hypothesis-driven.
* [x] Every candidate has full-history, yearly, drawdown, cost, and OOS results.
* [x] No candidate artifact overwrites official state.
* [x] A candidate is promoted only if every formal gate passes.
* [x] Normal auto signal and Web verification succeed after promotion, or a no-pass result is documented without changing official evidence.
* [x] Focused and full regression checks pass for any code/config changes.

## Guardrails

* Do not lower the 20% annual/yearly return targets.
* Do not weaken the -20% drawdown limits.
* Do not use `--force-official` or `--allow-low-quality` for promotion.
* Do not evaluate route candidates from standalone equity curves; rebuild the path-dependent backtest.
* Do not choose candidates based only on 2026 performance; require full-history and OOS evidence.
* Do not mutate `config/settings.yaml` or formal evidence until a complete gate pass is proven.

## Likely Evidence and Tools

* `outputs/auto_backtest_yearly_breakdown.csv`
* `outputs/auto_failure_analysis.json`
* `outputs/auto_research_*`
* `outputs/auto_annual_state_router_*`
* `scripts/run_annual_state_router_backtest.py`
* `scripts/run_annual_state_router_grid.py`
* `scripts/run_auto_signal.py`
* Existing formal router evidence files referenced by `config/settings.yaml`.

## Out of Scope

* Broad unconstrained parameter mining.
* New alternative-data sources.
* Manual editing of output JSON to manufacture a pass.
* Official promotion based on a single favorable period.

## Implementation Plan

* Audit baseline artifacts and reproduce the current gate.
* Build a persisted 2026 attribution report.
* Select a small candidate matrix from diagnosed failure drivers.
* Run formal candidate backtests and compare all gates.
* Promote only a complete pass, rerun auto signal, and verify Web output.
