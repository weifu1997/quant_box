# Improve Yearly Returns To At Least 20 Percent

## Goal

Improve the local A-share strategy research pipeline so the historical full-flow backtest can pass the project gate that every reported yearly annual return is at least 20%, while preserving data integrity, official-signal safeguards, and candidate-vs-official output boundaries.

## What I Already Know

* The latest full-flow run completed without data-health or governance issues.
* Current full-history backtest metrics are annual return `0.1872` and max drawdown `-0.2521`.
* Current yearly annual-return failures are `2016=-0.0233`, `2017=-0.0259`, `2018=-0.0658`, `2020=0.0406`, `2023=0.0231`, and `2026=-0.1700`.
* Current yearly drawdown failure is `2015=-0.2381`.
* Automatic parameter selection found no acceptable parameter set; the best summary has annual-return mean `0.2134`, annual-return min `-0.1918`, and worst drawdown `-0.2090`.
* Current baseline optimization grid is narrow: `factor_group in [momentum, factor:LOW0]`, `top_n in [7,10,15,20]`, monthly rebalance, and fixed drift threshold.

## Assumptions

* Success means historical backtest evidence, not a guarantee of future annual returns.
* The task must not relax `min_yearly_annual_return`, fabricate prices/factors, or force official outputs.
* A strategy change is acceptable only if it is encoded in the pipeline and can be re-run end to end.
* If the target cannot be achieved with the available data and strategy family, the deliverable should make that clear with reproducible evidence and the best next research path.

## Requirements

* Preserve data-health and governance gates.
* Preserve official-vs-candidate signal safeguards.
* Explore strategy, scoring, risk, regime, and optimization changes that directly improve failed yearly returns.
* Avoid calendar-year or candidate-level hindsight when proposing a strategy improvement.
* Prefer reusable pipeline improvements over one-off notebooks.
* Record experiment results under `outputs/` or task `research/` so future runs can inspect them.
* Keep changes testable with focused unit/regression coverage.

## Acceptance Criteria

* [ ] A full-flow or equivalent authoritative backtest report shows every yearly annual return is at least `0.20`.
* [ ] Overall backtest annual return is at least `0.20`.
* [ ] Max drawdown gates remain enforced and are not weakened.
* [ ] No official signal is produced unless quality gates pass.
* [ ] Tests pass for changed strategy/optimizer/reporting code.
* [ ] If the target remains unmet, the remaining blockers are backed by generated experiment artifacts and current status/report files.

## Definition Of Done

* Focused tests pass with `.\.venv\Scripts\python.exe`.
* `git diff --check` passes.
* Generated/private market data is not staged.
* PRD/research notes capture any non-obvious strategy decision or failed path.

## Out Of Scope

* Guaranteeing future live yearly returns.
* Lowering quality thresholds to make a failing strategy look successful.
* Manually editing generated metrics, price data, or factor data.
* Promoting candidate signals without passing gates.

## Technical Notes

* Latest evidence files: `outputs/auto_run_status.json`, `outputs/auto_backtest_yearly_breakdown.csv`, `outputs/auto_parameter_quality.json`, and `outputs/auto_failure_analysis.json`.
* Research evidence: `research/current-evidence.md`.
* Likely modules: `src/scoring.py`, `src/strategy.py`, `src/optimizer.py`, `src/auto_tuning.py`, `src/backtest.py`, `scripts/run_auto_signal.py`, and research scripts under `scripts/run_goal_*.py`.
* Relevant specs: `.trellis/spec/backend/index.md`, `.trellis/spec/backend/quality-guidelines.md`, `.trellis/spec/backend/database-guidelines.md`, `.trellis/spec/backend/logging-guidelines.md`.
