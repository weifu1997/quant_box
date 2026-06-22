# Validate annual_state_router Auto Signal Chain

## Goal

Verify that the promoted `annual_state_router` configuration works correctly in the formal auto-signal workflow before any manual trading review. This task is a production-chain validation task, not another backtest optimization task.

## What I Already Know

* The latest router research controls are committed and pushed through `cfaf7c0 feat: add route-specific annual router research controls`.
* `config/settings.yaml` enables `annual_state_router` and points to formal evidence files:
  * `outputs/codex_router_grid_20260614_beta20_exposure_hit_metrics.json`
  * `outputs/codex_router_grid_20260614_beta20_exposure_hit_years.csv`
* The formal evidence artifact family exists under `outputs/codex_router_grid_20260614_beta20_exposure_hit_*`.
* `scripts/run_auto_signal.py` has annual-router-specific paths:
  * `_annual_state_router_quality()` validates evidence metrics/yearly artifacts and config combo matching.
  * `_build_annual_state_router_runtime()` rebuilds score sources, routes annual state scores, and writes route artifacts.
  * `_run_backtest_stage()` uses `annual_state_router.routed.scores` and `annual_state_router.backtest_config` when router mode is enabled.
  * `_run_signal_stage()` uses routed scores for signal generation when annual router runtime is present.
* Backend spec documents the contract under "Annual State Router Auto-Signal Mode".

## Assumptions

* Validation should not tune strategy parameters, expand router grids, or alter evidence files.
* Candidate outputs are acceptable for validation. Official output should only be written if all gates pass and the workflow already allows it under existing safeguards.
* Existing local data/config secrets are sufficient to run the auto-signal workflow; if account or data gates fail, record the failure instead of bypassing it.

## Requirements

* Run the annual-router auto-signal workflow in a safe validation mode using existing configuration and evidence.
* Confirm `annual_state_router` evidence combo matches `config/settings.yaml`.
* Confirm the auto-signal workflow uses routed score panels and does not fall back to legacy strategy scores.
* Check data governance, parameter quality, backtest quality, and account gates from generated artifacts.
* Confirm generated signal/holding outputs are candidate or official according to existing gates, without manual promotion.
* Produce a short conclusion report answering whether the result can enter manual trading review.

## Acceptance Criteria

* [x] `scripts/run_auto_signal.py` is run without changing optimization/grid parameters.
* [x] `outputs/auto_parameter_quality.json` exists and records annual-router parameter quality.
* [x] `outputs/auto_backtest_quality.json` exists and records the rebuilt auto-backtest quality.
* [x] `outputs/auto_annual_state_router_score_routes.csv` and `outputs/auto_annual_state_router_year_routes.csv` exist and are non-empty.
* [x] `outputs/auto_signal_report.json` records `strategy_mode=annual_state_router`.
* [x] Evidence combo mismatches, if any, are recorded as blockers rather than ignored. (No mismatch: evidence combo matches config.)
* [x] Data governance, backtest quality, parameter quality, and account gate results are summarized. (See `conclusion.md`.)
* [x] The task outputs a clear "ready / not ready for manual trading review" conclusion. (NOT READY — data-health gate fails on stale factor coverage.)

## Definition of Done

* Focused validation command(s) have been run and results recorded.
* No parameter optimization, grid expansion, or signal promotion is performed as part of this task.
* Generated artifacts are inspected and summarized.
* Any workflow bug discovered during validation is fixed with focused tests.
* If behavior or contracts change, update `.trellis/spec/backend/database-guidelines.md`.
* Work is committed, pushed, and the Trellis task is finished.

## Technical Approach

Use the existing auto-signal workflow as the validation subject. Prefer running the same production entry point with current `config/settings.yaml`, then inspect the emitted JSON/CSV artifacts rather than adding new strategy logic. If the run fails because a gate blocks execution, treat that as validation evidence and summarize the blocking gate.

## Decision (ADR-lite)

Context: The router backtest evidence already passed, but official trading safety depends on the full auto-signal chain rebuilding routed scores and passing quality/data/account gates.

Decision: Validate the full auto-signal workflow end to end before any further parameter work or manual trading review.

Consequences: The task may conclude "not ready" even when backtest evidence is strong, because data/account/governance gates are authoritative for production readiness.

## Out of Scope

* Expanding the full router grid.
* Tuning `beta_top_n`, min-position guards, exposures, or turnover settings.
* Editing formal evidence artifacts to make gates pass.
* Promoting candidate outputs manually.
* Changing real market data or account files.

## Technical Notes

* Main entry point: `scripts/run_auto_signal.py`.
* Relevant config section: `annual_state_router` in `config/settings.yaml`.
* Relevant outputs:
  * `outputs/auto_parameter_quality.json`
  * `outputs/auto_backtest_quality.json`
  * `outputs/auto_annual_state_router_score_routes.csv`
  * `outputs/auto_annual_state_router_year_routes.csv`
  * `outputs/auto_signal_report.json`
  * `outputs/auto_run_status.json`
* Relevant spec: `.trellis/spec/backend/database-guidelines.md`, "Annual State Router Auto-Signal Mode".

## Open Questions

* ~~Should the first validation run be candidate-only, or may it write official outputs if every gate passes?~~ **Resolved (2026-06-21):** the first validation run is candidate-only. The workflow may generate candidate signals and reports, but must not write official `signal_<DATE>.csv`, must not overwrite the official latest-holdings file, and must not promote candidates — even when every gate passes. This is enforced by a new `--candidate-only` flag on `scripts/run_auto_signal.py`.
