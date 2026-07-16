# Extract Annual Router Engine into `src`

## Goal

Move the reusable annual market-state routing, score routing, and schedule construction logic out of CLI scripts and into a stable `src` module. This removes the current script-to-script circular dependency while preserving strategy behavior and every existing operational interface.

## What I Already Know

* The quality gate is currently green on Windows, Ubuntu, frontend build, and Chromium E2E.
* `scripts/run_annual_state_router_grid.py` imports router models and behavior from the backtest script.
* `scripts/run_annual_state_router_backtest.py` imports `definitions_for_turnover_mode` from the grid script inside `main()`, creating a deferred circular dependency.
* `scripts/run_annual_state_router_backtest.py` also imports `route_for_date` from the probe script, so reusable routing behavior currently depends on CLI modules.
* `scripts/run_auto_signal.py` imports annual-router behavior from both the backtest and grid scripts.
* Existing tests intentionally import public names from the scripts; those imports must continue to work during this extraction.

## Requirements

* Add a production-owned annual-router module under `src`.
* Move the engine contract, data models, market-state routing primitives, score routing, route adjustment, turnover-mode transformation, and route-derived schedule construction into that module.
* Ensure production code under `src` does not import from `scripts`.
* Update the backtest, grid, probe, and auto-signal entrypoints to consume the production module.
* Keep compatibility re-exports from existing scripts for callers and tests that still use the old import paths.
* Preserve CLI flags, default values, report/evidence schemas, cache fingerprints, route reasons, source selection, exposure values, and scheduling behavior.
* Keep file loading, argument parsing, grid iteration, score-source construction, report writing, and CLI orchestration in `scripts`.

## Acceptance Criteria

* [x] `src.annual_router` exposes the reusable annual-router contract and engine API.
* [x] `src.annual_router` has no imports from `scripts`.
* [x] The grid and backtest scripts no longer form a circular dependency.
* [x] `scripts/run_auto_signal.py` imports reusable annual-router behavior from `src` rather than the router CLI scripts.
* [x] Existing script import paths remain compatible for the extracted names.
* [x] Characterization tests show equivalent market-state decisions, routed scores, turnover transformations, and schedules.
* [x] Annual-router, auto-signal, and import-boundary tests pass.
* [x] The full Python test suite, frontend build, and Playwright suite pass.

## Definition of Done

* Tests added or updated for the new module boundary and compatibility exports.
* Full local quality gate passes.
* GitHub Actions passes on Windows backend, Ubuntu backend, frontend build, and Chromium E2E.
* Trellis specs are updated if the extraction establishes a durable module boundary.
* The task is archived after the verified commit is pushed.

## Technical Approach

Create a single `src/annual_router.py` module first, because the extracted surface is cohesive and already shares the same pandas-based data contracts. Move pure and production-layer behavior into it, then turn the existing script definitions into imports/re-exports. Implement the point-in-time score lookup locally so the production module does not depend on `scripts.run_quality_selector_gate_backtest.daily_score_for_date`.

The initial production API includes:

* `ANNUAL_ROUTER_ENGINE_CONTRACT`
* `ScoreSourceDefinition`, `RoutedScoreRun`, and the probe-compatible `AnnualRouterRun`
* `normalize_benchmark`, trailing metric helpers, `route_source`, `route_for_date`, and `run_annual_state_router`
* `run_annual_state_score_router`, point-in-time lookup, signal/trade-date mapping, annual decisions, and route adjustment
* `definitions_for_turnover_mode`
* selection, exposure, and risk-exit schedule builders
* `routed_backtest_config`, while keeping its dependencies inside `src`

## Decision (ADR-lite)

**Context**: Router behavior is shared by backtest, grid search, probe, and production signal generation, but it is currently owned by CLI scripts with a deferred circular import.

**Decision**: Establish `src/annual_router.py` as the single production owner. Keep CLI-specific I/O and orchestration in scripts and retain compatibility re-exports during migration.

**Consequences**: Imports become one-directional and later auto-signal splitting can depend on a stable engine API. The compatibility layer temporarily leaves some names visible from scripts, but avoids a disruptive all-at-once migration.

## Out of Scope

* Changing routing thresholds, route reasons, source definitions, or exposure policy.
* Changing annual-router evidence validation or cache-key semantics.
* Splitting `run_auto_signal.py`; that is the next roadmap task after this extraction is stable.
* Splitting Dashboard backend/frontend.
* Refactoring `src/backtest.py` or other high-risk calculation cores.
* Introducing new dependencies or a package hierarchy before the single-module boundary proves insufficient.

## Technical Notes

* Primary callers: `scripts/run_annual_state_router_backtest.py`, `scripts/run_annual_state_router_grid.py`, `scripts/run_annual_state_router_probe.py`, and `scripts/run_auto_signal.py`.
* Primary regression suites: `tests/test_run_annual_state_router_backtest.py`, `tests/test_run_annual_state_router_grid.py`, `tests/test_run_annual_state_router_probe.py`, and `tests/test_run_auto_signal.py`.
* Durable annual-router contracts are documented in `.trellis/spec/backend/database-guidelines.md`.
* Extraction analysis is recorded in `research/extraction-boundary.md`.
