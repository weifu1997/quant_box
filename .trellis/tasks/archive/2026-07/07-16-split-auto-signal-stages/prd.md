# Split Auto-Signal Stages

## Goal

Turn the 1,900-line `scripts/run_auto_signal.py` workflow into importable, testable stage modules while keeping the existing CLI, output artifacts, status semantics, quality gates, annual-router behavior, and candidate/official safety boundary unchanged.

## What I Already Know

* The previous task extracted the reusable annual router into `src.annual_router`; auto-signal now consumes that engine directly.
* `scripts/run_auto_signal.py` currently owns five stage functions:
  * `_run_data_preparation_stage`
  * `_run_optimization_stage`
  * `_run_backtest_stage`
  * `_run_signal_stage`
  * `_write_auto_report_stage`
* The script also contains annual-router runtime/evidence helpers, result dataclasses, status/artifact helpers, candidate promotion, and CLI orchestration.
* The main flow is linear: target-date resolution → data preparation → optimization/selection → parameter quality → backtest/diagnostics → signal/orders/failure analysis → report/archive → final status.
* `outputs/auto_run_status.json`, `outputs/auto_signal_report.json`, candidate files, official files, and route evidence files are established contracts.
* Existing tests import the script as a module and patch many of its globals. A split must either preserve those patch points through explicit compatibility wrappers or deliberately migrate tests to canonical stage modules.
* The repository is manual-trading-assist software; this task must not add broker execution or weaken official-output gates.

## Requirements

* Introduce an importable `src.auto_signal` package (or an equally clear `src` module boundary) with stage-specific modules.
* Extract stage result models into a stable importable module.
* Extract data preparation, optimization, backtest/diagnostics, signal/order generation, and report/archive stages into independently importable functions.
* Keep `scripts/run_auto_signal.py` responsible for CLI parsing, target-date resolution, dependency wiring, linear orchestration, top-level error handling, and user-facing logging.
* Preserve all existing CLI flags, defaults, skip behavior, status stage names/states/messages, output filenames, JSON/CSV schemas, archive behavior, quality gates, and annual-router route provenance.
* Preserve candidate-only, allow-low-quality, force-official, and promote-candidate safety semantics.
* Make stage dependencies explicit enough that unit tests can patch or inject collaborators without relying on hidden module globals.
* Keep legacy script-level stage/helper imports working where practical during migration; document any intentionally changed patch path.
* Add characterization tests for stage result contracts, stage ordering/status updates, annual-router and legacy strategy paths, candidate/official output behavior, and import boundaries.

## Acceptance Criteria

* [x] `src.auto_signal` contains independently importable stage functions and result models.
* [x] `scripts/run_auto_signal.py` is reduced to CLI/orchestration and compatibility exports; it no longer owns the full stage implementations.
* [x] Stage functions have explicit input/output contracts and do not parse CLI arguments directly.
* [x] Existing annual-router and legacy optimizer paths produce equivalent results and artifacts.
* [x] `auto_run_status.json` preserves stage names, state transitions, messages, and final gate fields.
* [x] Candidate outputs never overwrite official signal/holdings during the split.
* [x] Existing `tests/test_run_auto_signal.py` passes, with new direct stage/import-boundary tests.
* [x] Full Python tests, frontend build, Playwright, strict doctor, and GitHub CI pass.

## Definition of Done

* Tests added/updated for every extracted stage boundary and compatibility path.
* No generated market data, account data, or output artifacts committed.
* Trellis specs updated for the new stage ownership and status/artifact contracts if needed.
* Work committed, task archived, journal recorded, pushed, and CI verified.

## Technical Approach

Use a staged `src.auto_signal` package with one module per workflow concern:

* `models.py`: stage result dataclasses and shared runtime/context types.
* `status.py`: status artifact and stage transition primitives, preserving `_stage()` semantics.
* `router.py`: auto-signal-specific annual-router runtime construction, route signal config, evidence quality/provenance checks, and router report projection. Core route math remains in `src.annual_router`.
* `data_stage.py`: refresh, conversion reuse, factors, health, adjustment-factor metadata, and governance.
* `optimization_stage.py`: grid parsing, annual-router skip path, walk-forward optimization, timeout/progress, and selected config.
* `backtest_stage.py`: historical backtest, annual-router/legacy score selection, quality, and research diagnostics.
* `signal_stage.py`: signal generation, account/order templates, failure analysis, fundamental screen, and candidate writes.
* `report_stage.py`: report JSON/Markdown and optional archive.

The script will assemble a lightweight dependency/context object in `models.py` and call these modules in the same linear order. Stage functions will receive explicit context and collaborators (or a typed services bundle), so tests can inject fakes and the script can retain compatibility wrappers while migration is underway.

Migration order:

1. Add models/status contracts and characterization tests without changing behavior.
2. Extract router/evidence helpers and data preparation.
3. Extract optimization and backtest stages.
4. Extract signal and report stages.
5. Shrink the script to parser/orchestration, update imports/tests/specs, and run the full gate.

## Decision (ADR-lite)

**Context**: The current script mixes CLI parsing, stage logic, file I/O, quality policy, and shared mutable status. A mechanical move would create hidden imports and break test patchability.

**Options**:

* **A — Explicit stage package with typed context/services (Recommended)**: canonical stage modules own behavior; the script wires dependencies and keeps compatibility exports. More initial plumbing, but clear boundaries and test seams.
* **B — One `src.auto_signal_pipeline` module**: move the functions together and leave only a thin script wrapper. Lower migration risk, but stage coupling and future Dashboard reuse remain high.
* **C — Keep implementations in the script and add façade functions**: smallest diff, but does not actually establish stage ownership or reduce the long-term coupling.

**Decision**: Use option A: an explicit stage package with typed context/services and behavior-preserving script wrappers.

**Consequences**: The initial change adds more small modules and dependency-wiring code than a mechanical move, but each stage receives a stable test seam, production code no longer depends on CLI globals, and later Dashboard/worker orchestration can reuse stages without importing the command script. Compatibility wrappers remain temporary and can be removed only in a separate, explicitly approved cleanup.

## Out of Scope

* Changing strategy formulas, optimization grids, annual-router thresholds, or risk policy.
* Changing any CLI flag, config key, output filename, JSON/CSV schema, or gate meaning.
* Dashboard backend/frontend split; it follows this task.
* Refactoring `src/backtest.py` or other high-risk calculation cores.
* Adding asynchronous execution, a task queue, broker integration, or automatic order placement.

## Technical Notes

* Main source: `scripts/run_auto_signal.py` (1,902 lines).
* Main regression suite: `tests/test_run_auto_signal.py` (1,116 lines).
* Related contracts: `.trellis/spec/backend/directory-structure.md`, `.trellis/spec/backend/database-guidelines.md`, `.trellis/spec/backend/logging-guidelines.md`, `.trellis/spec/backend/quality-guidelines.md`.
* Dependency and seam analysis is in [`research/stage-boundary.md`](research/stage-boundary.md).
