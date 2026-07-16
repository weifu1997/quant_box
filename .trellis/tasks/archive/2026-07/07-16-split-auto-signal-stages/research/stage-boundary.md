# Auto-Signal Stage Boundary Research

## Current Shape

`run_auto_signal.py` is both a CLI entrypoint and the implementation owner. The main orchestration invokes five stages in order and writes a status artifact after each stage. The stage functions share these concerns:

* `status` and `artifacts` are mutable run-scoped state.
* Config/path resolution and output directories are passed through every stage.
* Annual-router runtime construction is used by both backtest and signal stages.
* Quality reports from optimization and backtest feed the final signal gate and report.
* Tests patch script globals such as `load_or_compute_factors`, `run_backtest`, `generate_signal`, and `run_walk_forward_grid_validation`.

## Stage Inputs and Outputs

| Stage | Inputs | Outputs | Primary side effects |
|---|---|---|---|
| Data preparation | CLI args, config, target end date | factor file/frame, prices, health, governance, gates | refresh/conversion/factor/health/governance artifacts and status updates |
| Optimization | args, config, factors, prices | selected config/params, validation, summary | validation CSV/summary and optimizer status |
| Backtest | args, selected config, factors, prices | backtest result/config/quality, router runtime, diagnostics | equity/holdings/trades/quality/diagnostic artifacts |
| Signal | args, selected config, factors, prices, gate results, backtest result | signal result, orders, failure analysis, account and gate reasons | candidate/official signal, holdings, orders, execution/failure artifacts |
| Report | all prior stage results and paths | report paths/payload | report JSON/Markdown and optional history archive |

## Dependency Risks

1. **Global monkeypatch coupling**: moving a function changes where its collaborators are looked up. Existing tests patch the script module, so direct relocation requires either test migration or an explicit services/context seam.
2. **Status ordering**: `_stage()` appends rows and writes `auto_run_status.json`; moving it must preserve duplicate stage rows, exact names, state values, timestamps, and messages.
3. **Router consistency**: backtest and signal must continue to consume the same `AnnualStateRouterRuntime` and routed score panel. Do not rebuild a different panel in the signal stage.
4. **Gate propagation**: data, governance, parameter, backtest, and account gates must remain separate inputs to signal/report stages; no stage may infer official eligibility from partial results.
5. **Memory release**: the main flow intentionally trims the factor frame before annual-router construction. The new context must not retain a second full factor copy.
6. **Candidate boundary**: candidate writes and promotion must remain explicit; report/archive extraction must not accidentally promote or overwrite official state.

## Recommended Boundary

Use a `src.auto_signal` package with a small `models.py` and one module per stage. Keep `main()` and argument parsing in the script. Pass a typed run context plus explicit service collaborators so tests can inject fakes without importing CLI internals.

Compatibility wrappers in `scripts/run_auto_signal.py` should re-export canonical stage functions and result models during the migration. Tests should gradually move from patching script globals to patching the canonical stage module or injecting services; no production stage should import the CLI script.

## Alternative Considered

A single `src.auto_signal_pipeline` module would preserve more global behavior and reduce initial files, but it would leave router/evidence, status, and report concerns coupled. It is acceptable as a temporary bridge only if a follow-up extraction plan is recorded.

## Verification Plan

* Add direct unit tests for each stage's result bundle and failure behavior.
* Add an import-boundary test asserting `src.auto_signal.*` does not import `scripts.run_auto_signal`.
* Run existing auto-signal tests unchanged first, then migrate patch points deliberately.
* Run full Python tests, strict doctor, frontend build, Playwright, and CI after the final shrink of the script.
