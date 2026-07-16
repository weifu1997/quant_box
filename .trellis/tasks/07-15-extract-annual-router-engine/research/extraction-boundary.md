# Annual Router Extraction Boundary

## Current Dependency Shape

The annual-router implementation is spread across three CLI scripts:

* The backtest script owns score routing, annual decisions, route adjustment, configuration construction, and schedule generation.
* The probe script owns the base market-state classifier and trailing benchmark calculations.
* The grid script owns turnover-mode transformations.

The grid imports the backtest module at import time. The backtest imports the grid helper inside `main()` to avoid an immediate circular-import failure. Auto-signal then imports from both modules. This makes CLI files de facto production libraries and prevents a clean staged split of auto-signal.

## Production-owned Behavior

The following behavior belongs in `src/annual_router.py` because it is deterministic engine logic shared across entrypoints:

* Engine contract and immutable source/run models.
* Benchmark normalization and trailing return/volatility calculations.
* Base route classification, route row construction, and route selection for a date.
* Annual equity routing used by the probe.
* Score point-in-time lookup, canonical routed signal dates, and next-trade-date mapping.
* Annual route decisions and configured decision adjustment.
* Score-source routing and fallback selection.
* Turnover-mode transformation.
* Selection, exposure, and risk-exit schedule construction.
* Routed backtest configuration assembly, since it composes production policies and does not perform CLI I/O.

## Script-owned Behavior

The following remains in `scripts`:

* CLI argument parsing and defaults.
* Config and data-file resolution.
* Factor/selector score-source construction.
* Grid enumeration, resumption, cache files, and combo keys.
* Audit, diagnostic, candidate, and report writing.
* Command-specific `main()` orchestration.

## Compatibility Strategy

Existing script modules will import extracted names at module scope. This preserves `from scripts... import ...` callers while making `src.annual_router` the only implementation owner. New production callers, especially auto-signal, import the shared engine directly from `src`.

## Point-in-time Score Lookup

The current helper delegates to a score accessor in another CLI script. The production implementation should select the latest eligible date from the first MultiIndex level and return that cross-section directly. Characterization tests must cover index names, normalized dates, empty inputs, and score names so the replacement is behaviorally equivalent.

## Risks and Controls

* **Semantic drift**: copy behavior before cleanup; add characterization tests around exact dictionaries/dataframes.
* **Import compatibility**: test both the canonical `src` import and legacy script imports.
* **Evidence invalidation**: do not change `ANNUAL_ROUTER_ENGINE_CONTRACT` or serialized source definitions.
* **Hidden circular imports**: add a static import-boundary assertion that `src/annual_router.py` contains no `scripts` import.
* **Pandas index drift**: retain date normalization, sorting, MultiIndex names, and output series names.

## Chosen Approach

Use one cohesive production module with compatibility re-exports. A multi-file package would add indirection before the engine boundary has stabilized, while leaving logic in scripts would not unblock the next auto-signal decomposition stage.
