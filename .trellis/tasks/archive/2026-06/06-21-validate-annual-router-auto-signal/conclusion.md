# Conclusion: annual_state_router Auto-Signal Candidate-Only Validation

**Date:** 2026-06-21
**Mode:** candidate-only (`scripts/run_auto_signal.py --skip-update --candidate-only --no-archive`)
**Auto-resolved target date:** 2026-06-18 (after latest-trade-date cutoff)
**Signal date produced:** 2026-06-09 (latest routed score date on/before target)

## Verdict: NOT READY for manual trading review

The router strategy evidence and the rebuilt full-history backtest are strong, but a
hard **data freshness gate fails**, so the run is correctly blocked from official output.
Production readiness depends on the data/governance gates, which are authoritative.

## Gate Results

| Gate | Result | Detail |
| --- | --- | --- |
| Data governance (point-in-time) | PASS | `point_in_time_ready` |
| Parameter quality (router evidence) | PASS | 12 windows, positive_return_rate=1.0, annual_return_mean=0.2645, sharpe=1.60, worst_drawdown=-0.177; evidence combo matches `config/settings.yaml` |
| Backtest quality (rebuilt routed scores) | PASS | annual_return=0.2603, max_drawdown=-0.1768, all yearly gates pass |
| Account | PASS | no account issues |
| **Data health** | **FAIL** | `factor_latest_coverage_below_threshold:0.0000<0.9500`, `factor_latest_before_end:2026-05-27<2026-06-18` |
| Candidate-only hold | applied | `candidate_only_requested` |

The factor cache only extends to **2026-05-27**, while the workflow auto-resolved the
target trade date to **2026-06-18**. Factor data is ~3 weeks stale relative to the target,
so the data-health gate blocks official output. This is a genuine production-readiness
finding, not a workflow defect.

## Chain Integrity (verified)

- `strategy_mode=annual_state_router` recorded in report and status; optimizer skipped with `annual_state_router_enabled`.
- Routed score panel was rebuilt (103,132 score rows, 138 routed signal dates, 12 year routes) and drove both backtest and signal generation — no fallback to legacy strategy scores.
- Rebuilt backtest reproduces the formal evidence (annual_return 0.260 vs 0.261, drawdown -0.177 vs -0.177), confirming `routed.scores` + `routed_backtest_config()` are used.
- Evidence combo matches config on every numeric/categorical field and reason set (no `annual_state_router_evidence_combo_mismatch`).

## Candidate-Only Safety (verified)

- `is_executable=false`, `status=blocked`.
- Only candidate artifacts written this run: `candidate_signal_2026-06-09.csv` (5 BUY rows), `candidate_holdings_2026-06-09.csv` (5 instruments), `manual_orders_candidate_2026-06-09.csv` (all timestamped 2026-06-21 23:21).
- No official outputs written this run: `signal_2026-06-09.csv` / `manual_orders_2026-06-09.csv` are stale leftovers from 2026-06-14.
- `outputs/latest_holdings.csv` was **not** overwritten (last modified 2026-06-20, the prior run). No promotion occurred.

## Artifacts

- `outputs/auto_parameter_quality.json` — router parameter quality (acceptable)
- `outputs/auto_backtest_quality.json` — rebuilt backtest quality (acceptable)
- `outputs/auto_annual_state_router_score_routes.csv` (138 rows), `outputs/auto_annual_state_router_year_routes.csv` (12 rows)
- `outputs/auto_signal_report.json` — `strategy_mode=annual_state_router`, `candidate_only=true`, `is_executable=false`
- `outputs/auto_run_status.json` — `status=blocked`, `block_reasons` includes data-health failures + `candidate_only_requested`

## Path to "ready"

1. Refresh raw + factor data through the latest trade date (run without `--skip-update`, or run the point-in-time data update), so factor coverage reaches the target date.
2. Re-run candidate-only to confirm the data-health gate clears and all other gates still pass.
3. Only then consider an official run (without `--candidate-only`) followed by a deliberate manual review before any `--promote-candidate`.

## Note on this validation run

A second `run_auto_signal` process overlapped this run's early stages (both auto-resolved to
the same target/config and both hit the same hard data gate). This run (started 22:59:30) was
the final writer and its status/report/artifacts are internally consistent; file mtimes confirm
the candidate files and route artifacts all come from this run, and no official outputs or the
latest-holdings file were modified. For a fully isolated re-run, ensure no other auto-signal
process is active.
