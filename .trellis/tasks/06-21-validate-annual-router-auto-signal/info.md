# Validation Report — annual_state_router Auto-Signal Chain

**Date:** 2026-06-21
**Task:** validate-annual-router-auto-signal
**Status of conclusion:** NOT READY for manual trading review (data staleness blocker)

---

## 1. Command Run

```bash
.\.venv\Scripts\python.exe scripts\run_auto_signal.py \
  --skip-update --skip-convert \
  --skip-optimize \
  --candidate-only
```

Decisions behind the flags (confirmed with user before running):

- `--skip-update --skip-convert` — validate router wiring on existing data; do not exercise the data-fetch path (no incremental value for "is the router correctly wired", adds network/account uncertainty).
- `--skip-optimize` — do not tune parameters or expand grids; use the current `config/settings.yaml` `annual_state_router` configuration (matches PRD Out of Scope).
- `--candidate-only` — per PRD Open Questions resolution (2026-06-21): first validation run is candidate-only; never write official `signal_<DATE>.csv`, never overwrite official latest holdings, never promote — even if all gates pass.
- No bypass flags (`--allow-unhealthy` / `--allow-low-quality` / `--force-official`) — if a gate blocks, record it as evidence rather than override.

Exit code: 0. Run window: 2026-06-21T22:54:21 → 23:12:45.
Target trade date resolved to **2026-06-18** (after latest-trade-date cutoff).

---

## 2. Acceptance Criteria — Item by Item

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Run without changing optimization/grid params | ✅ PASS | `optimize_params` stage `skipped` (reason `annual_state_router_enabled`); report `skip_optimize=true` |
| 2 | `auto_parameter_quality.json` exists, records router param quality | ✅ PASS | `is_acceptable=true`, 12 windows, annual_return_mean 0.2645, min 0.2017, sharpe 1.60 |
| 3 | `auto_backtest_quality.json` exists, records rebuilt backtest quality | ✅ PASS | `is_acceptable=true`, annual_return 0.2603, max_drawdown -0.1768, calmar 1.47 |
| 4 | Both route CSVs exist and non-empty | ✅ PASS | `auto_annual_state_router_score_routes.csv` 138 rows; `auto_annual_state_router_year_routes.csv` 12 yearly rows (2015–2026) |
| 5 | `auto_signal_report.json` records `strategy_mode=annual_state_router` | ✅ PASS | `strategy_mode=annual_state_router`, `selected_params_status=annual_state_router` |
| 6 | Evidence combo mismatches recorded as blockers | ✅ PASS (no mismatch) | `parameter_quality.issues=[]`; config combo matches evidence `codex_router_grid_20260614_beta20_exposure_hit_metrics.json` field-by-field |
| 7 | Data governance / backtest / parameter / account gates summarized | ✅ PASS | See section 3 |
| 8 | Clear ready / not ready conclusion | ✅ PASS | See section 4 |

---

## 3. Gate Results

| Gate | Result | Detail |
|------|--------|--------|
| **Router wiring** | ✅ Healthy | Used routed scores, no fallback to legacy strategy. 103,132 routed score rows, 138 routed signal dates, 12 yearly routes. `optimize_params` skipped because router mode active. |
| **Parameter quality** | ✅ PASS | annual_return_mean 0.2645 (≥0.20), min 0.2017 (≥0.20), positive_return_rate 1.0, sharpe 1.60, worst yearly drawdown -0.1768 (≥-0.20), annual_turnover 7.92 (≤20), trade_cost_ratio 0.178 (≤0.20). 12 windows. issues=[]. |
| **Backtest quality** | ✅ PASS | Rebuilt annual_return 0.2603, max_drawdown -0.1768, calmar 1.47, all years pass return target and drawdown limit. issues=[]. Rebuild reproduces evidence metrics (26.0% vs 26.1% evidence) — router rebuild is reproducible. |
| **Data governance** | ✅ PASS | `is_point_in_time_ready=true`. 1 non-blocking warning: `st_calendar_end_before_factor_end:2026-06-09<2026-06-12`. Index/historical universe month coverage 1.0. |
| **Account** | ✅ PASS | No account issues. Holdings loaded from `config/current_holdings.csv`, total_asset 1,000,000, validation clean. |
| **Data health** | ❌ **FAIL (blocker)** | `factor_latest_coverage_below_threshold:0.0000<0.9500` and `factor_latest_before_end:2026-05-27<2026-06-18`. Factor data stops at 2026-05-27, ~3 weeks behind target trade date 2026-06-18; latest-day factor coverage 0%. |

Non-gate research note: `research_diagnostics` reported `market_cap_asof_stale:6>5` (market-cap as-of 2026-06-12 is 6 days stale vs 5-day threshold) — diagnostic only, not a hard gate, but same root cause (stale local data).

---

## 4. Conclusion: NOT READY for manual trading review

**Chain validation itself: PASSED.** `annual_state_router` correctly rebuilds routed scores, runs the historical backtest, and generates signals inside the production auto-signal chain. Backtest / parameter / governance / account gates all pass. The production wiring is healthy at the code level.

**Cannot enter manual trading review** because the **data health gate failed**: factor data stops at 2026-05-27, behind target trade date 2026-06-18, with 0% latest-day factor coverage. The system correctly classified the run as `blocked` and wrote only candidate files — no official files touched. `--candidate-only` and the gates behaved exactly as designed.

The block has two layers, kept separate so neither masks the other:

1. **Stale real data** (factors not refreshed to latest) — this is the root-cause blocker.
2. `candidate_only_requested` — user-requested; would block official output regardless of data state, independent of (1).

No workflow bug was found. Chain behavior is correct, so no code change or focused test was required (Definition of Done item is N/A this run).

---

## 5. Scope Limitation of This Run

Because `--skip-update --skip-convert` skipped data refresh, this run validates **only** the router compute chain on existing (stale) data. It does **not** prove the data-update chain works. A full production-readiness validation requires a separate run with data update enabled.

---

## 6. Prerequisites for Re-Run (to reach review readiness)

1. Refresh data (drop `--skip-update --skip-convert`) so factors reach 2026-06-18.
   - At run time `now` was 2026-06-21 22:54, already past the 2026-06-18 close cutoff, so the target stays 2026-06-18; a data update should backfill factor coverage.
2. Re-run validation, ideally still `--candidate-only` for the first clean-data pass.
3. Confirm the data health gate clears (`factor_latest_coverage ≥ 0.95`, factor latest = target date).
4. Confirm `market_cap_asof_stale` clears once daily_basic is refreshed.
5. Only after data health passes and a human reviews the candidate output should promotion be considered (still out of scope for this task).

---

## 7. Candidate Artifacts Produced (no official files written)

- `outputs/candidate_signal_2026-06-09.csv` (5 BUY)
- `outputs/candidate_holdings_2026-06-09.csv`
- `outputs/manual_orders_candidate_2026-06-09.csv`
- `outputs/auto_signal_report.json` (full run record)
- `outputs/auto_parameter_quality.json`, `outputs/auto_backtest_quality.json`
- `outputs/auto_annual_state_router_score_routes.csv` (138 rows)
- `outputs/auto_annual_state_router_year_routes.csv` (12 rows)
- `outputs/auto_run_status.json` (status=blocked)
- `outputs/history/2026-06-09/` (archived run)

No `config/settings.yaml`, evidence files, official signal, or latest-holdings files were modified.
