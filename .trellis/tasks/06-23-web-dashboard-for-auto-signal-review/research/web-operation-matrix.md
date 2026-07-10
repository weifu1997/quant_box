# Web operation matrix and timing baseline

## Goal

Make the Web workspace the primary surface while preserving fixed-command safety and Windows/Ubuntu compatibility.

## Core workflow coverage

| Pipeline stage | Web action | Backend script | Typical warm-cache time | Cold/full time | Current status |
| --- | --- | --- | --- | --- | --- |
| Tushare configuration | Check Tushare config | `check_tushare_config.py` | <1 min | <1 min | Available |
| Raw daily data | Incremental market update | `run_update_data.py` | 5–20 min | 1–8+ hours | Available |
| Point-in-time data | Full PIT update / daily_basic repair | `run_update_point_in_time_data.py` | 5–30 min | 1–4+ hours | Available |
| Fundamentals | Missing-only fundamentals update | `run_update_fundamentals.py` | 10–30 min | 1–4+ hours | Available |
| Historical universe | Build historical universe | `run_build_universe.py` | 10–30 min | 30–90 min | Available |
| Price conversion | Convert raw data | `run_convert_data.py` | 2–8 min | 10–30 min | Available |
| Alpha158 | Calculate factors | `run_calc_factors.py` | 1–5 min if cache valid | 10–60 min | Available |
| Factor research | Factor diagnostics | `run_factor_diagnostics.py` | 2–10 min | 10–30 min | Available |
| Parameter validation | Walk-forward optimization | `run_optimize.py` | 5–20 min | 30–120+ min | Available |
| Realistic backtest | Backtest | `run_backtest.py` | 2–10 min | 10–30 min | Available |
| Research diagnosis | Quant/optimization/evidence reports | diagnostic scripts | 1–10 min | 10–30 min | Available |
| Fundamental review | Fundamental screen | `run_fundamental_screen.py` | 1–5 min | 5–15 min | Available |
| Candidate signal | Generate candidate signal | `run_daily_signal.py` | 1–5 min | 5–15 min | Available |
| Full signal pipeline | Candidate/normal auto signal | `run_auto_signal.py` | 10–45 min | 1–8+ hours | Available |
| Manual execution | Preview/apply fills | in-process backend service | <1 min | <1 min | Available |
| Final review | Latest dashboard snapshot | artifact API | <2 sec | <5 sec | Available |

Times are operational estimates, not guarantees. Network rate limits, cache freshness, symbol count, date span, and CPU/memory dominate actual duration.

## Remaining coverage

- Advanced research probes and grids under `scripts/run_*probe.py`, `run_*grid.py`, and goal-specific research scripts need an advanced workspace with validated parameter schemas rather than unrestricted command strings.
- Account/settings editing needs a secret-safe form contract and explicit validation before it can replace local YAML/CSV editing.
- Full browser automation is still required for every navigation, refresh, run, stop, preview, apply, artifact, and empty/error-state button.
- Real cold-cache timing must be measured on the target Windows machine and later on the Ubuntu server; estimates above are only planning baselines.

## Performance priorities

1. Reuse raw, price, factor, IC-weight, and router score caches by default.
2. Keep full refresh and force-recompute actions separate from daily actions.
3. Expose stage-level progress instead of making long tasks appear frozen.
4. Keep one expensive workflow active at a time until resource isolation is designed.
5. Record elapsed time per job and stage so future optimization targets measured bottlenecks.
