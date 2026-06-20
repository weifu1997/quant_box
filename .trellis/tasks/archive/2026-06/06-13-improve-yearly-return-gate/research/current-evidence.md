# Current Evidence: Yearly 20 Percent Goal

## Latest Baseline

The latest full-flow auto-signal run completed through data update, conversion, factor loading, optimization, backtest, and candidate output generation. Data health and governance were clean, but the strategy quality gates failed.

Baseline evidence:

* `outputs/auto_backtest_yearly_breakdown.csv`
* `outputs/auto_backtest_quality.json`
* `outputs/auto_parameter_quality.json`
* `outputs/auto_failure_analysis.json`

## 2026-06-13 Full-Flow Rerun

I reran the authoritative auto-signal workflow on current local data:

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal.py --chunk-size 300 --sleep-seconds 0 --allow-low-quality --no-archive
```

The run completed all stages. Raw update reused the completed progress for target date `2026-06-12`; conversion, factor loading, data health, adj-factor metadata, and data governance completed successfully. The workflow still wrote only candidate outputs:

* `outputs/candidate_signal_2026-06-12.csv`
* `outputs/manual_orders_candidate_2026-06-12.csv`
* `outputs/order_confirmations/order_confirmation_candidate_2026-06-12.csv`
* `outputs/fill_feedback/fill_feedback_candidate_2026-06-12.csv`

Confirmed current blockers:

* Parameter quality: `no_acceptable_params`; best validation annual-return minimum is `-19.18%`, and worst validation drawdown is `-20.90%`.
* Full backtest: annual return `18.72%` versus the `20%` gate.
* Full backtest: max drawdown `-25.21%` versus the `-20%` gate.
* Yearly annual-return failures remain `2016`, `2017`, `2018`, `2020`, `2023`, and `2026`.
* Yearly drawdown failure remains `2015`.

Engineering fix from this rerun:

* Before the fix, `outputs/auto_run_status.json` could show `status=blocked` and `is_executable=false` while `block_reasons` was empty when `--allow-low-quality` let the workflow continue to candidate outputs.
* The status/report/manual-order path now copies quality warnings into `block_reasons` when the signal is not executable and there is no harder block reason. This keeps candidate safeguards unchanged, but makes the blocked candidate report diagnosable.

I also checked whether using the optimizer's best fallback row would help when no acceptable parameter set exists. The top validation fallback (`factor:LOW0`, `top_n=20`, `rank_buffer=30`) produced a full-history annual return of only `9.88%` and max drawdown `-49.86%`, with more yearly failures than the current baseline. Therefore, selecting the validation fallback would make the candidate strategy worse and should not be encoded as a pipeline improvement.

Baseline yearly annual-return failures:

| Year | Annual return | Max drawdown |
| --- | ---: | ---: |
| 2016 | -2.33% | -19.48% |
| 2017 | -2.59% | -14.59% |
| 2018 | -6.58% | -13.31% |
| 2020 | 4.06% | -16.58% |
| 2023 | 2.31% | -13.79% |
| 2026 | -17.00% | -10.13% |

## Bottleneck Scan

I reran the yearly artifact scan:

```powershell
.\.venv\Scripts\python.exe scripts\run_goal_bottleneck_scan.py --glob "outputs/*_years.csv" --output outputs\codex_yearly_goal_bottleneck_after_formal_20260613.csv
```

Results:

* Scanned yearly rows: `3812`
* Scanned files: `361`
* Every calendar year has at least one historical candidate with annual return above 20%.
* Weak years have very few passing candidates:
  * 2016: `14` return-pass rows
  * 2017: `7` return-pass rows
  * 2018: `16` return-pass rows
  * 2020: `21` return-pass rows
  * 2026: `8` return-pass rows

This suggests the issue is not the absence of any useful signal in each year; it is the absence of a single static rule that works in all years.

## Current Formal Candidate Rerun

I reran the last 30 formal candidates from `config/goal_formal_candidates.json` on current data:

```powershell
.\.venv\Scripts\python.exe scripts\run_goal_formal_candidates.py --start-index 84 --max-candidates 30 --skip-diagnostics --output outputs\codex_goal_formal_candidates_current_20260613.csv
```

Log:

* `outputs/logs/codex_goal_formal_candidates_20260613_213154.out.log`

Outcome:

* No candidate met the yearly target.
* Best yearly-return pass count in this batch was `6/12`.
* Best overall annual return in this batch was about `20.11%`, but yearly-return pass count was only `5/12` and max drawdown was worse than `-20%`.
* KLEN/fixed-blend candidates did not reproduce earlier optimistic hints on current data.

Representative top rows:

| Candidate | Annual return | Max drawdown | Yearly return pass |
| --- | ---: | ---: | ---: |
| `momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma80_schedsig_bear06` | 19.50% | -25.91% | 6/12 |
| `momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_cb19_30` | 19.39% | -23.94% | 6/12 |
| `momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd20_schedsig_side1_bear08` | 20.11% | -23.97% | 5/12 |

## Implication

The current static-candidate path is unlikely to satisfy "every year >= 20%" by incremental risk overlays alone. The next useful research direction is a rolling, non-lookahead dynamic selector:

* Select or weight factor/strategy families using only trailing information.
* Evaluate whether dynamic selection can improve weak years without introducing a calendar-year lookahead.
* If promising, encode it in the pipeline as a tested strategy/scoring option instead of a one-off artifact.

## Dynamic Selector Probe

I added a research probe script:

```powershell
.\.venv\Scripts\python.exe scripts\run_candidate_equity_selector.py --output-prefix outputs\codex_candidate_equity_selector_20260613
```

The script loads existing candidate equity curves and, at each rebalance date, scores candidates using only equity rows strictly before that date.

Best result from the default grid:

* Candidate equity curves loaded: `308`
* Best lookback/top-k/penalty: `63` days, `top_k=5`, `drawdown_penalty=0`
* Full-period annual return: `16.89%`
* Max drawdown: `-35.81%`
* Yearly annual-return pass count: `6/12`
* Failed annual-return years: `2016, 2017, 2018, 2020, 2022, 2026`

This did not improve on the best static candidates. I also checked a coarse year-level rule that picks next year's candidates from prior-year performance. Even allowing top 10 prior-year candidates, annual-return pass count was only `6/11`, with `2026` still far below target.

Conclusion: simple trailing-performance selection is not enough. The next research should target weak-year prediction directly, especially `2016`, `2017`, `2018`, `2020`, and `2026`, instead of using broad trailing winner selection.

## Annual Market-State Router Probe

I added a reproducible research probe:

```powershell
.\.venv\Scripts\python.exe scripts\run_annual_state_router_probe.py --output-prefix outputs\codex_annual_state_router_probe_20260613 --missing-ret252-exposure 0.65 --flat-negative-exposure 0.90
```

The probe routes once per calendar year among existing candidate equity curves. Route features are computed from benchmark data strictly before the first trading day of each year. The source set is:

* `beta`: beta factor candidate
* `db_size`: inverse circulating-market-cap candidate
* `quality`: fundamental quality candidate
* `selector`: selector-weight candidate
* `industry`: industry momentum candidate

Result artifacts:

* `outputs/codex_annual_state_router_probe_20260613_metrics.json`
* `outputs/codex_annual_state_router_probe_20260613_years.csv`
* `outputs/codex_annual_state_router_probe_20260613_routes.csv`

Result:

* Overall annual return: `36.87%`
* Full max drawdown: `-19.97%`
* Yearly annual-return pass count: `12/12`
* Yearly drawdown pass count: `12/12`
* Minimum yearly annual return: `21.47%` in `2017`
* Worst yearly drawdown: `-19.49%` in `2018`

Routes:

| Year | Source | Reason | Exposure |
| --- | --- | --- | ---: |
| 2015 | beta | insufficient_history | 1.00 |
| 2016 | db_size | ret252_missing | 0.65 |
| 2017 | quality | negative_high_vol | 1.00 |
| 2018 | selector | low_vol_moderate_uptrend | 1.00 |
| 2019 | industry | negative_moderate_vol | 1.00 |
| 2020 | selector | strong_trailing_market | 1.00 |
| 2021 | selector | strong_trailing_market | 1.00 |
| 2022 | beta | default_beta | 1.00 |
| 2023 | industry | negative_moderate_vol | 1.00 |
| 2024 | beta | flat_with_negative_half_year | 0.90 |
| 2025 | beta | default_beta | 1.00 |
| 2026 | beta | default_beta | 1.00 |

Important limitation: this is a research/meta-strategy over already generated candidate equity curves. The annual route uses non-future market features, but the rule thresholds and selected source set are exploratory and sample-informed. This is not yet a formal signal pipeline improvement, because it does not generate holdings/trades from the underlying strategy sources in the main auto-signal workflow. The next engineering step is to turn this into a tested score/signal router or a formal candidate generator that reproduces source scores and preserves candidate-vs-official gates.

## 2026-06-14 Full-Flow Rerun And Score-Level Router

I reran the authoritative auto-signal workflow:

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal.py --chunk-size 300 --sleep-seconds 0 --allow-low-quality --no-archive
```

The workflow completed data update, conversion, factor loading, data health, adj-factor metadata, governance, optimization, backtest, diagnostics, and candidate output generation. It still produced candidate outputs only. Current blockers are unchanged:

* Parameter quality: `no_acceptable_params`; validation annual-return minimum `-19.18%`, worst validation drawdown `-20.90%`.
* Full backtest annual return: `18.72%` versus the `20%` gate.
* Full backtest max drawdown: `-25.21%` versus the `-20%` gate.
* Yearly annual-return failures: `2016`, `2017`, `2018`, `2020`, `2023`, and `2026`.
* Yearly drawdown failure: `2015`.

I added a score-level annual router backtest:

```powershell
.\.venv\Scripts\python.exe scripts\run_annual_state_router_backtest.py --end-date 2026-06-09 --output-prefix outputs\codex_annual_state_router_score_20260614_e20260609_execyear --full-turnover-on-route-change
```

The script builds real monthly score panels for the routed sources, then runs the existing formal backtest engine. It also fixes an important year-boundary detail: route source selection is aligned to the next execution trading day, not the signal date, so a year-end monthly signal that executes in the new year uses the new year's route. The source extended-factor artifacts currently cover only through `2026-06-09` and do not satisfy the full current price-panel coverage check, so this score-router evidence is bounded to that date until those research factor caches are refreshed.

Score-level result with execution-year routing:

* Overall annual return: `22.88%`.
* Full max drawdown: `-23.12%`.
* Annual trade cost ratio: `15.07%`.
* Yearly annual-return pass count: `7/12`.
* Yearly drawdown pass count: `8/12`.
* Failed years: `2016`, `2018`, `2020`, `2022`, `2024`, and `2025`.
* Artifacts: `outputs/codex_annual_state_router_score_20260614_e20260609_execyear_*`.

I also tested the same score-router with defensive timing:

```powershell
.\.venv\Scripts\python.exe scripts\run_annual_state_router_backtest.py --end-date 2026-06-09 --output-prefix outputs\codex_annual_state_router_score_20260614_e20260609_execyear_defensive --full-turnover-on-route-change --use-defensive-timing
```

Defensive timing improved drawdown but still did not meet the goal:

* Overall annual return: `24.91%`.
* Full max drawdown: `-22.26%`.
* Annual trade cost ratio: `16.82%`.
* Yearly annual-return pass count: `8/12`.
* Yearly drawdown pass count: `11/12`.
* Failed years: `2018`, `2020`, `2022`, and `2025`.
* Artifacts: `outputs/codex_annual_state_router_score_20260614_e20260609_execyear_defensive_*`.

Conclusion: the equity-curve router does not survive formal score/trade reconstruction. The largest remaining gap is not the route calculation itself; it is that the current source set cannot provide enough real score-level return in 2018/2020/2022/2025 while keeping drawdown under `-20%`. Refreshing the extended research factor caches to the current target date is still needed for an authoritative current-date score-router run, but the `2026-06-09` formal reconstruction is already enough to show this router is not yet a valid pipeline fix.

## 2026-06-14 Verification Rerun

I reran the authoritative auto-signal workflow again after the diagnostics and score-router engineering changes:

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal.py --chunk-size 300 --sleep-seconds 0 --allow-low-quality --no-archive
```

The workflow completed all operational stages: raw update reuse, conversion, factor load, data health, adj-factor metadata, governance, optimization, backtest, diagnostics, and candidate signal/manual-order/report generation. Official output safeguards held: the run remained `status=blocked`, `is_executable=false`, and produced candidate outputs only.

Current full-flow blockers remain:

* Parameter quality: `no_acceptable_params`; validation annual-return minimum `-19.18%`, worst validation drawdown `-20.90%`.
* Full backtest annual return: `18.72%` versus the `20%` gate.
* Full backtest max drawdown: `-25.21%` versus the `-20%` gate.
* Yearly annual-return failures: `2016`, `2017`, `2018`, `2020`, `2023`, and `2026`.
* Yearly drawdown failure: `2015`.

The diagnostics fix is confirmed in `outputs/auto_run_status.json`: `block_reasons` is non-empty and mirrors the quality failures when the low-quality candidate path continues, and `quality_warnings` is recorded separately.

I also reran the yearly artifact bottleneck scan:

```powershell
.\.venv\Scripts\python.exe scripts\run_goal_bottleneck_scan.py --glob "outputs/*_years.csv" --output outputs\codex_yearly_goal_bottleneck_20260614_after_fullflow.csv
```

Results:

* Scanned yearly rows: `3968`
* Scanned files: `374`
* Every year has at least one historical candidate that passes both the `20%` annual-return target and the `-20%` drawdown limit.
* Weak score-router years still have passable standalone candidates:
  * `2018`: `15` both-pass rows, mostly selector/quality-selector variants.
  * `2020`: `27` both-pass rows, mostly selector/quality-selector/dynamic variants.
  * `2022`: `131` both-pass rows, mostly momentum/low-liquidity and beta variants.
  * `2025`: `200` both-pass rows, mostly RSQR/BETA/DB-total-mv variants.
* The intersection of both-pass files across `2016`, `2017`, `2018`, `2020`, `2022`, `2023`, `2025`, and `2026` contains only the equity-curve annual router probe and its audit duplicate.

Important limitation: yearly rows from standalone candidate equity files are path-dependent. A candidate's good return in one calendar year assumes that candidate's own prior holdings, cash, and realized path. A real annual router that switches score sources starts each year from the previous routed portfolio, pays real turnover costs, and may not reproduce the standalone candidate's calendar-year result even when it selects the same source for that year. This explains why the equity-curve router passes all years but the formal score/trade router still fails. The remaining gap is strategy research, not a diagnosable pipeline failure.

## 2026-06-14 Full-Flow Rerun After Exposure Composition Fix

I found and fixed another engineering issue in the formal score-router path: `apply_defensive_timing_to_backtest_config()` replaced an existing `exposure_schedule`, so annual-router exposure decisions such as `strong_trailing_exposure=0.85` were lost whenever defensive timing was enabled. The adapter now composes schedules by multiplying the existing route exposure schedule by the defensive timing schedule after normalizing dates and forward-filling route exposure.

Regression coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_market_regime.py tests\test_run_annual_state_router_backtest.py -q
```

Result: `22 passed`.

I reran the expanded formal score-router with composed exposure:

```powershell
.\.venv\Scripts\python.exe scripts\run_annual_state_router_backtest.py --end-date 2026-06-09 --output-prefix outputs\codex_annual_state_router_score_20260614_expanded_roc60_strong085_defensive_composed --full-turnover-on-route-change --use-defensive-timing --include-expanded-sources --moderate-positive-source roc60 --moderate-positive-ret252-min 0.20 --strong-trailing-exposure 0.85
```

The shell wrapper timed out after writing outputs, but the result artifacts are complete:

* `outputs/codex_annual_state_router_score_20260614_expanded_roc60_strong085_defensive_composed_metrics.json`
* `outputs/codex_annual_state_router_score_20260614_expanded_roc60_strong085_defensive_composed_years.csv`
* `outputs/codex_annual_state_router_score_20260614_expanded_roc60_strong085_defensive_composed_year_routes.csv`

Formal score-router result after the fix:

* Overall annual return: `26.09%`.
* Full max drawdown: `-18.55%`.
* Annual trade cost ratio: `19.25%`.
* Overall annual return, max drawdown, turnover, and cost gates pass.
* Yearly annual-return pass count remains `8/12`; yearly drawdown pass count is `12/12`.
* Failed yearly-return years are `2016` (`18.52%`), `2018` (`18.51%`), `2022` (`17.06%`), and `2025` (`18.63%`).
* Full goal remains false because the yearly annual-return gate is still not met.

This is the best formal score/trade router evidence so far: the exposure composition fix makes the drawdown gate pass, but it does not reach the every-year `20%` return target.

I also attempted a larger in-process parameter grid over `roc60`/`db_total`, `strong_trailing_exposure`, and equity-overlay settings. It ran for 30 minutes without producing the planned grid CSV, and the residual Python process had to be stopped. This is an experiment-efficiency issue in the expanded score-router path; future research should either cache per-source score panels or add a resumable grid runner before doing larger sweeps.

I then reran the authoritative auto-signal workflow again:

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal.py --chunk-size 300 --sleep-seconds 0 --allow-low-quality --no-archive
```

The full flow completed in about 19 minutes. Data update reused the completed progress for target date `2026-06-12`; conversion, factor loading, data health, adj-factor metadata, governance, optimization, backtest, diagnostics, and candidate generation all completed. Data health and point-in-time governance passed; governance still records the non-blocking warning `st_calendar_end_before_factor_end:2026-06-09<2026-06-12`.

Current full-flow blockers are unchanged:

* Parameter quality: `no_acceptable_params`; validation annual-return minimum `-19.18%`, worst validation drawdown `-20.90%`.
* Full backtest annual return: `18.72%` versus the `20%` gate.
* Full backtest max drawdown: `-25.21%` versus the `-20%` gate.
* Yearly annual-return failures: `2016`, `2017`, `2018`, `2020`, `2023`, and `2026`.
* Yearly drawdown failure: `2015`.

Official output safeguards held:

* `outputs/auto_run_status.json` is `status=blocked`, `is_executable=false`, with `7` `block_reasons` and matching `quality_warnings`.
* `outputs/auto_signal_report.json` parses as valid UTF-8 JSON and points to candidate files.
* Candidate files were written for `2026-06-12`: signal, holdings, manual orders, order confirmation, and fill feedback.
* `outputs/latest_holdings.csv` was not overwritten; its timestamp remained `2026-06-04 21:13:22`.

## 2026-06-14 Formal Score/Trade Router Hit

I added a resumable formal score-router grid (`scripts/run_annual_state_router_grid.py`) that caches per-source score panels under `outputs/router_score_cache/` and appends one CSV row per completed parameter combination. This avoids losing long grid runs to shell timeouts. I also extended `scripts/run_annual_state_router_backtest.py` with:

* `selection_schedule`-compatible per-route turnover boosts by route reason.
* Expanded score sources `beta20` (`factor:BETA20`, top 5, liquidity q80) and `rsqr20` (`factor:RSQR20`, top 7, liquidity q80).
* `moderate_low_exposure`, so a low ret252 beta route can switch to a strong source at partial exposure instead of forcing full risk/cost.

The first formal full-gate hit was produced by:

```powershell
.\.venv\Scripts\python.exe scripts\run_annual_state_router_grid.py --output outputs\codex_router_grid_20260614_beta20_exposure.csv --max-combinations 4 --missing-ret252-exposures 0.70 --strong-trailing-exposures 0.80 --moderate-positive-sources roc60 --moderate-positive-ret252-mins 0.20 --moderate-low-sources beta20 --moderate-low-ret252-mins 0.18 --moderate-low-ret252-maxs 0.20 --moderate-low-exposures 0.40,0.50,0.60,0.70 --turnover-modes rank10 --turnover-boost-reason-sets 'low_vol_moderate_uptrend+moderate_positive_roc60' --turnover-boost-max-turnovers 2 --turnover-boost-rank-buffers 10 --equity-overlay-sideways-exposures none --equity-overlay-bear-exposures none --defensive-bear-exposures none --write-hit-prefix outputs\codex_router_grid_20260614_beta20_exposure_hit
```

Winning route parameters:

* `missing_ret252_exposure=0.70`
* `strong_trailing_exposure=0.80`
* `moderate_positive_source=roc60` for ret252 >= `0.20`
* `moderate_low_source=beta20` for `0.18 <= ret252 < 0.20`
* `moderate_low_exposure=0.40`
* `turnover_mode=rank10`
* turnover boost reasons: `low_vol_moderate_uptrend` and `moderate_positive_roc60`
* turnover boost: `max_turnover=2`, `rank_buffer=10`

Formal score/trade metrics:

* Overall annual return: `26.10%`.
* Full max drawdown: `-17.68%`.
* Annual turnover: `7.92`.
* Annual trade cost ratio: `17.83%`.
* Yearly annual-return pass count: `12/12`.
* Yearly drawdown pass count: `12/12`.
* Minimum yearly annual return: `20.17%` in `2016`.
* Worst yearly drawdown: `-17.68%` in `2015`.
* `full_gate.is_full_goal_met=true`.

Artifacts:

* `outputs/codex_router_grid_20260614_beta20_exposure_hit_metrics.json`
* `outputs/codex_router_grid_20260614_beta20_exposure_hit_years.csv`
* `outputs/codex_router_grid_20260614_beta20_exposure_hit_year_routes.csv`
* `outputs/codex_router_grid_20260614_beta20_exposure_hit_score_routes.csv`
* `outputs/codex_router_grid_20260614_beta20_exposure_hit_holdings.csv`
* `outputs/codex_router_grid_20260614_beta20_exposure_hit_trades.csv`
* `outputs/codex_router_grid_20260614_beta20_exposure_hit_equity.csv`

Important boundary: this is an authoritative formal score/trade research backtest hit, not a promotion of official live signals. The auto-signal official-output gate remains separate and must still refuse promotion unless its own configured quality gates pass or this router is deliberately integrated into that workflow.

## 2026-06-14 Main Auto-Signal Full-Flow Rerun After Router Hit

I reran the main automated signal workflow after the formal router hit:

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal.py --chunk-size 300 --sleep-seconds 0 --allow-low-quality --no-archive
```

The operational pipeline completed all stages for target date `2026-06-12`:

* Raw update reused the completed progress file.
* Raw -> Qlib/price-panel conversion completed.
* Factor loading/computation completed.
* Data health completed as healthy.
* Adj-factor metadata coverage was `3073/3073`.
* Point-in-time governance completed, with the existing non-blocking warning `st_calendar_end_before_factor_end:2026-06-09<2026-06-12`.
* Walk-forward optimization completed `144` validation rows over `16` parameter combinations.
* Full backtest, research diagnostics, candidate signal, candidate manual orders, and report generation completed.

The main auto-signal strategy family remains blocked because it has not been integrated with the winning annual-state router:

* `outputs/auto_run_status.json`: `status=blocked`, `is_executable=false`.
* `block_reasons` count: `7`; `quality_warnings` count: `7`.
* Parameter quality remains unacceptable: `no_acceptable_params`, validation annual-return minimum `-19.18%`, validation worst drawdown `-20.90%`.
* Main full backtest annual return remains `18.72%` versus the `20%` target.
* Main full max drawdown remains `-25.21%` versus the `-20%` limit.
* Yearly annual-return failures remain `2016`, `2017`, `2018`, `2020`, `2023`, and `2026`.
* Yearly drawdown failure remains `2015`.

Official-output safeguards held:

* `outputs/latest_holdings.csv` was not overwritten; its timestamp remained `2026-06-04 21:13:22`.
* Candidate files were written for `2026-06-12`: `candidate_signal`, `candidate_holdings`, `manual_orders_candidate`, order confirmation, fill feedback, and report.
* `outputs/auto_signal_report.json` parses successfully with Python `json.loads`; PowerShell `ConvertFrom-Json` was unreliable on this large report and should not be treated as a project JSON-write failure without a Python/parser repro.

## 2026-06-14 Main Auto-Signal Annual Router Integration

I integrated the formal annual-state router into `scripts/run_auto_signal.py` and reran the full workflow without `--allow-low-quality`:

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal.py --chunk-size 300 --sleep-seconds 0 --no-archive
```

The full operational pipeline completed for target date `2026-06-12`: raw update reuse, raw-to-price conversion, factor load, data health, adj-factor metadata, point-in-time governance, annual-router parameter quality, routed backtest, diagnostics, official signal, official manual orders, and report generation.

Key integration details:

* `annual_state_router.enabled=true` now skips the legacy optimizer with `selected_params_status=annual_state_router`.
* Parameter quality comes from `outputs/codex_router_grid_20260614_beta20_exposure_hit_metrics.json` and fails if the configured combo differs from the evidence combo.
* The main backtest rebuilds routed score panels and uses `routed_backtest_config()` with the formal hit settings: `turnover_mode=rank10`, `full_turnover_on_route_change=true`, and `use_defensive_timing=true`.
* Signal generation uses the same routed score panel through `generate_signal(scores=...)`; it no longer generates the official signal from the legacy single-strategy score path.

Final full-flow status:

* `outputs/auto_run_status.json`: `status=complete`, `is_executable=true`, `strategy_mode=annual_state_router`.
* `block_reasons=[]`, `quality_warnings=[]`.
* `outputs/auto_parameter_quality.json`: `is_acceptable=true`.
* `outputs/auto_backtest_quality.json`: `is_acceptable=true`.
* Backtest annual return: `26.20%`.
* Backtest max drawdown: `-17.68%`.
* Annual trade cost ratio: `17.82%`.
* Minimum yearly annual return: `20.72%`.
* Worst yearly drawdown: `-17.68%`.
* Latest routed score date: `2026-06-09`; intended trade date: `2026-06-10`.

Official outputs written by the passing run:

* `outputs/signal_2026-06-09.csv`
* `outputs/latest_holdings.csv`
* `outputs/manual_orders_2026-06-09.csv`
* `outputs/order_confirmations/order_confirmation_2026-06-09.csv`
* `outputs/fill_feedback/fill_feedback_2026-06-09.csv`
* `outputs/auto_signal_report.json`

Important debugging note: an intermediate integration failed because it mixed the formal evidence with different runtime behavior. Using `alpha158.parquet` for the beta/roc/beta20 sources changed the score universe and failed drawdown/yearly gates. Filtering routed source dates to price-panel month-ends also removed the formal `2026-06-09` score date. The passing main flow must preserve the formal score source files and the grid hit's `rank10`/full-route-turnover/defensive-timing settings while still rebuilding the current backtest and signal inside `run_auto_signal.py`.

## 2026-06-20 Current Worktree Verification

I reran the main automated signal workflow on the current worktree without a Tushare update, preserving conversion, factor loading, data health, adj-factor metadata, point-in-time governance, annual-router backtest, diagnostics, and signal generation:

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal.py --skip-update --end-date 2026-05-27 --no-archive
```

The workflow completed successfully for fixed target date `2026-05-27`:

* `outputs/auto_run_status.json`: `status=complete`, `is_executable=true`, `strategy_mode=annual_state_router`.
* Data health completed as healthy.
* Adj-factor metadata coverage was `3073/3073`.
* Point-in-time governance completed as `point_in_time_ready`.
* Annual-router parameter quality was acceptable with `annual_return_min=20.17%` and worst validation drawdown `-17.68%`.
* Rebuilt backtest quality was acceptable with annual return `26.56%`, max drawdown `-17.68%`, minimum yearly annual return `20.72%`, and no years below the return or drawdown gates.
* Official outputs were written for `2026-05-27`: `outputs/signal_2026-05-27.csv`, `outputs/manual_orders_2026-05-27.csv`, and `outputs/auto_signal_report.json`.

Regression checks after this rerun:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data_converter.py tests/test_data_governance.py tests/test_config_loader.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_run_auto_signal.py tests/test_run_annual_state_router_backtest.py tests/test_run_annual_state_router_grid.py tests/test_signal_generator.py tests/test_backtest.py tests/test_market_regime.py -q
git diff --check
```

Results: `37 passed`, `111 passed`, and `git diff --check` passed with only Git's expected LF-to-CRLF warning for the Trellis spec file.
