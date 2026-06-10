# Backtest Goal Findings - 2026-06-11

## Current Scope

The active goal is a single explainable strategy framework, not a hindsight multi-strategy switcher:

- Backtest from 2015 to the latest available date.
- Every calendar year annual return should be at least 20%.
- Every calendar year max drawdown should be no worse than -20%.
- Prefer fundamental quality, dividends, debt safety, and report explanation before complex models.

## Fundamental V1 Status

The fundamental data and report loop is in place:

- Latest screen output: `outputs/fundamental_screen_2026-06-05.csv`.
- Latest report: `outputs/fundamental_screen_report.md`.
- Tushare `fina_indicator` cache after small batches: 493 symbols.
- Tushare `dividend` cache after small batches: 485 symbols.
- Current small-sample screen result after percent normalization, annual-report preference, stricter dividend gating, and `--missing-only` expansions: 46 PASS, 133 WATCH, 314 REJECT, 5277 rows with insufficient fundamental data.
- Current fundamental coverage is 8.54%; dividend coverage is 7.56%.
- The screen now prefers the latest announced annual report for quality metrics, falling back to interim reports only when no recent annual report is available.
- `dividend_pass` now requires both current dividend yield and a multi-year dividend record. Stocks with a long record but weak current yield are kept as WATCH rather than core PASS.

This is directionally aligned with the Word notes: start with business quality, cash/dividend discipline, balance-sheet risk, and plain-language explanation before adding model complexity.

## Current Baseline Audit

The latest baseline audit is written by:

```powershell
.\.venv\Scripts\python.exe scripts\run_goal_audit.py --output-prefix outputs\goal_audit_current_20260611
```

Result:

- Full-period annual return: 18.61%.
- Full-period max drawdown: -25.21%.
- Yearly return pass count: 6/12.
- Yearly drawdown pass count: 11/12.
- Return-failing years: 2016, 2017, 2018, 2020, 2023, 2026.
- Drawdown-failing year: 2015.

## Fundamental Quality Backtest

A first single-framework fundamental quality candidate was added and run:

```powershell
.\.venv\Scripts\python.exe scripts\run_fundamental_quality_backtest.py --start-date 2015-01-01 --end-date 2026-06-09 --top-n 10 --output-prefix outputs\fundamental_quality_full_20260611
```

This candidate ranks stocks monthly using only fundamental data available as of each signal date. It is not a multi-strategy switcher.

Result:

- Full-period annual return: 3.45%.
- Full-period max drawdown: -23.60%.
- Yearly return pass count: 1/12.
- Yearly drawdown pass count: 12/12.
- Covered symbols in the current cache: 200.

After expanding the cache to 300 symbols and tightening `dividend_pass` so both current yield and dividend record are required, the same quality-only framework was rerun:

```powershell
.\.venv\Scripts\python.exe scripts\run_fundamental_quality_backtest.py --start-date 2015-01-01 --top-n 10 --output-prefix outputs\fundamental_quality_300_20260611
```

Result:

- Full-period annual return: 4.78%.
- Full-period max drawdown: -29.25%.
- Yearly return pass count: 1/12.
- Yearly drawdown pass count: 12/12.
- Failed return years: 2016 through 2026.
- Worst return year: 2022.

Interpretation: the fundamental quality framework is useful as a quality/risk layer, but current coverage and scoring are not a standalone return engine for the strict yearly 20% target. The stricter dividend gate improves explainability and candidate quality, but does not create enough return.

After expanding coverage to roughly 500 symbols, full-period quality-only rerun returned a non-zero exit code with no traceback in the shell capture. Shorter probes completed:

- 2015-2020 quality-only: 9.11% annual return, -20.13% max drawdown, yearly return 1/6, yearly drawdown 6/6.
- 2021-2026 quality-only: 1.42% annual return, -23.71% max drawdown, yearly return 1/6, yearly drawdown 6/6.
- 2024-2026 quality-only smoke: 1.95% annual return, -12.32% max drawdown, yearly return 0/3.

Interpretation: the 500-symbol quality-only probes reinforce the same conclusion. The framework is explainable and often drawdown-aware, but it still lacks enough return, especially in 2018, 2022, and 2026.

## Fundamental + Momentum Filter Backtest

A second single-framework candidate was added and run:

```powershell
.\.venv\Scripts\python.exe scripts\run_fundamental_quality_backtest.py --start-date 2015-01-01 --end-date 2026-06-09 --top-n 10 --combine-mode filter_price --price-factor-group momentum --output-prefix outputs\fundamental_momentum_filter_full_20260611
```

This candidate uses the fundamental quality pool as a monthly eligibility filter, then ranks eligible names by the existing momentum score. It is still one rule family and does not switch by year.

Result:

- Full-period annual return: 6.02%.
- Full-period max drawdown: -46.19%.
- Yearly return pass count: 1/12.
- Yearly drawdown pass count: 10/12.
- Return-failing years: 2016 through 2026.
- Drawdown-failing years: 2015 and 2016.

Interpretation: a hard fundamental quality filter currently removes too much return source and does not solve the weak-year target. The next iteration, if pursued, should test a softer quality blend or broader fundamental coverage before treating fundamentals as a hard veto.

A short 2024-2026 soft-blend smoke test with `--combine-mode blend --quality-weight 0.35` produced a similar result: 6.84% annual return, -12.40% max drawdown, and 0/3 yearly return pass count.

After expanding coverage to roughly 500 symbols, the same fixed blend framework was rerun:

```powershell
.\.venv\Scripts\python.exe scripts\run_fundamental_quality_backtest.py --start-date 2015-01-01 --top-n 10 --combine-mode blend --price-factor-group momentum --quality-weight 0.35 --output-prefix outputs\fundamental_momentum_blend_500_20260611
```

Result:

- Full-period annual return: 7.57%.
- Full-period max drawdown: -27.82%.
- Yearly return pass count: 2/12.
- Yearly drawdown pass count: 12/12.
- Failed return years: 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, and 2026.

Interpretation: the soft blend improves return versus pure quality but still does not solve the weak-year target. It also worsens 2017 versus the 300-symbol pure quality run, so simple quality+momentum blending is not enough.

## Industry Momentum Backtest

A single-framework industry-relative momentum family was also tested with a custom factor cache:

- Factor cache: `data/factors/codex_goal_industry_momentum_factors_20260611.parquet`.
- Combined summary: `outputs/codex_goal_industry_momentum_summary_20260611_combined.csv`.
- Best candidate by yearly return pass count: `ind_roc120_q65_top15_take035`.
- Best candidate annual return: 15.83%.
- Best candidate max drawdown: -32.18%.
- Best candidate yearly return pass count: 5/12.
- Best candidate yearly drawdown pass count: 9/12.

The rerun of initially errored candidates with the correct custom factor file also failed. The best rerun row, `ind_rev20_q65_top10_take035`, reached 12.82% annual return with 5/12 yearly return pass count and 9/12 yearly drawdown pass count.

Interpretation: industry-relative momentum is a valid single rule family, but it still does not solve the weak-year return constraint or drawdown constraint.

## Fast Screen Recheck

The weak-year fast screen had suggested that `inverse_factor:KLEN` with high-liquidity q0.65 and top 5 could lift 2017. Exact formal reruns did not confirm it:

```powershell
.\.venv\Scripts\python.exe scripts\run_goal_formal_candidates.py --candidates-file config\codex_goal_bottleneck_style_candidates_20260611.json --output outputs\codex_goal_bottleneck_style_summary_20260611.csv --skip-diagnostics
```

Best rows from this recheck:

- `klen_inv_q65_top5_turn1_take035`: 6.98% annual return, -48.52% max drawdown, yearly return 4/12, yearly drawdown 9/12.
- `klen_inv_q65_top7_turn1_take035`: 7.91% annual return, -44.50% max drawdown, yearly return 4/12, yearly drawdown 9/12.
- `max60_inv_q65_top5_turn1_take035`: 5.40% annual return, -44.65% max drawdown, yearly return 5/12, yearly drawdown 11/12.
- `klen_inv_q65_top5_turn1_notake_no_selrisk`: 7.49% annual return, -54.33% max drawdown, yearly return 4/12, yearly drawdown 9/12.

The exact KLEN q0.65/top5 formal row returned -0.98% in 2017 and -18.37% in 2018, far below the fast-screen indication. Trade logs show many blocked trades in 2017/2018, mostly `not_buyable`, while the fast screen intentionally omits formal tradability, capacity, risk-exit, and blocked-order mechanics.

Project logic was tightened so `scripts/run_goal_fast_factor_screen.py` now:

- Adds `formal_confirmation_required=True`.
- Adds an approximation note explaining that formal tradability/capacity/risk exits are ignored.
- Applies the configured `max_annual_turnover` gate in fast quality fields.

Interpretation: fast screens are still useful for rough triage, but they must not be treated as evidence that a candidate can satisfy the goal.

## Selector Weight Formal Recheck

Directly rerunning the DB/price dynamic selector through `run_goal_formal_candidates.py` with the extended factor file caused a hard process exit while recomputing rolling IC on the large factor cache. A lightweight bridge script was added:

```powershell
.\.venv\Scripts\python.exe scripts\run_selector_weight_backtest.py --selector-file <selector.csv> --factor-file data\factors\codex_goal_extended_factors_20260610.parquet --output-prefix <prefix>
```

This script reuses precomputed monthly selector weights, rebuilds signed factor scores, applies the configured liquidity filter, and then uses the formal backtest/audit path.

Key rechecks:

- `lb63_top3_pos_equal`, top 10 full period: 19.65% annual return, -32.74% max drawdown, yearly return 5/12, yearly drawdown 10/12.
- `lb63_top5_pos_proportional`, top 5 full period: 13.36% annual return, -42.03% max drawdown, yearly return 5/12, yearly drawdown 8/12.
- `lb63_top5_pos_proportional`, top 5 on 2018 only: 32.25% annual return, -19.34% max drawdown, 2018 passes both gates.

Fundamental quality top 5 on 2017-2018 was also probed:

- 2017: 28.08% annual return, -7.02% max drawdown.
- 2018: -6.10% annual return, -9.18% max drawdown.

A static quality/selector score blend on 2017-2018 did not solve both years. Low quality weights preserved the selector's 2018 success but left 2017 negative; high quality weights softened risk but lost the 2018 return.

Interpretation: 2017 and 2018 now each have a passing single-year rule family under formal-style checks, but a static single-score blend still fails to unify them. The next design question is whether a predeclared market-state gate can choose between "quality carry" and "tactical DB/price selector" without becoming hindsight multi-strategy switching.

## Best Existing Candidate Evidence

A formal bottleneck scan is now available:

```powershell
.\.venv\Scripts\python.exe scripts\run_goal_bottleneck_scan.py --output outputs\goal_bottleneck_scan_20260611.csv
```

It scanned 3154 yearly rows from 292 `*_years.csv` files. The scan now contains passing single-year rows for 2017 and 2018, but no full-period single framework satisfies every yearly gate.

Best annual return by year among existing yearly artifacts:

| Year | Best Annual Return | Max Drawdown On That Row |
| --- | ---: | ---: |
| 2015 | 121.48% | -30.82% |
| 2016 | 36.43% | -10.49% |
| 2017 | 28.08% | -7.02% |
| 2018 | 32.25% | -19.34% |
| 2019 | 52.56% | -8.54% |
| 2020 | 37.43% | -10.50% |
| 2021 | 122.60% | -9.35% |
| 2022 | 62.53% | -16.90% |
| 2023 | 48.51% | -8.30% |
| 2024 | 56.59% | -17.07% |
| 2025 | 79.64% | -10.62% |
| 2026 | 134.18% | -12.92% |

Counts of candidates passing both gates by year:

| Year | Passing Candidate Count |
| --- | ---: |
| 2015 | 46 |
| 2016 | 14 |
| 2017 | 2 |
| 2018 | 6 |
| 2019 | 172 |
| 2020 | 19 |
| 2021 | 206 |
| 2022 | 102 |
| 2023 | 69 |
| 2024 | 45 |
| 2025 | 167 |
| 2026 | 8 |

## Interpretation

The current blocker is not only drawdown control. Tighter risk controls can improve 2015 drawdown, but the hard constraint is finding one predeclared framework that handles both 2017's quality-led environment and 2018's tactical selector environment. Static blending did not do this.

The latest factor-momentum selector direction improved the full-period annual return to around 20%, but still failed the yearly gate:

- Best checked selector audit: return pass 7/12, drawdown pass 9/12.
- Remaining failed years: 2015, 2017, 2018, 2020, 2024, 2026.

## Recommended Next Step

Do not commit or promote a strategy as validated yet. Keep `require_is_acceptable: true` so automation blocks unacceptable candidates.

Next work should stay conservative:

- Expand fundamental coverage beyond the current roughly 500-symbol cache only if the data layer needs broader reports; strategy evidence says coverage alone is not enough.
- Use the fundamental screen as a reporting and veto layer first, not as a complex predictive model.
- Investigate whether a predeclared market-state gate can combine quality carry and tactical selector behavior without hindsight switching.
- Use `scripts/run_goal_audit.py` after every full-span rerun.

If the strict every-calendar-year 20/20 target remains mandatory, the evidence so far suggests the project needs a new return source, not just parameter tuning inside the current price/volume candidate space.
