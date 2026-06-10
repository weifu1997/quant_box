# Fundamental Screen Plan — Complete Roadmap

This roadmap keeps the project close to the Word notes' core idea: treat stocks as long-term business ownership first, and only then think about price signals.

---

## Bug Fixes — ✅ Completed (2026-06-11)

Three confirmed bugs were fixed before feature enhancements, per the code-review-first policy:

- **B2**: `_ratio_series` p75>1.5 statistical heuristic replaced with explicit `TUSHARE_PERCENT_FIELDS` lookup. Tushare fields known to store percentages (15.0 for 15%) are now divided by 100 deterministically; unknown fields pass through unchanged. This eliminates the risk of edge-case datasets producing wildly wrong normalization.
- **B3**: `valuation_pass` now treats NaN PE/PB as "no opinion" → pass, not as False. Pandas `NaN <= 30.0` evaluates to `False` (not NaN), so `.fillna()` was ineffective. Fixed by using `(pe_ttm <= threshold) | pe_ttm.isna()` logic.
- **B4**: `dividend_yield_ttm` now falls back to `ttm_cash_div_tax / close` when Tushare's `dv_ttm` is NaN. Previously, missing `dv_ttm` caused the entire dividend screen to fail even when computed dividend data was available.

3 new test cases added: `test_ratio_series_uses_explicit_field_config_not_heuristic`, `test_valuation_pass_treats_nan_pe_pb_as_true`, `test_dividend_yield_fallback_to_ttm_cash_div_tax_over_close`. All 446 existing tests pass with zero regressions.

---

## P1: Philosophy To Rules — ✅ Done

- Extract the durable principles: business ownership, dividends as payback, survivability under bad scenarios, low leverage, cash generation, and patience.
- Translate those principles into auditable checks instead of a complex predictive model.
- Keep this layer independent from official trading signals until it has enough data coverage and manual review.

**Deliverable**: docs/investment_principles.md (pending).

---

## P2: Fundamental Data Layer — ✅ V1 Done, Enhancement Pending

### Completed (V1)

- Cache `fina_indicator` for quality, cash-flow, ROE, and leverage metrics.
- Cache `dividend` for dividend history and recent cash dividend evidence.
- Reuse existing `daily_basic` for PE, PB, market value, and dividend yield.
- Support small smoke-test updates and resumable incremental coverage expansion (`--missing-only`).

### Enhancement Needed

- **Add three-statement data**: `income` (income_statement), `cashflow` (cashflow_statement), `balancesheet` (balancesheet).
  - Why: fina_indicator gives summary ratios, but the user wants capital expenditure intensity, equity dilution, and detailed cash-flow breakdown — these require raw statement fields.
  - New fields needed: `capital_expense`, `total_share`, `float_share`, `n_cashflow_act`, `n_cashflow_inv`, `n_cashflow_fnc`, `total_assets`, `total_liab`, `total_hldr_eqy_exc_min_int`.
  - Cache paths: `data/fundamentals/income.parquet`, `data/fundamentals/cashflow.parquet`, `data/fundamentals/balancesheet.parquet`.
  - Update: `scripts/run_update_fundamentals.py` to support `--skip-income`, `--skip-cashflow`, `--skip-balancesheet`.
  - Config: add `fundamentals.income_file`, `fundamentals.cashflow_file`, `fundamentals.balancesheet_file`.

Status: V1 done; enhancement in progress.

---

## P3: Quality, Dividend, Debt, And Valuation Screen — ✅ V1 Done, Enhancement Pending

### Completed (V1)

- Quality: require ROE and cash conversion/free cash flow evidence.
- Debt: reject high debt-to-assets companies by default.
- Dividend: require dividend yield or a multi-year positive dividend record.
- Valuation: keep PE/PB ceilings as a payback-period sanity check.
- Conservative missing-data behavior: missing fundamentals cannot pass.
- review_status: PASS / WATCH / REJECT / INSUFFICIENT_DATA.
- Company name and industry in report.
- Coverage summary, near misses, reason guide.

### Enhancement Needed — "散户乙风格" Core Factors

- **资本开支强度** (capex_intensity): `capital_expense / ocf_to_or * total_mv` or `n_cashflow_inv_act / total_assets`. Low capex = less reinvestment risk.
- **ROE 稳定性** (roe_stability): std(roe over lookback years) < threshold. Requires multi-year fina_indicator rows.
- **股本稀释** (equity_dilution): `(total_share_current - total_share_3y_ago) / total_share_3y_ago`. Dilution = shareholder value destruction.
- **分红稳定性评分** (dividend_stability_score): count of consecutive positive dividend years / lookback years. Not just "at least 2 years" but a ratio.
- **经营现金流/净利润** (ocf_to_netprofit): `n_cashflow_act / net_profit`. Already partially covered by `ocf_to_opincome`, but direct cash/profit ratio is more precise.
- **自由现金流收益率** (fcf_yield): already computed from fcff/market_cap or fcff_ps/close.
- **股息率** (dividend_yield_ttm): already computed from dv_ttm or ttm_cash_div_tax/close.
- **分红回本年限** (dividend_payback_years): already computed as 1/dividend_yield_ttm.
- **资产负债率** (debt_to_assets): already from fina_indicator.

Factor computation will live in `_add_screen_metrics()`, thresholds in `fundamental_screen` config block.

Status: V1 done; enhancement in progress.

---

## P4: Worst-Case Stress Test — 🆕 New

The core insight from 散户乙: "先看最坏情况". Not just whether the company is good today, but whether it survives when things go wrong.

### Scenarios

- **利润腰斩** (profit_half): if net_profit halves, does the company still cover interest and dividends?
- **估值压缩** (valuation_compress): if PE drops 50%, is the dividend yield still > 1.5%?
- **分红下降** (dividend_down): if dividend drops 50%, does dividend_payback_years stay below 30?
- **负债上升** (debt_up): if debt_to_assets increases 20%, is it still < 60%?
- **行业周期下行** (recession): if revenue drops 30% and margins compress, does OCF remain positive?
- **现金流转负** (cashflow_negative): if operating cash flow goes to zero, can the company survive 2 years on cash reserves?

### Implementation

- Compute a `stress_pass` flag per scenario.
- Aggregate into `worst_case_score` (0-6).
- Add `worst_case_pass` boolean (pass all scenarios).
- Report: add a "Worst-Case Stress Test" section.
- Config: add `fundamental_screen.stress_test.enabled`, scenario thresholds.

Status: pending.

---

## P5: Long-Term Stock Pool Rating — ✅ V1 Done (4-tier), Enhancement Pending (5-tier)

### Completed (V1)

- INSUFFICIENT_DATA: no fundamental data available.
- PASS: passes all quality, debt, dividend, valuation checks.
- WATCH: fails some checks but total_score >= watch_min_score (4).
- REJECT: fails enough checks.

### Enhancement Needed — 散户乙 五档评级

Replace the 4-tier system with a 5-tier system that separates "好公司" from "好价格":

- **禁止 (BAN)**: fundamental data missing, or worst-case stress test fails completely, or company has destructive traits (negative ROE, equity dilution > 20%, debt > 70%).
- **观察 (WATCH)**: quality is borderline, or dividend is unstable, or valuation is borderline. Needs more evidence before committing.
- **持有 (HOLD)**: good company (quality_pass, debt_pass, dividend_pass) but price is not cheap enough for new money. Hold existing positions, don't buy more.
- **增持 (ADD)**: good company and valuation is reasonable (PE < 20, dividend_yield > 3%, payback < 20y).
- **买入 (BUY)**: good company and price is genuinely cheap (PE < 15, dividend_yield > 4%, payback < 15y, worst-case stress test passes).

Rating computation will replace `_review_status()` with a more nuanced `_pool_rating()` function.
Config: add `fundamental_screen.rating` block with thresholds for each tier.

Status: V1 done; enhancement pending.

---

## P6: Investment Principles Document — 🆕 New

Write a document that formalizes what this project believes, in language that matches 散户乙's philosophy and is traceable to code:

- We are not an automatic trading system. We are a "长期股权筛选 + 交易辅助" system.
- A stock is a piece of a business. The first question is "will this business survive and return cash over 10 years?", not "will the price go up tomorrow?"
- We prioritize: survivability → cash generation → shareholder return → price.
- Missing data is treated conservatively: no fundamental data = no opinion.
- The fundamental layer is advisory, not mandatory, until data coverage and manual review confirm its reliability.

Deliverable: `docs/investment_principles.md`.

Status: pending.

---

## P7: Non-Blocking Integration Into Daily Report — 🆕 New

### Current State

- `scripts/run_auto_signal.py` builds `auto_signal_report.json`.
- `src/reporting.py` renders `daily_signal_report.md`.
- Fundamental screen is standalone: `scripts/run_fundamental_screen.py` → `outputs/fundamental_screen_report.md`.

### Enhancement

- Add a `fundamental_screen` section to `auto_signal_report.json` using `summarize_fundamental_screen_result()`.
- Add a "Fundamental Screen" section in `daily_signal_report.md` showing coverage, top pass/watch, and stress test summary.
- The fundamental summary is **read-only context** — it does not block or override official signals.
- Optionally: annotate each signal candidate with its `review_status` or `pool_rating`.

Implementation:
- In `run_auto_signal.py`, after generating the signal, call `build_fundamental_screen()` and `summarize_fundamental_screen_result()`, add result to the report dict.
- In `reporting.py`, add `_fundamental_screen_lines()` to render the summary section.

Status: first version done. `scripts/run_auto_signal.py` now attaches a non-blocking `fundamental_screen` summary to `auto_signal_report.json`, and `src/reporting.py` renders it in `daily_signal_report.md`.

---

## P8: Backtest Enhancement — 🆕 New (Longer-Term)

### Current State

- Backtest uses price-only equity curve, no dividend reinvestment.
- Holding is determined by rank-based rebalance, not fundamental holding logic.
- No concept of "holding period" or "dividend payback years realized".

### Enhancement

- **分红再投入**: Add dividend cash flows to the equity curve. When a held stock pays a dividend, add the dividend amount to cash and reinvest at next rebalance.
- **低换手持有**: Implement a "hold until fundamental deteriorates" mode. Instead of monthly rebalance, only exit when review_status drops from HOLD/ADD/BUY to WATCH/BAN.
- **现金储备**: Allow a cash allocation (e.g., 20%) when no stocks meet BUY/ADD criteria.
- **基本面恶化退出**: Track fundamental status over time. If a stock was BUY and becomes WATCH, reduce position. If it becomes BAN, exit.
- **持有年限统计**: Track how long each position was held, average holding period, dividend received per position.
- **分红回本实现**: Track cumulative dividends received vs. initial cost for each position.

This is a larger refactor and should be done after P2-P7 are stable.

Status: pending (longer-term).

---

## P9: Coverage Expansion — Pending Data Runs

- Run `scripts/run_update_fundamentals.py --missing-only --max-symbols N` repeatedly.
- Prefer targeted baskets: high dividend, coal, power, consumer staples, banks.
- After broad coverage, inspect industry concentration in the pass/watch pool.

Status: pending.

---

## P10: Validation And Maintenance — Pending

- Backtest whether using the screen as a filter improves drawdown, turnover, and holding quality.
- Add periodic freshness checks for fundamental caches.
- Split fetching and reporting modules if the file grows too large.
- Tune thresholds by industry only after enough evidence.

Status: pending.

---

## Execution Priority Order

| Priority | Phase | Scope | Effort |
|----------|-------|-------|--------|
| 1 | P6 | Investment principles doc | Small |
| 2 | P2-Enh | Three-statement data + config + CLI | Medium |
| 3 | P3-Enh | 散户乙 core factors (capex, ROE stability, dilution, dividend stability) | Medium |
| 4 | P4 | Worst-case stress test | Medium |
| 5 | P5-Enh | Five-tier pool rating | Medium |
| 6 | P7 | Daily report non-blocking integration | Small |
| 7 | P9 | Coverage expansion (data runs) | Manual |
| 8 | P8 | Backtest enhancement | Large |
| 9 | P10 | Validation and maintenance | Ongoing |
