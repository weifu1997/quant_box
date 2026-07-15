# Fix Dashboard Factor Freshness Precheck

## Goal

Make the Web precheck use the same factor-freshness contract as the authoritative data-health gate, so a small set of suspended or confirmed-no-new-data stocks does not create a misleading `artifact_before_target` warning when current-date factor coverage already passes the configured threshold.

## Requirements

* In `_precheck_factor_freshness`, treat `factor_*` issues from `data_health_report.json` as authoritative failures.
* Set the authoritative `quality.min_factor_coverage` default and project setting to 0.99; raw and price coverage thresholds remain unchanged.
* Use the current configured factor threshold in Web precheck even when the last data-health report was generated with an older threshold.
* When there are no factor issues and `factor_latest_target_coverage >= min_factor_coverage`, return a passing factor-freshness item even if the minimum per-symbol `factor_latest_date` is earlier than the target date.
* The passing summary must report the observed percentage and configured threshold.
* When counts are available, report the number of target symbols covered through the target date and the remainder that may be suspended or have no new market row.
* Continue to fail or warn when coverage is below the configured threshold, factor issues exist, or no factor evidence covers the target.
* Do not mutate `data_update_progress.json`, market data, factor caches, target dates, or quality thresholds.
* Do not change the separate backtest-quality gate; the current 2026 yearly-return failure remains a blocker.

## Acceptance Criteria

* [x] A health payload with 2705/2708 current symbols, coverage 0.998892, threshold 0.99, no factor issues, and minimum date 2026-06-29 produces `status=pass`.
* [x] The pass summary includes 99.89%, the 99.00% threshold, 2705/2708, and three symbols without a target-date row.
* [x] A factor coverage issue remains `status=fail` with its repair action.
* [x] Coverage below threshold cannot be converted into a pass solely because some factor rows exist.
* [x] Existing metadata-only fallback behavior remains available when `data_health_report.json` is absent.
* [x] The live `/api/dashboard/precheck` no longer returns `artifact_before_target:2026-06-29<2026-07-10` for the current healthy data.
* [x] Focused backend tests and the existing dashboard browser/build checks pass.

## Definition of Done

* Focused regression tests cover pass, fail, and missing-health behavior.
* No generated/private market data is edited or committed.
* `git diff --check` passes.
* The running dashboard service is restarted and its real precheck API is verified.

## Out of Scope

* Strategy optimization or changing the 2026 annual-return gate.
* Fabricating rows for suspended stocks.
* Lowering data-health thresholds.
* Changing the automatic data-fetch completion contract.

## Technical Notes

* Current evidence: target symbols 2708, factor-current symbols 2705, coverage 0.9988921713441654, configured minimum 0.99, data-health issues empty.
* The misleading date is the minimum per-symbol latest date, driven by `601369.SH` at 2026-06-29; the factor cache itself contains 2026-07-10 rows.
* Current code duplicates freshness policy in `src/dashboard.py` after `src/data_health.py` has already evaluated coverage and emitted factor-specific issues.
