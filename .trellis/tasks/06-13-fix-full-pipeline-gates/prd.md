# Fix Full Pipeline Gates

## Goal

Run the local A-share auto-signal pipeline end to end, identify the current hard failures and quality gates, and fix issues that are data or workflow defects without weakening official signal safeguards.

## Requirements

* Refresh missing governance data so point-in-time gates reflect current caches.
* Ensure adj-factor metadata checks only raw stock files that are expected to carry `adj_factor`.
* Re-run the full pipeline after fixes and inspect logs/status artifacts.
* Preserve candidate-vs-official output boundaries when strategy quality gates still fail.
* Do not expose local Tushare tokens or private account details in logs or summaries.

## Acceptance Criteria

* [x] Default `scripts/run_auto_signal.py` no longer fails because of stale `daily_basic` coverage.
* [x] `data_governance_report.json` has no hard issues caused by index raw files missing `adj_factor`.
* [x] Full pipeline is re-run after repairs with logs under `outputs/logs/`.
* [x] Remaining block reasons, if any, are backed by `outputs/auto_run_status.json` and report files.
* [x] Focused tests cover any code behavior change.

## Definition of Done

* Focused tests pass for changed modules.
* Pipeline run result is verified from status/report files.
* Generated/private data is not staged or committed.
* Any remaining quality failure is clearly separated from fixed data/workflow defects.

## Technical Approach

Use report-suggested repair commands for generated local data first, then make narrowly scoped code fixes only where the pipeline gate is using an incorrect contract. Keep strategy quality gates intact; do not force official outputs unless explicitly requested.

## Out of Scope

* Relaxing performance thresholds just to produce an official signal.
* Manually fabricating market data.
* Committing generated data under `data/` or `outputs/`.

## Technical Notes

* Initial run failed at `update_data` for `600717.SH:not_latest`.
* Repair command fixed `daily_basic` coverage to `2779/2779`.
* Index raw files `000300.SH` and `000905.SH` do not include `adj_factor`; governance previously counted them in adj-factor metadata.
* Relevant specs: `.trellis/spec/backend/index.md` and backend guideline files.
