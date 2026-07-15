# Fix Dashboard Quality Review and Quote Date Semantics

## Goal

Diagnose and fix the Web dashboard's failed-review explanation, the unusable quality-report action, and the stock modal's quote-date semantics so users can distinguish a real strategy-quality blocker from UI defects and can inspect the authoritative evidence needed to act.

## What I Already Know

* The current automatic signal is blocked by `backtest_yearly_annual_return_below_threshold:2026=-0.0719<0.2000`.
* This blocker is a real quality-gate outcome, not a frontend exception.
* The blocker repair center shows a report-related action that the user cannot click.
* The stock detail modal currently shows `行情日期`, but live `rt_k` responses may not provide an explicit market date.
* The existing dashboard and stock quote APIs must remain bounded and read-only for these views.

## Assumptions

* The 20% quality threshold must not be bypassed or silently lowered merely to make review pass.
* The dashboard should link to an authoritative existing quality artifact rather than fabricate a new verdict in the browser.
* A live quote's retrieval timestamp and its market-session date are different concepts and must not be conflated.

## Open Questions

* None. Runtime and source inspection resolved the ambiguous points.

## Requirements

* Explain the failed review using the exact authoritative quality-gate evidence and actionable context.
* Make the blocker center's report action actually usable when the referenced report exists, and present a controlled unavailable state when it does not.
* Preserve the real 2026 performance blocker instead of forcing an official signal.
* Correct the stock modal's quote-date label/value so live and fallback data are semantically honest.
* Add regression tests for all corrected behaviors.

## Acceptance Criteria

* [x] The dashboard distinguishes a valid quality-gate failure from a Web application failure.
* [x] The 2026 annual-return blocker is translated into a clear Chinese message with actual and required values.
* [x] The quality report button/link opens or downloads the authoritative report when available.
* [x] A missing report yields an explicit disabled/unavailable state rather than a broken click target.
* [x] No change bypasses, lowers, or fabricates the strategy quality gate.
* [x] Live quote date/time fields are labeled from real evidence; no placeholder claims a market date that the API did not provide.
* [x] Fallback quotes continue to show their actual local daily market date and non-live status.
* [x] Focused frontend/backend tests, production build, full regression suite, and real-page verification pass.

## Definition of Done

* Root causes and the selected repair approach are documented.
* Repair approach is reviewed for data truthfulness, security boundaries, stale-artifact behavior, and regression risk before implementation.
* Code and tests are updated.
* Multiple verification rounds pass with no unresolved issue in the stated scope.

## Out of Scope

* Lowering the configured 20% annual-return requirement.
* Forcing candidate output to become official.
* Inventing or backfilling market dates not returned by the quote source.
* Broad strategy research unrelated to diagnosing the current 2026 quality evidence.

## Technical Notes

* Likely surfaces: `src/dashboard.py`, dashboard artifact mapping/API routes, `web/src/App.tsx`, blocker/action components, `src/dashboard_stock.py`, `web/src/StockDetailWorkspace.tsx`, types, and tests.
* The current running dashboard at `http://127.0.0.1:8000` and files under `outputs/` are authoritative runtime evidence.

## Research Reference

* [`research/root-cause-and-repair.md`](research/root-cause-and-repair.md) records the runtime evidence, rejected alternatives, selected repair, and risk review.

## Decision (ADR-lite)

**Context**: The dashboard has a genuine current-year quality failure, but its report action is fake and its live quote date label overstates what `rt_k` provides.

**Decision**: Preserve the 20% quality gate and candidate-only outcome. Improve the blocker evidence and link it to the bounded authoritative quality artifact. Treat absent reports explicitly. For live quotes without a source date, display that the interface did not provide a date instead of claiming a current trading session.

**Consequences**: The Web page becomes actionable and semantically correct, but the official signal remains blocked until strategy evidence actually satisfies the 2026 gate. This is intentional risk behavior, not an unresolved UI bug.

## Implementation Plan

* Enrich backend blocker actions with quality evidence and report artifact metadata.
* Render a real report link or explicit unavailable state in the blocker center.
* Translate yearly backtest blockers into clear percentage-based Chinese text.
* Correct live quote date display and disclosure while preserving fallback dates.
* Add backend and Playwright regression coverage, then run focused, full, and real-page verification.
