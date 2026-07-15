# Manual Order Stock Detail and Current Price

## Goal

Make manual trade review more actionable by letting the user click a stock name or code in the manual order list and inspect a stock detail view that shows the latest available price.

## What I already know

* The Web workspace already has a manual trade/order experience.
* The user cannot currently see what price a stock is trading at from that workflow.
* Both the stock name and stock code should act as entry points to the detail view.
* The manual order table is rendered by `OrdersTable` in `web/src/App.tsx`; it currently formats every cell as plain text.
* The current FastAPI application has no stock-detail or quote endpoint.
* The configured Tushare proxy successfully supports `rt_k` for an individual A-share code and returns `pre_close`, `open`, `high`, `low`, `close`, `vol`, and `amount`.
* Intraday `rt_min` is not available with the current proxy permissions, so the MVP cannot promise minute-by-minute history.
* Local unadjusted close data is available through 2026-07-10 and can provide a safe fallback when the live quote request fails.

## Assumptions

* The first version stays inside the existing Web workspace rather than linking to an external finance website.
* “Current price” clearly discloses its retrieval/market timestamp and whether it is a live quote or the latest locally available close.

## Open Questions

* None.

## Requirements (evolving)

* Make stock names and stock codes in manual orders clickable.
* Open a compact in-app modal over the manual-order review page for the selected security; do not navigate away from the dashboard.
* Request the configured Tushare `rt_k` quote when the detail view opens and whenever the user presses refresh.
* Show current price, change percentage, previous close, open, high, low, volume, and quote retrieval time.
* If the live quote fails, fall back automatically to the latest local unadjusted close and label it with its market date and an explicit non-live status.
* Provide an explicit refresh action without reloading the whole dashboard.
* Provide a visible close button and support Escape-key and backdrop-click dismissal.
* Keep keyboard focus and screen-reader semantics appropriate for a modal dialog.
* Preserve the existing security boundary: the browser supplies only a normalized stock code, while the backend owns the allowed quote call and local file paths.

## Acceptance Criteria (evolving)

* [x] Clicking a stock name opens the corresponding compact modal without leaving the dashboard.
* [x] Clicking a stock code opens the same compact modal.
* [x] The detail experience displays a price, its effective market time, and an explicit freshness/status label.
* [x] A successful live quote displays current price, change percentage, previous close, open, high, low, volume, and retrieval time.
* [x] The refresh action requests a new quote and updates the detail view without reloading the whole page.
* [x] A remote quote failure returns the latest local unadjusted close, market date, and a visible non-live/fallback label.
* [x] Missing or unavailable quote data is shown as a controlled empty/error state.
* [x] The modal closes through its close button, Escape key, and backdrop click.
* [x] Desktop and mobile modal layouts remain usable without page-level overflow.

## Definition of Done (team quality bar)

* Tests added or updated at the API and UI layers where appropriate.
* Frontend type-check/build and focused backend tests pass.
* Browser behavior is verified for desktop and mobile layouts.
* Documentation or operational notes are updated if the quote source changes deployment requirements.

## Out of Scope (explicit)

* Placing broker orders directly from the stock detail view.
* Public Internet exposure or authentication changes.
* Advanced charting and full fundamental research until the MVP scope is confirmed.
* Minute-level intraday charts, because the configured Tushare proxy does not grant `rt_min` access.

## Technical Notes

* Task created from the user request on 2026-07-11.
* Likely implementation surfaces: `src/dashboard_api.py`, a focused quote/detail backend module, `web/src/api.ts`, `web/src/types.ts`, `web/src/App.tsx`, `web/src/styles.css`, and API/UI tests.
* Feasible approach A (recommended): request `rt_k` on detail open/refresh, label the retrieval time and quote source, and fall back to the latest local unadjusted close with its market date when the remote quote is unavailable.
* Feasible approach B: use only the latest local close; simpler and fully offline, but it does not satisfy the ordinary expectation of “现在价格” during trading hours.

## Decision (ADR-lite)

**Context**: Manual order review needs a price that is useful during trading hours, while the Web workspace must remain honest and usable when the remote quote service is unavailable.

**Decision**: Use a backend-owned Tushare `rt_k` request for the live quote and automatically fall back to the latest local unadjusted close. Present the selected stock in a compact modal over the manual-order dashboard, with refresh/dismiss actions and explicit source/freshness labeling.

**Consequences**: The normal path depends on the configured Tushare proxy, but the page remains usable offline. Minute-level charts remain out of scope because `rt_min` permission is unavailable.

## Implementation Plan

* Add a narrowly scoped, read-only stock quote/detail backend service and API route.
* Add frontend quote types/API access and an accessible in-app stock detail modal.
* Make both stock name and code cells in manual orders open that modal.
* Add focused backend and browser tests, build the production UI, and verify live and fallback behavior.
