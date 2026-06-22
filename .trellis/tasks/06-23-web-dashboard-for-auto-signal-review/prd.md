# Web Dashboard for Auto-Signal Review

## Goal

Build a local web dashboard for the daily auto-signal review workflow so the user can quickly see whether the latest signal is ready for manual trading review, which gates passed or failed, and which candidate/official artifacts were generated. The first version should be read-only and Apple-inspired in visual style.

## What I Already Know

* The user wants to formally proceed with a separate Trellis task named `web-dashboard-for-auto-signal-review`.
* The dashboard should not be mixed into the current annual router validation task.
* The recommended MVP is a read-only "daily signal review dashboard" before adding run control.
* The user wants an Apple-style UI.
* The user chose the FastAPI + React/Vite implementation path for the MVP.
* The user chose a review-console first screen: readiness verdict first, then gates, block reasons, and order summary.
* The user chose latest-run-only scope for the MVP; historical run browsing is deferred.
* The user chose structured report summary plus a link to the existing Markdown report; full inline Markdown rendering is deferred.
* The user wants the dashboard UI localized in Chinese, including user-facing status labels and common gate/block reasons.
* The project is currently a local Python quant pipeline that emits JSON, CSV, Markdown, and log artifacts under `outputs/`.
* Key workflow artifacts already exist:
  * `outputs/auto_run_status.json`
  * `outputs/auto_signal_report.json`
  * `outputs/daily_signal_report.md`
  * `outputs/data_health_report.json`
  * `outputs/data_governance_report.json`
  * `outputs/auto_parameter_quality.json`
  * `outputs/auto_backtest_quality.json`
  * `outputs/manual_orders_*.csv` / `outputs/manual_orders_candidate_*.csv`
  * `outputs/order_confirmations/*.csv`
  * `outputs/fill_feedback/*.csv`
* `scripts/run_auto_signal_supervised.py` already exposes local job status and log-tail behavior that can inform a future run-control version.

## Assumptions

* MVP is local-only, accessed from the user's machine, not deployed publicly.
* MVP is read-only and must not trigger pipeline runs, edit configs, promote signals, or write official outputs.
* Missing `outputs/` artifacts are normal and should render as friendly empty/error states.
* Apple-style means Apple-inspired interaction and visual language, not copying Apple branding or assets.
* The dashboard should be useful for daily manual review, not a marketing/landing page.

## Requirements

* Provide a local web dashboard focused on the latest auto-signal review.
* MVP reads and displays only the latest/current run artifacts from the configured `outputs/` directory.
* Show a clear readiness verdict: ready / blocked / candidate-only / missing artifacts.
* Present user-facing UI copy in Chinese and avoid describing quality-gate blocks as frontend/application errors.
* Use a review-console first-screen layout:
  * top area: manual-review readiness verdict and key dates;
  * next area: gate status cards and block reasons;
  * next area: signal/order summary and manual order preview;
  * supporting area: artifact links and report sections.
* Surface key dates: generated time, signal date, intended trade date, requested/target date, and latest stage if available.
* Show strategy mode, including annual-state-router details when present.
* Show gate status for data health, data governance, parameter quality, backtest quality, account/holdings, and candidate-only state.
* Show block reasons and quality warnings prominently.
* Show signal action summary and manual order table from the generated CSV artifacts.
* Show artifact links/paths for the generated report, signal, holdings, manual orders, order confirmation, fill feedback, and quality files.
* Show a structured summary derived from JSON/CSV artifacts and expose the latest `daily_signal_report.md` as an original report artifact link/path.
* Use an Apple-inspired UI:
  * light-first, calm, polished desktop-app feel;
  * system font stack and crisp typography;
  * quiet sidebar or toolbar navigation;
  * restrained surfaces, subtle borders/shadows, and semantic status color;
  * compact financial tables optimized for scanning;
  * no marketing hero page, no decorative blobs/orbs, and no oversized promo layout.
* Keep the first screen as the usable dashboard.

## Acceptance Criteria

* [ ] A FastAPI backend and React/Vite frontend can be started from documented local commands.
* [ ] The first screen shows the latest readiness verdict without requiring navigation.
* [ ] MVP reads the latest/current run only and does not expose a history selector.
* [ ] The first screen follows the review-console order: verdict, gates, blockers, order summary, then supporting details.
* [ ] If `outputs/auto_signal_report.json` exists, the dashboard displays strategy mode, executable status, block reasons, key gate summaries, signal date, intended trade date, and artifact paths.
* [ ] If expected outputs are missing or malformed, the UI shows a clear non-crashing empty/error state.
* [ ] Manual orders are displayed in a readable table when a manual order CSV exists.
* [ ] The daily report section shows structured summary information and links to the original Markdown report without rendering the full Markdown inline.
* [ ] The UI follows the agreed Apple-inspired style and remains dashboard-like, not landing-page-like.
* [ ] User-facing labels, section titles, readiness copy, and common gate/block reasons are localized in Chinese.
* [ ] MVP does not execute scripts, edit configs, write outputs, or promote official signals.
* [ ] Focused tests cover backend artifact parsing and missing-file behavior.
* [ ] README or a small docs note explains how to start the dashboard.

## Definition of Done

* Requirements are confirmed by the user.
* Relevant backend/frontend specs are loaded before implementation.
* Implementation is scoped to read-only dashboard behavior.
* Tests are added or updated for artifact parsing and API view-model behavior.
* Local dev startup is verified.
* Documentation is updated for dashboard startup and MVP limitations.
* Trellis check and project tests pass for the touched areas.

## Technical Approach

Chosen direction: FastAPI backend plus React/Vite frontend.

* Backend:
  * Add a small local API that reads existing output artifacts and returns dashboard-oriented JSON.
  * Normalize missing or malformed artifacts into explicit status objects.
  * Keep command execution out of MVP.
* Frontend:
  * Add a `web/` app using React, TypeScript, and Vite.
  * Build an Apple-inspired dashboard shell with overview, gates, orders, artifacts, and report sections.
  * Use familiar controls such as tabs/segmented controls, icon buttons with tooltips, status badges, and dense data tables.
* Integration:
  * Provide one documented local startup path.
  * Avoid moving quant logic into the frontend.
  * Preserve current CLI and `.bat` workflows.

## Research References

* [`research/local-web-dashboard-stack.md`](research/local-web-dashboard-stack.md) - recommends a thin FastAPI API plus React/Vite UI for a polished but safe local dashboard.

## Decision (ADR-lite)

Context: The project already has a mature local batch pipeline and rich output artifacts, but no web surface. The first user value is faster manual review, not remote deployment or automated trading control.

Decision: Start with a read-only local dashboard that visualizes existing artifacts using FastAPI + React/Vite.

Consequences: This adds a web layer without changing strategy logic. It introduces frontend tooling, but keeps the MVP safe by excluding command execution and output mutation.

## Out of Scope

* Running the auto-signal pipeline from the web UI.
* Editing `config/settings.yaml`, account files, holdings, or Tushare secrets.
* Promoting candidate signals to official outputs.
* Applying fill feedback or updating holdings from the web UI.
* Historical run browsing or archive comparison.
* Full inline Markdown report rendering.
* Remote deployment, multi-user authentication, or broker integration.
* Reworking the quant pipeline, strategy selection, or annual router logic.

## Technical Notes

* Current task directory: `.trellis/tasks/06-23-web-dashboard-for-auto-signal-review`.
* Existing active annual-router validation changes are separate and should not be edited for this task.
* Relevant code inspected:
  * `scripts/run_auto_signal.py`
  * `scripts/run_auto_signal_supervised.py`
  * `scripts/export_auto_status_metrics.py`
  * `src/reporting.py`
  * `src/manual_orders.py`
  * `config/settings.yaml`
  * `.trellis/spec/backend/index.md`
* No existing `package.json`, Vite config, or Python project config file was found.
* The first implementation style question was resolved: use FastAPI + React/Vite.
* The first-screen emphasis question was resolved: use a review-console layout.
* The history scope question was resolved: latest run only for MVP.
* The report rendering question was resolved: structured summary plus Markdown artifact link.

## Implementation Plan

* PR1: Add read-only backend artifact reader and API view model with tests for present, missing, and malformed latest-run artifacts.
* PR2: Add React/Vite dashboard shell and Apple-inspired review-console UI consuming the backend API.
* PR3: Add documentation/start commands, polish empty states, and verify local startup.

## Open Questions

* None before final user confirmation.
