# Web Dashboard for Auto-Signal Review

## Goal

Build a local web dashboard for the daily auto-signal review workflow so the user can quickly see whether the latest signal is ready for manual trading review, which gates passed or failed, and which candidate/official artifacts were generated. The first version was planned as read-only and Apple-inspired; the user later expanded the scope to include controlled local run actions for real data repair and signal reruns.

## Scope Update: Controlled Run Actions

On 2026-06-24 the user explicitly requested that the frontend be able to trigger real point-in-time `daily_basic` repair and switch/re-run auto signal output mode from the dashboard. This supersedes the original read-only limitation while keeping the safety boundary narrow:

* The dashboard may trigger only whitelisted backend actions, not arbitrary commands.
* The dashboard may start real `daily_basic` point-in-time repair through `scripts/run_update_point_in_time_data.py`.
* The dashboard may re-run auto signal in either candidate output mode or normal gated output mode.
* Normal gated output mode must not force official output; existing auto-signal gates still decide whether official artifacts are written.
* Long-running jobs must expose status and log-tail output in the frontend so the user can tell the task is still active.
* The review report panel should align visually with other dashboard panels instead of being cramped.

## Scope Update: Web-First Project Workspace

On 2026-07-10 the user clarified that the Web application will become the primary surface for future viewing and operations. The dashboard must therefore remain an efficient daily review console while also explaining the full system clearly enough to serve as the user's main project workspace.

* Keep the latest-run review console as the default operational screen.
* Add a dedicated project-overview workspace instead of mixing long explanatory content into the trading review flow.
* Explain the four decisions the project supports, the end-to-end data flow, data and point-in-time governance, factor research, realistic backtesting, quality gates, and the manual trading/fill-feedback loop.
* Make the two workspaces easy to switch from a persistent navigation surface.
* Preserve the controlled-action safety boundary: becoming Web-first does not authorize arbitrary commands, forced official output, automatic trading, or direct holdings mutation.
* Keep the UI polished, Chinese-localized, responsive, and practical for repeated daily use.
* Add a controlled trade-execution workspace that edits the latest official fill-feedback template, previews holdings changes, and applies validated fills to current holdings after explicit confirmation.
* Treat instrument, side, and planned order size as immutable backend-owned fields; candidate templates must never be applicable to official holdings.
* Add a safe account-and-holdings workspace so official-order prerequisites can be configured without manual YAML/CSV editing.
* Restrict account editing to whitelisted trading inputs, never expose Tushare credentials or arbitrary configuration, and back up existing account files before replacement.

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
* The original MVP was read-only, but current accepted scope allows controlled local repair/rerun actions through backend white-listed jobs.
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
* Provide controlled frontend actions to:
  * repair the `daily_basic` point-in-time data gap with a real backend command;
  * rerun auto signal in candidate output mode;
  * rerun auto signal in normal gated output mode.
* Show active/recent dashboard job status and log tail in the frontend.
* Keep report-summary UI width aligned with the main dashboard panels.
* Use an Apple-inspired UI:
  * light-first, calm, polished desktop-app feel;
  * system font stack and crisp typography;
  * quiet sidebar or toolbar navigation;
  * restrained surfaces, subtle borders/shadows, and semantic status color;
  * compact financial tables optimized for scanning;
  * no marketing hero page, no decorative blobs/orbs, and no oversized promo layout.
* Keep the first screen as the usable dashboard.
* Provide a project-overview workspace covering the system purpose, pipeline, data layer, factor research, realistic backtest constraints, quality architecture, and manual execution loop.
* Keep project education separate from the operational review hierarchy so explanatory content does not hide the latest readiness verdict or run controls.
* Treat the Web dashboard as the primary user-facing workspace while retaining the existing safe CLI and batch workflows as underlying execution paths.
* Support the manual execution loop from Web: enter fill status/quantity/price/costs, preview validation, and explicitly apply valid fills to current holdings with an audit artifact.

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
* [ ] Dashboard run controls execute only whitelisted backend commands and do not expose arbitrary command execution.
* [ ] The frontend can trigger real `daily_basic` point-in-time repair and show status/log output while it runs.
* [ ] The frontend can select candidate vs normal gated auto-signal output mode before rerun.
* [ ] Normal gated output mode does not add `--candidate-only` and does not add `--force-official`.
* [ ] The review report panel width is aligned with other main dashboard panels.
* [ ] Focused tests cover backend artifact parsing and missing-file behavior.
* [ ] README or a small docs note explains how to start the dashboard.
* [ ] Persistent navigation switches between the daily review console and the project-overview workspace.
* [ ] The project-overview workspace covers all seven user-specified project areas and clearly states the candidate-vs-official safety boundary.
* [ ] The overview remains readable on desktop and narrow/mobile layouts without degrading the operational dashboard.
* [ ] A trade-execution workspace loads only the latest official fill-feedback template and excludes candidate templates.
* [ ] Web users can edit only execution-result fields; instrument, side, and planned shares remain unchanged even if a request attempts to override them.
* [ ] Applying fills requires a successful preview/validation and explicit confirmation, writes an audit JSON, and rejects pending, over-planned, or over-position sells without modifying holdings.
* [ ] The Web account workspace can preview and save total asset, cash, position cap, lot sizes, and normalized current holdings.
* [ ] Account saves require explicit confirmation, reject non-finite/invalid/duplicate/non-lot inputs, back up existing files, and never expose unrelated local settings or secrets.
* [ ] Playwright browser tests cover workspace navigation, workflow run/stop requests, execution preview/apply, account preview/save, and narrow-screen overflow.
* [ ] Browser tests can run against system Chrome on Windows and Playwright Chromium on Ubuntu.
* [ ] Workflow cards with configurable behavior render backend-provided parameter schemas and submit values under a structured `parameters` object.
* [ ] The backend rejects unknown fields, invalid types, non-finite/out-of-range values, invalid dates, and malformed list text; file/output paths and arbitrary commands remain unavailable.
* [ ] Advanced risk refinement, regime blend, rebalance drift, and annual-router grid workflows are available with bounded defaults.

## Definition of Done

* Requirements are confirmed by the user.
* Relevant backend/frontend specs are loaded before implementation.
* Implementation is scoped to latest-run review plus controlled local repair/rerun actions.
* Tests are added or updated for artifact parsing and API view-model behavior.
* Local dev startup is verified.
* Documentation is updated for dashboard startup and MVP limitations.
* Trellis check and project tests pass for the touched areas.

## Technical Approach

Chosen direction: FastAPI backend plus React/Vite frontend.

* Backend:
  * Add a small local API that reads existing output artifacts and returns dashboard-oriented JSON.
  * Normalize missing or malformed artifacts into explicit status objects.
  * Expose only whitelisted dashboard jobs for `daily_basic` repair and candidate/normal auto-signal rerun.
  * Persist dashboard job status under `outputs/dashboard_jobs/` and logs under `outputs/logs/dashboard_job_*.log`.
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

* Arbitrary command execution from the web UI.
* Editing `config/settings.yaml`, account files, holdings, or Tushare secrets.
* Promoting candidate signals to official outputs or forcing official output from the dashboard.
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

* PR1: Add backend artifact reader and API view model with tests for present, missing, and malformed latest-run artifacts.
* PR2: Add React/Vite dashboard shell and Apple-inspired review-console UI consuming the backend API.
* PR3: Add controlled run actions, job logs, documentation/start commands, polish empty states, and verify local startup.

## Open Questions

* None before final user confirmation.
