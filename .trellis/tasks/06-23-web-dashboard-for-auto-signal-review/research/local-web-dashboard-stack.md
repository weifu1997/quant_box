# Local Web Dashboard Stack Research

## Question

What is the safest first web architecture for a local Python quant pipeline that currently emits JSON, CSV, Markdown, and log artifacts?

## Repo Constraints

* The project is currently a Python batch pipeline, not a web service.
* Core workflow entry points live in `scripts/`, with business logic in `src/`.
* The auto-signal workflow already emits machine-readable artifacts under `outputs/`, including `auto_run_status.json`, `auto_signal_report.json`, quality JSON files, manual order CSVs, and Markdown reports.
* Private files and generated outputs are gitignored, so the dashboard must treat missing files as normal local state.
* The first MVP should be read-only to avoid accidental signal promotion, order mutation, or secret exposure.

## Feasible Approaches

### Approach A: FastAPI backend + React/Vite frontend (recommended)

How it works:
* Add a small Python HTTP API that reads existing artifacts and normalizes them into dashboard view models.
* Add a `web/` frontend using Vite, React, and TypeScript.
* Keep the first MVP read-only; later versions can add whitelisted commands.

Pros:
* Fits the existing Python codebase while allowing a polished, Apple-inspired UI.
* Clean separation between artifact parsing, API contracts, and UI components.
* Easier to expand later into run control, log streaming, and order confirmation workflows.

Cons:
* Adds Node tooling and frontend dependencies.
* Requires a small local dev-server story for Windows users.

### Approach B: FastAPI backend + server-rendered HTML

How it works:
* Use Python templates and static CSS/JS for the dashboard.
* Avoid a separate frontend build system.

Pros:
* Fewer dependencies and simpler deployment.
* Good enough for a mostly static read-only report.

Cons:
* Harder to build rich tables, responsive interactions, segmented views, and chart-like UI polish.
* Future interactive workflows become more awkward.

### Approach C: Static page only

How it works:
* Generate static HTML from the latest outputs.

Pros:
* Minimal runtime complexity.

Cons:
* Weak fit for local artifact discovery, refresh, logs, and future run control.
* Browser file access restrictions make direct local JSON/CSV reads brittle.

## Recommendation

Use Approach A for the product direction, but keep the first implementation narrow:

* Read-only API endpoints.
* No command execution in MVP.
* No config editing in MVP.
* No official signal promotion from the web UI.
* Apple-inspired visual language: quiet system typography, restrained surfaces, dense but readable financial tables, semantic status color, sidebar + toolbar structure, and no marketing landing page.

