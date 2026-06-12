# Backend Development Guidelines

> Entry point for backend work in this project.

---

## Overview

This project is a single-repo Python quant research pipeline. Backend work means changes to importable modules in `src/`, CLI workflows in `scripts/`, config/data/report contracts, tests, and generated-file boundaries.

Read this index first, then read the specific guideline files that match the work.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Module organization and file layout | Filled |
| [Database Guidelines](./database-guidelines.md) | File-based storage contracts, cache formats, migrations | Filled |
| [Error Handling](./error-handling.md) | Error types, handling strategies | Filled |
| [Quality Guidelines](./quality-guidelines.md) | Code standards, forbidden patterns | Filled |
| [Logging Guidelines](./logging-guidelines.md) | Structured logging, status files, log levels | Filled |

---

## Pre-Development Checklist

Before editing backend code or workflow docs:

- Read [Directory Structure](./directory-structure.md) for where the change belongs.
- Read [Database Guidelines](./database-guidelines.md) for any change touching config, CSV, parquet, JSON, reports, caches, or output paths.
- Read [Error Handling](./error-handling.md) for new validation, external calls, missing files, or CLI failures.
- Read [Logging Guidelines](./logging-guidelines.md) for long-running scripts, run status, progress files, or diagnostics.
- Read [Quality Guidelines](./quality-guidelines.md) for tests, forbidden patterns, generated data, and review checks.
- Use `.\.venv\Scripts\python.exe` for project commands.
- Search before adding a new helper, constant, config key, or artifact path.

---

## Quality Check

Before reporting backend work as done:

- Confirm generated/private data was not added outside ignored paths.
- Confirm factors/scores/prices still follow the documented pandas contracts.
- Confirm new or changed config keys are present in `DEFAULT_CONFIG`, validated when risky, and tested.
- Confirm user-visible scripts, `.bat` files, README, and CHANGELOG are updated when workflow behavior changes.
- Run focused tests with `.\.venv\Scripts\python.exe -m pytest <tests> -q`.
- For docs-only Trellis spec updates, run `git diff --check` and inspect the rendered Markdown content.

---

## Language

Write Trellis spec files in English. User-facing README and batch-script text may remain Chinese where the existing project uses Chinese.
