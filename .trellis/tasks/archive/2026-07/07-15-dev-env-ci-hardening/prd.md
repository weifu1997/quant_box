# Harden Development Environment Sync and CI

## Goal

Make a freshly cloned or freshly pulled checkout reproducibly runnable on Windows and Ubuntu, and make GitHub Actions enforce the backend, frontend, and browser contracts introduced by the cross-platform Web workspace.

## What I Already Know

* The repository supports Windows desktop development and Ubuntu/systemd deployment.
* Python 3.11 is the supported runtime and Node 22 LTS is the CI baseline.
* The current local virtual environment is missing FastAPI and Uvicorn even though both are listed in `requirements-lock.txt`.
* `web/node_modules` and `web/dist` are intentionally ignored and currently absent.
* The current CI runs only Windows/Python tests; it does not build the frontend or run Playwright.
* The existing Windows installer uses `requirements.txt`, while CI uses `requirements-lock.txt`.
* The existing Linux launcher builds only when `web/dist` is absent, so a pull can leave a stale production bundle.

## Requirements

* Add one cross-platform, standard-library Python entry point with `sync` and `doctor` subcommands.
* `sync` must be idempotent and must:
  * create the project virtual environment when missing;
  * install the Python versions from `requirements-lock.txt`;
  * install frontend dependencies with `npm ci`;
  * optionally build the production frontend;
  * optionally install the Playwright Chromium browser;
  * record dependency/build input hashes so unchanged environments can be checked quickly.
* `doctor` must report actionable failures without printing secrets and support strict non-zero exit behavior for automation.
* Doctor checks must cover Python version/imports/locked direct versions, Node/npm availability, frontend dependency state, and optional production-build freshness.
* The Windows installation entry point must delegate to the shared sync implementation instead of maintaining separate dependency logic.
* The Windows dashboard launcher must detect or repair dependency drift before starting the development servers.
* The Ubuntu production launcher must not perform package installation or a network-dependent build during service startup; it must fail with an actionable sync/build command when runtime artifacts are missing or stale.
* Deployment and README instructions must use the shared sync/doctor workflow.
* GitHub Actions must run:
  * the full Python regression suite on Windows and Ubuntu with Python 3.11;
  * frontend type-check and production build with Node 22;
  * Playwright E2E with Chromium on Ubuntu;
  * minimal workflow permissions, concurrency cancellation, and bounded timeouts.
* Failed browser jobs must retain the Playwright report/test results when available.
* Existing CLI, API, data, signal, and official/candidate output behavior must remain unchanged.

## Acceptance Criteria

* [x] A clean Windows checkout can run the shared sync command and pass strict doctor checks.
* [x] A clean Ubuntu checkout can run the same Python sync/doctor contract using POSIX virtual-environment paths.
* [x] Re-running sync with unchanged lock files is safe and does not unnecessarily reinstall dependencies.
* [x] Missing FastAPI/Uvicorn, missing npm dependencies, changed lock files, and stale/missing production builds produce actionable doctor failures.
* [x] The Windows installer and dashboard launcher use the shared environment contract.
* [x] The Ubuntu start script performs validation only and never installs/builds implicitly.
* [x] CI defines Windows and Ubuntu backend tests, frontend build, and Playwright jobs.
* [x] Python unit tests cover environment hashing/version/doctor decision logic without modifying the developer's real environment.
* [x] The full Python suite passes after sync.
* [x] `npm run build` passes.
* [x] `npm run test:e2e` passes with Chromium.
* [x] README and deployment documentation describe sync, doctor, build, and start as separate steps.

## Definition of Done

* Focused environment-tool tests pass on Windows.
* Full backend, frontend build, and browser checks pass locally or in equivalent CI commands.
* `git diff --check` passes.
* No generated environment, frontend build, Playwright report, secret, market data, or output artifact is committed.
* Trellis specs are updated if the new environment contract becomes a durable project convention.

## Technical Approach

Use `scripts/dev_env.py` as the single cross-platform implementation. Keep it standard-library-only so it can create/check `.venv` before project dependencies exist. Use small platform adapters only for locating the virtual-environment Python executable and resolving npm commands. Store ignored hash stamps inside `.venv/`, `web/node_modules/`, and `web/dist/`.

CI remains explicit instead of calling a mutation-heavy sync command: it installs from the same lock files, then runs doctor/tests/build/E2E. This keeps CI logs clear while sharing validation rules.

## Decision (ADR-lite)

**Context**: Windows batch scripts, Ubuntu deployment instructions, and CI currently install different dependency sets and validate different product surfaces.

**Decision**: Introduce a standard-library Python sync/doctor tool, retain pip/npm for this task, and split production build from service start.

**Consequences**: Environment behavior becomes consistent without a package-manager migration. `requirements-lock.txt` still pins direct rather than all transitive dependencies; migration to `pyproject.toml`/`uv.lock` remains future work.

## Research References

* [`research/environment-ci-audit.md`](research/environment-ci-audit.md) — repository-specific drift, CI gaps, and selected implementation boundary.

## Out of Scope

* Migrating to `uv`, Poetry, pip-tools, or a new Python packaging layout.
* Docker or Kubernetes deployment.
* Changing quant strategy, data, backtest, signal, or account behavior.
* Extracting the annual router or splitting the auto-signal/Dashboard code; those are dependent follow-up tasks.

## Technical Notes

* Relevant files: `requirements*.txt`, `00_安装依赖环境.bat`, `15_启动Web仪表盘.bat`, `scripts/start_dashboard.sh`, `web/package*.json`, `.github/workflows/ci.yml`, `README.md`, and `docs/web-deployment.md`.
* Project commands must continue to use Python 3.11.
* The production dashboard serves `web/dist` through FastAPI.
* Playwright API mutations are mocked, so E2E must not touch real market/account data.

## Verification

* Local strict doctor: 8 pass, 0 warn, 0 fail.
* Local backend: 713 tests and 6 subtests passed.
* Local frontend: production build and 10 Playwright Chrome tests passed.
* GitHub Actions run `29412248561`: Windows backend, Ubuntu backend, Node 22 frontend build, and Playwright Chromium all passed.
