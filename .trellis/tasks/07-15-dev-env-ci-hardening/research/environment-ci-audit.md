# Environment and CI audit

## Observed drift

* The local `.venv` matches every direct locked version except FastAPI and Uvicorn, which are missing.
* Python test collection finds 656 tests before four FastAPI-dependent modules fail; the repository contains 704 Python test methods, including 48 in those modules.
* `web/node_modules` and `web/dist` are absent, while the Playwright browser cache exists.
* The Windows installer installs `requirements.txt`; CI installs `requirements-lock.txt`.
* The direct-dependency lock does not freeze transitive packages.

## CI coverage gap

* `.github/workflows/ci.yml` was last changed before the July cross-platform Web workspace work.
* CI runs only Windows/Python 3.11 and `pytest -q`.
* There is no Ubuntu backend job, frontend build, or Playwright job.
* The E2E suite contains ten tests and mocks mutating API calls, making it appropriate for CI.

## Startup gap

* The Windows launcher installs frontend dependencies only when the whole `node_modules` directory is absent.
* The Ubuntu launcher builds only when `web/dist/index.html` is absent, so a Git pull can leave stale frontend output.
* Ubuntu service startup can currently perform network-dependent installation/build work.

## Selected boundary

Implement a standard-library `scripts/dev_env.py` so it can operate before project packages are installed. Use lock/input hashes for idempotency and stale-build detection. Keep Python package-manager migration out of this task. Make CI exercise both supported operating systems plus the frontend and browser contracts.
