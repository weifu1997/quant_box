#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
HOST="${DASHBOARD_HOST:-127.0.0.1}"
PORT="${DASHBOARD_PORT:-8000}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python virtual environment not found: $PYTHON_BIN" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "web/dist is missing and npm is unavailable. Install Node.js or build the frontend before deployment." >&2
    exit 1
  fi
  echo "Building dashboard frontend..."
  (
    cd "$ROOT/web"
    npm ci
    npm run build
  )
fi

echo "Starting quant_box Web workspace on http://$HOST:$PORT"
exec "$PYTHON_BIN" scripts/run_dashboard.py --host "$HOST" --port "$PORT"
