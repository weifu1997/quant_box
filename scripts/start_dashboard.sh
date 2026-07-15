#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
HOST="${DASHBOARD_HOST:-127.0.0.1}"
PORT="${DASHBOARD_PORT:-8000}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python virtual environment not found: $PYTHON_BIN" >&2
  echo "Synchronize it with: python3.11 scripts/dev_env.py sync --build-web" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/dev_env.py doctor --strict --runtime-only

echo "Starting quant_box Web workspace on http://$HOST:$PORT"
exec "$PYTHON_BIN" scripts/run_dashboard.py --host "$HOST" --port "$PORT"
