"""Export auto_run_status.json as Prometheus text metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import resolve_path
from src.monitoring import write_auto_status_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export automatic signal run status metrics.")
    parser.add_argument("--status-file", default="outputs/auto_run_status.json")
    parser.add_argument("--output", default="outputs/auto_run_metrics.prom")
    args = parser.parse_args()

    status_path = resolve_path(args.status_file)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    output_path = write_auto_status_metrics(status, resolve_path(args.output))
    print(output_path)


if __name__ == "__main__":
    main()
