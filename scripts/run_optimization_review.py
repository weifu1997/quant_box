"""Command-line entry point for post-diagnostic optimization review."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization_review import build_optimization_review, write_optimization_review


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review style routing, risk exposure, and trading constraints after quant diagnostics pass."
    )
    parser.add_argument("--artifact-dir", default="outputs", help="Directory containing auto-run and diagnostic artifacts.")
    parser.add_argument("--out-dir", default="outputs", help="Directory where the optimization review is written.")
    args = parser.parse_args()

    report = build_optimization_review(args.artifact_dir)
    paths = write_optimization_review(report, args.out_dir)
    logger.info("Optimization review written: json=%s markdown=%s", paths["json"], paths["markdown"])
    print(
        "Optimization review: "
        f"status={report['status']}; "
        f"diagnostics_ready={report['diagnostics_ready']}; "
        f"json={paths['json']}; markdown={paths['markdown']}"
    )


if __name__ == "__main__":
    main()
