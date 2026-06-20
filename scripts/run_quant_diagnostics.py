"""Command-line entry point for the five-layer quant diagnostic report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quant_diagnostics import build_quant_diagnostic_report, write_quant_diagnostic_report


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a five-layer diagnostic report for quant backtest artifacts.")
    parser.add_argument("--artifact-dir", default="outputs", help="Directory containing backtest and diagnostic artifacts.")
    parser.add_argument("--compare-dir", default="", help="Optional second artifact directory for reproducibility checks.")
    parser.add_argument("--out-dir", default="outputs", help="Directory where the diagnostic report is written.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Numeric tolerance for accounting invariants.")
    args = parser.parse_args()

    report = build_quant_diagnostic_report(
        args.artifact_dir,
        compare_dir=args.compare_dir or None,
        tolerance=args.tolerance,
    )
    paths = write_quant_diagnostic_report(report, args.out_dir)
    logger.info("Quant diagnostics written: json=%s markdown=%s", paths["json"], paths["markdown"])
    print(
        "Quant diagnostics: "
        f"optimization_ready={report['optimization_ready']}; "
        f"json={paths['json']}; markdown={paths['markdown']}"
    )


if __name__ == "__main__":
    main()
