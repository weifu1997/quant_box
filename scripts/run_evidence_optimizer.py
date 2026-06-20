"""Command-line entry point for evidence-backed optimization planning."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence_optimizer import build_evidence_optimization_plan, write_evidence_optimization_plan


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evidence-backed style, risk, and trading optimization plan."
    )
    parser.add_argument("--artifact-dir", default="outputs", help="Directory containing diagnostics and router grid artifacts.")
    parser.add_argument("--out-dir", default="outputs", help="Directory where the optimization plan is written.")
    parser.add_argument("--grid-glob", default="*router_grid*.csv", help="Glob for router grid CSV evidence.")
    parser.add_argument("--max-industry-weight-target", type=float, default=0.35)
    parser.add_argument("--annual-trade-cost-ratio-target", type=float, default=0.20)
    args = parser.parse_args()

    report = build_evidence_optimization_plan(
        args.artifact_dir,
        grid_glob=args.grid_glob,
        max_industry_weight_target=args.max_industry_weight_target,
        annual_trade_cost_ratio_target=args.annual_trade_cost_ratio_target,
    )
    paths = write_evidence_optimization_plan(report, args.out_dir)
    logger.info("Evidence optimization plan written: json=%s markdown=%s", paths["json"], paths["markdown"])
    print(
        "Evidence optimization plan: "
        f"status={report['status']}; "
        f"json={paths['json']}; markdown={paths['markdown']}"
    )


if __name__ == "__main__":
    main()
