"""模块说明：提供 run_apply_fills 命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.manual_orders import apply_fill_feedback, load_current_holdings, save_updated_holdings


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    config = load_config()
    parser = argparse.ArgumentParser(description="Apply manually entered fill feedback to current holdings.")
    parser.add_argument("fill_file", help="Path to fill_feedback_YYYY-MM-DD.csv exported by the daily signal flow.")
    parser.add_argument("--dry-run", action="store_true", help="Write an audit file without updating current_holdings.csv.")
    parser.add_argument("--output", help="Optional holdings output path. Defaults to account.current_holdings_file.")
    args = parser.parse_args()

    fill_path = resolve_path(args.fill_file)
    if not fill_path.exists():
        raise FileNotFoundError(f"Fill feedback file not found: {fill_path}")
    fills = pd.read_csv(fill_path)
    current = load_current_holdings(config)
    updated = apply_fill_feedback(current, fills)

    output_path = resolve_path(args.output) if args.output else resolve_path(
        config.get("account", {}).get("current_holdings_file", "config/current_holdings.csv")
    )
    if not args.dry_run:
        if args.output:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            updated.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            output_path = save_updated_holdings(updated, config)

    audit = _audit_payload(fill_path, output_path, current, updated, fills, dry_run=args.dry_run)
    out_dir = resolve_path(config.get("outputs", {}).get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / f"fill_apply_audit_{_signal_date(fills, fill_path)}.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Loaded fill feedback: %s", fill_path)
    logger.info("Updated holdings rows: %s", len(updated))
    logger.info("Holdings output: %s%s", output_path, " (dry-run)" if args.dry_run else "")
    logger.info("Audit output: %s", audit_path)


def _audit_payload(
    fill_path: Path,
    output_path: Path,
    current: pd.DataFrame,
    updated: pd.DataFrame,
    fills: pd.DataFrame,
    dry_run: bool,
) -> dict[str, object]:
    """函数说明：处理 audit_payload 的内部辅助逻辑。"""
    status = fills.get("fill_status", pd.Series(dtype=object)).fillna("").astype(str).str.strip().str.upper()
    applied = status.isin({"FILLED", "PARTIAL"})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fill_file": str(fill_path),
        "holdings_output": str(output_path),
        "dry_run": bool(dry_run),
        "current_positions": int(len(current)),
        "updated_positions": int(len(updated)),
        "fill_rows": int(len(fills)),
        "applied_fill_rows": int(applied.sum()),
        "fill_status_counts": status.value_counts().to_dict(),
    }


def _signal_date(fills: pd.DataFrame, fill_path: Path) -> str:
    """函数说明：处理 signal_date 的内部辅助逻辑。"""
    if "signal_date" in fills.columns and not fills["signal_date"].dropna().empty:
        return str(fills["signal_date"].dropna().iloc[0])
    stem = fill_path.stem
    parts = stem.split("_")
    return parts[-1] if parts else datetime.now().strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
