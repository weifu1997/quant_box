"""Tests for the dashboard artifact view model."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.dashboard import build_dashboard_snapshot


class DashboardTests(unittest.TestCase):
    def test_build_dashboard_snapshot_summarizes_latest_report_and_orders(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            orders = out_dir / "manual_orders_candidate_2026-06-09.csv"
            orders.write_text(
                "instrument,action,is_order_actionable,target_value,note\n"
                "001268.SZ,BUY,false,120000,blocked\n"
                "603116.SH,BUY,true,120000,ready\n",
                encoding="utf-8-sig",
            )
            (out_dir / "daily_signal_report.md").write_text("# Daily report\n", encoding="utf-8")
            (out_dir / "auto_run_status.json").write_text(
                json.dumps({"status": "blocked", "stages": [{"name": "generate_signal", "state": "complete"}]}),
                encoding="utf-8",
            )
            (out_dir / "auto_signal_report.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-21T23:21:57",
                        "strategy_mode": "annual_state_router",
                        "signal_date": "2026-06-09",
                        "intended_trade_date": "2026-06-10",
                        "is_executable": False,
                        "candidate_only": True,
                        "block_reasons": ["candidate_only_requested"],
                        "quality_warnings": [],
                        "signal_summary": {"BUY": 2, "HOLD": 0, "SELL": 0},
                        "data_health": {"is_healthy": True, "issues": []},
                        "data_governance": {"is_point_in_time_ready": True, "issues": [], "warnings": []},
                        "parameter_quality": {"is_acceptable": True, "issues": [], "windows": 12},
                        "backtest_quality": {"is_acceptable": True, "issues": [], "annual_return": 0.26},
                        "account": {"holdings_loaded": True},
                        "files": {"manual_orders": str(orders)},
                    }
                ),
                encoding="utf-8",
            )

            snapshot = build_dashboard_snapshot(out_dir)

            self.assertEqual(snapshot["readiness"]["status"], "candidate_only")
            self.assertEqual(snapshot["latest_run"]["strategy_mode"], "annual_state_router")
            self.assertEqual(snapshot["latest_run"]["latest_stage"]["name"], "generate_signal")
            self.assertEqual(snapshot["signal_summary"]["BUY"], 2)
            self.assertEqual(snapshot["orders"]["total_rows"], 2)
            self.assertEqual(snapshot["orders"]["action_counts"], {"BUY": 2})
            self.assertEqual(snapshot["orders"]["actionable_count"], 1)
            self.assertTrue(any(gate["id"] == "data_health" and gate["status"] == "pass" for gate in snapshot["gates"]))
            self.assertTrue(any(item["id"] == "daily_report" and item["exists"] for item in snapshot["artifacts"]))

    def test_build_dashboard_snapshot_handles_missing_latest_report(self) -> None:
        with TemporaryDirectory() as tmp:
            snapshot = build_dashboard_snapshot(Path(tmp))

            self.assertEqual(snapshot["readiness"]["status"], "missing")
            self.assertEqual(snapshot["orders"]["exists"], False)
            self.assertTrue(any(item["id"] == "auto_signal_report" and not item["exists"] for item in snapshot["artifacts"]))

    def test_build_dashboard_snapshot_reports_malformed_json(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "auto_signal_report.json").write_text("{not-json", encoding="utf-8")

            snapshot = build_dashboard_snapshot(out_dir)

            self.assertEqual(snapshot["readiness"]["status"], "error")
            self.assertTrue(snapshot["errors"])


if __name__ == "__main__":
    unittest.main()
