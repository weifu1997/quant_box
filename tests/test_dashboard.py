"""Tests for the dashboard artifact view model."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.dashboard import build_dashboard_precheck, build_dashboard_snapshot


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

    def test_build_dashboard_snapshot_uses_newer_governance_after_repair(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "auto_signal_report.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-23T01:40:28",
                        "is_executable": False,
                        "candidate_only": True,
                        "block_reasons": [
                            "governance:daily_basic_date_coverage_below_required:2779/2784<1.00",
                            "candidate_only_requested",
                        ],
                        "quality_warnings": [
                            "governance:daily_basic_date_coverage_below_required:2779/2784<1.00",
                        ],
                        "data_governance": {
                            "generated_at": "2026-06-23T01:40:28",
                            "is_point_in_time_ready": False,
                            "issues": ["daily_basic_date_coverage_below_required:2779/2784<1.00"],
                            "warnings": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "data_governance_report.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-24T00:36:04",
                        "is_point_in_time_ready": True,
                        "issues": [],
                        "warnings": [],
                        "daily_basic_covered_dates": 2784,
                        "daily_basic_expected_dates": 2784,
                        "daily_basic_date_coverage": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            snapshot = build_dashboard_snapshot(out_dir)
            governance_gate = next(gate for gate in snapshot["gates"] if gate["id"] == "data_governance")

            self.assertEqual(governance_gate["status"], "pass")
            self.assertEqual(governance_gate["issues"], [])
            self.assertTrue(governance_gate["details"]["supersedes_auto_report"])
            self.assertEqual(
                governance_gate["details"]["resolved_auto_report_issues"],
                ["daily_basic_date_coverage_below_required:2779/2784<1.00"],
            )
            self.assertEqual(snapshot["block_reasons"], ["candidate_only_requested"])
            self.assertEqual(snapshot["blocker_actions"][0]["action"]["action"], "run_auto_signal")
            self.assertEqual(snapshot["blocker_actions"][0]["action"]["mode"], "normal")
            self.assertEqual(snapshot["quality_warnings"], [])
            self.assertEqual(snapshot["freshness_notes"], ["data_governance_repaired_after_auto_report"])
            self.assertEqual(snapshot["blocker_actions"][1]["source"], "freshness_note")
            self.assertEqual(snapshot["blocker_actions"][1]["action"]["action"], "run_auto_signal")

    def test_build_dashboard_snapshot_maps_daily_basic_blocker_to_repair_action(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "auto_signal_report.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-23T01:40:28",
                        "is_executable": False,
                        "block_reasons": ["governance:daily_basic_date_coverage_below_required:2779/2784<1.00"],
                        "quality_warnings": [],
                        "data_governance": {
                            "generated_at": "2026-06-23T01:40:28",
                            "is_point_in_time_ready": False,
                            "issues": ["daily_basic_date_coverage_below_required:2779/2784<1.00"],
                            "warnings": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            snapshot = build_dashboard_snapshot(out_dir)

            self.assertEqual(snapshot["blocker_actions"][0]["title"], "补齐 daily_basic 点时数据")
            self.assertEqual(snapshot["blocker_actions"][0]["action"]["action"], "repair_point_in_time")

    def test_build_dashboard_precheck_passes_with_current_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            out_dir.mkdir()
            holdings = root / "current_holdings.csv"
            holdings.write_text("instrument,shares\n000001.SZ,100\n", encoding="utf-8")
            (out_dir / "data_health_report.json").write_text(
                json.dumps(
                    {
                        "requested_end_date": "2026-06-23",
                        "is_healthy": True,
                        "issues": [],
                        "raw_latest_date": "2026-06-23",
                        "price_latest_date": "2026-06-23",
                        "factor_latest_date": "2026-06-23",
                        "factor_latest_target_coverage": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "data_governance_report.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-24T09:00:00",
                        "is_point_in_time_ready": True,
                        "issues": [],
                        "warnings": [],
                        "daily_basic_end_date": "2026-06-23",
                        "daily_basic_date_coverage": 1.0,
                        "st_calendar_end_date": "2026-06-23",
                        "factor_cache_meta_available": True,
                        "factor_cache_meta_end_date": "2026-06-23",
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "data": {"end_date": "2026-06-23"},
                "account": {"current_holdings_file": str(holdings), "total_asset": 1_000_000, "cash": 100_000},
                "outputs": {"dir": str(out_dir)},
            }

            precheck = build_dashboard_precheck(out_dir=out_dir, config=config)

            self.assertEqual(precheck["status"], "pass")
            self.assertTrue(precheck["can_run_normal"])
            self.assertTrue(all(item["status"] == "pass" for item in precheck["items"]))

    def test_build_dashboard_precheck_maps_blockers_to_actions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            out_dir.mkdir()
            holdings = root / "missing_holdings.csv"
            (out_dir / "data_health_report.json").write_text(
                json.dumps(
                    {
                        "requested_end_date": "2026-06-23",
                        "is_healthy": False,
                        "issues": ["factor_latest_before_end:2026-06-22<2026-06-23"],
                        "factor_latest_date": "2026-06-22",
                        "factor_latest_target_coverage": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "data_governance_report.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-24T09:00:00",
                        "is_point_in_time_ready": False,
                        "issues": ["daily_basic_date_coverage_below_required:2779/2784<1.00"],
                        "warnings": [],
                        "factor_cache_meta_available": True,
                        "factor_cache_meta_end_date": "2026-06-23",
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "data": {"end_date": "2026-06-23"},
                "account": {"current_holdings_file": str(holdings), "total_asset": 1_000_000, "cash": 100_000},
                "outputs": {"dir": str(out_dir)},
            }

            precheck = build_dashboard_precheck(out_dir=out_dir, config=config)
            by_id = {item["id"]: item for item in precheck["items"]}

            self.assertEqual(precheck["status"], "fail")
            self.assertFalse(precheck["can_run_normal"])
            self.assertEqual(by_id["data_governance"]["action"]["action"], "repair_point_in_time")
            self.assertEqual(by_id["factor_freshness"]["action"]["action"], "run_auto_signal")
            self.assertEqual(by_id["account"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
