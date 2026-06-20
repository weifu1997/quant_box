"""Tests for the five-layer quant diagnostic report."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.quant_diagnostics import (
    build_quant_diagnostic_report,
    render_quant_diagnostic_markdown,
    write_quant_diagnostic_report,
)


class QuantDiagnosticsTests(unittest.TestCase):
    def test_report_passes_when_all_layers_have_required_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            compare = Path(tmp) / "compare"
            root.mkdir()
            compare.mkdir()
            _write_complete_artifacts(root)
            _write_complete_artifacts(compare)

            with patch("src.quant_diagnostics.resolve_path", side_effect=lambda value: Path(value)):
                report = build_quant_diagnostic_report(root, compare_dir=compare)

            self.assertTrue(report["optimization_ready"])
            self.assertEqual(report["layers"]["backtest_engine"]["status"], "pass")
            self.assertEqual(report["layers"]["data"]["status"], "pass")
            self.assertEqual(report["layers"]["factor"]["status"], "pass")
            self.assertEqual(report["layers"]["portfolio"]["status"], "pass")
            self.assertEqual(report["layers"]["optimization"]["status"], "pass")

    def test_missing_artifacts_are_reported_as_caveats_and_block_optimization(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()

            with patch("src.quant_diagnostics.resolve_path", side_effect=lambda value: Path(value)):
                report = build_quant_diagnostic_report(root)

            checks = {check["name"]: check for check in report["checks"]}
            self.assertFalse(report["optimization_ready"])
            self.assertEqual(checks["required_artifacts"]["status"], "fail")
            self.assertIn("backtest_equity.csv", checks["required_artifacts"]["evidence"]["missing"])
            self.assertEqual(checks["optimization_gate"]["status"], "fail")

    def test_accounting_invariants_fail_when_costs_or_holdings_do_not_reconcile(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_complete_artifacts(root)
            (root / "backtest_metrics.json").write_text('{"trade_cost": 99.0}', encoding="utf-8")
            pd.DataFrame(
                [{"date": "2024-01-02", "instrument": "000001.SZ", "shares": 20, "price": 10.0, "value": 200.0}]
            ).to_csv(root / "backtest_holdings.csv", index=False, encoding="utf-8-sig")

            with patch("src.quant_diagnostics.resolve_path", side_effect=lambda value: Path(value)):
                report = build_quant_diagnostic_report(root)

            checks = {check["name"]: check for check in report["checks"]}
            self.assertEqual(checks["trade_cost_invariant"]["status"], "fail")
            self.assertEqual(checks["holding_roll_forward"]["status"], "fail")
            self.assertFalse(report["optimization_ready"])

    def test_reproducibility_fails_when_compare_artifact_differs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            compare = Path(tmp) / "compare"
            root.mkdir()
            compare.mkdir()
            _write_complete_artifacts(root)
            _write_complete_artifacts(compare)
            (compare / "backtest_metrics.json").write_text('{"trade_cost": 2.0}', encoding="utf-8")

            with patch("src.quant_diagnostics.resolve_path", side_effect=lambda value: Path(value)):
                report = build_quant_diagnostic_report(root, compare_dir=compare)

            checks = {check["name"]: check for check in report["checks"]}
            self.assertEqual(checks["same_input_reproducibility"]["status"], "fail")
            self.assertIn("backtest_metrics.json", checks["same_input_reproducibility"]["evidence"]["mismatched"])

    def test_reproducibility_ignores_run_summary_timestamp_noise(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            compare = Path(tmp) / "compare"
            root.mkdir()
            compare.mkdir()
            _write_complete_artifacts(root)
            _write_complete_artifacts(compare)
            (compare / "backtest_run_summary.json").write_text('{"log_file": "outputs/logs/backtest_later.log"}', encoding="utf-8")

            with patch("src.quant_diagnostics.resolve_path", side_effect=lambda value: Path(value)):
                report = build_quant_diagnostic_report(root, compare_dir=compare)

            checks = {check["name"]: check for check in report["checks"]}
            self.assertEqual(checks["same_input_reproducibility"]["status"], "pass")

    def test_writer_persists_json_and_markdown(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_complete_artifacts(root)

            with patch("src.quant_diagnostics.resolve_path", side_effect=lambda value: Path(value)):
                report = build_quant_diagnostic_report(root, compare_dir=root)
                paths = write_quant_diagnostic_report(report, root)

            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["markdown"]).exists())
            self.assertIn("Quant Diagnostic Report", render_quant_diagnostic_markdown(report))
            self.assertIn("Optimization ready: True", Path(paths["markdown"]).read_text(encoding="utf-8"))


def _write_complete_artifacts(root: Path) -> None:
    pd.Series([1000.0], index=pd.to_datetime(["2024-01-02"]), name="equity").to_csv(root / "backtest_equity.csv")
    pd.DataFrame(
        [{"date": "2024-01-02", "instrument": "000001.SZ", "shares": 10, "price": 10.0, "value": 100.0}]
    ).to_csv(root / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "signal_date": "2024-01-01",
                "date": "2024-01-02",
                "instrument": "000001.SZ",
                "side": "BUY",
                "shares": 10,
                "price": 10.0,
                "cash": 100.0,
                "status": "filled",
                "commission_cost": 1.0,
                "tax_cost": 0.0,
                "transfer_fee_cost": 0.0,
                "slippage_cost": 0.0,
            }
        ]
    ).to_csv(root / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    (root / "backtest_metrics.json").write_text('{"trade_cost": 1.0, "annual_return": 0.2}', encoding="utf-8")
    pd.DataFrame([{"year": 2024, "annual_return": 0.20, "max_drawdown": -0.10}]).to_csv(
        root / "backtest_yearly.csv", index=False, encoding="utf-8-sig"
    )
    (root / "backtest_run_summary.json").write_text('{"costs": {"total_trade_cost": 1.0}}', encoding="utf-8")
    (root / "data_health_report.json").write_text('{"is_healthy": true, "issues": []}', encoding="utf-8")
    (root / "data_governance_report.json").write_text(
        """
        {
          "is_point_in_time_ready": true,
          "issues": [],
          "warnings": [],
          "adj_factor_meta_available": true,
          "st_filter_mode": "calendar",
          "historical_universe_available": true,
          "index_constituents_available": true
        }
        """,
        encoding="utf-8",
    )
    pd.DataFrame([{"factor": "alpha", "mean_ic": 0.03, "ic_ir": 0.5, "positive_ratio": 0.6, "count": 120}]).to_csv(
        root / "factor_ic_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"year": 2024, "factor": "alpha", "mean_ic": 0.03}]).to_csv(
        root / "factor_ic_yearly.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"date": "2024-01-02", "top_minus_bottom": 0.01}]).to_csv(
        root / "factor_group_returns.csv", index=False, encoding="utf-8-sig"
    )
    (root / "auto_research_diagnostics.json").write_text(
        """
        {
          "exposure": {"enabled": true, "latest_top_position_weight": 0.1},
          "turnover_attribution": {"annual_turnover_estimate": 1.0},
          "cost_attribution": {"total_trade_cost": 1.0},
          "drawdown": {"enabled": true, "max_drawdown": -0.1}
        }
        """,
        encoding="utf-8",
    )
    (root / "auto_failure_analysis.json").write_text(
        '{"enabled": true, "primary_failure_area": "none", "failure_scope_summary": {"primary_scope": "none"}}',
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
