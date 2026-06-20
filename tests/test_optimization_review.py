"""Tests for post-diagnostic optimization review."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.optimization_review import (
    build_optimization_review,
    render_optimization_review_markdown,
    write_optimization_review,
)


class OptimizationReviewTests(unittest.TestCase):
    def test_review_summarizes_router_performance_risk_and_trading_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_review_artifacts(root)

            with patch("src.optimization_review.resolve_path", side_effect=lambda value: Path(value)):
                report = build_optimization_review(root)

            self.assertEqual(report["status"], "ready")
            self.assertTrue(report["diagnostics_ready"])
            self.assertEqual(report["strategy_mode"], "annual_state_router")
            self.assertAlmostEqual(report["performance"]["annual_return_delta"], 0.245)
            self.assertAlmostEqual(report["performance"]["drawdown_improvement"], 0.32)
            self.assertEqual(report["style_recognition"]["latest_source"], "beta20")
            self.assertEqual(report["style_recognition"]["source_counts"]["beta20"], 2)
            self.assertIn("low_position_count:3<5", report["risk_exposure"]["flags"])
            self.assertIn("high_industry_concentration:0.3754>0.35", report["risk_exposure"]["flags"])
            self.assertIn("small_cap_concentration:1.0000>0.80", report["risk_exposure"]["flags"])
            self.assertIn(
                "annual_trade_cost_ratio_above_target:0.2166>0.20",
                report["trading_constraints"]["flags"],
            )
            self.assertIn("Use annual_state_router", report["recommendations"][0])
            self.assertIn("Do not increase turnover", " ".join(report["recommendations"]))

    def test_review_blocks_readiness_when_diagnostics_or_auto_signal_are_not_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_review_artifacts(root)
            _write_json(root / "quant_diagnostic_report.json", {"optimization_ready": False})
            _write_json(root / "auto_run_status.json", {"is_executable": False, "strategy_mode": "annual_state_router"})

            with patch("src.optimization_review.resolve_path", side_effect=lambda value: Path(value)):
                report = build_optimization_review(root)

            self.assertEqual(report["status"], "review")
            self.assertFalse(report["diagnostics_ready"])
            self.assertFalse(report["auto_signal_executable"])

    def test_missing_auto_quality_blocks_ready_status(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_review_artifacts(root)
            (root / "auto_backtest_quality.json").unlink()

            with patch("src.optimization_review.resolve_path", side_effect=lambda value: Path(value)):
                report = build_optimization_review(root)

            self.assertEqual(report["status"], "review")
            self.assertFalse(report["auto_backtest_acceptable"])

    def test_writer_persists_json_and_markdown(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_review_artifacts(root)

            with patch("src.optimization_review.resolve_path", side_effect=lambda value: Path(value)):
                report = build_optimization_review(root)
                paths = write_optimization_review(report, root)

            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["markdown"]).exists())
            saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "ready")
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("Optimization Review", markdown)
            self.assertIn("Risk flags:", render_optimization_review_markdown(report))


def _write_review_artifacts(root: Path) -> None:
    _write_json(root / "quant_diagnostic_report.json", {"optimization_ready": True})
    _write_json(
        root / "backtest_metrics.json",
        {"annual_return": 0.018, "max_drawdown": -0.50, "sharpe": 0.19, "calmar": 0.04},
    )
    _write_json(
        root / "auto_backtest_metrics.json",
        {"annual_return": 0.263, "max_drawdown": -0.18, "sharpe": 1.61, "calmar": 1.49},
    )
    _write_json(root / "auto_backtest_quality.json", {"is_acceptable": True, "issues": []})
    _write_json(root / "auto_run_status.json", {"is_executable": True, "strategy_mode": "annual_state_router"})
    _write_json(
        root / "auto_research_diagnostics.json",
        {
            "exposure": {
                "latest_position_count": 3,
                "latest_max_industry_weight": 0.3754,
                "latest_top_position_weight": 0.34,
                "market_cap_buckets": [{"bucket": "small", "weight": 1.0}],
            },
            "turnover_attribution": {
                "annual_turnover_estimate": 9.0,
                "annual_turnover_without_rebalance_trims_estimate": 7.0,
                "rebalance_trim_cost_share_of_total_trade_cost": 0.12,
            },
            "cost_attribution": {"cost_drag_on_initial_equity": 0.11},
        },
    )
    pd.DataFrame(
        [
            {"year": 2025, "source": "beta", "reason": "default_beta", "exposure": 1.0},
            {"year": 2026, "source": "beta20", "reason": "moderate_low_beta20", "exposure": 0.4},
        ]
    ).to_csv(root / "auto_annual_state_router_year_routes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "date": "2026-01-31",
                "source": "beta20",
                "reason": "moderate_low_beta20",
                "top_n": 5,
                "max_turnover": 2,
                "rank_buffer": 8,
                "exposure": 0.4,
            },
            {
                "date": "2026-02-28",
                "source": "beta20",
                "reason": "moderate_low_beta20",
                "top_n": 5,
                "max_turnover": 2,
                "rank_buffer": 8,
                "exposure": 0.4,
            },
        ]
    ).to_csv(root / "auto_annual_state_router_score_routes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"year": 2025, "annual_trade_cost_ratio": 0.18},
            {"year": 2026, "annual_trade_cost_ratio": 0.2166},
        ]
    ).to_csv(root / "auto_backtest_yearly_breakdown.csv", index=False, encoding="utf-8-sig")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
