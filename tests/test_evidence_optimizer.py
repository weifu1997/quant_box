"""Tests for evidence-backed optimization planning."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.evidence_optimizer import (
    build_evidence_optimization_plan,
    render_evidence_optimization_markdown,
    write_evidence_optimization_plan,
)


class EvidenceOptimizerTests(unittest.TestCase):
    def test_plan_selects_full_goal_candidate_and_generates_constraints(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "ready")
            self.assertEqual(plan["style_recognition"]["status"], "ready")
            self.assertEqual(plan["risk_exposure"]["status"], "review")
            self.assertEqual(plan["trading_constraints"]["status"], "ready")
            candidate = plan["style_recognition"]["candidate"]
            self.assertEqual(candidate["turnover_mode"], "rank10")
            self.assertAlmostEqual(candidate["annual_return"], 0.26)
            self.assertEqual(plan["candidate_evidence"]["evaluated_grid_rows"], 2)
            self.assertEqual(plan["candidate_evidence"]["full_goal_rows"], 2)
            self.assertEqual(plan["candidate_evidence"]["cost_eligible_full_goal_rows"], 1)
            self.assertEqual(
                plan["candidate_evidence"]["best_rejected_candidate"]["reject_reason"],
                "trade_cost_above_target",
            )
            self.assertEqual(plan["risk_exposure"]["max_industry_weight"], 0.35)
            self.assertEqual(plan["risk_exposure"]["target_min_positions"], 5)
            self.assertEqual(plan["risk_exposure"]["small_cap_action"], "reduce_small_cap_concentration")
            self.assertEqual(plan["trading_constraints"]["turnover_action"], "do_not_increase_turnover")
            self.assertEqual(plan["trading_constraints"]["candidate_turnover_boost_max_turnover"], 2)
            self.assertTrue(
                any("Adopt the selected annual-state-router style candidate" in item for item in plan["optimization_decisions"])
            )
            self.assertTrue(any("Do not loosen turnover" in item for item in plan["optimization_decisions"]))
            self.assertIn("--max-industry-weights none,0.35", plan["next_commands"][0])
            self.assertIn("--rebalance-after-risk-exit-options false,true", plan["next_commands"][0])

    def test_plan_warns_when_no_cost_eligible_candidate_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root, annual_trade_cost_ratio=0.25)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "review")
            self.assertEqual(plan["style_recognition"]["candidate"], {})
            self.assertIn("No full-goal router grid candidate", " ".join(plan["caveats"]))

    def test_failed_risk_overlay_keeps_plan_in_review(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root, include_failed_risk_overlay=True, include_failed_refill_overlay=True)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "review")
            self.assertEqual(plan["risk_exposure"]["overlay_validation"]["status"], "fail")
            self.assertEqual(plan["risk_exposure"]["risk_exit_refill_validation"]["status"], "fail")
            self.assertTrue(any("Reject the tested hard max-industry-weight overlay" in item for item in plan["optimization_decisions"]))
            self.assertTrue(any("Reject same-day refill" in item for item in plan["optimization_decisions"]))
            self.assertIn("did not pass the full router gate", " ".join(plan["caveats"]))

    def test_refill_overlay_is_not_required_without_low_position_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root, low_position_flag=False, include_failed_refill_overlay=True)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "ready")
            self.assertIsNone(plan["risk_exposure"]["target_min_positions"])
            self.assertEqual(plan["risk_exposure"]["risk_exit_refill_validation"]["status"], "not_required")

    def test_writer_persists_json_and_markdown(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)
                paths = write_evidence_optimization_plan(plan, root)

            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["markdown"]).exists())
            saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "ready")
            self.assertIn("Evidence Optimization Plan", render_evidence_optimization_markdown(plan))


def _write_evidence(
    root: Path,
    *,
    annual_trade_cost_ratio: float = 0.18,
    low_position_flag: bool = True,
    include_failed_risk_overlay: bool = False,
    include_failed_refill_overlay: bool = False,
) -> None:
    risk_flags = [
        "high_industry_concentration:0.3754>0.35",
        "small_cap_concentration:1.0000>0.80",
    ]
    if low_position_flag:
        risk_flags.insert(0, "low_position_count:3<5")
    _write_json(
        root / "optimization_review.json",
        {
            "status": "ready",
            "strategy_mode": "annual_state_router",
            "risk_exposure": {
                "flags": risk_flags
            },
            "trading_constraints": {
                "flags": ["annual_trade_cost_ratio_above_target:0.9497>0.20"]
            },
        },
    )
    _write_json(
        root / "auto_selected_params.json",
        {
            "strategy_mode": "annual_state_router",
            "initial_source": "beta",
            "turnover_mode": "rank10",
            "turnover_boost_reasons": ["low_vol_moderate_uptrend", "moderate_positive_roc60"],
        },
    )
    rows = [
        {
            "key": "bad",
            "annual_return": 0.30,
            "max_drawdown": -0.19,
            "annual_trade_cost_ratio": 0.40,
            "full_goal": True,
            "turnover_mode": "full",
        },
        {
            "key": "good",
            "annual_return": 0.26,
            "max_drawdown": -0.17,
            "annual_trade_cost_ratio": annual_trade_cost_ratio,
            "full_goal": annual_trade_cost_ratio <= 0.20,
            "missing_ret252_exposure": 0.7,
            "strong_trailing_exposure": 0.8,
            "moderate_positive_source": "roc60",
            "moderate_positive_ret252_min": 0.2,
            "moderate_low_source": "beta20",
            "moderate_low_ret252_min": 0.18,
            "moderate_low_ret252_max": 0.2,
            "moderate_low_exposure": 0.4,
            "turnover_mode": "rank10",
            "turnover_boost_reasons": "low_vol_moderate_uptrend+moderate_positive_roc60",
            "turnover_boost_max_turnover": 2,
            "turnover_boost_rank_buffer": 10,
        },
    ]
    if include_failed_risk_overlay:
        rows.append(
            {
                "key": "risk-overlay",
                "annual_return": 0.28,
                "max_drawdown": -0.22,
                "annual_trade_cost_ratio": 0.25,
                "full_goal": False,
                "turnover_mode": "rank10",
                "max_industry_weight": 0.35,
            }
        )
    if include_failed_refill_overlay:
        rows.append(
            {
                "key": "risk-refill",
                "annual_return": 0.27,
                "max_drawdown": -0.26,
                "annual_trade_cost_ratio": 0.28,
                "full_goal": False,
                "turnover_mode": "rank10",
                "rebalance_after_risk_exit": True,
            }
        )
    pd.DataFrame(rows).to_csv(root / "sample_router_grid.csv", index=False, encoding="utf-8-sig")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
