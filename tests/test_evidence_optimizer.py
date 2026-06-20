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

            self.assertEqual(plan["status"], "review")
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

    def test_candidate_research_diagnostics_override_stale_auto_risk_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root)
            _write_candidate_research_diagnostics(root, latest_position_count=4, top_industry_weight=0.26, small_weight=0.24)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "review")
            self.assertEqual(plan["risk_exposure"]["source_flags"], ["low_position_count:4<5"])
            self.assertIn("high_industry_concentration:0.3754>0.35", plan["risk_exposure"]["review_source_flags"])
            self.assertIsNone(plan["risk_exposure"]["max_industry_weight"])
            self.assertEqual(plan["risk_exposure"]["small_cap_action"], "monitor")
            self.assertEqual(plan["risk_exposure"]["target_min_positions"], 5)
            self.assertEqual(plan["risk_exposure"]["overlay_validation"]["status"], "not_required")
            self.assertEqual(plan["risk_exposure"]["risk_exit_refill_validation"]["status"], "not_tested")
            self.assertNotIn("--max-industry-weights", plan["next_commands"][0])
            self.assertIn("--rebalance-after-risk-exit-options false,true", plan["next_commands"][0])
            self.assertTrue(any("Do not add a hard industry cap" in item for item in plan["optimization_decisions"]))
            markdown = render_evidence_optimization_markdown(plan)
            self.assertIn("Selected candidate top industry weight: 26.00%", markdown)
            self.assertIn("Selected candidate small-cap weight: 24.00%", markdown)

    def test_grid_exposure_columns_prefer_risk_ready_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root)
            pd.DataFrame(
                [
                    {
                        "key": "higher-return-low-position",
                        "annual_return": 0.30,
                        "max_drawdown": -0.16,
                        "annual_trade_cost_ratio": 0.18,
                        "full_goal": True,
                        "turnover_mode": "rank10",
                        "latest_position_count": 4,
                        "latest_max_industry_weight": 0.25,
                        "market_cap_small_weight": 0.20,
                    },
                    {
                        "key": "risk-ready",
                        "annual_return": 0.27,
                        "max_drawdown": -0.17,
                        "annual_trade_cost_ratio": 0.17,
                        "full_goal": True,
                        "turnover_mode": "rank10",
                        "latest_position_count": 5,
                        "latest_max_industry_weight": 0.24,
                        "market_cap_small_weight": 0.20,
                    },
                ]
            ).to_csv(root / "sample_router_grid.csv", index=False, encoding="utf-8-sig")

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "ready")
            self.assertEqual(plan["style_recognition"]["candidate"]["key"], "risk-ready")
            self.assertEqual(plan["risk_exposure"]["status"], "ready")
            self.assertEqual(plan["risk_exposure"]["source_flags"], [])
            self.assertEqual(plan["candidate_evidence"]["risk_ready_cost_eligible_full_goal_rows"], 1)

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

    def test_failed_source_top_n_expansion_keeps_position_risk_in_review(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root, include_failed_source_top_n=True)
            _write_candidate_research_diagnostics(root, latest_position_count=4, top_industry_weight=0.26, small_weight=0.24)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "review")
            self.assertEqual(plan["risk_exposure"]["source_top_n_validation"]["status"], "fail")
            self.assertEqual(plan["risk_exposure"]["source_top_n_validation"]["candidate"]["beta_top_n"], 6)
            self.assertTrue(any("Reject the tested source top_n expansion" in item for item in plan["optimization_decisions"]))
            self.assertIn("source top_n expansion did not pass", " ".join(plan["caveats"]))

    def test_passing_risk_exit_min_position_guard_can_clear_position_risk(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root, include_passing_min_position_guard=True)
            _write_candidate_research_diagnostics(root, latest_position_count=4, top_industry_weight=0.26, small_weight=0.24)

            with patch("src.evidence_optimizer.resolve_path", side_effect=lambda value: Path(value)):
                plan = build_evidence_optimization_plan(root)

            self.assertEqual(plan["status"], "ready")
            self.assertEqual(plan["risk_exposure"]["risk_exit_min_positions_validation"]["status"], "pass")
            self.assertEqual(plan["risk_exposure"]["risk_exit_min_positions_validation"]["candidate"]["risk_exit_min_positions"], 5)
            self.assertTrue(any("Keep the tested risk-exit min-position guard" in item for item in plan["optimization_decisions"]))

    def test_refill_overlay_is_not_required_without_low_position_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            _write_evidence(root, low_position_flag=False, high_small_flags=False, include_failed_refill_overlay=True)

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
            self.assertEqual(saved["status"], "review")
            self.assertIn("Evidence Optimization Plan", render_evidence_optimization_markdown(plan))


def _write_evidence(
    root: Path,
    *,
    annual_trade_cost_ratio: float = 0.18,
    low_position_flag: bool = True,
    high_small_flags: bool = True,
    include_failed_risk_overlay: bool = False,
    include_failed_refill_overlay: bool = False,
    include_failed_source_top_n: bool = False,
    include_passing_min_position_guard: bool = False,
) -> None:
    risk_flags = []
    if low_position_flag:
        risk_flags.append("low_position_count:3<5")
    if high_small_flags:
        risk_flags.extend(
            [
                "high_industry_concentration:0.3754>0.35",
                "small_cap_concentration:1.0000>0.80",
            ]
        )
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
    if include_failed_source_top_n:
        rows.append(
            {
                "key": "source-top-n",
                "annual_return": 0.24,
                "max_drawdown": -0.21,
                "annual_trade_cost_ratio": 0.15,
                "full_goal": False,
                "turnover_mode": "rank10",
                "beta_top_n": 6,
                "failed_years": "2015,2016",
            }
        )
    if include_passing_min_position_guard:
        rows.append(
            {
                "key": "risk-exit-min-positions",
                "annual_return": 0.27,
                "max_drawdown": -0.16,
                "annual_trade_cost_ratio": 0.16,
                "full_goal": True,
                "turnover_mode": "rank10",
                "risk_exit_min_positions": 5,
            }
        )
    pd.DataFrame(rows).to_csv(root / "sample_router_grid.csv", index=False, encoding="utf-8-sig")


def _write_candidate_research_diagnostics(
    root: Path,
    *,
    latest_position_count: int,
    top_industry_weight: float,
    small_weight: float,
) -> None:
    _write_json(
        root / "evidence_optimized_router_hit_research_diagnostics.json",
        {
            "exposure": {
                "latest_position_count": latest_position_count,
                "latest_max_industry_weight": top_industry_weight,
                "latest_top_position_weight": top_industry_weight,
                "market_cap_buckets": [
                    {"bucket": "small", "weight": small_weight},
                    {"bucket": "mid", "weight": 1.0 - small_weight},
                ],
            }
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
