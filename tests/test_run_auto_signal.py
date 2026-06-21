"""模块说明：覆盖 test_run_auto_signal 相关行为的测试用例。"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


class RunAutoSignalTests(unittest.TestCase):
    """类说明：组织 RunAutoSignalTests 测试用例。"""
    def test_validation_progress_message_includes_window_and_params(self) -> None:
        """函数说明：验证 test_validation_progress_message_includes_window_and_params 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")

        message = module._validation_progress_message(
            {
                "test_start": pd.Timestamp("2024-01-01"),
                "test_end": pd.Timestamp("2024-12-31"),
                "factor_group": "momentum",
                "top_n": 7,
                "rebalance_freq": "monthly",
            },
            3,
        )

        self.assertIn("3 results", message)
        self.assertIn("2024-01-01..2024-12-31", message)
        self.assertIn("factor_group=momentum", message)

    def test_signal_output_date_infers_latest_factor_date_for_empty_signal(self) -> None:
        """函数说明：验证 test_signal_output_date_infers_latest_factor_date_for_empty_signal 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [1.0, 2.0]}, index=index)

        output_date = module._signal_output_date(pd.DataFrame(columns=["date", "instrument", "action"]), "latest", factors=factors)

        self.assertEqual(output_date, "2024-01-03")

    def test_backtest_stage_skip_writes_metrics_and_quality(self) -> None:
        """Verify the extracted backtest stage keeps skip-backtest artifacts stable."""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            prices = pd.read_parquet(config["ic"]["price_file"])
            args = SimpleNamespace(skip_backtest=True, allow_low_quality=False, start_date="2024-01-01")
            status = module._new_status()
            artifacts: list[Path] = []

            result = module._run_backtest_stage(args, config, factors, prices, "2024-01-04", root, status, artifacts)

            metrics = json.loads((root / "auto_backtest_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["backtest_skipped"])
            self.assertTrue((root / "auto_backtest_quality.json").exists())
            self.assertEqual(result.research_diagnostics["issues"], ["backtest_skipped"])
            self.assertIn(root / "auto_backtest_metrics.json", artifacts)
            self.assertEqual([stage["state"] for stage in status["stages"] if stage["name"] == "backtest"], ["skipped"])

    def test_annual_state_router_quality_uses_formal_evidence(self) -> None:
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            years_path = root / "router_years.csv"
            pd.DataFrame(
                [
                    {"year": 2023, "annual_return": 0.22, "max_drawdown": -0.10},
                    {"year": 2024, "annual_return": 0.24, "max_drawdown": -0.12},
                    {"year": 2025, "annual_return": 0.21, "max_drawdown": -0.11},
                ]
            ).to_csv(years_path, index=False)
            metrics_path = root / "router_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "annual_return": 0.25,
                            "sharpe": 1.2,
                            "max_drawdown": -0.12,
                            "annual_turnover": 5.0,
                            "annual_trade_cost_ratio": 0.02,
                        },
                        "audit": {"year_count": 3, "min_yearly_annual_return": 0.21, "worst_yearly_drawdown": -0.12},
                        "full_gate": {"is_full_goal_met": True},
                        "combo": {
                            "missing_ret252_exposure": 0.7,
                            "strong_trailing_exposure": 0.8,
                            "moderate_positive_source": "roc60",
                            "moderate_positive_ret252_min": 0.2,
                            "moderate_positive_exposure": 1.0,
                            "moderate_low_source": "beta20",
                            "moderate_low_ret252_min": 0.18,
                            "moderate_low_ret252_max": 0.2,
                            "moderate_low_exposure": 0.4,
                            "moderate_lower_source": None,
                            "moderate_lower_ret252_min": 0.16,
                            "moderate_lower_ret252_max": 0.18,
                            "moderate_lower_exposure": 1.0,
                            "turnover_boost_reasons": "low_vol_moderate_uptrend+moderate_positive_roc60",
                            "turnover_boost_max_turnover": 2,
                            "turnover_boost_rank_buffer": 10,
                            "risk_exit_min_positions": 5,
                            "risk_exit_min_positions_reasons": "default_beta",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "annual_state_router": {
                    "enabled": True,
                    "missing_ret252_exposure": 0.7,
                    "strong_trailing_exposure": 0.8,
                    "moderate_positive_source": "roc60",
                    "moderate_positive_ret252_min": 0.2,
                    "moderate_positive_exposure": 1.0,
                    "moderate_low_source": "beta20",
                    "moderate_low_ret252_min": 0.18,
                    "moderate_low_ret252_max": 0.2,
                    "moderate_low_exposure": 0.4,
                    "moderate_lower_source": None,
                    "moderate_lower_ret252_min": 0.16,
                    "moderate_lower_ret252_max": 0.18,
                    "moderate_lower_exposure": 1.0,
                    "turnover_boost_reasons": ["low_vol_moderate_uptrend", "moderate_positive_roc60"],
                    "turnover_boost_max_turnover": 2,
                    "turnover_boost_rank_buffer": 10,
                    "risk_exit_min_positions": 5,
                    "risk_exit_min_positions_reasons": ["default_beta"],
                    "evidence_metrics_file": str(metrics_path),
                    "evidence_years_file": str(years_path),
                }
            }
            quality = {
                "min_validation_windows": 3,
                "min_positive_return_rate": 0.5,
                "min_optimizer_annual_return": 0.20,
                "max_drawdown_limit": -0.20,
                "max_annual_turnover": 20.0,
                "max_annual_trade_cost_ratio": 0.2,
            }

            report = module._annual_state_router_quality(config, quality)

            self.assertTrue(report.is_acceptable)
            self.assertEqual(report.windows, 3)
            self.assertAlmostEqual(report.annual_return_min, 0.21)

            config["annual_state_router"]["risk_exit_min_positions_reasons"] = ["moderate_low_beta20"]
            mismatched = module._annual_state_router_quality(config, quality)

            self.assertFalse(mismatched.is_acceptable)
            self.assertIn(
                "annual_state_router_evidence_combo_mismatch:risk_exit_min_positions_reasons",
                mismatched.issues,
            )

    def test_backtest_stage_uses_annual_state_router_scores_when_enabled(self) -> None:
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            config["annual_state_router"] = {"enabled": True}
            prices = pd.read_parquet(config["ic"]["price_file"])
            score_index = pd.MultiIndex.from_product([[pd.Timestamp("2024-01-03")], ["A", "B"]], names=["date", "instrument"])
            routed_scores = pd.Series([2.0, 1.0], index=score_index, name="score")
            routed = module.RoutedScoreRun(
                scores=routed_scores,
                score_routes=pd.DataFrame(
                    [{"date": "2024-01-03", "source": "beta", "top_n": 2, "max_turnover": 2, "rank_buffer": 0}]
                ),
                year_routes=pd.DataFrame([{"decision_date": "2024-01-03", "exposure": 1.0}]),
            )
            runtime = module.AnnualStateRouterRuntime(
                routed=routed,
                source_definitions={},
                backtest_config={"initial_capital": 100000, "top_n": 2, "selection_schedule": {"2024-01-03": {"top_n": 2}}},
                files={},
            )
            result = module.BacktestResult(
                equity_curve=pd.Series([100000.0, 130000.0], index=pd.to_datetime(["2024-01-03", "2024-01-04"]), name="equity"),
                holdings=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={"annual_return": 0.30, "max_drawdown": -0.10, "calmar": 3.0},
            )
            args = SimpleNamespace(skip_backtest=False, allow_low_quality=False, start_date="2024-01-01")
            status = module._new_status()
            artifacts: list[Path] = []

            with patch.object(module, "_build_annual_state_router_runtime", return_value=runtime), patch.object(
                module,
                "build_strategy_scores",
            ) as legacy_scores, patch.object(module, "run_backtest", return_value=result) as backtest, patch.object(
                module,
                "build_research_diagnostics",
                return_value=({"enabled": False}, {}),
            ), patch.object(module, "write_research_diagnostics", return_value={}):
                stage = module._run_backtest_stage(args, config, factors, prices, "2024-01-04", root, status, artifacts)

            legacy_scores.assert_not_called()
            self.assertIs(stage.annual_state_router, runtime)
            self.assertIs(backtest.call_args.args[0], routed_scores)
            self.assertEqual(backtest.call_args.args[4]["selection_schedule"], {"2024-01-03": {"top_n": 2}})

    def test_skip_optimize_defaults_to_candidate_outputs(self) -> None:
        """函数说明：验证 test_skip_optimize_defaults_to_candidate_outputs 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")

            with _patched_auto_run(module, config, factors, ["run_auto_signal.py", "--skip-update", "--skip-convert", "--skip-optimize", "--skip-backtest", "--no-archive"]):
                module.main()

            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertTrue((root / "manual_orders_candidate_2024-01-03.csv").exists())
            self.assertTrue((root / "data_governance_report.json").exists())
            self.assertTrue((root / "order_confirmations" / "order_confirmation_candidate_2024-01-03.csv").exists())
            self.assertTrue((root / "fill_feedback" / "fill_feedback_candidate_2024-01-03.csv").exists())
            self.assertEqual(latest.read_text(encoding="utf-8"), "instrument\nOLD.SZ\n")
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "blocked")
            self.assertIn("params:parameter_validation_skipped", status["block_reasons"])
            json_report = json.loads((root / "auto_signal_report.json").read_text(encoding="utf-8"))
            self.assertEqual(json_report["previous_holdings_source"], "outputs.holdings_file")
            report = (root / "daily_signal_report.md").read_text(encoding="utf-8")
            self.assertIn("## Data Governance", report)
            self.assertIn("## Execution Loop", report)

    def test_auto_signal_reuses_factor_cache_by_default(self) -> None:
        """函数说明：验证自动流程默认不强制重算因子缓存。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)

            with _patched_auto_run(
                module,
                config,
                factors,
                ["run_auto_signal.py", "--skip-update", "--skip-convert", "--skip-optimize", "--skip-backtest", "--no-archive"],
            ) as patches:
                module.main()

            kwargs = patches["load_or_compute_factors"].call_args.kwargs
            self.assertFalse(kwargs["force"])

    def test_auto_signal_report_includes_optional_fundamental_screen(self) -> None:
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            fundamental_summary = {
                "enabled": True,
                "status": "ok",
                "as_of_date": "2024-01-03",
                "rows": 2,
                "covered_rows": 1,
                "passed": 0,
                "watch": 1,
                "fundamental_coverage": "50.00%",
                "dividend_coverage": "50.00%",
                "top_watch": [{"ts_code": "000001.SZ", "name": "Alpha Bank", "industry": "Bank", "total_score": 4}],
            }
            fundamental_files = {"fundamental_screen_report": str(root / "fundamental_screen_report.md")}

            with _patched_auto_run(
                module,
                config,
                factors,
                ["run_auto_signal.py", "--skip-update", "--skip-convert", "--skip-optimize", "--skip-backtest", "--no-archive"],
            ), patch.object(module, "_maybe_build_fundamental_screen", return_value=(fundamental_summary, fundamental_files)):
                module.main()

            json_report = json.loads((root / "auto_signal_report.json").read_text(encoding="utf-8"))
            self.assertEqual(json_report["fundamental_screen"]["watch"], 1)
            self.assertEqual(json_report["files"]["fundamental_screen_report"], str(root / "fundamental_screen_report.md"))
            markdown_report = (root / "daily_signal_report.md").read_text(encoding="utf-8")
            self.assertIn("## Fundamental Screen", markdown_report)
            self.assertIn("- Covered rows: 1/2", markdown_report)

    def test_empty_latest_signal_uses_factor_date_for_candidate_outputs(self) -> None:
        """函数说明：验证 test_empty_latest_signal_uses_factor_date_for_candidate_outputs 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\n", encoding="utf-8")
            empty_signal = pd.DataFrame(columns=["date", "instrument", "action"])

            with _patched_auto_run(
                module,
                config,
                factors,
                ["run_auto_signal.py", "--skip-update", "--skip-convert", "--skip-optimize", "--skip-backtest", "--no-archive"],
            ), patch.object(module, "generate_signal", return_value=(empty_signal, [])):
                module.main()

            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertFalse((root / "candidate_signal_latest.csv").exists())
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "blocked")

    def test_skip_optimize_uses_validated_strategy_evidence(self) -> None:
        """函数说明：验证 test_skip_optimize_uses_validated_strategy_evidence 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            (root / "current_holdings.csv").write_text("instrument,shares\n", encoding="utf-8")
            years_path = root / "validated_years.csv"
            pd.DataFrame(
                [
                    {"year": 2022, "annual_return": 0.22, "max_drawdown": -0.10},
                    {"year": 2023, "annual_return": 0.24, "max_drawdown": -0.12},
                    {"year": 2024, "annual_return": 0.21, "max_drawdown": -0.18},
                ]
            ).to_csv(years_path, index=False)
            summary_path = root / "validated_summary.csv"
            pd.DataFrame(
                [
                    {
                        "candidate": "validated_candidate",
                        "annual_return": 0.21,
                        "sharpe": 1.1,
                        "max_drawdown": -0.18,
                        "annual_turnover": 5.0,
                        "annual_trade_cost_ratio": 0.02,
                        "is_acceptable": True,
                        "years_path": str(years_path),
                    }
                ]
            ).to_csv(summary_path, index=False)
            config["validated_strategy"] = {
                "enabled": True,
                "candidate": "validated_candidate",
                "summary_file": str(summary_path),
                "require_is_acceptable": True,
            }
            config["quality"] = {
                "min_validation_windows": 3,
                "min_positive_return_rate": 0.5,
                "min_optimizer_annual_return": 0.20,
                "max_drawdown_limit": -0.20,
                "min_backtest_annual_return": 0.20,
                "max_backtest_drawdown_limit": -0.20,
                "max_annual_turnover": 20.0,
                "max_annual_trade_cost_ratio": 0.2,
            }
            good_result = module.BacktestResult(
                equity_curve=pd.Series(
                    [100000.0, 130000.0],
                    index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
                    name="equity",
                ),
                holdings=pd.DataFrame([{"date": "2024-01-04", "instrument": "E", "value": 100000.0}]),
                trades=pd.DataFrame(),
                metrics={"annual_return": 0.30, "max_drawdown": -0.10, "calmar": 3.0},
            )

            with _patched_auto_run(
                module,
                config,
                factors,
                ["run_auto_signal.py", "--skip-update", "--skip-convert", "--skip-optimize", "--no-archive"],
            ), patch.object(module, "run_backtest", return_value=good_result):
                module.main()

            self.assertTrue((root / "signal_2024-01-03.csv").exists())
            self.assertTrue((root / "manual_orders_2024-01-03.csv").exists())
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")
            quality = json.loads((root / "auto_parameter_quality.json").read_text(encoding="utf-8"))
            self.assertTrue(quality["is_acceptable"])
            self.assertEqual(quality["windows"], 3)

    def test_allow_low_quality_keeps_skip_optimize_run_as_candidate_outputs(self) -> None:
        """函数说明：验证 test_allow_low_quality_keeps_skip_optimize_run_as_candidate_outputs 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            (root / "current_holdings.csv").write_text("instrument,shares\n", encoding="utf-8")

            argv = [
                "run_auto_signal.py",
                "--skip-update",
                "--skip-convert",
                "--skip-optimize",
                "--skip-backtest",
                "--allow-low-quality",
                "--no-archive",
            ]
            with _patched_auto_run(module, config, factors, argv):
                module.main()

            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertTrue((root / "manual_orders_candidate_2024-01-03.csv").exists())
            self.assertFalse((root / "signal_2024-01-03.csv").exists())
            self.assertEqual(latest.read_text(encoding="utf-8"), "instrument\nOLD.SZ\n")
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            report = json.loads((root / "auto_signal_report.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "blocked")
            self.assertFalse(status["is_executable"])
            self.assertEqual(status["block_reasons"], status["quality_warnings"])
            self.assertEqual(report["block_reasons"], report["quality_warnings"])
            self.assertTrue(status["block_reasons"])

    def test_force_official_promotes_allowed_low_quality_run(self) -> None:
        """函数说明：验证 test_force_official_promotes_allowed_low_quality_run 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            (root / "current_holdings.csv").write_text("instrument,shares\n", encoding="utf-8")

            argv = [
                "run_auto_signal.py",
                "--skip-update",
                "--skip-convert",
                "--skip-optimize",
                "--skip-backtest",
                "--allow-low-quality",
                "--force-official",
                "--no-archive",
            ]
            with _patched_auto_run(module, config, factors, argv):
                module.main()

            self.assertTrue((root / "signal_2024-01-03.csv").exists())
            self.assertTrue((root / "manual_orders_2024-01-03.csv").exists())
            self.assertTrue((root / "order_confirmations" / "order_confirmation_2024-01-03.csv").exists())
            self.assertTrue((root / "fill_feedback" / "fill_feedback_2024-01-03.csv").exists())
            latest_frame = pd.read_csv(latest)
            self.assertEqual(latest_frame["instrument"].tolist(), ["E"])

    def test_update_partial_stage_is_not_marked_complete(self) -> None:
        """函数说明：验证 test_update_partial_stage_is_not_marked_complete 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            update_result = SimpleNamespace(
                to_status_dict=lambda: {
                    "status": "partial",
                    "written_symbols": 1,
                    "failed_symbols": 0,
                    "remaining_symbols": 2,
                    "progress_path": str(root / "progress.json"),
                    "last_error": "",
                }
            )
            argv = [
                "run_auto_signal.py",
                "--skip-convert",
                "--skip-optimize",
                "--skip-backtest",
                "--no-archive",
            ]

            with _patched_auto_run(module, config, factors, argv), patch.object(
                module,
                "update_daily_data_resumable",
                return_value=update_result,
            ):
                module.main()

            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            update_stages = [stage for stage in status["stages"] if stage["name"] == "update_data"]
            self.assertEqual(update_stages[-1]["state"], "partial")
            self.assertIn("remaining=2", update_stages[-1]["message"])

    def test_update_error_fails_run_without_allow_unhealthy(self) -> None:
        """函数说明：验证 test_update_error_fails_run_without_allow_unhealthy 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            update_result = SimpleNamespace(
                to_status_dict=lambda: {
                    "status": "error",
                    "written_symbols": 0,
                    "failed_symbols": 1,
                    "remaining_symbols": 1,
                    "last_error": "000001.SZ:not_written",
                }
            )
            argv = [
                "run_auto_signal.py",
                "--skip-convert",
                "--skip-optimize",
                "--skip-backtest",
                "--no-archive",
            ]

            with _patched_auto_run(module, config, factors, argv), patch.object(
                module,
                "update_daily_data_resumable",
                return_value=update_result,
            ):
                with self.assertRaisesRegex(RuntimeError, "Data update failed"):
                    module.main()

            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            update_stages = [stage for stage in status["stages"] if stage["name"] == "update_data"]
            self.assertEqual(update_stages[-1]["state"], "error")
            self.assertIn("000001.SZ:not_written", status["last_error"])

    def test_data_governance_issues_block_official_outputs_even_with_force_official(self) -> None:
        """函数说明：验证 test_data_governance_issues_block_official_outputs_even_with_force_official 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            (root / "current_holdings.csv").write_text("instrument,shares\n", encoding="utf-8")
            governance = _governance_report(ready=False, issues=["daily_basic_start_after_point_in_time_start:2026-05-06>2015-01-01"])
            argv = [
                "run_auto_signal.py",
                "--skip-update",
                "--skip-convert",
                "--skip-optimize",
                "--skip-backtest",
                "--allow-low-quality",
                "--force-official",
                "--no-archive",
            ]

            with _patched_auto_run(module, config, factors, argv, governance=governance):
                module.main()

            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertTrue((root / "manual_orders_candidate_2024-01-03.csv").exists())
            self.assertFalse((root / "signal_2024-01-03.csv").exists())
            self.assertEqual(latest.read_text(encoding="utf-8"), "instrument\nOLD.SZ\n")
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "blocked")
            self.assertIn("governance:daily_basic_start_after_point_in_time_start:2026-05-06>2015-01-01", status["block_reasons"])
            report = (root / "daily_signal_report.md").read_text(encoding="utf-8")
            self.assertIn("Repair action: daily_basic", report)
            self.assertIn("scripts\\run_update_point_in_time_data.py", report)

    def test_backtest_quality_blocks_official_outputs(self) -> None:
        """函数说明：验证 test_backtest_quality_blocks_official_outputs 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            config["ic"].update({"horizon": 3, "method": "pearson", "min_obs": 4})
            config["quality"] = {
                "min_validation_windows": 3,
                "min_positive_return_rate": 0.5,
                "min_optimizer_annual_return": 0.20,
                "max_drawdown_limit": -0.20,
                "min_backtest_annual_return": 0.20,
                "max_backtest_drawdown_limit": -0.20,
            }
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            validation = pd.DataFrame(
                [
                    {
                        "factor_group": "momentum",
                        "top_n": 1,
                        "max_turnover": 1,
                        "rank_buffer": 0,
                        "rebalance_freq": "daily",
                        "rebalance_drift_threshold": 0.0,
                        "optimization_score": 1.0,
                        "annual_return": 0.30,
                        "sharpe": 1.2,
                        "max_drawdown": -0.10,
                        "annual_turnover": 1.0,
                        "annual_trade_cost_ratio": 0.01,
                    }
                ]
                * 3
            )
            bad_result = module.BacktestResult(
                equity_curve=pd.Series([100000.0, 90000.0], index=pd.to_datetime(["2024-01-03", "2024-01-04"]), name="equity"),
                holdings=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={"annual_return": 0.19, "max_drawdown": -0.30, "calmar": 1.0},
            )

            with _patched_auto_run(
                module,
                config,
                factors,
                ["run_auto_signal.py", "--skip-update", "--skip-convert", "--no-archive"],
            ), patch.object(module, "run_walk_forward_grid_validation", return_value=validation) as validate, patch.object(
                module,
                "run_backtest",
                return_value=bad_result,
            ):
                module.main()

            kwargs = validate.call_args.kwargs
            self.assertEqual(kwargs["ic_horizon"], 3)
            self.assertEqual(kwargs["ic_method"], "pearson")
            self.assertEqual(kwargs["ic_min_obs"], 4)
            self.assertTrue(callable(kwargs["on_result"]))
            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertFalse((root / "signal_2024-01-03.csv").exists())
            self.assertEqual(latest.read_text(encoding="utf-8"), "instrument\nOLD.SZ\n")
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "blocked")
            self.assertTrue(any(reason.startswith("backtest:backtest_max_drawdown_worse_than_limit") for reason in status["block_reasons"]))
            report = (root / "daily_signal_report.md").read_text(encoding="utf-8")
            self.assertIn("## Backtest Quality", report)
            self.assertIn("## Research Diagnostics", report)
            self.assertIn("## Failure Analysis", report)
            self.assertIn("Parameter/backtest mismatch: True", report)
            self.assertIn("failure scope: cross_window_full_history", report)
            failure = json.loads((root / "auto_failure_analysis.json").read_text(encoding="utf-8"))
            self.assertTrue(failure["parameter_backtest_mismatch"])
            self.assertEqual(failure["failure_scope_summary"]["primary_scope"], "cross_window_full_history")
            self.assertAlmostEqual(failure["backtest_threshold_gaps"]["annual_return_gap"], -0.01)
            self.assertEqual(failure["drawdown_summary"]["trough_date"], "2024-01-04")
            self.assertAlmostEqual(failure["drawdown_summary"]["strategy_return_peak_to_trough"], -0.1)
            comparison = pd.read_csv(root / "auto_validation_vs_backtest.csv")
            self.assertIn("failure_driver", comparison.columns)
            self.assertIn("validation_window", comparison["row_type"].tolist())
            self.assertIn("backtest_segment", comparison["row_type"].tolist())
            self.assertIn("max_drawdown", ",".join(comparison["failure_driver"].astype(str).tolist()))
            yearly = pd.read_csv(root / "auto_backtest_yearly_breakdown.csv")
            self.assertEqual(yearly["year"].tolist(), [2024])

    def test_auto_signal_writes_partial_validation_outputs_on_optimizer_timeout(self) -> None:
        """函数说明：验证优化超时时自动流程会保留部分验证结果。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            partial_validation = pd.DataFrame(
                [
                    {
                        "factor_group": "momentum",
                        "top_n": 1,
                        "max_turnover": 1,
                        "rank_buffer": 0,
                        "rebalance_freq": "daily",
                        "rebalance_drift_threshold": 0.02,
                        "optimization_score": 1.0,
                        "annual_return": 0.10,
                        "sharpe": 1.0,
                        "max_drawdown": -0.05,
                        "annual_turnover": 1.0,
                        "annual_trade_cost_ratio": 0.01,
                    }
                ]
            )
            timeout = module.OptimizationTimeoutError(
                "Walk-forward grid validation timed out after 0 completed windows and 1 completed combinations.",
                partial_results=partial_validation,
                completed_windows=0,
                completed_combinations=1,
            )

            with self.assertRaises(module.OptimizationTimeoutError):
                with _patched_auto_run(
                    module,
                    config,
                    factors,
                    [
                        "run_auto_signal.py",
                        "--skip-update",
                        "--skip-convert",
                        "--skip-backtest",
                        "--no-archive",
                        "--optimize-timeout-seconds",
                        "30",
                        "--max-optimize-combinations",
                        "8",
                    ],
                ), patch.object(module, "run_walk_forward_grid_validation", side_effect=timeout) as validate:
                    module.main()

            kwargs = validate.call_args.kwargs
            self.assertEqual(kwargs["timeout_seconds"], 30)
            self.assertEqual(kwargs["max_grid_combinations"], 8)
            validation = pd.read_csv(root / "auto_validation_windows.csv")
            self.assertEqual(len(validation), 1)
            summary = pd.read_csv(root / "auto_parameter_summary.csv")
            self.assertEqual(len(summary), 1)
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["optimizer_timeout"]["completed_combinations"], 1)
            timeout_stages = [stage for stage in status["stages"] if stage["name"] == "optimize_params" and stage["state"] == "timeout"]
            self.assertEqual(len(timeout_stages), 1)

    def test_no_acceptable_optimized_params_falls_back_to_current_config(self) -> None:
        """函数说明：验证 test_no_acceptable_optimized_params_falls_back_to_current_config 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
            config["strategy"].update({"factor_group": "momentum", "top_n": 1, "rank_buffer": 0})
            config["quality"] = {
                "min_validation_windows": 3,
                "min_positive_return_rate": 0.5,
                "min_optimizer_annual_return": 0.20,
                "max_drawdown_limit": -0.20,
                "min_backtest_annual_return": 0.20,
                "max_backtest_drawdown_limit": -0.20,
            }
            validation = pd.DataFrame(
                [
                    {
                        "factor_group": "factor:LOW0",
                        "top_n": 10,
                        "max_turnover": 1,
                        "rank_buffer": 30,
                        "rebalance_freq": "monthly",
                        "rebalance_drift_threshold": 0.02,
                        "optimization_score": 10.0,
                        "annual_return": 0.30,
                        "sharpe": 2.0,
                        "max_drawdown": -0.35,
                        "annual_turnover": 1.0,
                        "annual_trade_cost_ratio": 0.01,
                    }
                ]
                * 3
            )
            result = module.BacktestResult(
                equity_curve=pd.Series([100000.0, 130000.0], index=pd.to_datetime(["2024-01-03", "2024-01-04"]), name="equity"),
                holdings=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={"annual_return": 0.30, "max_drawdown": -0.10, "calmar": 3.0},
            )

            with _patched_auto_run(
                module,
                config,
                factors,
                ["run_auto_signal.py", "--skip-update", "--skip-convert", "--no-archive"],
            ), patch.object(module, "run_walk_forward_grid_validation", return_value=validation), patch.object(
                module,
                "run_backtest",
                return_value=result,
            ) as backtest:
                module.main()

            bt_config = backtest.call_args.args[4]
            self.assertEqual(bt_config["factor_group"], "momentum")
            self.assertEqual(bt_config["top_n"], 1)
            selected = json.loads((root / "auto_selected_params.json").read_text(encoding="utf-8"))
            self.assertEqual(selected, {})
            quality = json.loads((root / "auto_parameter_quality.json").read_text(encoding="utf-8"))
            self.assertIn("no_acceptable_params", quality["issues"])
            report = json.loads((root / "auto_signal_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["selected_params_status"], "no_acceptable_params")

    def test_promote_candidate_writes_official_signal_and_latest_holdings(self) -> None:
        """函数说明：验证 test_promote_candidate_writes_official_signal_and_latest_holdings 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            pd.DataFrame([{"date": "2024-01-03", "instrument": "E", "action": "BUY"}]).to_csv(
                root / "candidate_signal_2024-01-03.csv",
                index=False,
            )
            pd.DataFrame({"instrument": ["E"]}).to_csv(root / "candidate_holdings_2024-01-03.csv", index=False)
            config = {
                "outputs": {"dir": str(root), "holdings_file": str(latest)},
                "data": {"start_date": "2024-01-01", "end_date": "2024-01-03"},
            }

            with patch.object(sys, "argv", ["run_auto_signal.py", "--promote-candidate", "2024-01-03"]), patch(
                "scripts.run_auto_signal.load_config",
                return_value=config,
            ):
                module.main()

            self.assertTrue((root / "signal_2024-01-03.csv").exists())
            latest_frame = pd.read_csv(latest)
            self.assertEqual(latest_frame["instrument"].tolist(), ["E"])


def _auto_config_and_factors(root: Path) -> tuple[dict, pd.DataFrame]:
    """函数说明：处理 auto_config_and_factors 的内部辅助逻辑。"""
    price_path = root / "prices.parquet"
    dates = pd.to_datetime(["2024-01-03", "2024-01-04"])
    prices = pd.concat(
        {
            "close": pd.DataFrame({code: [10.0 + idx, 11.0 + idx] for idx, code in enumerate(["A", "B", "C", "D", "E"])}, index=dates),
            "open": pd.DataFrame({code: [10.0 + idx, 11.0 + idx] for idx, code in enumerate(["A", "B", "C", "D", "E"])}, index=dates),
            "volume": pd.DataFrame({code: [1000.0, 1000.0] for code in ["A", "B", "C", "D", "E"]}, index=dates),
        },
        axis=1,
    )
    prices.to_parquet(price_path)
    factor_index = pd.MultiIndex.from_product([[pd.Timestamp("2024-01-03")], ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
    factors = pd.DataFrame({"ROC5": [1, 2, 3, 4, 5]}, index=factor_index)
    config = {
        "data": {"start_date": "2024-01-01", "end_date": "2024-01-03", "target_date_cutoff_time": "20:00", "timezone": "Asia/Shanghai"},
        "strategy": {"factor_group": "momentum", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
        "factors": {"cache_file": str(root / "factors.parquet")},
        "ic": {"price_file": str(price_path)},
        "outputs": {"dir": str(root), "holdings_file": str(root / "latest_holdings.csv")},
        "quality": {},
        "account": {"current_holdings_file": str(root / "current_holdings.csv"), "total_asset": 100000, "cash": 10000},
        "reports": {"history_dir": str(root / "history")},
        "backtest": {"initial_capital": 100000},
    }
    return config, factors


@contextmanager
def _patched_auto_run(module, config: dict, factors: pd.DataFrame, argv: list[str], governance=None):
    """函数说明：处理 patched_auto_run 的内部辅助逻辑。"""
    health = SimpleNamespace(is_healthy=True, issues=[], to_dict=lambda: {"is_healthy": True, "issues": []})
    adj_meta = SimpleNamespace(files_with_adj_factor=1, raw_file_count=1)
    governance = governance or _governance_report(ready=True, issues=[])
    root = Path(config["outputs"]["dir"])
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(module, "load_config", return_value=config))
        load_factors = stack.enter_context(patch.object(module, "load_or_compute_factors", return_value=factors))
        stack.enter_context(patch.object(module, "build_data_health_report", return_value=health))
        stack.enter_context(patch.object(module, "write_data_health_report", side_effect=lambda *args, **kwargs: _write_health_files(root)))
        stack.enter_context(patch.object(module, "build_adj_factor_metadata", return_value=adj_meta))
        stack.enter_context(patch.object(module, "write_adj_factor_metadata", side_effect=lambda *args, **kwargs: _write_adj_meta_file(root)))
        stack.enter_context(patch.object(module, "build_data_governance_report", return_value=governance))
        stack.enter_context(patch.object(module, "write_data_governance_report", side_effect=lambda report, *_args, **_kwargs: _write_governance_file(root, report)))
        yield {"load_or_compute_factors": load_factors}


def _write_health_files(root: Path) -> tuple[Path, Path]:
    """函数说明：写入 write_health_files 的内部辅助逻辑。"""
    json_path = root / "data_health_report.json"
    csv_path = root / "data_health_report.csv"
    json_path.write_text('{"is_healthy": true, "issues": []}', encoding="utf-8")
    csv_path.write_text("issue\n", encoding="utf-8")
    return json_path, csv_path


def _governance_report(ready: bool, issues: list[str]):
    """函数说明：处理 governance_report 的内部辅助逻辑。"""
    repair_actions = (
        [
            {
                "component": "daily_basic",
                "reason": "daily_basic_history_or_freshness_incomplete",
                "commands": [
                    r".\.venv\Scripts\python.exe scripts\run_update_point_in_time_data.py --start-date 2012-01-01 --end-date 2026-06-05 --skip-index-constituents --skip-st-calendar"
                ],
            }
        ]
        if issues
        else []
    )
    return SimpleNamespace(
        is_point_in_time_ready=ready,
        issues=issues,
        warnings=[],
        to_dict=lambda: {
            "is_point_in_time_ready": ready,
            "issues": issues,
            "warnings": [],
            "st_filter_mode": "historical_calendar",
            "repair_actions": repair_actions,
        },
    )


def _write_governance_file(root: Path, report) -> Path:
    """函数说明：写入 write_governance_file 的内部辅助逻辑。"""
    path = root / "data_governance_report.json"
    path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
    return path


def _write_adj_meta_file(root: Path) -> Path:
    """函数说明：写入 write_adj_meta_file 的内部辅助逻辑。"""
    path = root / "adj_factor_meta.json"
    path.write_text('{"source": "raw_csv_adj_factor", "digest": "test"}', encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
