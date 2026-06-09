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
            report = (root / "daily_signal_report.md").read_text(encoding="utf-8")
            self.assertIn("## Data Governance", report)
            self.assertIn("## Execution Loop", report)

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
            failure = json.loads((root / "auto_failure_analysis.json").read_text(encoding="utf-8"))
            self.assertTrue(failure["parameter_backtest_mismatch"])
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
        stack.enter_context(patch.object(module, "load_or_compute_factors", return_value=factors))
        stack.enter_context(patch.object(module, "build_data_health_report", return_value=health))
        stack.enter_context(patch.object(module, "write_data_health_report", side_effect=lambda *args, **kwargs: _write_health_files(root)))
        stack.enter_context(patch.object(module, "build_adj_factor_metadata", return_value=adj_meta))
        stack.enter_context(patch.object(module, "write_adj_factor_metadata", side_effect=lambda *args, **kwargs: _write_adj_meta_file(root)))
        stack.enter_context(patch.object(module, "build_data_governance_report", return_value=governance))
        stack.enter_context(patch.object(module, "write_data_governance_report", side_effect=lambda report, *_args, **_kwargs: _write_governance_file(root, report)))
        yield


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
