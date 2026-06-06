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
    def test_skip_optimize_defaults_to_candidate_outputs(self) -> None:
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
            self.assertEqual(latest.read_text(encoding="utf-8"), "instrument\nOLD.SZ\n")
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "blocked")
            self.assertIn("params:parameter_validation_skipped", status["block_reasons"])

    def test_allow_low_quality_keeps_skip_optimize_run_as_candidate_outputs(self) -> None:
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
            latest_frame = pd.read_csv(latest)
            self.assertEqual(latest_frame["instrument"].tolist(), ["E"])

    def test_backtest_quality_blocks_official_outputs(self) -> None:
        module = importlib.import_module("scripts.run_auto_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, factors = _auto_config_and_factors(root)
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
                        "max_weight_per_stock": None,
                        "stop_loss_pct": None,
                        "take_profit_pct": None,
                        "circuit_breaker_drawdown": None,
                        "circuit_breaker_cooldown_days": None,
                        "circuit_breaker_target_exposure": None,
                        "target_vol": None,
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
                metrics={"annual_return": 0.30, "max_drawdown": -0.30, "calmar": 1.0},
            )

            with _patched_auto_run(
                module,
                config,
                factors,
                ["run_auto_signal.py", "--skip-update", "--skip-convert", "--no-archive"],
            ), patch.object(module, "run_walk_forward_grid_validation", return_value=validation), patch.object(
                module,
                "run_backtest",
                return_value=bad_result,
            ):
                module.main()

            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertFalse((root / "signal_2024-01-03.csv").exists())
            self.assertEqual(latest.read_text(encoding="utf-8"), "instrument\nOLD.SZ\n")
            status = json.loads((root / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "blocked")
            self.assertTrue(any(reason.startswith("backtest:backtest_max_drawdown_worse_than_limit") for reason in status["block_reasons"]))
            report = (root / "daily_signal_report.md").read_text(encoding="utf-8")
            self.assertIn("## Backtest Quality", report)

    def test_promote_candidate_writes_official_signal_and_latest_holdings(self) -> None:
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
def _patched_auto_run(module, config: dict, factors: pd.DataFrame, argv: list[str]):
    health = SimpleNamespace(is_healthy=True, issues=[], to_dict=lambda: {"is_healthy": True, "issues": []})
    root = Path(config["outputs"]["dir"])
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(module, "load_config", return_value=config))
        stack.enter_context(patch.object(module, "load_or_compute_factors", return_value=factors))
        stack.enter_context(patch.object(module, "build_data_health_report", return_value=health))
        stack.enter_context(patch.object(module, "write_data_health_report", side_effect=lambda *args, **kwargs: _write_health_files(root)))
        yield


def _write_health_files(root: Path) -> tuple[Path, Path]:
    json_path = root / "data_health_report.json"
    csv_path = root / "data_health_report.csv"
    json_path.write_text('{"is_healthy": true, "issues": []}', encoding="utf-8")
    csv_path.write_text("issue\n", encoding="utf-8")
    return json_path, csv_path


if __name__ == "__main__":
    unittest.main()
