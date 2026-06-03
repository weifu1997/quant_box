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

    def test_allow_low_quality_promotes_skip_optimize_run_to_official_outputs(self) -> None:
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

            self.assertTrue((root / "signal_2024-01-03.csv").exists())
            self.assertTrue((root / "manual_orders_2024-01-03.csv").exists())
            latest_frame = pd.read_csv(latest)
            self.assertEqual(latest_frame["instrument"].tolist(), ["E"])

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
