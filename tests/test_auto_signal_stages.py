"""Tests for the importable auto-signal stage package boundary."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.auto_signal import models
from src.auto_signal.backtest_stage import run_backtest_stage
from src.auto_signal.data_stage import run_data_preparation_stage
from src.auto_signal.optimization_stage import run_optimization_stage
from src.auto_signal.report_stage import write_auto_report_stage
from src.auto_signal.signal_stage import run_signal_stage
from src.auto_signal.status import new_status, stage


class AutoSignalStageBoundaryTests(unittest.TestCase):
    def test_stage_modules_do_not_import_cli_script(self) -> None:
        package_dir = Path(models.__file__).resolve().parent
        imported_modules: set[str] = set()
        for module_path in package_dir.glob("*.py"):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            imported_modules.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )

        self.assertNotIn("scripts.run_auto_signal", imported_modules)

    def test_all_stage_entrypoints_are_importable(self) -> None:
        for entrypoint in [
            run_data_preparation_stage,
            run_optimization_stage,
            run_backtest_stage,
            run_signal_stage,
            write_auto_report_stage,
        ]:
            self.assertTrue(callable(entrypoint))
            self.assertTrue(entrypoint.__module__.startswith("src.auto_signal."))

    def test_script_reexports_stage_result_models(self) -> None:
        script = importlib.import_module("scripts.run_auto_signal")

        self.assertIs(script.DataPreparationStageResult, models.DataPreparationStageResult)
        self.assertIs(script.OptimizationStageResult, models.OptimizationStageResult)
        self.assertIs(script.BacktestStageResult, models.BacktestStageResult)
        self.assertIs(script.SignalStageResult, models.SignalStageResult)
        self.assertIs(script.ReportStageResult, models.ReportStageResult)

    def test_status_stage_contract_remains_append_and_write(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            status = new_status({"target_date": "2026-07-16"})

            stage(status, out_dir, "compute_factors", "running", "loading cache")

            payload = json.loads((out_dir / "auto_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["target_date_resolution"], {"target_date": "2026-07-16"})
            self.assertEqual(payload["stages"][-1]["name"], "compute_factors")
            self.assertEqual(payload["stages"][-1]["state"], "running")
            self.assertEqual(payload["stages"][-1]["message"], "loading cache")


if __name__ == "__main__":
    unittest.main()
