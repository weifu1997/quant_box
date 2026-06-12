"""Tests for monitoring metric exports."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.monitoring import build_auto_status_metrics, write_auto_status_metrics


class MonitoringTests(unittest.TestCase):
    def test_build_auto_status_metrics_exports_status_and_latest_stage_state(self) -> None:
        metrics = build_auto_status_metrics(
            {
                "status": "blocked",
                "is_executable": False,
                "block_reasons": ["params:parameter_validation_skipped"],
                "stages": [
                    {"name": "backtest", "state": "running"},
                    {"name": "backtest", "state": "skipped"},
                    {"name": "generate_signal", "state": "complete"},
                ],
            }
        )

        self.assertIn('quant_box_auto_run_status{status="blocked"} 1', metrics)
        self.assertIn("quant_box_auto_run_block_reasons_total 1.0", metrics)
        self.assertIn("quant_box_auto_run_is_executable 0.0", metrics)
        self.assertIn('quant_box_auto_run_stage_state{stage="backtest",state="skipped"} 1', metrics)
        self.assertIn('quant_box_auto_run_stage_updates_total{stage="backtest"} 2.0', metrics)

    def test_build_auto_status_metrics_escapes_labels(self) -> None:
        metrics = build_auto_status_metrics({"status": 'bad"state', "stages": [{"name": "x\\y", "state": "a\nb"}]})

        self.assertIn('status="bad\\"state"', metrics)
        self.assertIn('stage="x\\\\y"', metrics)
        self.assertIn('state="a\\nb"', metrics)

    def test_write_auto_status_metrics_creates_parent_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics" / "auto.prom"

            path = write_auto_status_metrics({"status": "complete"}, output)

            self.assertEqual(path, output)
            self.assertTrue(output.exists())
            self.assertIn('quant_box_auto_run_status{status="complete"} 1', output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
