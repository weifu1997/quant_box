"""Tests for the controlled dashboard fill-feedback workflow."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from src.dashboard_api import create_dashboard_app
from src.dashboard_execution import apply_execution_feedback, build_execution_workspace, preview_execution_feedback


class DashboardExecutionTests(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        return {
            "outputs": {"dir": str(root / "outputs")},
            "manual_orders": {"fill_feedback_dir": str(root / "outputs" / "fill_feedback")},
            "account": {"current_holdings_file": str(root / "current_holdings.csv")},
        }

    def _write_fixture(self, root: Path) -> tuple[dict, Path]:
        config = self._config(root)
        fill_dir = root / "outputs" / "fill_feedback"
        fill_dir.mkdir(parents=True)
        fill_path = fill_dir / "fill_feedback_2026-07-10.csv"
        pd.DataFrame(
            [
                {
                    "signal_date": "2026-07-10",
                    "instrument": "000001.SZ",
                    "side": "BUY",
                    "planned_order_shares": 200,
                    "fill_status": "PENDING",
                    "executed_shares": None,
                    "executed_price": None,
                },
                {
                    "signal_date": "2026-07-10",
                    "instrument": "600519.SH",
                    "side": "SELL",
                    "planned_order_shares": -100,
                    "fill_status": "PENDING",
                    "executed_shares": None,
                    "executed_price": None,
                },
            ]
        ).to_csv(fill_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {"instrument": ["000001.SZ", "600519.SH"], "shares": [100, 100]}
        ).to_csv(root / "current_holdings.csv", index=False, encoding="utf-8-sig")
        return config, fill_path

    def test_workspace_ignores_candidate_fill_templates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            fill_dir = root / "outputs" / "fill_feedback"
            fill_dir.mkdir(parents=True)
            (fill_dir / "fill_feedback_candidate_2026-07-10.csv").write_text("instrument,fill_status\n000001.SZ,PENDING\n", encoding="utf-8")

            workspace = build_execution_workspace(config)

            self.assertEqual(workspace["status"], "missing")
            self.assertIsNone(workspace["source_id"])

    def test_preview_preserves_immutable_order_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _fill_path = self._write_fixture(root)
            workspace = build_execution_workspace(config)
            rows = workspace["rows"]
            rows[0].update({"instrument": "999999.SH", "side": "SELL", "planned_order_shares": 9999, "fill_status": "FILLED", "executed_shares": 200})
            rows[1].update({"fill_status": "CANCELLED", "executed_shares": None})

            preview = preview_execution_feedback({"source_id": workspace["source_id"], "rows": rows}, config)

            self.assertTrue(preview["valid"])
            updated = {row["instrument"]: row["shares"] for row in preview["updated_holdings"]}
            self.assertEqual(updated["000001.SZ"], 300.0)
            self.assertEqual(updated["600519.SH"], 100.0)
            self.assertNotIn("999999.SH", updated)

    def test_apply_requires_confirmation_and_updates_holdings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, fill_path = self._write_fixture(root)
            workspace = build_execution_workspace(config)
            rows = workspace["rows"]
            rows[0].update({"fill_status": "FILLED", "executed_shares": 200, "executed_price": 12.5})
            rows[1].update({"fill_status": "FILLED", "executed_shares": 100, "executed_price": 1500})
            payload = {"source_id": workspace["source_id"], "rows": rows}

            with self.assertRaisesRegex(ValueError, "confirm must be true"):
                apply_execution_feedback(payload, config)
            result = apply_execution_feedback({**payload, "confirm": True}, config)

            self.assertEqual(result["status"], "applied")
            holdings = pd.read_csv(root / "current_holdings.csv")
            self.assertEqual(holdings.to_dict(orient="records"), [{"instrument": "000001.SZ", "shares": 300.0}])
            saved_fills = pd.read_csv(fill_path)
            self.assertEqual(saved_fills["fill_status"].tolist(), ["FILLED", "FILLED"])
            self.assertTrue(Path(result["audit_path"]).exists())

    def test_apply_rejects_stale_source_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _fill_path = self._write_fixture(root)
            workspace = build_execution_workspace(config)

            with self.assertRaisesRegex(ValueError, "template changed"):
                preview_execution_feedback({"source_id": "fill_feedback_old.csv", "rows": workspace["rows"]}, config)

    def test_execution_api_maps_validation_errors(self) -> None:
        with patch("src.dashboard_api.preview_execution_feedback", side_effect=ValueError("bad fills")):
            response = TestClient(create_dashboard_app()).post(
                "/api/dashboard/execution/preview",
                json={"source_id": "fill_feedback.csv", "rows": []},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "bad fills")

    def test_execution_api_serializes_empty_numeric_cells_as_null(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _fill_path = self._write_fixture(root)
            with patch("src.dashboard_execution.load_config", return_value=config):
                response = TestClient(create_dashboard_app()).get("/api/dashboard/execution")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "needs_input")
        self.assertIsNone(body["rows"][0]["executed_shares"])
        self.assertIsNone(body["rows"][0]["executed_price"])


if __name__ == "__main__":
    unittest.main()
