"""Tests for Web account and holdings management."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd
import yaml

from src.dashboard_account import apply_account_update, build_account_workspace, preview_account_update
from src.dashboard_api import create_dashboard_app


class DashboardAccountTests(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        return {
            "account": {
                "file": str(root / "account.yaml"),
                "current_holdings_file": str(root / "current_holdings.csv"),
                "total_asset": 1_000_000,
                "cash": 0,
                "max_position_pct": None,
                "lot_size": 100,
                "star_market_lot_size": 200,
            },
            "outputs": {"dir": str(root / "outputs")},
        }

    def _payload(self) -> dict:
        return {
            "account": {
                "total_asset": 1_000_000,
                "cash": 100_000,
                "max_position_pct": 0.2,
                "lot_size": 100,
                "star_market_lot_size": 200,
            },
            "holdings": [
                {"instrument": "000001.sz", "shares": 500},
                {"instrument": "600519.SH", "shares": 100},
            ],
        }

    def test_preview_normalizes_and_validates_holdings(self) -> None:
        with TemporaryDirectory() as tmp:
            result = preview_account_update(self._payload(), self._config(Path(tmp)))

        self.assertTrue(result["valid"])
        self.assertEqual(result["holdings"][0]["instrument"], "000001.SZ")
        self.assertEqual(result["position_count"], 2)

    def test_preview_rejects_duplicate_or_invalid_lot_holdings(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = self._payload()
            payload["holdings"] = [
                {"instrument": "000001.SZ", "shares": 150},
                {"instrument": "000001.sz", "shares": 100},
            ]
            result = preview_account_update(payload, self._config(Path(tmp)))

        self.assertFalse(result["valid"])
        self.assertIn("shares_not_lot_multiple:000001.SZ", result["issues"])
        self.assertIn("duplicate_instrument:000001.SZ", result["issues"])

    def test_preview_rejects_non_finite_account_values(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = self._payload()
            payload["account"]["total_asset"] = "inf"

            with self.assertRaisesRegex(ValueError, "total_asset must be a finite number"):
                preview_account_update(payload, self._config(Path(tmp)))

    def test_preview_rejects_boolean_and_collection_numbers(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            boolean_payload = self._payload()
            boolean_payload["account"]["cash"] = True
            with self.assertRaisesRegex(ValueError, "cash must be a number"):
                preview_account_update(boolean_payload, config)

            collection_payload = self._payload()
            collection_payload["account"]["max_position_pct"] = []
            with self.assertRaisesRegex(ValueError, "max_position_pct must be a number"):
                preview_account_update(collection_payload, config)

    def test_apply_requires_confirmation_writes_files_and_backups(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            (root / "account.yaml").write_text("total_asset: 500000\ncash: 50000\n", encoding="utf-8")
            pd.DataFrame({"instrument": ["000001.SZ"], "shares": [100]}).to_csv(root / "current_holdings.csv", index=False)

            with self.assertRaisesRegex(ValueError, "confirm must be true"):
                apply_account_update(self._payload(), config)
            result = apply_account_update({**self._payload(), "confirm": True}, config)

            account = yaml.safe_load((root / "account.yaml").read_text(encoding="utf-8"))
            self.assertEqual(account["total_asset"], 1_000_000)
            holdings = pd.read_csv(root / "current_holdings.csv")
            self.assertEqual(len(holdings), 2)
            backup_dir = Path(result["backup_dir"])
            self.assertTrue((backup_dir / "account.yaml").exists())
            self.assertTrue((backup_dir / "current_holdings.csv").exists())

    def test_workspace_does_not_expose_tushare_or_other_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            config["tushare"] = {"token": "secret-token"}

            workspace = build_account_workspace(config)

        self.assertNotIn("tushare", workspace)
        self.assertNotIn("secret-token", str(workspace))

    def test_account_api_maps_invalid_payload_to_400(self) -> None:
        with patch("src.dashboard_api.preview_account_update", side_effect=ValueError("bad account")):
            response = TestClient(create_dashboard_app()).post("/api/dashboard/account/preview", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "bad account")


if __name__ == "__main__":
    unittest.main()
