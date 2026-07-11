from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from src.dashboard_api import create_dashboard_app
from src.dashboard_stock import build_stock_detail


class _FakeClient:
    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self.frame = frame if frame is not None else pd.DataFrame()
        self.error = error
        self.calls: list[tuple[str, dict, object]] = []

    def call(self, api_name: str, params: dict | None = None, fields: object = None) -> pd.DataFrame:
        self.calls.append((api_name, params or {}, fields))
        if self.error is not None:
            raise self.error
        return self.frame.copy()


class DashboardStockTests(unittest.TestCase):
    def test_build_stock_detail_returns_live_rt_k_quote(self) -> None:
        client = _FakeClient(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "pre_close": 10.0,
                        "open": 10.1,
                        "high": 10.8,
                        "low": 9.9,
                        "close": 10.5,
                        "vol": 123456,
                        "amount": 9876543,
                    }
                ]
            )
        )

        result = build_stock_detail(
            "000001.sz",
            config={"data": {}},
            client=client,
            now=datetime(2026, 7, 11, 10, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(client.calls[0][0], "rt_k")
        self.assertEqual(client.calls[0][1], {"ts_code": "000001.SZ"})
        self.assertEqual(result["status"], "live")
        self.assertTrue(result["is_live"])
        self.assertEqual(result["name"], "平安银行")
        self.assertEqual(result["price"], 10.5)
        self.assertAlmostEqual(result["change_pct"], 5.0)
        self.assertIsNone(result["market_date"])
        self.assertEqual(result["retrieved_at"], "2026-07-11T18:30:00+08:00")
        self.assertIn("接口未提供行情日期", result["message"])

    def test_build_stock_detail_falls_back_to_latest_local_daily_row(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "trade_date": "20260709", "open": 9.8, "high": 10.1, "low": 9.7, "close": 10.0, "vol": 100, "amount": 1000},
                    {"ts_code": "000001.SZ", "trade_date": "20260710", "open": 10.1, "high": 10.8, "low": 10.0, "close": 10.5, "vol": 200, "amount": 2100},
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False, encoding="utf-8-sig")
            universe = root / "stocks.csv"
            pd.DataFrame([{"ts_code": "000001.SZ", "name": "平安银行"}]).to_csv(
                universe, index=False, encoding="utf-8-sig"
            )
            client = _FakeClient(error=RuntimeError("proxy unavailable"))

            result = build_stock_detail(
                "000001.SZ",
                config={"data": {"raw_dir": str(raw_dir), "constituents_file": str(universe)}},
                client=client,
                now=datetime(2026, 7, 11, 18, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(result["status"], "fallback")
        self.assertFalse(result["is_live"])
        self.assertEqual(result["source"], "local_daily")
        self.assertEqual(result["name"], "平安银行")
        self.assertEqual(result["price"], 10.5)
        self.assertEqual(result["pre_close"], 10.0)
        self.assertAlmostEqual(result["change_pct"], 5.0)
        self.assertEqual(result["market_date"], "2026-07-10")
        self.assertIn("非实时", result["message"])

    def test_build_stock_detail_rejects_unbounded_instrument_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid stock instrument"):
            build_stock_detail("../../config/settings.local.yaml", config={"data": {}}, client=_FakeClient())

    def test_stock_api_maps_validation_and_unavailable_errors(self) -> None:
        app = create_dashboard_app()
        client = TestClient(app)
        with patch("src.dashboard_api.build_stock_detail", side_effect=ValueError("bad code")):
            response = client.get("/api/dashboard/stocks/not-a-code")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "bad code")

        with patch("src.dashboard_api.build_stock_detail", side_effect=RuntimeError("quote unavailable")):
            response = client.get("/api/dashboard/stocks/000001.SZ")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "quote unavailable")


if __name__ == "__main__":
    unittest.main()
