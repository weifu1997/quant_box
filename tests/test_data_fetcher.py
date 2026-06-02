from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.data_fetcher import DAILY_FIELDS, fetch_daily_stocks, update_daily_data


class FakeTushareClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, list[str] | str | None]] = []

    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        self.calls.append((api_name, params.copy(), fields))
        codes = str(params.get("ts_code", "")).split(",")
        rows = []
        for code in codes:
            if not code:
                continue
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vol": 1000.0,
                    "amount": 10000.0,
                }
            )
        if api_name == "daily":
            return pd.DataFrame(rows, columns=DAILY_FIELDS)
        if api_name == "adj_factor":
            return pd.DataFrame(
                [{"ts_code": row["ts_code"], "trade_date": row["trade_date"], "adj_factor": 1.0} for row in rows],
                columns=["ts_code", "trade_date", "adj_factor"],
            )
        raise AssertionError(f"Unexpected API call: {api_name}")


class MissingAdjFactorClient(FakeTushareClient):
    missing_code = "600519.SH"

    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        if api_name != "adj_factor":
            return super().call(api_name, params=params, fields=fields)

        params = params or {}
        self.calls.append((api_name, params.copy(), fields))
        codes = str(params.get("ts_code", "")).split(",")
        rows = []
        for code in codes:
            if not code or code == self.missing_code:
                continue
            rows.append({"ts_code": code, "trade_date": "20240102", "adj_factor": 1.0})
        return pd.DataFrame(rows, columns=["ts_code", "trade_date", "adj_factor"])


class DataFetcherTests(unittest.TestCase):
    def test_fetch_daily_stocks_uses_comma_separated_batch_request(self) -> None:
        client = FakeTushareClient()

        df = fetch_daily_stocks(["000001.SZ", "600519.SH"], "2024-01-01", "2024-01-03", client=client)

        daily_calls = [call for call in client.calls if call[0] == "daily"]
        self.assertEqual(len(daily_calls), 1)
        self.assertEqual(daily_calls[0][1]["ts_code"], "000001.SZ,600519.SH")
        self.assertEqual(sorted(df["ts_code"].unique().tolist()), ["000001.SZ", "600519.SH"])
        self.assertIn("adj_factor", df.columns)

    def test_fetch_daily_stocks_splits_long_range_into_date_windows(self) -> None:
        client = FakeTushareClient()

        fetch_daily_stocks(
            ["000001.SZ", "600519.SH"],
            "2024-01-01",
            "2024-01-05",
            client=client,
            window_days=2,
        )

        daily_calls = [call for call in client.calls if call[0] == "daily"]
        self.assertEqual(len(daily_calls), 3)
        self.assertEqual([call[1]["start_date"] for call in daily_calls], ["20240101", "20240103", "20240105"])
        self.assertEqual([call[1]["end_date"] for call in daily_calls], ["20240102", "20240104", "20240105"])

    def test_update_daily_data_writes_each_symbol_from_batched_response(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "max_new_symbols_per_run": 100,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.TushareHttpClient.from_config", return_value=client
            ):
                written = update_daily_data(
                    stock_codes=["000001.SZ", "600519.SH"],
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    raw_dir=raw_dir,
                )

            self.assertEqual(set(written), {"000001.SZ", "600519.SH"})
            for code in written:
                path = raw_dir / f"{code}.csv"
                self.assertTrue(path.exists())
                df = pd.read_csv(path)
                self.assertEqual(df["ts_code"].tolist(), [code])
                self.assertIn("adj_factor", df.columns)

        daily_calls = [call for call in client.calls if call[0] == "daily"]
        self.assertEqual(len(daily_calls), 1)
        self.assertEqual(daily_calls[0][1]["ts_code"], "000001.SZ,600519.SH")

    def test_fetch_daily_stocks_skips_symbol_with_incomplete_adj_factor(self) -> None:
        client = MissingAdjFactorClient()

        df = fetch_daily_stocks(["000001.SZ", "600519.SH"], "2024-01-01", "2024-01-03", client=client)

        self.assertEqual(df["ts_code"].unique().tolist(), ["000001.SZ"])
        self.assertIn("adj_factor", df.columns)
        self.assertFalse(df["adj_factor"].isna().any())

    def test_update_daily_data_records_failed_symbol_when_adj_factor_is_incomplete(self) -> None:
        client = MissingAdjFactorClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "max_new_symbols_per_run": 100,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.TushareHttpClient.from_config", return_value=client
            ):
                written = update_daily_data(
                    stock_codes=["000001.SZ", "600519.SH"],
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    raw_dir=raw_dir,
                )

            self.assertEqual(set(written), {"000001.SZ"})
            self.assertTrue((raw_dir / "000001.SZ.csv").exists())
            self.assertFalse((raw_dir / "600519.SH.csv").exists())
            failed = pd.read_csv(raw_dir / "failed_fetches.csv")
            self.assertEqual(failed["ts_code"].tolist(), ["600519.SH"])
            self.assertEqual(failed["reason"].tolist(), ["empty_or_failed_fetch"])

    def test_update_daily_data_limits_new_symbol_backfill_but_keeps_existing_incremental_updates(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "max_new_symbols_per_run": 1,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            existing = raw_dir / "000001.SZ.csv"
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-01",
                        "open": 9.0,
                        "high": 9.0,
                        "low": 9.0,
                        "close": 9.0,
                        "vol": 100.0,
                        "amount": 900.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(existing, index=False)

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.TushareHttpClient.from_config", return_value=client
            ):
                written = update_daily_data(
                    stock_codes=["000001.SZ", "600519.SH", "000002.SZ"],
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    raw_dir=raw_dir,
                )

            self.assertEqual(set(written), {"000001.SZ", "600519.SH"})
            self.assertTrue((raw_dir / "000001.SZ.csv").exists())
            self.assertTrue((raw_dir / "600519.SH.csv").exists())
            self.assertFalse((raw_dir / "000002.SZ.csv").exists())


if __name__ == "__main__":
    unittest.main()
