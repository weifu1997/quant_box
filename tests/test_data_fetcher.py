from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.data_fetcher import (
    DAILY_BASIC_FIELDS,
    DAILY_FIELDS,
    INDEX_WEIGHT_FIELDS,
    fetch_daily_basic,
    fetch_index_constituents,
    fetch_daily_stock,
    fetch_daily_stocks,
    fetch_st_calendar,
    fetch_hs300_stocks,
    fetch_stock_universe,
    filter_universe_frame,
    normalize_daily_frame,
    normalize_daily_basic_frame,
    normalize_index_constituents_frame,
    normalize_st_calendar_frame,
    _raw_latest_date,
    update_daily_basic_data,
    update_daily_data,
    update_daily_data_resumable,
    update_index_constituents_data,
    update_st_calendar_data,
)


class FakeTushareClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, list[str] | str | None]] = []

    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        self.calls.append((api_name, params.copy(), fields))
        codes = str(params.get("ts_code", "")).split(",")
        trade_date = str(params.get("end_date", "20240102"))
        rows = []
        for code in codes:
            if not code:
                continue
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
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
        trade_date = str(params.get("end_date", "20240102"))
        rows = []
        for code in codes:
            if not code or code == self.missing_code:
                continue
            rows.append({"ts_code": code, "trade_date": trade_date, "adj_factor": 1.0})
        return pd.DataFrame(rows, columns=["ts_code", "trade_date", "adj_factor"])


class EmptyTushareClient(FakeTushareClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        self.calls.append((api_name, (params or {}).copy(), fields))
        if api_name == "daily":
            return pd.DataFrame(columns=DAILY_FIELDS)
        if api_name == "adj_factor":
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        raise AssertionError(f"Unexpected API call: {api_name}")


class DailyBasicClient(FakeTushareClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        self.calls.append((api_name, params.copy(), fields))
        if api_name == "daily_basic":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": params.get("trade_date", "20240102"),
                        "total_mv": 1000.0,
                        "circ_mv": 800.0,
                        "pb": 1.2,
                    }
                ]
            )
        return super().call(api_name, params=params, fields=fields)


class FlakyDailyBasicClient(DailyBasicClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        if api_name == "daily_basic" and params.get("trade_date") == "20240103":
            self.calls.append((api_name, params.copy(), fields))
            raise RuntimeError("daily_basic limited")
        return super().call(api_name, params=params, fields=fields)


class PointInTimeClient(FakeTushareClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        self.calls.append((api_name, params.copy(), fields))
        if api_name == "index_weight":
            trade_date = params.get("end_date") or params.get("trade_date") or "20240103"
            return pd.DataFrame(
                [
                    {
                        "index_code": params.get("index_code", "000300.SH"),
                        "con_code": "000001.SZ",
                        "trade_date": trade_date,
                        "weight": 1.23,
                    }
                ]
            )
        if api_name == "namechange":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "name": "*ST TEST",
                        "start_date": "20240102",
                        "end_date": "",
                        "ann_date": "20240101",
                        "change_reason": "特别处理",
                    },
                    {
                        "ts_code": "000002.SZ",
                        "name": "NORMAL",
                        "start_date": "20240102",
                        "end_date": "",
                        "ann_date": "20240101",
                        "change_reason": "更名",
                    },
                ]
            )
        return super().call(api_name, params=params, fields=fields)


class FailingTushareClient(FakeTushareClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        self.calls.append((api_name, (params or {}).copy(), fields))
        raise RuntimeError("limited")


class FlakyIndexClient(PointInTimeClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        if api_name == "index_weight" and params.get("start_date") == "20240201":
            self.calls.append((api_name, params.copy(), fields))
            raise RuntimeError("window limited")
        return super().call(api_name, params=params, fields=fields)


class FallbackIndexClient(PointInTimeClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        if api_name == "index_weight" and params.get("index_code") == "000300.SH":
            self.calls.append((api_name, params.copy(), fields))
            return pd.DataFrame(columns=INDEX_WEIGHT_FIELDS)
        return super().call(api_name, params=params, fields=fields)


class DataFetcherTests(unittest.TestCase):
    def test_raw_latest_date_reads_latest_value_from_csv_tail(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "000001.SZ.csv"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                    "close": [10.0, 10.5, 11.0],
                }
            ).to_csv(path, index=False)

            self.assertEqual(_raw_latest_date(path), pd.Timestamp("2024-01-04"))

    def test_normalize_daily_basic_frame_keeps_market_cap_fields(self) -> None:
        frame = normalize_daily_basic_frame(
            pd.DataFrame(
                [
                    {
                        "ts_code": " 000001.sz ",
                        "trade_date": "20240102",
                        "total_mv": "1000.5",
                        "circ_mv": "800.25",
                    }
                ]
            )
        )

        self.assertEqual(frame["ts_code"].iloc[0], "000001.SZ")
        self.assertEqual(frame["trade_date"].iloc[0], pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(float(frame["total_mv"].iloc[0]), 1000.5)
        self.assertIn("circ_mv", frame.columns)

    def test_symbol_normalizers_strip_whitespace_and_case(self) -> None:
        daily = normalize_daily_frame(
            pd.DataFrame(
                [
                    {
                        "ts_code": " 000001.sz ",
                        "trade_date": "20240102",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "vol": 1000,
                        "amount": 10000,
                    }
                ]
            )
        )
        index = normalize_index_constituents_frame(
            pd.DataFrame(
                [
                    {"index_code": " 000300.sh ", "con_code": " 600519.sh ", "trade_date": "20240102", "weight": 1.5},
                ]
            )
        )
        st_calendar = normalize_st_calendar_frame(
            pd.DataFrame(
                [
                    {"ts_code": " 000001.sz ", "name": "*ST TEST", "start_date": " 20240102 "},
                ]
            )
        )

        self.assertEqual(daily["ts_code"].tolist(), ["000001.SZ"])
        self.assertEqual(index[["index_code", "con_code"]].iloc[0].tolist(), ["000300.SH", "600519.SH"])
        self.assertEqual(st_calendar["ts_code"].tolist(), ["000001.SZ"])

    def test_normalize_daily_frame_deduplicates_symbol_date_pairs(self) -> None:
        daily = normalize_daily_frame(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20240102",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "vol": 1000,
                        "amount": 10000,
                        "adj_factor": 1.0,
                    },
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20240102",
                        "open": 11.0,
                        "high": 11.5,
                        "low": 10.8,
                        "close": 11.2,
                        "vol": 1200,
                        "amount": 12000,
                        "adj_factor": 1.1,
                    },
                ]
            )
        )

        self.assertEqual(len(daily), 1)
        self.assertAlmostEqual(float(daily["close"].iloc[0]), 11.2)
        self.assertAlmostEqual(float(daily["adj_factor"].iloc[0]), 1.1)

    def test_fetch_daily_basic_requests_tushare_daily_basic_fields(self) -> None:
        client = DailyBasicClient()

        frame = fetch_daily_basic("2024-01-02", client=client, retries=1)

        self.assertEqual(client.calls[0][0], "daily_basic")
        self.assertEqual(client.calls[0][1]["trade_date"], "20240102")
        self.assertEqual(client.calls[0][2], DAILY_BASIC_FIELDS)
        self.assertEqual(frame["trade_date"].iloc[0], pd.Timestamp("2024-01-02"))

    def test_fetch_index_constituents_requests_tushare_index_weight_fields(self) -> None:
        client = PointInTimeClient()

        frame = fetch_index_constituents("000300.SH", "2024-01-01", "2024-01-03", client=client, retries=1)

        self.assertEqual(client.calls[0][0], "index_weight")
        self.assertEqual(client.calls[0][1]["index_code"], "000300.SH")
        self.assertEqual(client.calls[0][1]["start_date"], "20240101")
        self.assertEqual(client.calls[0][1]["end_date"], "20240103")
        self.assertEqual(client.calls[0][2], INDEX_WEIGHT_FIELDS)
        self.assertEqual(frame["trade_date"].iloc[0], pd.Timestamp("2024-01-03"))

    def test_fetch_st_calendar_keeps_only_st_namechange_rows(self) -> None:
        client = PointInTimeClient()

        frame = fetch_st_calendar(client=client, retries=1)

        self.assertEqual(client.calls[0][0], "namechange")
        self.assertEqual(frame["ts_code"].tolist(), ["000001.SZ"])
        self.assertEqual(frame["st_start_date"].iloc[0], pd.Timestamp("2024-01-02"))

    def test_update_daily_basic_data_writes_incremental_parquet_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "daily_basic.parquet"
            price_file = root / "prices.parquet"
            prices = pd.DataFrame(
                {"close": [1.0, 2.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            )
            prices.to_parquet(price_file)
            config = {
                "data": {
                    "start_date": "2024-01-02",
                    "end_date": "2024-01-03",
                    "daily_basic_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                },
                "ic": {"price_file": str(price_file)},
            }
            client = DailyBasicClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                path = update_daily_basic_data(client=client)

            cached = pd.read_parquet(path)
            self.assertEqual(len(cached), 2)
            self.assertEqual([call[1]["trade_date"] for call in client.calls], ["20240102", "20240103"])

    def test_update_daily_basic_data_supports_history_start_and_max_dates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "daily_basic.parquet"
            price_file = root / "prices.parquet"
            prices = pd.DataFrame(
                {"close": [1.0, 2.0, 3.0]},
                index=pd.to_datetime(["2021-01-04", "2021-01-05", "2024-01-02"]),
            )
            prices.to_parquet(price_file)
            config = {
                "data": {
                    "start_date": "2024-01-01",
                    "history_start_date": "2021-01-01",
                    "end_date": "2024-01-03",
                    "daily_basic_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                },
                "ic": {"price_file": str(price_file)},
            }
            client = DailyBasicClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                path = update_daily_basic_data(client=client, max_dates=1)

            cached = pd.read_parquet(path)
            self.assertEqual(len(cached), 1)
            self.assertEqual([call[1]["trade_date"] for call in client.calls], ["20210104"])

    def test_update_daily_basic_data_skips_failed_dates_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "daily_basic.parquet"
            price_file = root / "prices.parquet"
            prices = pd.DataFrame(
                {"close": [1.0, 2.0, 3.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            )
            prices.to_parquet(price_file)
            config = {
                "data": {
                    "start_date": "2024-01-02",
                    "end_date": "2024-01-04",
                    "daily_basic_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                },
                "ic": {"price_file": str(price_file)},
            }
            client = FlakyDailyBasicClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                path = update_daily_basic_data(client=client)

            cached = pd.read_parquet(path)
            self.assertEqual(cached["trade_date"].dt.strftime("%Y%m%d").tolist(), ["20240102", "20240104"])

    def test_update_daily_basic_data_can_fail_on_date_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "daily_basic.parquet"
            price_file = root / "prices.parquet"
            prices = pd.DataFrame(
                {"close": [1.0, 2.0, 3.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            )
            prices.to_parquet(price_file)
            config = {
                "data": {
                    "start_date": "2024-01-02",
                    "end_date": "2024-01-04",
                    "daily_basic_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                },
                "ic": {"price_file": str(price_file)},
            }
            client = FlakyDailyBasicClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                with self.assertRaises(ValueError):
                    update_daily_basic_data(client=client, skip_failed=False)

    def test_update_index_constituents_data_writes_point_in_time_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "hs300_constituents.csv"
            config = {
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "hs300_constituents_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                }
            }
            client = PointInTimeClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                path = update_index_constituents_data(client=client, max_windows=1)

            cached = pd.read_csv(path)
            self.assertEqual(cached["con_code"].tolist(), ["000001.SZ"])
            self.assertEqual(client.calls[0][0], "index_weight")

    def test_update_index_constituents_data_defaults_to_month_sized_windows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "hs300_constituents.csv"
            config = {
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-03-10",
                    "hs300_constituents_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                }
            }
            client = PointInTimeClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                update_index_constituents_data(client=client, max_windows=2)

            calls = [call for call in client.calls if call[0] == "index_weight"]
            self.assertEqual([call[1]["start_date"] for call in calls], ["20240101", "20240201"])
            self.assertEqual([call[1]["end_date"] for call in calls], ["20240131", "20240302"])

    def test_update_index_constituents_data_uses_fallback_code_when_primary_is_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "hs300_constituents.csv"
            config = {
                "data": {
                    "start_date": "2015-01-01",
                    "end_date": "2015-01-31",
                    "hs300_constituents_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                },
                "data_governance": {"index_fallback_codes": ["399300.SZ"]},
            }
            client = FallbackIndexClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                path = update_index_constituents_data(client=client, max_windows=1)

            cached = pd.read_csv(path)
            self.assertEqual(cached["index_code"].tolist(), ["399300.SZ"])
            self.assertEqual([call[1]["index_code"] for call in client.calls if call[0] == "index_weight"], ["000300.SH", "399300.SZ"])

    def test_update_index_constituents_data_skips_failed_windows_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "hs300_constituents.csv"
            config = {
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-03-10",
                    "hs300_constituents_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                }
            }
            client = FlakyIndexClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                path = update_index_constituents_data(client=client, max_windows=3)

            cached = pd.read_csv(path)
            self.assertEqual(cached["trade_date"].tolist(), ["2024-01-31", "2024-03-10"])

    def test_update_index_constituents_data_can_fail_on_window_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "hs300_constituents.csv"
            config = {
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-03-10",
                    "hs300_constituents_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                }
            }
            client = FlakyIndexClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                with self.assertRaises(ValueError):
                    update_index_constituents_data(client=client, max_windows=3, skip_failed=False)

    def test_update_st_calendar_data_writes_st_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_file = root / "st_calendar.csv"
            config = {
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "st_calendar_file": str(out_file),
                    "retries": 1,
                    "retry_max_wait": 0,
                }
            }
            client = PointInTimeClient()

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ):
                path = update_st_calendar_data(client=client)

            cached = pd.read_csv(path)
            self.assertEqual(cached["ts_code"].tolist(), ["000001.SZ"])
            self.assertIn("st_start_date", cached.columns)

    def test_fetch_daily_stock_defaults_to_five_retries_and_caps_wait(self) -> None:
        client = FailingTushareClient()

        with patch("src.data_fetcher.random.uniform", return_value=0.0), patch("src.data_fetcher.time.sleep") as sleep:
            with self.assertRaises(ValueError):
                fetch_daily_stock("000001.SZ", "2024-01-01", "2024-01-03", client=client, retry_max_wait=1.5)

        self.assertEqual(len(client.calls), 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 1.5, 1.5, 1.5])

    def test_hs300_universe_uses_hs300_constituents_not_mainboard_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mainboard_file = root / "mainboard_a_stocks.csv"
            hs300_file = root / "hs300_constituents.csv"
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "name": "MAINBOARD_A"},
                    {"ts_code": "000002.SZ", "name": "MAINBOARD_B"},
                ]
            ).to_csv(mainboard_file, index=False)
            pd.DataFrame(
                [
                    {"con_code": "600000.SH"},
                    {"con_code": "000300.SZ"},
                ]
            ).to_csv(hs300_file, index=False)
            config = {
                "data": {
                    "universe": "hs300",
                    "constituents_file": str(mainboard_file),
                    "hs300_constituents_file": str(hs300_file),
                    "end_date": "2024-01-03",
                }
            }

            with patch("src.data_fetcher.load_config", return_value=config):
                codes = fetch_stock_universe()

            self.assertEqual(codes, ["000300.SZ", "600000.SH"])

    def test_fetch_hs300_stocks_filters_local_constituents_as_of_date(self) -> None:
        with TemporaryDirectory() as tmp:
            hs300_file = Path(tmp) / "hs300_constituents.csv"
            pd.DataFrame(
                [
                    {"index_code": "000300.SH", "con_code": "000001.SZ", "trade_date": "20240103", "weight": 1.0},
                    {"index_code": "000300.SH", "con_code": "600519.SH", "trade_date": "20240201", "weight": 1.0},
                ]
            ).to_csv(hs300_file, index=False)

            january = fetch_hs300_stocks(date="2024-01-31", local_file=hs300_file)
            february = fetch_hs300_stocks(date="2024-02-01", local_file=hs300_file)

            self.assertEqual(january, ["000001.SZ"])
            self.assertEqual(february, ["600519.SH"])

    def test_filter_universe_frame_excludes_delisted_before_as_of_date(self) -> None:
        universe = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "PINGAN", "list_status": "L", "list_date": "19910403", "delist_date": ""},
                {"ts_code": "000003.SZ", "name": "DELISTED", "list_status": "D", "list_date": "19910403", "delist_date": "20020614"},
                {"ts_code": "000015.SZ", "name": "FUTURE_EXIT", "list_status": "D", "list_date": "19910403", "delist_date": "20251231"},
            ]
        )

        filtered = filter_universe_frame(universe, universe="mainboard_a", as_of_date="2024-01-01", exclude_st=True)

        self.assertEqual(filtered["ts_code"].tolist(), ["000001.SZ", "000015.SZ"])

    def test_filter_universe_frame_normalizes_symbol_whitespace_and_case(self) -> None:
        universe = pd.DataFrame(
            [
                {"ts_code": " 000001.sz ", "name": "PINGAN", "list_status": " l ", "list_date": " 19910403 ", "delist_date": ""},
            ]
        )

        filtered = filter_universe_frame(universe, universe="mainboard_a", as_of_date="2024-01-01", exclude_st=True)

        self.assertEqual(filtered["ts_code"].tolist(), ["000001.SZ"])

    def test_filter_universe_frame_uses_point_in_time_st_calendar(self) -> None:
        universe = pd.DataFrame(
            [
                {"ts_code": " 000001.sz ", "name": "ST_STATIC_NAME", "list_status": "L", "list_date": "19910403", "delist_date": ""},
                {"ts_code": "000002.SZ", "name": "NORMAL", "list_status": "L", "list_date": "19910403", "delist_date": ""},
            ]
        )
        st_calendar = pd.DataFrame(
            [
                {"ts_code": " 000001.sz ", "st_start_date": "20240601", "st_end_date": ""},
            ]
        )

        before = filter_universe_frame(
            universe,
            universe="mainboard_a",
            as_of_date="2024-05-31",
            exclude_st=True,
            st_calendar=st_calendar,
        )
        during = filter_universe_frame(
            universe,
            universe="mainboard_a",
            as_of_date="2024-06-01",
            exclude_st=True,
            st_calendar=st_calendar,
        )

        self.assertIn("000001.SZ", before["ts_code"].tolist())
        self.assertNotIn("000001.SZ", during["ts_code"].tolist())

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

    def test_update_daily_data_defaults_to_history_start_date(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "history_start_date": "2021-01-01",
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
                update_daily_data(
                    stock_codes=["000001.SZ"],
                    end_date="2024-01-03",
                    raw_dir=raw_dir,
                )

        daily_calls = [call for call in client.calls if call[0] == "daily"]
        self.assertEqual(daily_calls[0][1]["start_date"], "20210101")

    def test_fetch_daily_stocks_skips_symbol_with_incomplete_adj_factor(self) -> None:
        client = MissingAdjFactorClient()

        df = fetch_daily_stocks(["000001.SZ", "600519.SH"], "2024-01-01", "2024-01-03", client=client)

        self.assertEqual(df["ts_code"].unique().tolist(), ["000001.SZ"])
        self.assertIn("adj_factor", df.columns)
        self.assertFalse(df["adj_factor"].isna().any())

    def test_fetch_daily_stock_drops_rows_with_missing_adj_factor(self) -> None:
        client = MissingAdjFactorClient()

        df = fetch_daily_stock("600519.SH", "2024-01-01", "2024-01-03", client=client, retries=1)

        self.assertTrue(df.empty)
        self.assertIn("adj_factor", df.columns)

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

    def test_resumable_update_prioritizes_missing_symbols_and_writes_progress(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "update_chunk_size": 1,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-03",
                        "open": 9.0,
                        "high": 9.0,
                        "low": 9.0,
                        "close": 9.0,
                        "vol": 100.0,
                        "amount": 900.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.TushareHttpClient.from_config", return_value=client):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH", "000002.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=1,
                    sleep_seconds=0,
                    max_chunks=1,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(set(written), {"600519.SH"})
            self.assertTrue((raw_dir / "600519.SH.csv").exists())
            self.assertFalse((raw_dir / "000002.SZ.csv").exists())
            self.assertEqual(int(progress["initial_existing"]), 1)
            self.assertEqual(int(progress["initial_latest_symbols"]), 1)
            self.assertEqual(int(progress["pending_symbols"]), 2)
            self.assertEqual(int(progress["completed_symbols"]), 1)
            self.assertEqual(int(progress["remaining_symbols"]), 1)
            self.assertEqual(int(progress["latest_symbols"]), 2)
            self.assertAlmostEqual(float(progress["latest_coverage"]), 2 / 3)

    def test_resumable_update_refreshes_stale_existing_symbols(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "update_chunk_size": 10,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 9.0,
                        "high": 9.0,
                        "low": 9.0,
                        "close": 9.0,
                        "vol": 100.0,
                        "amount": 900.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.TushareHttpClient.from_config", return_value=client):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=10,
                    sleep_seconds=0,
                )

            progress = pd.read_json(progress_file, typ="series")
            updated = pd.read_csv(raw_dir / "000001.SZ.csv", parse_dates=["trade_date"])

            self.assertEqual(set(written), {"000001.SZ"})
            self.assertEqual(updated["trade_date"].max(), pd.Timestamp("2024-01-03"))
            self.assertEqual(int(progress["initial_existing"]), 1)
            self.assertEqual(int(progress["initial_latest_symbols"]), 0)
            self.assertEqual(int(progress["pending_symbols"]), 1)
            self.assertEqual(int(progress["latest_symbols"]), 1)
            self.assertEqual(int(progress["remaining_symbols"]), 0)

    def test_resumable_update_marks_existing_empty_fetch_as_confirmed_no_new_data(self) -> None:
        client = EmptyTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "update_chunk_size": 10,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 9.0,
                        "high": 9.0,
                        "low": 9.0,
                        "close": 9.0,
                        "vol": 100.0,
                        "amount": 900.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.TushareHttpClient.from_config", return_value=client):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=10,
                    sleep_seconds=0,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(set(written), {"000001.SZ"})
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(int(progress["confirmed_no_new_data_symbols"]), 1)
            self.assertEqual(int(progress["fresh_or_confirmed_symbols"]), 1)
            self.assertEqual(int(progress["failed_symbols"]), 0)
            self.assertEqual(int(progress["remaining_symbols"]), 0)
            self.assertEqual(int(progress["latest_symbols"]), 0)

    def test_resumable_update_skips_previous_confirmed_no_new_data_symbols(self) -> None:
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "update_chunk_size": 10,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 9.0,
                        "high": 9.0,
                        "low": 9.0,
                        "close": 9.0,
                        "vol": 100.0,
                        "amount": 900.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            progress_file.write_text(
                '{"target_end_date":"2024-01-03","confirmed_no_new_data":["000001.SZ"]}',
                encoding="utf-8",
            )

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.update_daily_data") as update:
                update_daily_data_resumable(
                    stock_codes=["000001.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=10,
                    sleep_seconds=0,
                )

            progress = pd.read_json(progress_file, typ="series")

            update.assert_not_called()
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(int(progress["pending_symbols"]), 0)
            self.assertEqual(int(progress["confirmed_no_new_data_symbols"]), 1)
            self.assertEqual(int(progress["fresh_or_confirmed_symbols"]), 1)

    def test_resumable_update_marks_error_when_symbol_is_not_written(self) -> None:
        client = EmptyTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "update_chunk_size": 1,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.TushareHttpClient.from_config", return_value=client):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=1,
                    sleep_seconds=0,
                    max_chunks=1,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(written, {})
            self.assertEqual(progress["status"], "error")
            self.assertEqual(int(progress["failed_symbols"]), 1)
            self.assertIn("not_written", progress["last_error"])

    def test_resumable_update_calls_update_daily_data_once_per_chunk_start_group(self) -> None:
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "update_chunk_size": 3,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }
        calls: list[list[str]] = []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"

            def fake_update_daily_data(stock_codes, start_date=None, end_date=None, raw_dir=None, force_full=False):
                codes = list(stock_codes)
                calls.append(codes)
                written = {}
                for code in codes:
                    path = Path(raw_dir) / f"{code}.csv"
                    pd.DataFrame({"ts_code": [code], "trade_date": [end_date], "close": [10.0]}).to_csv(
                        path, index=False
                    )
                    written[code] = path
                return written

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.update_daily_data", side_effect=fake_update_daily_data):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH", "000002.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=3,
                    sleep_seconds=0,
                    max_chunks=1,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(calls, [["000001.SZ", "600519.SH", "000002.SZ"]])
            self.assertEqual(set(written), {"000001.SZ", "600519.SH", "000002.SZ"})
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(int(progress["completed_symbols"]), 3)

    def test_resumable_update_batches_existing_stale_symbols_despite_different_list_dates(self) -> None:
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "update_chunk_size": 10,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }
        calls: list[tuple[list[str], str | None]] = []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            for code in ["000001.SZ", "600519.SH"]:
                pd.DataFrame({"ts_code": [code], "trade_date": ["2024-01-02"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                    raw_dir / f"{code}.csv", index=False
                )

            def fake_update_daily_data(stock_codes, start_date=None, end_date=None, raw_dir=None, force_full=False):
                codes = list(stock_codes)
                calls.append((codes, start_date))
                for code in codes:
                    pd.DataFrame({"ts_code": [code], "trade_date": [end_date], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                        Path(raw_dir) / f"{code}.csv", index=False
                    )
                return {code: Path(raw_dir) / f"{code}.csv" for code in codes}

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.update_daily_data", side_effect=fake_update_daily_data):
                update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=10,
                    sleep_seconds=0,
                    max_chunks=1,
                )

            self.assertEqual(calls, [(["000001.SZ", "600519.SH"], "2024-01-01")])

    def test_resumable_update_force_full_batches_existing_symbols(self) -> None:
        config = {
            "data": {
                "start_date": "2019-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "update_chunk_size": 10,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }
        calls: list[tuple[list[str], str | None, bool]] = []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            for code in ["000001.SZ", "600519.SH"]:
                pd.DataFrame({"ts_code": [code], "trade_date": ["2024-01-03"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                    raw_dir / f"{code}.csv", index=False
                )

            def fake_update_daily_data(stock_codes, start_date=None, end_date=None, raw_dir=None, force_full=False):
                codes = list(stock_codes)
                calls.append((codes, start_date, force_full))
                return {code: Path(raw_dir) / f"{code}.csv" for code in codes}

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.update_daily_data", side_effect=fake_update_daily_data):
                update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=10,
                    sleep_seconds=0,
                    max_chunks=1,
                    force_full=True,
                )

            self.assertEqual(calls, [(["000001.SZ", "600519.SH"], "2019-01-01", True)])

    def test_resumable_update_include_existing_tracks_processed_symbols(self) -> None:
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "update_chunk_size": 1,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }
        calls: list[list[str]] = []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            for code in ["000001.SZ", "600519.SH"]:
                (raw_dir / f"{code}.csv").write_text("", encoding="utf-8")

            def fake_update_daily_data(stock_codes, start_date=None, end_date=None, raw_dir=None, force_full=False):
                codes = list(stock_codes)
                calls.append(codes)
                for code in codes:
                    pd.DataFrame({"ts_code": [code], "trade_date": [end_date], "close": [10.0]}).to_csv(
                        Path(raw_dir) / f"{code}.csv", index=False
                    )
                return {code: Path(raw_dir) / f"{code}.csv" for code in codes}

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.update_daily_data", side_effect=fake_update_daily_data):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=1,
                    sleep_seconds=0,
                    max_chunks=1,
                    include_existing=True,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(calls, [["000001.SZ"]])
            self.assertEqual(set(written), {"000001.SZ"})
            self.assertEqual(progress["status"], "partial")
            self.assertEqual(int(progress["completed_symbols"]), 1)
            self.assertEqual(int(progress["remaining_symbols"]), 1)


if __name__ == "__main__":
    unittest.main()
