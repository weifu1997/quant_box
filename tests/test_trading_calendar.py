"""模块说明：覆盖 test_trading_calendar 相关行为的测试用例。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.trading_calendar import next_business_day, resolve_target_date


class TradingCalendarTests(unittest.TestCase):
    """类说明：组织 TradingCalendarTests 测试用例。"""
    def test_auto_target_date_uses_previous_trade_date_before_cutoff(self) -> None:
        """函数说明：验证 test_auto_target_date_uses_previous_trade_date_before_cutoff 覆盖的行为场景。"""
        resolution = resolve_target_date(
            "auto",
            config=_config(),
            now=datetime(2024, 1, 3, 19, 59),
            calendar=["2024-01-02", "2024-01-03"],
        )

        self.assertEqual(resolution.target_date, "2024-01-02")
        self.assertEqual(resolution.latest_trade_date, "2024-01-03")
        self.assertEqual(resolution.reason, "before_latest_trade_date_cutoff")

    def test_auto_target_date_uses_latest_trade_date_after_cutoff(self) -> None:
        """函数说明：验证 test_auto_target_date_uses_latest_trade_date_after_cutoff 覆盖的行为场景。"""
        resolution = resolve_target_date(
            "auto",
            config=_config(),
            now=datetime(2024, 1, 3, 20, 1),
            calendar=["2024-01-02", "2024-01-03"],
        )

        self.assertEqual(resolution.target_date, "2024-01-03")
        self.assertEqual(resolution.reason, "after_latest_trade_date_cutoff")

    def test_auto_target_date_parses_tushare_compact_dates(self) -> None:
        """函数说明：验证 test_auto_target_date_parses_tushare_compact_dates 覆盖的行为场景。"""
        resolution = resolve_target_date(
            "auto",
            config=_config(),
            now=datetime(2024, 1, 3, 20, 1),
            calendar=["20240102", "20240103"],
        )

        self.assertEqual(resolution.target_date, "2024-01-03")

    def test_auto_target_date_prefers_tushare_trade_cal(self) -> None:
        """函数说明：验证 test_auto_target_date_prefers_tushare_trade_cal 覆盖的行为场景。"""
        class Response:
            """类说明：封装 Response 相关数据和行为。"""
            def raise_for_status(self) -> None:
                """函数说明：处理 raise_for_status 主要逻辑。"""
                return None

            def json(self) -> dict:
                """函数说明：处理 json 主要逻辑。"""
                return {
                    "code": 0,
                    "data": {
                        "fields": ["cal_date", "is_open"],
                        "items": [["20240102", 1], ["20240103", 1]],
                    },
                }

        with patch("requests.post", return_value=Response()):
            resolution = resolve_target_date(
                "auto",
                config=_config(http_url="http://example.test"),
                now=datetime(2024, 1, 3, 20, 1),
            )

        self.assertEqual(resolution.target_date, "2024-01-03")
        self.assertEqual(resolution.calendar_source, "tushare_trade_cal")

    def test_auto_target_date_uses_a_trade_calendar_when_tushare_is_unconfigured(self) -> None:
        """函数说明：验证 test_auto_target_date_uses_a_trade_calendar_when_tushare_is_unconfigured 覆盖的行为场景。"""
        resolution = resolve_target_date(
            "auto",
            config=_config(),
            now=datetime(2024, 2, 18, 12, 0),
        )

        self.assertEqual(resolution.target_date, "2024-02-08")
        self.assertEqual(resolution.calendar_source, "a_trade_calendar")

    def test_auto_target_date_falls_back_to_a_trade_calendar_when_tushare_fails(self) -> None:
        """函数说明：验证 test_auto_target_date_falls_back_to_a_trade_calendar_when_tushare_fails 覆盖的行为场景。"""
        with patch("requests.post", side_effect=RuntimeError("bad proxy")), patch(
            "src.trading_calendar._a_trade_calendar",
            return_value=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        ):
            resolution = resolve_target_date(
                "auto",
                config=_config(http_url="http://example.test"),
                now=datetime(2024, 1, 3, 20, 1),
            )

        self.assertEqual(resolution.target_date, "2024-01-03")
        self.assertEqual(resolution.calendar_source, "a_trade_calendar")
        self.assertIn("tushare_trade_cal_unavailable", resolution.calendar_warnings)

    def test_next_business_day_uses_a_share_trade_calendar(self) -> None:
        """函数说明：验证 test_next_business_day_uses_a_share_trade_calendar 覆盖的行为场景。"""
        self.assertEqual(str(next_business_day("2024-02-08").date()), "2024-02-19")

    def test_next_business_day_rejects_missing_or_invalid_date(self) -> None:
        """函数说明：验证 test_next_business_day_rejects_missing_or_invalid_date 覆盖的行为场景。"""
        with self.assertRaisesRegex(ValueError, "date is required"):
            next_business_day(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Invalid date"):
            next_business_day("not-a-date")

    def test_next_business_day_uses_configured_calendar_when_library_is_unavailable(self) -> None:
        """函数说明：验证 test_next_business_day_uses_configured_calendar_when_library_is_unavailable 覆盖的行为场景。"""
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "qlib_data"
            calendar_dir = provider / "calendars"
            calendar_dir.mkdir(parents=True)
            (calendar_dir / "day.txt").write_text("2024-02-08\n2024-02-19\n", encoding="utf-8")

            with patch("src.trading_calendar._a_trade_calendar_next_trade_date", return_value=None):
                next_date = next_business_day(
                    "2024-02-08",
                    config={"qlib": {"provider_uri": str(provider)}, "ic": {"price_file": str(Path(tmp) / "missing.parquet")}},
                )

        self.assertEqual(str(next_date.date()), "2024-02-19")

    def test_next_business_day_normalizes_calendar_is_open_values(self) -> None:
        """函数说明：验证 test_next_business_day_normalizes_calendar_is_open_values 覆盖的行为场景。"""
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "qlib_data"
            calendar_dir = provider / "calendars"
            calendar_dir.mkdir(parents=True)
            (calendar_dir / "day.txt").write_text(
                "cal_date,is_open\n2024-02-08, TRUE \n2024-02-09, false \n2024-02-19,1\n",
                encoding="utf-8",
            )
            config = {"qlib": {"provider_uri": str(provider)}, "ic": {"price_file": str(Path(tmp) / "missing.parquet")}}

            with patch("src.trading_calendar._a_trade_calendar_next_trade_date", return_value=None):
                first = next_business_day("2024-02-07", config=config)
                second = next_business_day("2024-02-08", config=config)

        self.assertEqual(str(first.date()), "2024-02-08")
        self.assertEqual(str(second.date()), "2024-02-19")

    def test_next_business_day_strict_raises_without_trade_calendar(self) -> None:
        """函数说明：验证 test_next_business_day_strict_raises_without_trade_calendar 覆盖的行为场景。"""
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.trading_calendar._a_trade_calendar_next_trade_date", return_value=None):
                with self.assertRaises(ValueError):
                    next_business_day(
                        "2024-02-08",
                        config={
                            "qlib": {"provider_uri": str(Path(tmp) / "missing_qlib")},
                            "ic": {"price_file": str(Path(tmp) / "missing.parquet")},
                        },
                        strict=True,
                    )

    def test_fixed_target_date_bypasses_cutoff(self) -> None:
        """函数说明：验证 test_fixed_target_date_bypasses_cutoff 覆盖的行为场景。"""
        resolution = resolve_target_date(
            "2024-01-03",
            config=_config(),
            now=datetime(2024, 1, 3, 10, 0),
            calendar=["2024-01-02", "2024-01-03"],
        )

        self.assertEqual(resolution.target_date, "2024-01-03")
        self.assertEqual(resolution.reason, "fixed_end_date")


def _config(http_url: str = "") -> dict:
    """函数说明：处理 config 的内部辅助逻辑。"""
    return {
        "data": {
            "end_date": "auto",
            "target_date_cutoff_time": "20:00",
            "timezone": "Asia/Shanghai",
        },
        "tushare": {"http_url": http_url, "token": "", "timeout": 30},
    }


if __name__ == "__main__":
    unittest.main()
