"""模块说明：覆盖 test_data_governance 相关行为的测试用例。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.adj_factor_metadata import build_adj_factor_metadata, write_adj_factor_metadata
from src.data_governance import build_data_governance_report, write_data_governance_report


class DataGovernanceTests(unittest.TestCase):
    """类说明：组织 DataGovernanceTests 测试用例。"""
    def test_build_adj_factor_metadata_records_digest_and_date_range(self) -> None:
        """函数说明：验证 test_build_adj_factor_metadata_records_digest_and_date_range 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "trade_date": ["2024-01-04", "2024-01-03"],
                    "close": [10.5, 10.0],
                    "adj_factor": [1.1, 1.0],
                }
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            path = write_adj_factor_metadata(metadata, path=root / "adj_factor_meta.json")
            saved = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(metadata.raw_file_count, 1)
            self.assertEqual(metadata.files_with_adj_factor, 1)
            self.assertEqual(metadata.symbol_count, 1)
            self.assertEqual(metadata.start_date, "2024-01-03")
            self.assertEqual(metadata.end_date, "2024-01-04")
            self.assertEqual(len(saved["digest"]), 64)
            self.assertEqual(saved["symbols"][0]["last_adj_factor"], 1.1)

    def test_build_adj_factor_metadata_ignores_index_raw_files_without_adj_factor(self) -> None:
        """函数说明：验证指数 raw 文件没有复权因子时不会污染股票复权元数据。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["2024-01-03"],
                    "close": [10.0],
                    "adj_factor": [1.0],
                }
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            pd.DataFrame(
                {
                    "ts_code": ["000300.SH"],
                    "trade_date": ["2024-01-03"],
                    "close": [3500.0],
                }
            ).to_csv(raw_dir / "000300.SH.csv", index=False)
            pd.DataFrame(
                {
                    "ts_code": ["000905.SH"],
                    "trade_date": ["2024-01-03"],
                    "close": [5500.0],
                }
            ).to_csv(raw_dir / "000905.SH.csv", index=False)

            metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})

            self.assertEqual(metadata.raw_file_count, 1)
            self.assertEqual(metadata.files_with_adj_factor, 1)
            self.assertEqual(metadata.symbol_count, 1)
            self.assertEqual(metadata.issues, [])

    def test_build_data_governance_report_tracks_point_in_time_evidence(self) -> None:
        """函数说明：验证 test_build_data_governance_report_tracks_point_in_time_evidence 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "hs300_constituents.csv"
            factor_cache = root / "factors" / "alpha158.parquet"
            factor_cache.parent.mkdir()
            adj_meta = root / "factors" / "adj_factor_meta.json"
            daily_basic_file = root / "factors" / "daily_basic.parquet"

            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "name": ["A", "B"],
                    "industry": ["Bank", "Property"],
                    "list_status": ["L", "D"],
                    "list_date": ["20200101", "20200101"],
                    "delist_date": ["", "20240131"],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20220101"], "st_end_date": ["20240103"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "con_code": ["000001.SZ"],
                    "trade_date": ["20240103"],
                    "weight": [1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-03"],
                    "ts_code": ["000001.SZ"],
                    "circ_mv": [100.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-01-03"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"end_date": "2024-01-03", "symbols": ["000001.SZ"]}),
                encoding="utf-8",
            )
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            config = {
                "data": {
                    "raw_dir": str(raw_dir),
                    "constituents_file": str(universe_file),
                    "st_calendar_file": str(st_calendar),
                    "exclude_st": True,
                    "daily_basic_file": str(daily_basic_file),
                },
                "factors": {"cache_file": str(factor_cache)},
                "data_governance": {
                    "index_constituents_file": str(index_file),
                    "adj_factor_meta_file": str(adj_meta),
                },
                "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
            }

            report = build_data_governance_report(config, sample_raw_files=1)
            path = write_data_governance_report(report, root)

            self.assertTrue(report.is_point_in_time_ready)
            self.assertEqual(report.st_filter_mode, "historical_calendar")
            self.assertEqual(report.delisted_rows, 1)
            self.assertTrue(report.index_constituents_has_trade_date)
            self.assertTrue(report.index_constituents_has_weight)
            self.assertTrue(report.daily_basic_available)
            self.assertTrue(report.daily_basic_has_trade_date)
            self.assertTrue(report.daily_basic_has_ts_code)
            self.assertTrue(report.daily_basic_has_market_cap)
            self.assertTrue(report.st_calendar_has_ts_code)
            self.assertEqual(report.st_calendar_start_date, "2022-01-01")
            self.assertEqual(report.st_calendar_end_date, "2024-01-03")
            self.assertEqual(report.daily_basic_end_date, "2024-01-03")
            self.assertEqual(report.raw_adj_factor_files_with_column, 1)
            self.assertTrue(report.factor_cache_meta_available)
            self.assertTrue(report.adj_factor_meta_available)
            self.assertEqual(report.adj_factor_meta_source, "raw_csv_adj_factor")
            self.assertEqual(report.adj_factor_meta_raw_file_count, 1)
            self.assertEqual(report.adj_factor_meta_files_with_adj_factor, 1)
            self.assertEqual(report.adj_factor_meta_end_date, "2024-01-03")
            self.assertEqual(len(report.adj_factor_meta_digest), 64)
            self.assertEqual(report.repair_actions, [])
            self.assertTrue(path.exists())
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["repair_actions"], [])

    def test_build_data_governance_report_flags_current_name_st_filter(self) -> None:
        """函数说明：验证 test_build_data_governance_report_flags_current_name_st_filter 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                    "delist_date": [""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-01-03"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )

            report = build_data_governance_report(
                {
                    "data": {
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "daily_basic_file": str(root / "missing_daily_basic.parquet"),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(root / "missing.parquet")},
                    "research": {"exposure": {"daily_basic_file": str(root / "missing_daily_basic.parquet")}},
                },
                sample_raw_files=1,
            )

            self.assertFalse(report.is_point_in_time_ready)
            self.assertIn("st_calendar_missing_current_name_filter_only", report.issues)
            self.assertIn("daily_basic_missing_market_cap_exposure_unavailable", report.issues)
            self.assertIn("universe_industry_missing", report.warnings)

    def test_build_data_governance_report_flags_partial_daily_basic_history(self) -> None:
        """函数说明：验证 test_build_data_governance_report_flags_partial_daily_basic_history 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "hs300_constituents.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"

            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "industry": ["Bank"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                    "delist_date": [""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20240101"]}).to_csv(st_calendar, index=False)
            pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "con_code": ["000001.SZ"],
                    "trade_date": ["20240103"],
                    "weight": [1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-03"],
                    "ts_code": ["000001.SZ"],
                    "circ_mv": [100.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-01-03"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"start_date": "2024-01-02", "end_date": "2024-02-05", "symbols": ["000001.SZ"]}),
                encoding="utf-8",
            )
            adj_meta = factor_dir / "adj_factor_meta.json"
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-01",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "data_governance": {
                        "index_constituents_file": str(index_file),
                        "adj_factor_meta_file": str(adj_meta),
                    },
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=1,
            )

            self.assertFalse(report.is_point_in_time_ready)
            self.assertEqual(report.factor_cache_meta_start_date, "2024-01-02")
            self.assertEqual(report.point_in_time_start_date, "2024-01-02")
            self.assertIn("daily_basic_start_after_point_in_time_start:2024-01-03>2024-01-02", report.issues)
            self.assertIn("index_constituents_month_coverage_below_required:1/2<1.00", report.issues)
            self.assertIn("index_constituents_end_before_factor_end:2024-01-03<2024-02-05", report.issues)
            self.assertIn("daily_basic_end_before_factor_end:2024-01-03<2024-02-05", report.warnings)
            repair_components = {action["component"] for action in report.repair_actions}
            self.assertIn("daily_basic", repair_components)
            self.assertIn("index_constituents", repair_components)
            daily_action = next(action for action in report.repair_actions if action["component"] == "daily_basic")
            self.assertEqual(daily_action["start_date"], "2024-01-02")
            self.assertEqual(daily_action["end_date"], "2024-02-05")
            self.assertIn("--skip-index-constituents --skip-st-calendar", daily_action["commands"][0])

    def test_build_data_governance_report_flags_point_in_time_coverage_gaps(self) -> None:
        """函数说明：验证 test_build_data_governance_report_flags_point_in_time_coverage_gaps 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "hs300_constituents.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"
            price_file = root / "prices.parquet"

            price_dates = pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-02-01", "2024-03-01"]
            )
            pd.DataFrame({"close": range(len(price_dates))}, index=price_dates).to_parquet(price_file)
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "industry": ["Bank"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                    "delist_date": [""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20230101"], "st_end_date": ["20240301"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "con_code": ["000001.SZ"],
                    "trade_date": ["20240102"],
                    "weight": [1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-02", "2024-03-01"],
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "circ_mv": [100.0, 103.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-03-01"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"start_date": "2024-01-02", "end_date": "2024-03-01", "symbols": ["000001.SZ"]}),
                encoding="utf-8",
            )
            adj_meta = factor_dir / "adj_factor_meta.json"
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-02",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "ic": {"price_file": str(price_file)},
                    "data_governance": {
                        "index_constituents_file": str(index_file),
                        "adj_factor_meta_file": str(adj_meta),
                    },
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=1,
            )

            self.assertFalse(report.is_point_in_time_ready)
            self.assertEqual(report.daily_basic_expected_dates, 6)
            self.assertEqual(report.daily_basic_covered_dates, 2)
            self.assertEqual(report.daily_basic_missing_dates, 4)
            self.assertEqual(report.index_constituents_expected_months, 3)
            self.assertEqual(report.index_constituents_observed_months, 1)
            self.assertEqual(report.index_constituents_missing_months, 2)
            self.assertIn("daily_basic_date_coverage_below_required:2/6<1.00", report.issues)
            self.assertIn("index_constituents_month_coverage_below_required:1/3<1.00", report.issues)

    def test_build_data_governance_report_flags_historical_universe_source_gaps(self) -> None:
        """Verify enabled historical universe coverage is checked per source label."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "index_constituents.csv"
            historical_universe = raw_dir / "historical_universe.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"

            months = ["2024-01-02", "2024-02-01", "2024-03-01"]
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "industry": ["Bank"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                    "delist_date": [""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20230101"], "st_end_date": ["20240301"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "index_code": ["000300.SH", "000300.SH", "000300.SH"],
                    "con_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "trade_date": months,
                    "weight": [1.0, 1.0, 1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                [
                    {"trade_date": "2024-01-02", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                    {"trade_date": "2024-02-01", "instrument": "000001.SZ", "sources": "hs300|csi1000"},
                    {"trade_date": "2024-03-01", "instrument": "000001.SZ", "sources": "hs300|csi1000"},
                ]
            ).to_csv(historical_universe, index=False)
            pd.DataFrame(
                {
                    "trade_date": months,
                    "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "circ_mv": [100.0, 101.0, 102.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-03-01"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"start_date": "2024-01-02", "end_date": "2024-03-01", "symbols": ["000001.SZ"]}),
                encoding="utf-8",
            )
            adj_meta = factor_dir / "adj_factor_meta.json"
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-02",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "data_governance": {
                        "index_constituents_file": str(index_file),
                        "adj_factor_meta_file": str(adj_meta),
                    },
                    "universe_builder": {
                        "enabled": True,
                        "output_file": str(historical_universe),
                    },
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=1,
            )

            self.assertFalse(report.is_point_in_time_ready)
            self.assertTrue(report.historical_universe_available)
            self.assertEqual(report.historical_universe_expected_months, 3)
            self.assertEqual(report.historical_universe_sources, ["csi1000", "csi500", "hs300"])
            self.assertEqual(report.historical_universe_source_coverage["hs300"]["observed_months"], 3)
            self.assertEqual(report.historical_universe_source_coverage["csi1000"]["observed_months"], 3)
            self.assertEqual(report.historical_universe_source_coverage["csi500"]["observed_months"], 1)
            self.assertIn("historical_universe_source_month_coverage_below_required:csi500:1/3<1.00", report.issues)
            repair_components = {action["component"] for action in report.repair_actions}
            self.assertIn("historical_universe", repair_components)

    def test_build_data_governance_report_caps_coverage_at_configured_end_date(self) -> None:
        """Verify fixed-date runs are not blocked by later factor-cache metadata."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "index_constituents.csv"
            historical_universe = raw_dir / "historical_universe.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"
            price_file = root / "prices.parquet"

            price_dates = pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"])
            pd.DataFrame({"close": range(len(price_dates))}, index=price_dates).to_parquet(price_file)
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "industry": ["Bank"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                    "delist_date": [""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20230101"], "st_end_date": ["20240215"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "index_code": ["000300.SH", "000300.SH"],
                    "con_code": ["000001.SZ", "000001.SZ"],
                    "trade_date": ["2024-01-02", "2024-02-01"],
                    "weight": [1.0, 1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                [
                    {"trade_date": "2024-01-02", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                    {"trade_date": "2024-02-01", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                ]
            ).to_csv(historical_universe, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-02", "2024-02-01"],
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "circ_mv": [100.0, 101.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-02-01"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"start_date": "2024-01-02", "end_date": "2024-03-01", "symbols": ["000001.SZ"]}),
                encoding="utf-8",
            )
            adj_meta = factor_dir / "adj_factor_meta.json"
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-02",
                        "end_date": "2024-02-15",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "ic": {"price_file": str(price_file)},
                    "data_governance": {
                        "index_constituents_file": str(index_file),
                        "adj_factor_meta_file": str(adj_meta),
                    },
                    "universe_builder": {
                        "enabled": True,
                        "output_file": str(historical_universe),
                    },
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=1,
            )

            self.assertTrue(report.is_point_in_time_ready)
            self.assertEqual(report.index_constituents_expected_months, 2)
            self.assertEqual(report.historical_universe_expected_months, 2)
            self.assertEqual(report.daily_basic_expected_dates, 2)
            self.assertEqual(report.factor_cache_meta_end_date, "2024-03-01")
            self.assertEqual(report.issues, [])

    def test_build_data_governance_report_carries_prior_snapshot_for_terminal_partial_month(self) -> None:
        """Verify latest prior source snapshots cover only the terminal partial month."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "index_constituents.csv"
            historical_universe = raw_dir / "historical_universe.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"

            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "industry": ["Bank"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                    "delist_date": [""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20230101"], "st_end_date": ["20240315"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "index_code": ["000300.SH", "000300.SH", "000300.SH"],
                    "con_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "trade_date": ["2024-01-31", "2024-02-29", "2024-03-01"],
                    "weight": [1.0, 1.0, 1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                [
                    {"trade_date": "2024-01-31", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                    {"trade_date": "2024-02-29", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                    {"trade_date": "2024-03-01", "instrument": "000001.SZ", "sources": "hs300"},
                ]
            ).to_csv(historical_universe, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-31", "2024-02-29", "2024-03-01"],
                    "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "circ_mv": [100.0, 101.0, 102.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-03-01"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"start_date": "2024-01-31", "end_date": "2024-03-01", "symbols": ["000001.SZ"]}),
                encoding="utf-8",
            )
            adj_meta = factor_dir / "adj_factor_meta.json"
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-31",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "data_governance": {
                        "index_constituents_file": str(index_file),
                        "adj_factor_meta_file": str(adj_meta),
                    },
                    "universe_builder": {
                        "enabled": True,
                        "output_file": str(historical_universe),
                    },
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=1,
            )

            self.assertTrue(report.is_point_in_time_ready)
            self.assertEqual(report.historical_universe_source_coverage["csi500"]["observed_months"], 3)
            self.assertEqual(report.historical_universe_source_coverage["csi500"]["carried_forward_month_values"], ["2024-03"])
            self.assertEqual(report.historical_universe_source_coverage["csi1000"]["carried_forward_month_values"], ["2024-03"])

    def test_build_data_governance_report_does_not_invent_end_date_without_factor_metadata(self) -> None:
        """Verify configured end date does not replace missing factor-cache metadata."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "index_constituents.csv"
            historical_universe = raw_dir / "historical_universe.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"
            price_file = root / "prices.parquet"

            price_dates = pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"])
            pd.DataFrame({"close": range(len(price_dates))}, index=price_dates).to_parquet(price_file)
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "name": ["A", "B"],
                    "industry": ["Bank", "Tech"],
                    "list_status": ["L", "D"],
                    "list_date": ["20200101", "20200101"],
                    "delist_date": ["", "20240115"],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20230101"], "st_end_date": ["20240301"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "index_code": ["000300.SH", "000300.SH", "000300.SH"],
                    "con_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "trade_date": ["2024-01-02", "2024-02-01", "2024-03-01"],
                    "weight": [1.0, 1.0, 1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                [
                    {"trade_date": "2024-01-02", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                    {"trade_date": "2024-02-01", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                    {"trade_date": "2024-03-01", "instrument": "000001.SZ", "sources": "hs300|csi500|csi1000"},
                ]
            ).to_csv(historical_universe, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-02", "2024-02-01", "2024-03-01"],
                    "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "circ_mv": [100.0, 101.0, 102.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["2024-03-01"],
                    "close": [10.0],
                    "adj_factor": [1.0],
                }
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            adj_meta = factor_dir / "adj_factor_meta.json"
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-02",
                        "end_date": "2024-02-15",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "ic": {"price_file": str(price_file)},
                    "data_governance": {
                        "index_constituents_file": str(index_file),
                        "adj_factor_meta_file": str(adj_meta),
                    },
                    "universe_builder": {
                        "enabled": True,
                        "output_file": str(historical_universe),
                    },
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=1,
            )

            self.assertEqual(report.factor_cache_meta_end_date, "")
            self.assertEqual(report.warnings, ["factor_cache_meta_missing"])
            self.assertEqual(report.issues, [])
            self.assertEqual(report.index_constituents_expected_months, 0)
            self.assertEqual(report.historical_universe_expected_months, 0)
            self.assertEqual(report.repair_actions, [])

    def test_build_data_governance_report_flags_partial_st_calendar_history(self) -> None:
        """函数说明：验证 test_build_data_governance_report_flags_partial_st_calendar_history 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"

            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "industry": ["Bank"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                    "delist_date": [""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"symbol": ["000001.SZ"], "st_start_date": ["20240103"], "st_end_date": ["20240104"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-01"],
                    "ts_code": ["000001.SZ"],
                    "circ_mv": [100.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-01-01"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"end_date": "2024-01-05", "symbols": ["000001.SZ"]}),
                encoding="utf-8",
            )
            adj_meta = factor_dir / "adj_factor_meta.json"
            adj_metadata = build_adj_factor_metadata({"data": {"raw_dir": str(raw_dir)}})
            write_adj_factor_metadata(adj_metadata, path=adj_meta)

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-01",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "data_governance": {"adj_factor_meta_file": str(adj_meta)},
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=1,
            )

            self.assertFalse(report.is_point_in_time_ready)
            self.assertFalse(report.st_calendar_has_ts_code)
            self.assertEqual(report.st_calendar_start_date, "2024-01-03")
            self.assertEqual(report.st_calendar_end_date, "2024-01-04")
            self.assertIn("st_calendar_ts_code_missing", report.issues)
            self.assertIn("st_calendar_start_after_data_start:2024-01-03>2024-01-01", report.issues)
            self.assertIn("st_calendar_end_before_factor_end:2024-01-04<2024-01-05", report.warnings)

    def test_build_data_governance_report_flags_incomplete_adj_factor_metadata_as_issue(self) -> None:
        """函数说明：验证 test_build_data_governance_report_flags_incomplete_adj_factor_metadata_as_issue 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            factor_dir = root / "factors"
            factor_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            st_calendar = raw_dir / "st_calendar.csv"
            index_file = raw_dir / "hs300_constituents.csv"
            daily_basic_file = factor_dir / "daily_basic.parquet"
            factor_cache = factor_dir / "alpha158.parquet"
            adj_meta = factor_dir / "adj_factor_meta.json"

            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "name": ["A", "B"],
                    "industry": ["Bank", "Tech"],
                    "list_status": ["L", "L"],
                    "list_date": ["20200101", "20200101"],
                    "delist_date": ["", ""],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame({"ts_code": ["000001.SZ"], "st_start_date": ["20230101"], "st_end_date": ["20240105"]}).to_csv(
                st_calendar,
                index=False,
            )
            pd.DataFrame(
                {
                    "index_code": ["000300.SH", "000300.SH"],
                    "con_code": ["000001.SZ", "000001.SZ"],
                    "trade_date": ["20240101", "20240105"],
                    "weight": [1.0, 1.0],
                }
            ).to_csv(index_file, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-01", "2024-01-05"],
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "circ_mv": [100.0, 101.0],
                }
            ).to_parquet(daily_basic_file)
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["2024-01-05"], "close": [10.0], "adj_factor": [1.0]}).to_csv(
                raw_dir / "000001.SZ.csv",
                index=False,
            )
            pd.DataFrame({"ts_code": ["000002.SZ"], "trade_date": ["2024-01-05"], "close": [20.0]}).to_csv(
                raw_dir / "000002.SZ.csv",
                index=False,
            )
            (Path(str(factor_cache) + ".meta.json")).write_text(
                json.dumps({"end_date": "2024-01-05", "symbols": ["000001.SZ", "000002.SZ"]}),
                encoding="utf-8",
            )
            adj_meta.write_text(
                json.dumps(
                    {
                        "source": "raw_csv_adj_factor",
                        "raw_file_count": 2,
                        "files_with_adj_factor": 1,
                        "symbol_count": 1,
                        "end_date": "2024-01-05",
                        "digest": "",
                        "issues": ["adj_factor_missing:000002.SZ"],
                    }
                ),
                encoding="utf-8",
            )

            report = build_data_governance_report(
                {
                    "data": {
                        "start_date": "2024-01-01",
                        "raw_dir": str(raw_dir),
                        "constituents_file": str(universe_file),
                        "st_calendar_file": str(st_calendar),
                        "exclude_st": True,
                    },
                    "factors": {"cache_file": str(factor_cache)},
                    "data_governance": {
                        "index_constituents_file": str(index_file),
                        "adj_factor_meta_file": str(adj_meta),
                    },
                    "research": {"exposure": {"daily_basic_file": str(daily_basic_file), "market_cap_field": "circ_mv"}},
                },
                sample_raw_files=2,
            )

            self.assertFalse(report.is_point_in_time_ready)
            self.assertIn("raw_adj_factor_missing_in_sample:1/2", report.issues)
            self.assertIn("adj_factor_version_meta_missing_files:1/2", report.issues)
            self.assertIn("adj_factor_version_meta_digest_missing", report.issues)
            self.assertIn("adj_factor_version_meta_issues:adj_factor_missing:000002.SZ", report.issues)
            self.assertEqual(report.adj_factor_meta_missing_symbols, ["000002.SZ"])
            adj_action = next(action for action in report.repair_actions if action["component"] == "adj_factor_version")
            self.assertEqual(adj_action["output"], str(adj_meta))
            self.assertEqual(adj_action["missing_symbols"], ["000002.SZ"])
            self.assertIn("scripts\\run_update_data.py --codes 000002.SZ", adj_action["commands"][0])
            self.assertIn("--force-full", adj_action["commands"][0])
            self.assertNotIn("--include-existing", adj_action["commands"][0])
            self.assertIn("scripts\\run_build_adj_factor_meta.py", adj_action["commands"][-1])


if __name__ == "__main__":
    unittest.main()
