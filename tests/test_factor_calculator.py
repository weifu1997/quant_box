"""模块说明：覆盖 test_factor_calculator 相关行为的测试用例。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.factor_calculator as factor_calculator
from src.factor_calculator import _ensure_qlib_initialized, _replace_infinite_factors_in_place, load_or_compute_factors
from tests.fixtures.real_data import require_real_market_data


class FakeQlib:
    """类说明：提供 FakeQlib 测试替身实现。"""
    def __init__(self) -> None:
        """函数说明：初始化实例状态。"""
        self.calls: list[tuple[str, str, int | None]] = []

    def init(self, provider_uri: str, region: str, kernels: int | None = None) -> None:
        """函数说明：处理 init 主要逻辑。"""
        self.calls.append((provider_uri, region, kernels))


class FactorCalculatorTests(unittest.TestCase):
    """类说明：组织 FactorCalculatorTests 测试用例。"""
    def tearDown(self) -> None:
        """函数说明：清理测试用例运行后的临时状态。"""
        factor_calculator._QLIB_INIT_STATE = None

    def test_ensure_qlib_initialized_reuses_matching_provider_and_region(self) -> None:
        """函数说明：验证 test_ensure_qlib_initialized_reuses_matching_provider_and_region 覆盖的行为场景。"""
        fake = FakeQlib()
        provider = Path("data/qlib_data")

        _ensure_qlib_initialized(fake, provider, "cn")
        _ensure_qlib_initialized(fake, provider, "cn")

        self.assertEqual(fake.calls, [(str(provider), "cn", None)])

    def test_ensure_qlib_initialized_passes_bounded_kernel_count(self) -> None:
        fake = FakeQlib()
        provider = Path("data/qlib_data")

        _ensure_qlib_initialized(fake, provider, "cn", kernels=4)

        self.assertEqual(fake.calls, [(str(provider), "cn", 4)])

    def test_replace_infinite_factors_cleans_columns_in_place(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "F1": np.array([np.inf, 1.0], dtype=np.float64),
                "F2": np.array([-np.inf, 2.0], dtype=np.float32),
                "label": ["x", "y"],
            },
            index=index,
        )

        cleaned = _replace_infinite_factors_in_place(factors)

        self.assertIs(cleaned, factors)
        self.assertTrue(pd.isna(cleaned.iloc[0]["F1"]))
        self.assertTrue(pd.isna(cleaned.iloc[0]["F2"]))
        self.assertEqual(cleaned["label"].tolist(), ["x", "y"])

    def test_load_or_compute_factors_recomputes_when_price_panel_has_new_symbol(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_recomputes_when_price_panel_has_new_symbol 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            cached_market = require_real_market_data(
                instruments=["000001.sz"],
                start="2024-01-02",
                end="2024-01-05",
                factor_columns=("LOW0",),
            )
            full_market = require_real_market_data(
                instruments=["000001.sz", "000002.sz"],
                start="2024-01-02",
                end="2024-01-05",
                factor_columns=("LOW0",),
            )
            cached_market.factors.to_parquet(cache_path)
            full_market.prices.to_parquet(price_path)
            config = {"factors": {"cache_file": str(cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors", return_value=full_market.factors) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_called_once()
        self.assertEqual(set(factors.index.get_level_values("instrument")), set(full_market.instruments))

    def test_load_or_compute_factors_reuses_matching_cache(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_reuses_matching_cache 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            market = require_real_market_data(
                instruments=["000001.sz", "000002.sz"],
                start="2024-01-02",
                end="2024-01-05",
                factor_columns=("LOW0",),
            )
            market.factors.to_parquet(cache_path)
            market.prices.to_parquet(price_path)
            config = {"factors": {"cache_file": str(cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_not_called()
        expected_dates = {pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")}
        self.assertEqual(set(pd.to_datetime(factors.index.get_level_values("datetime")).normalize()), expected_dates)
        self.assertEqual(set(factors.index.get_level_values("instrument")), set(market.instruments))

    def test_load_or_compute_factors_normalizes_cache_symbol_coverage(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_normalizes_cache_symbol_coverage 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            index = pd.MultiIndex.from_product([dates, [" 000001.sz "]], names=["datetime", "instrument"])
            cached = pd.DataFrame({"F1": [1.0, 2.0]}, index=index)
            cached.to_parquet(cache_path)
            prices = pd.concat(
                {
                    "close": pd.DataFrame({"000001.SZ": [10.0, 10.1]}, index=dates),
                },
                axis=1,
            )
            prices.to_parquet(price_path)
            config = {"factors": {"cache_file": str(cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_not_called()
        self.assertEqual(len(factors), len(cached))

    def test_load_or_compute_factors_rejects_flat_ohlcv_price_frame(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_rejects_flat_ohlcv_price_frame 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
            pd.DataFrame({"F1": [1.0, 2.0]}, index=index).to_parquet(cache_path)
            pd.DataFrame(
                {
                    "open": [10.0, 10.1],
                    "close": [10.2, 10.3],
                    "volume": [1000.0, 1100.0],
                },
                index=dates,
            ).to_parquet(price_path)
            config = {"factors": {"cache_file": str(cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                with self.assertRaisesRegex(ValueError, "close-price panel"):
                    load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_not_called()

    def test_load_or_compute_factors_recomputes_when_requested_columns_are_missing(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_recomputes_when_requested_columns_are_missing 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            cached_market = require_real_market_data(
                instruments=["000001.sz", "000002.sz"],
                start="2024-01-02",
                end="2024-01-05",
                factor_columns=("LOW0",),
            )
            recomputed_market = require_real_market_data(
                instruments=["000001.sz", "000002.sz"],
                start="2024-01-02",
                end="2024-01-05",
                factor_columns=("LOW0", "ROC5"),
            )
            cached_market.factors.to_parquet(cache_path)
            recomputed_market.prices.to_parquet(price_path)
            config = {"factors": {"cache_file": str(cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors", return_value=recomputed_market.factors) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path, columns=["LOW0", "ROC5"])

        compute.assert_called_once()
        self.assertEqual(factors.columns.tolist(), ["LOW0", "ROC5"])

    def test_load_or_compute_factors_reuses_custom_cache_without_meta(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_cache_path = root / "alpha158.parquet"
            custom_cache_path = root / "extended_factors.parquet"
            price_path = root / "ohlcv.parquet"
            dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
            instruments = ["A", "B"]
            index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
            cached = pd.DataFrame(
                {
                    "DB_pb": range(len(index)),
                    "ROC60": range(100, 100 + len(index)),
                },
                index=index,
            )
            cached.to_parquet(custom_cache_path)
            prices = pd.concat(
                {
                    "close": pd.DataFrame(
                        {
                            "A": [10.0, 10.1, 10.2],
                            "B": [20.0, 20.1, 20.2],
                        },
                        index=dates,
                    ),
                },
                axis=1,
            )
            prices.to_parquet(price_path)
            config = {"factors": {"cache_file": str(default_cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors(
                    "2024-01-02",
                    "2024-01-03",
                    cache_file=custom_cache_path,
                    columns=["DB_pb"],
                )

        compute.assert_not_called()
        self.assertEqual(factors.columns.tolist(), ["DB_pb"])
        self.assertEqual(set(factors.index.get_level_values("instrument")), set(instruments))
        self.assertEqual(
            set(pd.to_datetime(factors.index.get_level_values("datetime")).normalize()),
            {pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")},
        )

    def test_load_or_compute_factors_rejects_custom_cache_missing_requested_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_cache_path = root / "alpha158.parquet"
            custom_cache_path = root / "extended_factors.parquet"
            price_path = root / "ohlcv.parquet"
            dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
            pd.DataFrame({"DB_pb": [1.0, 2.0]}, index=index).to_parquet(custom_cache_path)
            pd.concat(
                {
                    "close": pd.DataFrame({"A": [10.0, 10.1]}, index=dates),
                },
                axis=1,
            ).to_parquet(price_path)
            config = {"factors": {"cache_file": str(default_cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                with self.assertRaisesRegex(ValueError, "missing requested columns: DB_missing"):
                    load_or_compute_factors(
                        "2024-01-02",
                        "2024-01-03",
                        cache_file=custom_cache_path,
                        columns=["DB_pb", "DB_missing"],
                    )
            remaining_columns = pd.read_parquet(custom_cache_path).columns.tolist()

        compute.assert_not_called()
        self.assertEqual(remaining_columns, ["DB_pb"])

    def test_load_or_compute_factors_reuses_cache_when_request_starts_before_first_trading_day(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_reuses_cache_when_request_starts_before_first_trading_day 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            provider = root / "qlib"
            trading_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            index = pd.MultiIndex.from_product([trading_dates, ["A", "B"]], names=["datetime", "instrument"])
            cached = pd.DataFrame({"F1": range(len(index))}, index=index)
            cached.to_parquet(cache_path)
            prices = pd.concat(
                {
                    "close": pd.DataFrame({"A": [10.0, 10.1], "B": [20.0, 20.1]}, index=trading_dates),
                },
                axis=1,
            )
            prices.to_parquet(price_path)
            (cache_path.with_name(f"{cache_path.name}.meta.json")).write_text(
                json.dumps(
                    {
                        "provider_uri": str(provider),
                        "region": "cn",
                        "instruments": "mainboard_a",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-03",
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "factors": {"cache_file": str(cache_path)},
                "ic": {"price_file": str(price_path)},
                "qlib": {"provider_uri": str(provider), "region": "cn", "instruments": "mainboard_a"},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors("2024-01-01", "2024-01-03", cache_file=cache_path)

        compute.assert_not_called()
        self.assertEqual(set(pd.to_datetime(factors.index.get_level_values("datetime")).date), set(trading_dates.date))

    def test_load_or_compute_factors_reuses_superset_cache_and_slices_dates(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_reuses_superset_cache_and_slices_dates 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            provider = root / "qlib"
            requested_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            market = require_real_market_data(
                instruments=["000001.sz", "000002.sz"],
                start="2024-01-02",
                end="2024-01-05",
                factor_columns=("LOW0",),
            )
            market.factors.to_parquet(cache_path)
            market.prices.loc[requested_dates].to_parquet(price_path)
            (cache_path.with_name(f"{cache_path.name}.meta.json")).write_text(
                json.dumps(
                    {
                        "provider_uri": str(provider),
                        "region": "cn",
                        "instruments": "mainboard_a",
                        "start_date": "2024-01-02",
                        "end_date": "2024-01-05",
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "factors": {"cache_file": str(cache_path)},
                "ic": {"price_file": str(price_path)},
                "qlib": {"provider_uri": str(provider), "region": "cn", "instruments": "mainboard_a"},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_not_called()
        self.assertEqual(set(pd.to_datetime(factors.index.get_level_values("datetime")).date), set(requested_dates.date))
        self.assertEqual(set(factors.index.get_level_values("instrument")), set(market.instruments))

    def test_load_or_compute_factors_reuses_cache_with_short_stale_tail(self) -> None:
        """函数说明：验证因子缓存只缺少量尾部日期时不会全量重算。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            provider = root / "qlib"
            factor_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            price_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
            index = pd.MultiIndex.from_product([factor_dates, ["A", "B"]], names=["datetime", "instrument"])
            cached = pd.DataFrame({"F1": range(len(index))}, index=index)
            cached.to_parquet(cache_path)
            prices = pd.concat(
                {
                    "close": pd.DataFrame(
                        {"A": [10.0, 10.1, 10.2, 10.3], "B": [20.0, 20.1, 20.2, 20.3]},
                        index=price_dates,
                    ),
                },
                axis=1,
            )
            prices.to_parquet(price_path)
            (cache_path.with_name(f"{cache_path.name}.meta.json")).write_text(
                json.dumps(
                    {
                        "provider_uri": str(provider),
                        "region": "cn",
                        "instruments": "mainboard_a",
                        "start_date": "2024-01-02",
                        "end_date": "2024-01-03",
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "factors": {
                    "cache_file": str(cache_path),
                    "allow_stale_tail": True,
                    "max_stale_tail_days": 10,
                    "max_stale_tail_sessions": 2,
                },
                "ic": {"price_file": str(price_path)},
                "qlib": {"provider_uri": str(provider), "region": "cn", "instruments": "mainboard_a"},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-05", cache_file=cache_path)

        compute.assert_not_called()
        self.assertEqual(set(pd.to_datetime(factors.index.get_level_values("datetime")).date), set(factor_dates.date))

    def test_load_or_compute_factors_recomputes_when_stale_tail_exceeds_limit(self) -> None:
        """函数说明：验证因子缓存尾部缺口超过阈值时仍然重算。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            provider = root / "qlib"
            factor_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            price_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
            cached_index = pd.MultiIndex.from_product([factor_dates, ["A"]], names=["datetime", "instrument"])
            cached = pd.DataFrame({"F1": [1.0, 2.0]}, index=cached_index)
            cached.to_parquet(cache_path)
            pd.concat(
                {
                    "close": pd.DataFrame({"A": [10.0, 10.1, 10.2, 10.3]}, index=price_dates),
                },
                axis=1,
            ).to_parquet(price_path)
            (cache_path.with_name(f"{cache_path.name}.meta.json")).write_text(
                json.dumps(
                    {
                        "provider_uri": str(provider),
                        "region": "cn",
                        "instruments": "mainboard_a",
                        "start_date": "2024-01-02",
                        "end_date": "2024-01-03",
                    }
                ),
                encoding="utf-8",
            )
            recomputed_index = pd.MultiIndex.from_product([price_dates, ["A"]], names=["datetime", "instrument"])
            recomputed = pd.DataFrame({"F1": range(len(recomputed_index))}, index=recomputed_index)
            config = {
                "factors": {
                    "cache_file": str(cache_path),
                    "allow_stale_tail": True,
                    "max_stale_tail_days": 10,
                    "max_stale_tail_sessions": 1,
                },
                "ic": {"price_file": str(price_path)},
                "qlib": {"provider_uri": str(provider), "region": "cn", "instruments": "mainboard_a"},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors", return_value=recomputed) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-05", cache_file=cache_path)

        compute.assert_called_once()
        self.assertEqual(set(pd.to_datetime(factors.index.get_level_values("datetime")).date), set(price_dates.date))

    def test_load_or_compute_factors_recomputes_stale_tail_when_latest_end_required(self) -> None:
        """函数说明：验证最新日期质量门槛开启时不会复用尾部缺口缓存。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            provider = root / "qlib"
            factor_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            price_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
            cached_index = pd.MultiIndex.from_product([factor_dates, ["A"]], names=["datetime", "instrument"])
            pd.DataFrame({"F1": [1.0, 2.0]}, index=cached_index).to_parquet(cache_path)
            pd.concat(
                {
                    "close": pd.DataFrame({"A": [10.0, 10.1, 10.2]}, index=price_dates),
                },
                axis=1,
            ).to_parquet(price_path)
            (cache_path.with_name(f"{cache_path.name}.meta.json")).write_text(
                json.dumps(
                    {
                        "provider_uri": str(provider),
                        "region": "cn",
                        "instruments": "mainboard_a",
                        "start_date": "2024-01-02",
                        "end_date": "2024-01-03",
                    }
                ),
                encoding="utf-8",
            )
            recomputed_index = pd.MultiIndex.from_product([price_dates, ["A"]], names=["datetime", "instrument"])
            recomputed = pd.DataFrame({"F1": range(len(recomputed_index))}, index=recomputed_index)
            config = {
                "factors": {
                    "cache_file": str(cache_path),
                    "allow_stale_tail": True,
                    "max_stale_tail_days": 10,
                    "max_stale_tail_sessions": 2,
                },
                "quality": {"require_latest_end_date": True},
                "ic": {"price_file": str(price_path)},
                "qlib": {"provider_uri": str(provider), "region": "cn", "instruments": "mainboard_a"},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors", return_value=recomputed) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-04", cache_file=cache_path)

        compute.assert_called_once()
        self.assertEqual(set(pd.to_datetime(factors.index.get_level_values("datetime")).date), set(price_dates.date))

    def test_load_or_compute_factors_does_not_overwrite_default_cache_for_partial_range(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_does_not_overwrite_default_cache_for_partial_range 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            price_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
            pd.DataFrame({"A": [10.0, 10.1, 10.2]}, index=price_dates).to_parquet(price_path)
            computed_index = pd.MultiIndex.from_product(
                [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
                names=["datetime", "instrument"],
            )
            computed = pd.DataFrame({"F1": [1.0, 2.0]}, index=computed_index)
            config = {
                "data": {"start_date": "2015-01-01", "end_date": "auto"},
                "factors": {"cache_file": str(cache_path)},
                "ic": {"price_file": str(price_path)},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors", return_value=computed) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_called_once()
        self.assertEqual(len(factors), 2)
        self.assertFalse(cache_path.exists())
        self.assertFalse(cache_path.with_name(f"{cache_path.name}.meta.json").exists())

    def test_load_or_compute_factors_writes_default_cache_when_request_matches_auto_target(self) -> None:
        """函数说明：验证盘中 auto 目标日前的完整请求会写入默认缓存。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            price_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
            pd.DataFrame({"A": [10.0, 10.1, 10.2]}, index=price_dates).to_parquet(price_path)
            computed_index = pd.MultiIndex.from_product(
                [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
                names=["datetime", "instrument"],
            )
            computed = pd.DataFrame({"F1": [1.0, 2.0]}, index=computed_index)
            config = {
                "data": {"start_date": "2024-01-02", "end_date": "auto"},
                "factors": {"cache_file": str(cache_path)},
                "ic": {"price_file": str(price_path)},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.resolve_target_date_value", return_value="2024-01-03"), patch(
                "src.factor_calculator.compute_alpha158_factors", return_value=computed
            ) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

            compute.assert_called_once()
            self.assertEqual(len(factors), 2)
            self.assertTrue(cache_path.exists())
            meta = json.loads(cache_path.with_name(f"{cache_path.name}.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["end_date"], "2024-01-03")

    def test_load_or_compute_factors_writes_default_cache_from_first_available_price_date(self) -> None:
        """函数说明：验证配置历史早于可用价格时不会误判为部分区间。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            price_dates = pd.to_datetime(["2015-01-05", "2015-01-06"])
            pd.DataFrame({"A": [10.0, 10.1]}, index=price_dates).to_parquet(price_path)
            computed_index = pd.MultiIndex.from_product(
                [price_dates, ["A"]],
                names=["datetime", "instrument"],
            )
            computed = pd.DataFrame({"F1": [1.0, 2.0]}, index=computed_index)
            config = {
                "data": {"history_start_date": "2012-01-01", "end_date": "auto"},
                "factors": {"cache_file": str(cache_path)},
                "ic": {"price_file": str(price_path)},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.resolve_target_date_value", return_value="2015-01-06"), patch(
                "src.factor_calculator.compute_alpha158_factors", return_value=computed
            ) as compute:
                load_or_compute_factors("2015-01-05", "2015-01-06", cache_file=cache_path)

            compute.assert_called_once()
            self.assertTrue(cache_path.exists())

    def test_load_or_compute_factors_reuses_cache_from_first_available_price_date(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            provider = root / "qlib"
            dates = pd.to_datetime(["2015-01-05", "2015-01-06"])
            index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
            pd.DataFrame({"F1": [1.0, 2.0]}, index=index).to_parquet(cache_path)
            pd.DataFrame({"A": [10.0, 10.1]}, index=dates).to_parquet(price_path)
            cache_path.with_name(f"{cache_path.name}.meta.json").write_text(
                json.dumps(
                    {
                        "provider_uri": str(provider),
                        "region": "cn",
                        "instruments": "mainboard_a",
                        "start_date": "2015-01-05",
                        "end_date": "2015-01-06",
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "data": {"history_start_date": "2012-01-01", "end_date": "auto"},
                "factors": {"cache_file": str(cache_path)},
                "ic": {"price_file": str(price_path)},
                "qlib": {"provider_uri": str(provider), "region": "cn", "instruments": "mainboard_a"},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors("2012-01-01", "2015-01-06", cache_file=cache_path)

            compute.assert_not_called()
            self.assertEqual(len(factors), 2)

    def test_load_or_compute_factors_recomputes_when_qlib_metadata_changes(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_recomputes_when_qlib_metadata_changes 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            provider = root / "qlib"
            dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
            cached = pd.DataFrame({"F1": [1.0, 2.0]}, index=index)
            cached.to_parquet(cache_path)
            pd.DataFrame({"A": [10.0, 10.1]}, index=dates).to_parquet(price_path)
            (cache_path.with_name(f"{cache_path.name}.meta.json")).write_text(
                json.dumps(
                    {
                        "provider_uri": str(provider),
                        "region": "cn",
                        "instruments": "old_universe",
                        "start_date": "2024-01-02",
                        "end_date": "2024-01-03",
                    }
                ),
                encoding="utf-8",
            )
            recomputed = pd.DataFrame({"F1": [3.0, 4.0]}, index=index)
            config = {
                "factors": {"cache_file": str(cache_path)},
                "ic": {"price_file": str(price_path)},
                "qlib": {"provider_uri": str(provider), "region": "cn", "instruments": "mainboard_a"},
            }

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors", return_value=recomputed) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_called_once()
        self.assertEqual(float(factors.iloc[0]["F1"]), 3.0)

    def test_load_or_compute_factors_reuses_cache_when_meta_exists_but_qlib_config_is_missing(self) -> None:
        """函数说明：验证 test_load_or_compute_factors_reuses_cache_when_meta_exists_but_qlib_config_is_missing 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
            cached = pd.DataFrame({"F1": [1.0, 2.0]}, index=index)
            cached.to_parquet(cache_path)
            pd.DataFrame({"A": [10.0, 10.1]}, index=dates).to_parquet(price_path)
            (cache_path.with_name(f"{cache_path.name}.meta.json")).write_text(
                json.dumps(
                    {
                        "provider_uri": str(root / "qlib"),
                        "region": "cn",
                        "instruments": "mainboard_a",
                        "start_date": "2024-01-02",
                        "end_date": "2024-01-03",
                    }
                ),
                encoding="utf-8",
            )
            config = {"factors": {"cache_file": str(cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors") as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_not_called()
        self.assertEqual(float(factors.iloc[0]["F1"]), 1.0)


if __name__ == "__main__":
    unittest.main()
