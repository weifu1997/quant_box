from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import src.factor_calculator as factor_calculator
from src.factor_calculator import _ensure_qlib_initialized, load_or_compute_factors
from tests.fixtures.real_data import require_real_market_data


class FakeQlib:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def init(self, provider_uri: str, region: str) -> None:
        self.calls.append((provider_uri, region))


class FactorCalculatorTests(unittest.TestCase):
    def tearDown(self) -> None:
        factor_calculator._QLIB_INIT_STATE = None

    def test_ensure_qlib_initialized_reuses_matching_provider_and_region(self) -> None:
        fake = FakeQlib()
        provider = Path("data/qlib_data")

        _ensure_qlib_initialized(fake, provider, "cn")
        _ensure_qlib_initialized(fake, provider, "cn")

        self.assertEqual(fake.calls, [(str(provider), "cn")])

    def test_load_or_compute_factors_recomputes_when_price_panel_has_new_symbol(self) -> None:
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

    def test_load_or_compute_factors_reuses_cache_when_request_starts_before_first_trading_day(self) -> None:
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

    def test_load_or_compute_factors_does_not_overwrite_default_cache_for_partial_range(self) -> None:
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

    def test_load_or_compute_factors_recomputes_when_qlib_metadata_changes(self) -> None:
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
