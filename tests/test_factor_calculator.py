from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import src.factor_calculator as factor_calculator
from src.factor_calculator import _ensure_qlib_initialized, load_or_compute_factors


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
            dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            cached_index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
            pd.DataFrame({"F1": [1.0, 2.0]}, index=cached_index).to_parquet(cache_path)
            prices = pd.concat(
                {
                    "close": pd.DataFrame({"A": [10.0, 10.1], "B": [20.0, 20.1]}, index=dates),
                },
                axis=1,
            )
            prices.to_parquet(price_path)
            recomputed_index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
            recomputed = pd.DataFrame({"F1": [1.0, 2.0, 3.0, 4.0]}, index=recomputed_index)
            config = {"factors": {"cache_file": str(cache_path)}, "ic": {"price_file": str(price_path)}}

            with patch("src.factor_calculator.load_config", return_value=config), patch(
                "src.factor_calculator.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.factor_calculator.compute_alpha158_factors", return_value=recomputed) as compute:
                factors = load_or_compute_factors("2024-01-02", "2024-01-03", cache_file=cache_path)

        compute.assert_called_once()
        self.assertEqual(set(factors.index.get_level_values("instrument")), {"A", "B"})

    def test_load_or_compute_factors_reuses_matching_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "alpha158.parquet"
            price_path = root / "ohlcv.parquet"
            dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
            index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
            cached = pd.DataFrame({"F1": [1.0, 2.0, 3.0, 4.0]}, index=index)
            cached.to_parquet(cache_path)
            prices = pd.concat(
                {
                    "close": pd.DataFrame({"A": [10.0, 10.1], "B": [20.0, 20.1]}, index=dates),
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
