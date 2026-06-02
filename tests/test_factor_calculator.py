from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.factor_calculator import load_or_compute_factors


class FactorCalculatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
