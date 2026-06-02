from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.signal_generator import generate_signal


class SignalGeneratorTests(unittest.TestCase):
    def test_generate_signal_rejects_stale_factor_cache(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
        config = {
            "data": {"start_date": "2024-01-01"},
            "strategy": {"factor_group": "momentum", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors", return_value=factors
        ):
            with self.assertRaises(ValueError):
                generate_signal("2024-01-03", previous_holdings=[])

    def test_generate_signal_latest_uses_factor_cache_latest_date(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-03"},
            "strategy": {"factor_group": "momentum", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors", return_value=factors
        ):
            signal, holdings = generate_signal("latest", previous_holdings=[])

        self.assertEqual(holdings, ["E"])
        self.assertEqual(signal["date"].unique().tolist(), ["2024-01-02"])


    def test_generate_signal_falls_back_to_close_price_file_for_ic_weights(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            close_path = tmp_path / "close.parquet"
            pd.DataFrame({"A": [10.0, 10.1]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])).to_parquet(close_path)
            missing_ohlcv = tmp_path / "ohlcv.parquet"
            index = pd.MultiIndex.from_product(
                [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
                names=["datetime", "instrument"],
            )
            factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
            config = {
                "data": {"start_date": "2024-01-01"},
                "strategy": {"factor_group": "ic_weighted", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
                "ic": {"price_file": str(missing_ohlcv), "top_k": 1, "min_abs_ic": 0.0, "min_periods": 1},
                "factors": {"cache_file": "unused.parquet"},
                "outputs": {"holdings_file": "unused.csv"},
            }

            def fake_resolve_path(value):
                return Path(value)

            with patch("src.signal_generator.load_config", return_value=config), patch(
                "src.signal_generator.resolve_path", side_effect=fake_resolve_path
            ), patch("src.signal_generator.load_or_compute_factors", return_value=factors), patch(
                "src.signal_generator.calculate_rolling_ic", return_value=pd.DataFrame()
            ), patch(
                "src.signal_generator.make_rolling_ic_weights",
                return_value={pd.Timestamp("2024-01-02"): pd.Series({"ROC5": 1.0})},
            ):
                _signal, holdings = generate_signal("2024-01-02", previous_holdings=[])

        self.assertEqual(holdings, ["E"])


if __name__ == "__main__":
    unittest.main()
