from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.signal_generator import generate_signal


class SignalGeneratorTests(unittest.TestCase):
    def test_generate_signal_falls_back_to_latest_cache_date_before_request(self) -> None:
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
            signal, holdings = generate_signal("2024-01-03", previous_holdings=[])

        self.assertEqual(holdings, ["E"])
        self.assertEqual(signal["date"].unique().tolist(), ["2024-01-02"])

    def test_generate_signal_rejects_when_no_cache_date_is_on_or_before_request(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-04")], ["A", "B", "C", "D", "E"]],
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


if __name__ == "__main__":
    unittest.main()
