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

    def test_generate_signal_matches_previous_holdings_case_insensitively(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["000001.sz", "600519.sh"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [2.0, 1.0]}, index=index)
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-02"},
            "strategy": {
                "factor_group": "momentum",
                "top_n": 1,
                "max_turnover": 1,
                "rank_buffer": 0,
                "min_cross_section_obs": 1,
            },
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        signal, holdings = generate_signal("latest", previous_holdings=["000001.SZ"], config=config, factors=factors)

        self.assertEqual(holdings, ["000001.SZ"])
        self.assertEqual(signal[["instrument", "action"]].to_dict("records"), [{"instrument": "000001.SZ", "action": "HOLD"}])

    def test_empty_signal_keeps_effective_signal_date_metadata(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-03"},
            "strategy": {"factor_group": "momentum", "top_n": 0, "max_turnover": 0, "rank_buffer": 0},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors",
            return_value=factors,
        ):
            signal, holdings = generate_signal("latest", previous_holdings=[])

        self.assertEqual(holdings, [])
        self.assertTrue(signal.empty)
        self.assertEqual(signal.columns.tolist(), ["date", "instrument", "action"])
        self.assertEqual(signal.attrs["signal_date"], "2024-01-02")

    def test_generate_signal_applies_max_industry_weight(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [10, 9, 8, 7]}, index=index)
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-02"},
            "strategy": {
                "factor_group": "momentum",
                "top_n": 3,
                "max_turnover": 3,
                "rank_buffer": 0,
                "min_cross_section_obs": 1,
                "max_industry_weight": 0.5,
            },
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }
        industry = pd.Series({"A": "bank", "B": "bank", "C": "tech", "D": "health"}, name="industry")

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors", return_value=factors
        ), patch("src.signal_generator.load_industry_group_map", return_value=industry):
            _signal, holdings = generate_signal("latest", previous_holdings=[])

        self.assertEqual(holdings, ["A", "C", "D"])

    def test_generate_signal_applies_selection_risk_filter(self) -> None:
        date = pd.Timestamp("2024-01-03")
        index = pd.MultiIndex.from_product([[date], ["A", "B", "C"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"ROC5": [3.0, 2.0, 1.0]}, index=index)
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 9.0], "B": [10.0, 10.2], "C": [10.0, 10.1]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 9.0], "B": [10.0, 10.2], "C": [10.0, 10.1]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 9.0], "B": [10.0, 10.1], "C": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0], "B": [1000.0, 1000.0], "C": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-03"},
            "strategy": {
                "factor_group": "momentum",
                "top_n": 1,
                "max_turnover": 1,
                "rank_buffer": 0,
                "min_cross_section_obs": 1,
            },
            "selection_risk_filter": {
                "enabled": True,
                "lookback_sessions": 2,
                "required_price_fields": ["open", "close"],
                "max_missing_price_sessions": 0,
                "max_limit_down_days": 0,
                "require_positive_volume": True,
            },
            "backtest": {"limit_down_threshold": 0.099},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        _signal, holdings = generate_signal("latest", previous_holdings=[], config=config, factors=factors, price_df=prices)

        self.assertEqual(holdings, ["B"])


if __name__ == "__main__":
    unittest.main()
