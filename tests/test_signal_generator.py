from __future__ import annotations

import unittest

import pandas as pd

from src.selection_constraints import load_industry_group_map
from src.signal_generator import generate_signal
from tests.fixtures.real_data import require_real_market_data


class SignalGeneratorTests(unittest.TestCase):
    def test_generate_signal_falls_back_to_latest_cache_date_before_request(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-01-05")
        config = _signal_config(start_date="2024-01-02", end_date="2024-01-06")

        signal, holdings = generate_signal("2024-01-06", previous_holdings=[], config=config, factors=market.factors)

        self.assertEqual(len(holdings), 1)
        self.assertIn(holdings[0], market.instruments)
        self.assertEqual(signal["date"].unique().tolist(), ["2024-01-05"])

    def test_generate_signal_rejects_when_no_cache_date_is_on_or_before_request(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-01-05")
        config = _signal_config(start_date="2024-01-02", end_date="2024-01-05")

        with self.assertRaises(ValueError):
            generate_signal("2024-01-01", previous_holdings=[], config=config, factors=market.factors)

    def test_generate_signal_latest_uses_factor_cache_latest_date(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-01-05")
        config = _signal_config(start_date="2024-01-02", end_date="2024-01-05")

        signal, holdings = generate_signal("latest", previous_holdings=[], config=config, factors=market.factors)

        self.assertEqual(len(holdings), 1)
        self.assertIn(holdings[0], market.instruments)
        self.assertEqual(signal["date"].unique().tolist(), ["2024-01-05"])

    def test_generate_signal_uses_latest_intraday_factors_for_signal_date(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02 09:30"), "A"),
                (pd.Timestamp("2024-01-02 09:30"), "B"),
                (pd.Timestamp("2024-01-02 15:00"), "A"),
                (pd.Timestamp("2024-01-02 15:00"), "B"),
            ],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [100.0, 1.0, 1.0, 50.0]}, index=index)
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

        signal, holdings = generate_signal("latest", previous_holdings=[], config=config, factors=factors)

        self.assertEqual(holdings, ["B"])
        self.assertEqual(signal[["date", "instrument", "action"]].to_dict("records"), [{"date": "2024-01-02", "instrument": "B", "action": "BUY"}])

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

    def test_generate_signal_keeps_highest_score_when_normalized_codes_duplicate(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], [" a ", "A", "B"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [100.0, 1.0, 50.0]}, index=index)
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

        signal, holdings = generate_signal("latest", previous_holdings=[], config=config, factors=factors)

        self.assertEqual(holdings, ["A"])
        self.assertEqual(signal[["instrument", "action"]].to_dict("records"), [{"instrument": "A", "action": "BUY"}])

    def test_empty_signal_keeps_effective_signal_date_metadata(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-01-05")
        config = _signal_config(start_date="2024-01-02", end_date="2024-01-05", top_n=0, max_turnover=0)

        signal, holdings = generate_signal("latest", previous_holdings=[], config=config, factors=market.factors)

        self.assertEqual(holdings, [])
        self.assertTrue(signal.empty)
        self.assertEqual(signal.columns.tolist(), ["date", "instrument", "action"])
        self.assertEqual(signal.attrs["signal_date"], "2024-01-05")

    def test_generate_signal_applies_max_industry_weight(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-01-05")
        config = _signal_config(
            start_date="2024-01-02",
            end_date="2024-01-05",
            top_n=3,
            max_turnover=3,
            max_industry_weight=0.5,
        )
        industry = load_industry_group_map(config)

        _signal, holdings = generate_signal("latest", previous_holdings=[], config=config, factors=market.factors)

        self.assertLessEqual(len(holdings), 3)
        self.assertTrue(set(holdings).issubset(set(market.instruments)))
        if not industry.empty and holdings:
            industry_counts = pd.Series(holdings).map(industry).value_counts(normalize=True)
            self.assertTrue((industry_counts <= 0.5).all())

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


def _signal_config(
    start_date: str,
    end_date: str,
    top_n: int = 1,
    max_turnover: int = 1,
    max_industry_weight: float | None = None,
) -> dict:
    strategy = {
        "factor_group": "factor:LOW0",
        "top_n": top_n,
        "max_turnover": max_turnover,
        "rank_buffer": 0,
        "min_cross_section_obs": 1,
    }
    if max_industry_weight is not None:
        strategy["max_industry_weight"] = max_industry_weight
    return {
        "data": {
            "start_date": start_date,
            "end_date": end_date,
            "constituents_file": "data/raw/mainboard_a_stocks.csv",
        },
        "strategy": strategy,
        "factors": {"cache_file": "data/factors/alpha158.parquet"},
        "outputs": {"holdings_file": "outputs/current_holdings.csv"},
        "liquidity_filter": {"enabled": False},
        "regime_score_blend": {"enabled": False},
        "regime_score_filter": {"enabled": False},
    }


if __name__ == "__main__":
    unittest.main()
