from __future__ import annotations

import unittest

import pandas as pd

from src.neutralization import neutralize_score_panel


class NeutralizationTests(unittest.TestCase):
    def test_neutralize_score_panel_respects_disabled_config(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-31"]), ["A", "B"]],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([1.0, 3.0], index=index, name="score")
        industry = pd.Series({"A": "bank", "B": "bank"})

        neutralized, summary = neutralize_score_panel(
            scores,
            industry_map=industry,
            config={"enabled": False, "industry": True, "market_cap": False},
        )

        pd.testing.assert_series_equal(neutralized, scores)
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["dates_neutralized"], 0)

    def test_neutralize_score_panel_removes_industry_means(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-31"]), ["A", "B", "C", "D"]],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([1.0, 3.0, 10.0, 14.0], index=index, name="score")
        industry = pd.Series({"A": "bank", "B": "bank", "C": "tech", "D": "tech"})

        neutralized, summary = neutralize_score_panel(
            scores,
            industry_map=industry,
            config={"enabled": True, "industry": True, "market_cap": False},
        )

        daily = neutralized.droplevel(0)
        self.assertAlmostEqual(float(daily.loc["A"]), -1.0)
        self.assertAlmostEqual(float(daily.loc["B"]), 1.0)
        self.assertAlmostEqual(float(daily.loc["C"]), -2.0)
        self.assertAlmostEqual(float(daily.loc["D"]), 2.0)
        self.assertEqual(summary["industry_dates"], 1)

    def test_neutralize_score_panel_matches_industry_case_insensitively(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-31"]), ["a", "b", "c", "d"]],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([1.0, 3.0, 10.0, 14.0], index=index, name="score")
        industry = pd.Series({"A": "bank", "B": "bank", "C": "tech", "D": "tech"})

        neutralized, summary = neutralize_score_panel(
            scores,
            industry_map=industry,
            config={"enabled": True, "industry": True, "market_cap": False},
        )

        daily = neutralized.droplevel(0)
        self.assertAlmostEqual(float(daily.loc["a"]), -1.0)
        self.assertAlmostEqual(float(daily.loc["b"]), 1.0)
        self.assertAlmostEqual(float(daily.loc["c"]), -2.0)
        self.assertAlmostEqual(float(daily.loc["d"]), 2.0)
        self.assertEqual(summary["industry_dates"], 1)

    def test_neutralize_score_panel_can_residualize_market_cap(self) -> None:
        date = pd.Timestamp("2024-01-31")
        instruments = ["A", "B", "C", "D"]
        index = pd.MultiIndex.from_product([[date], instruments], names=["datetime", "instrument"])
        scores = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="score")
        daily_basic = pd.DataFrame(
            {"circ_mv": [10.0, 20.0, 30.0, 40.0]},
            index=pd.MultiIndex.from_product([[date], instruments], names=["trade_date", "ts_code"]),
        )

        neutralized, summary = neutralize_score_panel(
            scores,
            daily_basic=daily_basic,
            config={"enabled": True, "industry": False, "market_cap": True, "market_cap_field": "circ_mv", "min_obs": 3},
        )

        cap = pd.Series([10.0, 20.0, 30.0, 40.0], index=instruments).apply(lambda value: __import__("math").log1p(value))
        corr = neutralized.droplevel(0).corr(cap)
        self.assertAlmostEqual(float(corr), 0.0, places=10)
        self.assertEqual(summary["market_cap_dates"], 1)

    def test_neutralize_score_panel_matches_market_cap_case_insensitively(self) -> None:
        date = pd.Timestamp("2024-01-31")
        score_instruments = ["a", "b", "c", "d"]
        basic_instruments = ["A", "B", "C", "D"]
        index = pd.MultiIndex.from_product([[date], score_instruments], names=["datetime", "instrument"])
        scores = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="score")
        daily_basic = pd.DataFrame(
            {"circ_mv": [10.0, 20.0, 30.0, 40.0]},
            index=pd.MultiIndex.from_product([[date], basic_instruments], names=["trade_date", "ts_code"]),
        )

        neutralized, summary = neutralize_score_panel(
            scores,
            daily_basic=daily_basic,
            config={"enabled": True, "industry": False, "market_cap": True, "market_cap_field": "circ_mv", "min_obs": 3},
        )

        cap = pd.Series([10.0, 20.0, 30.0, 40.0], index=score_instruments).apply(lambda value: __import__("math").log1p(value))
        corr = neutralized.droplevel(0).corr(cap)
        self.assertAlmostEqual(float(corr), 0.0, places=10)
        self.assertEqual(summary["market_cap_dates"], 1)


if __name__ == "__main__":
    unittest.main()
