from __future__ import annotations

import unittest

import pandas as pd

from src.score_blending import apply_regime_score_blend, apply_regime_score_filter


class ScoreBlendingTests(unittest.TestCase):
    def test_regime_score_blend_uses_defensive_components_in_bear_state(self) -> None:
        date = pd.Timestamp("2024-01-31")
        index = pd.MultiIndex.from_product([[date], ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([1.0, 0.0], index=index, name="score")
        factors = pd.DataFrame(
            {
                "STD20": [2.0, 0.1],
                "BETA20": [2.0, 0.1],
                "ROC20": [-0.1, 0.2],
            },
            index=index,
        )
        regimes = pd.Series(["bear"], index=[date], name="market_regime")

        blended, summary = apply_regime_score_blend(
            scores,
            factors,
            regimes,
            {"enabled": True, "bear_defensive_weight": 1.0},
        )

        daily = blended.xs(date, level=0)
        self.assertGreater(float(daily.loc["B"]), float(daily.loc["A"]))
        self.assertEqual(summary["dates_blended"], 1)

    def test_regime_score_blend_matches_factor_instruments_case_insensitively(self) -> None:
        date = pd.Timestamp("2024-01-31")
        score_index = pd.MultiIndex.from_product(
            [[date], ["000001.SZ", "600519.SH"]],
            names=["datetime", "instrument"],
        )
        factor_index = pd.MultiIndex.from_product(
            [[date], ["000001.sz", "600519.sh"]],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([0.0, 1.0], index=score_index, name="score")
        factors = pd.DataFrame({"ROC20": [0.5, -0.5]}, index=factor_index)
        regimes = pd.Series(["bear"], index=[date], name="market_regime")

        blended, summary = apply_regime_score_blend(
            scores,
            factors,
            regimes,
            {
                "enabled": True,
                "bear_defensive_weight": 1.0,
                "defensive_components": [{"column": "ROC20", "direction": 1.0}],
            },
        )

        daily = blended.xs(date, level=0)
        self.assertFalse(daily.isna().any())
        self.assertGreater(float(daily.loc["000001.SZ"]), float(daily.loc["600519.SH"]))
        self.assertEqual(summary["dates_blended"], 1)

    def test_regime_score_blend_keeps_highest_score_when_normalized_codes_duplicate(self) -> None:
        date = pd.Timestamp("2024-01-31")
        scores = pd.Series(
            [10.0, 1.0, 5.0],
            index=pd.MultiIndex.from_tuples(
                [(date, " a "), (date, "A"), (date, "B")],
                names=["datetime", "instrument"],
            ),
            name="score",
        )
        factor_index = pd.MultiIndex.from_product([[date], ["A", "B"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"ROC20": [1.0, 1.0]}, index=factor_index)
        regimes = pd.Series(["bear"], index=[date], name="market_regime")

        blended, _summary = apply_regime_score_blend(
            scores,
            factors,
            regimes,
            {
                "enabled": True,
                "bear_defensive_weight": 0.5,
                "defensive_components": [{"column": "ROC20", "direction": 1.0}],
            },
        )

        daily = blended.xs(date, level=0)
        self.assertGreater(float(daily.loc["A"]), float(daily.loc["B"]))

    def test_regime_score_filter_masks_weak_bear_candidates(self) -> None:
        date = pd.Timestamp("2024-01-31")
        index = pd.MultiIndex.from_product([[date], ["A", "B", "C"]], names=["datetime", "instrument"])
        scores = pd.Series([1.0, 0.9, 0.8], index=index, name="score")
        factors = pd.DataFrame({"ROC20": [-0.5, 0.1, 0.5]}, index=index)
        regimes = pd.Series(["bear"], index=[date], name="market_regime")

        filtered, summary = apply_regime_score_filter(
            scores,
            factors,
            regimes,
            {
                "enabled": True,
                "rules": [
                    {
                        "regime": "bear",
                        "components": [{"column": "ROC20", "direction": 1.0}],
                        "min_score": 0.0,
                    }
                ],
            },
        )

        daily = filtered.xs(date, level=0)
        self.assertTrue(pd.isna(daily.loc["A"]))
        self.assertFalse(pd.isna(daily.loc["B"]))
        self.assertFalse(pd.isna(daily.loc["C"]))
        self.assertEqual(summary["dates_filtered"], 1)
        self.assertEqual(summary["rows_removed"], 1)

    def test_regime_score_filter_matches_factor_instruments_case_insensitively(self) -> None:
        date = pd.Timestamp("2024-01-31")
        score_index = pd.MultiIndex.from_product(
            [[date], ["000001.SZ", "600519.SH"]],
            names=["datetime", "instrument"],
        )
        factor_index = pd.MultiIndex.from_product(
            [[date], ["000001.sz", "600519.sh"]],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([1.0, 0.9], index=score_index, name="score")
        factors = pd.DataFrame({"ROC20": [0.5, -0.5]}, index=factor_index)
        regimes = pd.Series(["bear"], index=[date], name="market_regime")

        filtered, summary = apply_regime_score_filter(
            scores,
            factors,
            regimes,
            {
                "enabled": True,
                "rules": [
                    {
                        "regime": "bear",
                        "components": [{"column": "ROC20", "direction": 1.0}],
                        "min_score": 0.5,
                    }
                ],
            },
        )

        daily = filtered.xs(date, level=0)
        self.assertFalse(pd.isna(daily.loc["000001.SZ"]))
        self.assertTrue(pd.isna(daily.loc["600519.SH"]))
        self.assertEqual(summary["rows_removed"], 1)


if __name__ == "__main__":
    unittest.main()
