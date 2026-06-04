from __future__ import annotations

import unittest

import pandas as pd

from src.factor_ic import calculate_factor_ic, make_ic_weights, summarize_ic


class FactorICTests(unittest.TestCase):
    def test_calculate_factor_ic_and_weights(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["a", "b", "c"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1, 2, 3, 2, 3, 4, 3, 4, 5]}, index=index)
        prices = pd.DataFrame(
            {
                "a": [10.0, 10.5, 11.0],
                "b": [10.0, 11.0, 12.0],
                "c": [10.0, 12.0, 14.0],
            },
            index=dates,
        )

        ic = calculate_factor_ic(factors, prices, min_obs=2)
        summary = summarize_ic(ic)
        weights = make_ic_weights(summary, top_k=1)

        self.assertIn("F1", ic.columns)
        self.assertIn("F1", weights.index)

    def test_calculate_factor_ic_respects_min_obs_after_vectorized_corr(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["a", "b", "c"]], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "F1": [1, 2, 3, 2, 3, 4, 3, 4, 5],
                "F2": [3, 2, 1, 4, 3, 2, 5, 4, 3],
            },
            index=index,
        )
        prices = pd.DataFrame(
            {
                "a": [10.0, 10.5, 11.0],
                "b": [10.0, 11.0, 12.0],
                "c": [10.0, 12.0, 14.0],
            },
            index=dates,
        )

        ic = calculate_factor_ic(factors, prices, min_obs=4)

        self.assertTrue(ic[["F1", "F2"]].isna().all().all())

    def test_calculate_factor_ic_matches_pairwise_spearman_with_missing_values(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        instruments = ["a", "b", "c", "d"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "F1": [1.0, 2.0, 3.0, 4.0, 2.0, None, 4.0, 5.0],
                "F2": [4.0, 3.0, None, 1.0, 1.0, 2.0, 3.0, 4.0],
            },
            index=index,
        )
        prices = pd.DataFrame(
            {
                "a": [10.0, 10.1, 10.0],
                "b": [10.0, 10.4, 10.7],
                "c": [10.0, 10.2, 10.5],
                "d": [10.0, 10.8, 11.2],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

        ic = calculate_factor_ic(factors, prices, method="spearman", min_obs=3)

        for factor in ["F1", "F2"]:
            for date in dates:
                daily = factors.xs(date, level="datetime")[[factor]].copy()
                daily["forward_return"] = prices.shift(-1).div(prices).sub(1).loc[date]
                pair = daily.dropna()
                expected = pair[factor].corr(pair["forward_return"], method="spearman") if len(pair) >= 3 else float("nan")
                actual = ic.loc[date, factor]
                if pd.isna(expected):
                    self.assertTrue(pd.isna(actual))
                else:
                    self.assertAlmostEqual(float(actual), float(expected))

    def test_calculate_factor_ic_rejects_flat_ohlcv_price_frame(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates[:2], ["a", "b", "c"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": range(6)}, index=index)
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.1, 10.2],
                "close": [10.0, 10.2, 10.4],
                "volume": [1000.0, 1200.0, 1300.0],
            },
            index=dates,
        )

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            calculate_factor_ic(factors, prices, min_obs=2)


if __name__ == "__main__":
    unittest.main()
