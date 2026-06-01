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


if __name__ == "__main__":
    unittest.main()
