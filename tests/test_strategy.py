from __future__ import annotations

import unittest

import pandas as pd

from src.strategy import composite_factor, select_stocks


class StrategyTests(unittest.TestCase):
    def test_select_stocks_limits_turnover(self) -> None:
        scores = pd.Series(
            [10, 9, 8, 7, 6, 5],
            index=["D", "E", "A", "B", "C", "F"],
        )
        previous = ["A", "B", "C"]

        selected = select_stocks(scores, top_n=3, previous_holdings=previous, max_turnover=1)

        self.assertEqual(len(selected), 3)
        self.assertLessEqual(len(set(selected) - set(previous)), 1)
        self.assertIn("D", selected)

    def test_select_stocks_uses_rank_buffer(self) -> None:
        scores = pd.Series(
            [10, 9, 8, 7, 6, 5],
            index=["A", "B", "C", "D", "E", "F"],
        )
        previous = ["A", "B", "F"]

        selected = select_stocks(scores, top_n=3, previous_holdings=previous, max_turnover=1, rank_buffer=3)

        self.assertIn("F", selected)
        self.assertLessEqual(len(set(selected) - set(previous)), 1)

    def test_composite_factor_returns_score_series(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A", "B"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "ROC5": [1.0, 2.0, 3.0, 4.0],
                "MOM10": [2.0, 3.0, 4.0, 5.0],
                "OTHER": [100.0, 100.0, 100.0, 100.0],
            },
            index=index,
        )

        scores = composite_factor(factors, method="momentum")

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(index))
        self.assertTrue(scores.notna().all())

    def test_composite_factor_supports_ic_weighted(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A", "B"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": [1.0, 2.0, 3.0, 4.0], "F2": [4.0, 3.0, 2.0, 1.0]}, index=index)

        scores = composite_factor(factors, method="ic_weighted", factor_weights={"F1": 1.0, "F2": -0.5})

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(index))


if __name__ == "__main__":
    unittest.main()
