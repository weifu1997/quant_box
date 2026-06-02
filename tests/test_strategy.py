from __future__ import annotations

import unittest

import pandas as pd

from src.strategy import composite_factor, resample_signals, select_stocks


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

    def test_select_stocks_deduplicates_previous_holdings(self) -> None:
        scores = pd.Series([10, 9, 8, 7], index=["A", "B", "C", "D"])
        selected = select_stocks(scores, top_n=3, previous_holdings=["A", "A", "B"], max_turnover=1)

        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 3)

    def test_composite_factor_returns_score_series(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "ROC5": range(1, 11),
                "MOM10": range(2, 12),
                "OTHER": [100.0] * 10,
            },
            index=index,
        )

        scores = composite_factor(factors, method="momentum")

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(index))
        self.assertTrue(scores.notna().all())

    def test_composite_factor_supports_ic_weighted(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": range(1, 11), "F2": range(10, 0, -1)}, index=index)

        scores = composite_factor(factors, method="ic_weighted", factor_weights={"F1": 1.0, "F2": -0.5})

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(index))

    def test_composite_factor_tolerates_partial_missing_values(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [1, 2, 3, 4, 5], "MOM10": [2, 3, None, 5, 6]}, index=index)

        scores = composite_factor(factors, method="momentum")

        self.assertTrue(pd.notna(scores.loc[(pd.Timestamp("2024-01-02"), "C")]))

    def test_resample_signals_supports_monthly_with_pandas_me_alias(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"])
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
        scores = pd.Series(range(len(index)), index=index, name="score")

        sampled = resample_signals(scores, "monthly")

        self.assertEqual(
            sorted(sampled.index.get_level_values(0).unique().strftime("%Y-%m-%d").tolist()),
            ["2024-01-31", "2024-02-29"],
        )


if __name__ == "__main__":
    unittest.main()
