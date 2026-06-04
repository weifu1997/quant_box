from __future__ import annotations

import unittest

import pandas as pd

from src.ml_strategy import build_ml_scores


class MLStrategyTests(unittest.TestCase):
    def test_build_ml_scores_uses_only_labels_known_before_signal_date(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=35)
        instruments = ["A", "B", "C", "D"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "F1": [float(i % 11) for i in range(len(index))],
                "F2": [float((i * 3) % 17) for i in range(len(index))],
            },
            index=index,
        )
        close = pd.DataFrame(
            {
                "A": range(100, 100 + len(dates)),
                "B": range(80, 80 + len(dates)),
                "C": range(60, 60 + len(dates)),
                "D": range(40, 40 + len(dates)),
            },
            index=dates,
            dtype=float,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 3,
                "min_train_rows": 8,
                "max_train_rows": 100,
                "feature_limit": None,
                "min_feature_fraction": 0.5,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.empty)
        self.assertEqual(set(result.scores.index.get_level_values("datetime")), {dates[-1]})
        row = result.diagnostics.iloc[0]
        self.assertTrue(bool(row["no_lookahead"]))
        self.assertLess(pd.Timestamp(row["max_label_end"]), pd.Timestamp(row["signal_date"]))
        self.assertEqual(row["model_used"], "ridge_numpy")
        self.assertGreater(int(row["train_rows_used"]), 0)

    def test_build_ml_scores_auto_model_falls_back_to_available_model(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=28)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": range(len(index)), "F2": range(len(index), 0, -1)}, index=index, dtype=float)
        close = pd.DataFrame(10.0, index=dates, columns=instruments)
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "auto",
                "train_years": 1,
                "label_horizon_sessions": 2,
                "min_train_rows": 6,
                "max_train_rows": 50,
                "feature_limit": None,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.empty)
        self.assertIn(result.diagnostics.iloc[0]["model_used"], {"lightgbm", "xgboost", "sklearn_gbdt", "ridge_numpy"})


if __name__ == "__main__":
    unittest.main()
