"""Tests for selector-weight formal backtest helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_selector_weight_backtest import apply_selector_directions, selector_weights_from_frame


class RunSelectorWeightBacktestTests(unittest.TestCase):
    def test_selector_weights_from_frame_maps_known_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "w_roc60": 2.0,
                    "w_db_pb_inv": 1.0,
                    "w_unknown": 99.0,
                }
            ]
        )

        weights = selector_weights_from_frame(frame)

        series = weights[pd.Timestamp("2024-01-31")]
        self.assertAlmostEqual(float(series["ROC60"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(series["DB_pb"]), 1.0 / 3.0)
        self.assertNotIn("unknown", {str(item).lower() for item in series.index})

    def test_apply_selector_directions_flips_inverse_factors(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-31")], ["000001.SZ"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC60": [1.5], "DB_pb": [2.0]}, index=index)
        weights = {pd.Timestamp("2024-01-31"): pd.Series({"ROC60": 0.5, "DB_pb": 0.5})}

        signed = apply_selector_directions(factors, weights)

        self.assertAlmostEqual(float(signed["ROC60"].iloc[0]), 1.5)
        self.assertAlmostEqual(float(signed["DB_pb"].iloc[0]), -2.0)


if __name__ == "__main__":
    unittest.main()
