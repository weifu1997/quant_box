from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.factor_ic import (
    calculate_factor_ic,
    calculate_rolling_ic,
    cluster_correlated_factors,
    make_rolling_ic_weights,
)
from src.strategy import composite_factor
from tests.fixtures.real_data import require_real_market_data


def _math_factor_price_fixture(days: int = 12, instruments: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    names = [f"S{i}" for i in range(instruments)]
    factors = []
    returns = []
    for i, date in enumerate(dates):
        base = np.arange(instruments, dtype=float) + i * 0.01
        factors.extend({"datetime": date, "instrument": name, "F1": value} for name, value in zip(names, base))
        returns.append((base - base.mean()) / 100)
    factor_df = pd.DataFrame(factors).set_index(["datetime", "instrument"])

    prices = pd.DataFrame(100.0, index=dates, columns=names)
    for i in range(days - 1):
        prices.iloc[i + 1] = prices.iloc[i] * (1 + returns[i])
    return factor_df, prices


class RollingICTests(unittest.TestCase):
    def test_calculate_rolling_ic_with_real_market_data(self) -> None:
        market = require_real_market_data(
            start="2024-01-02",
            end="2024-04-30",
            factor_columns=("LOW0", "ROC5", "ROC20"),
        )

        rolling_ic = calculate_rolling_ic(market.factors, market.close, window=10, min_periods=5, min_obs=2)

        self.assertFalse(rolling_ic.empty)
        self.assertTrue({"LOW0", "ROC5", "ROC20"}.issubset(set(rolling_ic.columns)))
        self.assertGreater(int(rolling_ic.notna().sum().sum()), 0)

    def test_rolling_ic_uses_prior_window(self) -> None:
        factors, prices = _math_factor_price_fixture()
        daily_ic = calculate_factor_ic(factors, prices, min_obs=3)
        rolling_ic = calculate_rolling_ic(factors, prices, window=5, min_periods=3, min_obs=3)

        self.assertTrue(pd.isna(rolling_ic.iloc[0]["F1"]))
        expected = daily_ic.iloc[0:5]["F1"].mean()
        self.assertAlmostEqual(float(rolling_ic.iloc[5]["F1"]), float(expected))

    def test_rolling_ic_does_not_use_same_day_ic(self) -> None:
        factors, prices = _math_factor_price_fixture()
        changed = factors.copy()
        date = pd.Timestamp("2024-01-06")
        changed.loc[(date, slice(None)), "F1"] = list(reversed(changed.loc[(date, slice(None)), "F1"].to_list()))

        original = calculate_rolling_ic(factors, prices, window=5, min_periods=3, min_obs=3)
        mutated = calculate_rolling_ic(changed, prices, window=5, min_periods=3, min_obs=3)

        self.assertAlmostEqual(float(original.loc[date, "F1"]), float(mutated.loc[date, "F1"]))

    def test_rolling_ic_lags_by_forward_horizon(self) -> None:
        factors, prices = _math_factor_price_fixture(days=14)
        daily_ic = calculate_factor_ic(factors, prices, horizon=3, min_obs=3)
        rolling_ic = calculate_rolling_ic(factors, prices, horizon=3, window=5, min_periods=3, min_obs=3)

        expected = daily_ic.iloc[0:3]["F1"].mean()
        self.assertAlmostEqual(float(rolling_ic.iloc[5]["F1"]), float(expected))

    def test_cluster_correlated_factors_keeps_one_representative(self) -> None:
        frame = pd.DataFrame(
            {
                "A": [0.02, 0.03, 0.04, 0.05, 0.06],
                "B": [0.021, 0.031, 0.041, 0.051, 0.061],
                "C": [0.06, -0.01, 0.03, -0.02, 0.04],
            }
        )

        clusters = cluster_correlated_factors(frame, threshold=0.7)

        self.assertEqual(len({"A", "B"} & set(clusters)), 1)

    def test_rolling_weights_filter_weak_and_correlated_factors(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        daily_ic = pd.DataFrame(
            {
                "strong": [0.03, 0.04, 0.05, 0.04, 0.05, 0.06, 0.05, 0.04],
                "duplicate": [0.031, 0.041, 0.051, 0.041, 0.051, 0.061, 0.051, 0.041],
                "weak": [0.001, -0.002, 0.001, 0.0, 0.002, -0.001, 0.0, 0.001],
            },
            index=dates,
        )
        rolling_ic = daily_ic.shift(1).rolling(window=3, min_periods=3).mean()
        rolling_ic.attrs["daily_ic"] = daily_ic
        rolling_ic.attrs["window"] = 3

        weights = make_rolling_ic_weights(rolling_ic, top_k=3, min_abs_ic=0.02, min_periods=3, correlation_threshold=0.7)
        latest = weights[pd.Timestamp("2024-01-05")]

        self.assertNotIn("weak", latest.index)
        self.assertEqual(len({"strong", "duplicate"} & set(latest.index)), 1)
        self.assertAlmostEqual(float(latest.abs().sum()), 1.0)

    def test_rolling_weights_require_daily_ic_attrs(self) -> None:
        rolling_ic = pd.DataFrame({"F1": [0.03, 0.04]}, index=pd.date_range("2024-01-01", periods=2))

        with self.assertRaises(ValueError):
            make_rolling_ic_weights(rolling_ic, min_periods=1)

    def test_rolling_weights_smoothing_and_turnover_cap_reduce_weight_jumps(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily_ic = pd.DataFrame(
            {
                "F1": [0.08, 0.08, 0.08, 0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08],
                "F2": [-0.08, -0.08, -0.08, -0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
            },
            index=dates,
        )
        rolling_ic = daily_ic.shift(1).rolling(window=3, min_periods=3).mean()
        rolling_ic.attrs["daily_ic"] = daily_ic
        rolling_ic.attrs["window"] = 3

        raw = make_rolling_ic_weights(
            rolling_ic,
            top_k=2,
            min_abs_ic=0.0,
            min_periods=3,
            correlation_threshold=2.0,
        )
        stable = make_rolling_ic_weights(
            rolling_ic,
            top_k=2,
            min_abs_ic=0.0,
            min_periods=3,
            correlation_threshold=2.0,
            weight_smoothing=0.6,
            max_weight_turnover=0.5,
        )

        self.assertLess(_max_weight_delta(stable), _max_weight_delta(raw))

    def test_rolling_weights_reuses_correlation_clusters_between_rebalance_sessions(self) -> None:
        dates = pd.date_range("2024-01-01", periods=9, freq="D")
        daily_ic = pd.DataFrame(
            {
                "F1": [0.06, 0.07, 0.08, 0.07, 0.08, 0.09, 0.08, 0.07, 0.08],
                "F2": [0.061, 0.071, 0.081, 0.071, 0.081, 0.091, 0.081, 0.071, 0.081],
            },
            index=dates,
        )
        rolling_ic = daily_ic.shift(1).rolling(window=2, min_periods=1).mean()
        rolling_ic.attrs["daily_ic"] = daily_ic
        rolling_ic.attrs["window"] = 2

        with patch("src.factor_ic.cluster_correlated_factors", wraps=cluster_correlated_factors) as cluster:
            weights = make_rolling_ic_weights(
                rolling_ic,
                top_k=2,
                min_abs_ic=0.0,
                min_periods=1,
                correlation_threshold=0.7,
                correlation_rebalance_sessions=3,
        )

        self.assertEqual(cluster.call_count, 3)
        self.assertEqual(len(weights), len(dates) - 2)

    def test_composite_factor_accepts_dynamic_weights(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": range(10), "F2": range(10, 20)}, index=index)
        weights = {dates[0]: pd.Series({"F1": 1.0}), dates[1]: pd.Series({"F2": 1.0})}

        scores = composite_factor(factors, method="ic_weighted", factor_weights_dynamic=weights)

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(index))

def _max_weight_delta(weights_by_date: dict[pd.Timestamp, pd.Series]) -> float:
    previous: pd.Series | None = None
    max_delta = 0.0
    for date in sorted(weights_by_date):
        current = weights_by_date[date]
        if previous is not None:
            index = current.index.union(previous.index)
            max_delta = max(max_delta, float((current.reindex(index, fill_value=0.0) - previous.reindex(index, fill_value=0.0)).abs().sum()))
        previous = current
    return max_delta


if __name__ == "__main__":
    unittest.main()
