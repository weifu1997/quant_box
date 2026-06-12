"""Tests for centralized risk policy accessors."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.risk_policy import RiskPolicy


class RiskPolicyTests(unittest.TestCase):
    def test_apply_to_backtest_config_carries_risk_controls(self) -> None:
        policy = RiskPolicy(
            {
                "selection_risk_filter": {"enabled": True, "lookback_sessions": 3},
                "strategy": {"stop_loss_pct": 0.08, "take_profit_pct": 0.35},
                "backtest": {"slippage": 0.001, "capacity_window": 10},
            }
        )

        result = policy.apply_to_backtest_config({"top_n": 1})

        self.assertEqual(result["selection_risk_filter"]["lookback_sessions"], 3)
        self.assertEqual(result["stop_loss_pct"], 0.08)
        self.assertEqual(result["take_profit_pct"], 0.35)
        self.assertEqual(result["slippage"], 0.001)
        self.assertEqual(result["capacity_window"], 10)

    def test_apply_to_backtest_config_preserves_explicit_runtime_values(self) -> None:
        policy = RiskPolicy(
            {
                "strategy": {"stop_loss_pct": 0.08, "take_profit_pct": 0.35},
                "backtest": {"slippage": 0.001},
            }
        )

        result = policy.apply_to_backtest_config({"stop_loss_pct": None, "slippage": 0.0})

        self.assertIsNone(result["stop_loss_pct"])
        self.assertEqual(result["slippage"], 0.0)

    def test_industry_group_map_loads_only_when_cap_active(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "industry.csv"
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "industry": "bank"},
                    {"ts_code": "000002.SZ", "industry": "tech"},
                ]
            ).to_csv(path, index=False)

            inactive = RiskPolicy({"industry_file": str(path), "strategy": {}}).industry_group_map()
            active = RiskPolicy({"industry_file": str(path), "strategy": {"max_industry_weight": 0.5}}).industry_group_map()

        self.assertIsNone(inactive)
        self.assertEqual(active.loc["000001.SZ"], "bank")
        self.assertEqual(active.loc["000002.SZ"], "tech")

    def test_apply_to_backtest_config_can_force_grid_industry_map(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "industry.csv"
            pd.DataFrame([{"ts_code": "000001.SZ", "industry": "bank"}]).to_csv(path, index=False)

            result = RiskPolicy({"industry_file": str(path)}).apply_to_backtest_config(
                {"top_n": 1},
                force_industry_map=True,
            )

        self.assertEqual(result["industry_map"].loc["000001.SZ"], "bank")

    def test_filter_selection_scores_delegates_existing_risk_filter(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 9.0], "B": [10.0, 10.2]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 9.0], "B": [10.0, 10.2]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 9.0], "B": [10.0, 10.1]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0], "B": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        scores = pd.Series([2.0, 1.0], index=["A", "B"], name="score")
        policy = RiskPolicy(
            {
                "selection_risk_filter": {
                    "enabled": True,
                    "lookback_sessions": 2,
                    "required_price_fields": ["open", "close"],
                    "max_missing_price_sessions": 0,
                    "max_limit_down_days": 0,
                    "require_positive_volume": True,
                },
                "backtest": {"limit_down_threshold": 0.099},
            }
        )

        filtered = policy.filter_selection_scores(scores, prices, "2024-01-03")

        self.assertTrue(pd.isna(filtered.loc["A"]))
        self.assertEqual(float(filtered.loc["B"]), 1.0)

    def test_risk_properties_fall_back_to_flat_runtime_config(self) -> None:
        policy = RiskPolicy({"stop_loss_pct": 0.05, "take_profit_pct": 0.25, "max_industry_weight": 0.4})

        self.assertEqual(policy.stop_loss_pct, 0.05)
        self.assertEqual(policy.take_profit_pct, 0.25)
        self.assertEqual(policy.max_industry_weight, 0.4)


if __name__ == "__main__":
    unittest.main()
