from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.run_risk_refine import (
    _best_rows,
    _combo_key,
    _completed_keys,
    _factor_group_values,
    _liquidity_filter_state,
    _requested_factor_columns_for_groups,
    _target_quality_fields,
    _with_timing_overrides,
)


class RunRiskRefineTests(unittest.TestCase):
    def test_combo_key_includes_factor_group(self) -> None:
        base = _combo_key("factor:MIN60", "low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        changed = _combo_key("inverse_factor:KLEN", "low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)

        self.assertNotEqual(base, changed)

    def test_combo_key_includes_timing_exposure_and_drawdown_trigger(self) -> None:
        base = _combo_key("factor:MIN60", "low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        changed = _combo_key("factor:MIN60", "low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.0, 0.08, 0.0, 0.5, 1.0)

        self.assertNotEqual(base, changed)

    def test_combo_key_includes_score_blend_weights(self) -> None:
        base = _combo_key("factor:MIN60", "low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        changed = _combo_key("factor:MIN60", "low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.0, 1.0)

        self.assertNotEqual(base, changed)

    def test_combo_key_includes_industry_weight_cap(self) -> None:
        base = _combo_key("factor:MIN60", "low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        changed = _combo_key("factor:MIN60", "low", 0.35, 15, 20, 0.25, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)

        self.assertNotEqual(base, changed)

    def test_completed_keys_reads_extended_timing_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk_refine.csv"
            pd.DataFrame(
                [
                    {
                        "factor_group": "factor:MIN60",
                        "liquidity_side": "low",
                        "liquidity_quantile": 0.35,
                        "top_n": 15,
                        "rank_buffer": 20,
                        "max_industry_weight": 0.25,
                        "stop_loss_pct": None,
                        "take_profit_pct": 0.65,
                        "circuit_breaker_drawdown": 0.12,
                        "circuit_breaker_cooldown_days": 60,
                        "circuit_breaker_target_exposure": 0.3,
                        "rebalance_drift_threshold": 0.02,
                        "defensive_timing": "enabled",
                        "bull_exposure": 1.0,
                        "sideways_exposure": 0.6,
                        "bear_exposure": 0.3,
                        "bear_drawdown_threshold": 0.08,
                        "bull_defensive_weight": 0.0,
                        "sideways_defensive_weight": 0.5,
                        "bear_defensive_weight": 1.0,
                    }
                ]
            ).to_csv(path, index=False)

            keys = _completed_keys(path)

        expected = _combo_key("factor:MIN60", "low", 0.35, 15, 20, 0.25, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        self.assertIn(expected, keys)

    def test_factor_group_values_falls_back_to_config_group(self) -> None:
        config = {"strategy": {"factor_group": "factor:MIN60"}}

        self.assertEqual(_factor_group_values("", config), ["factor:MIN60"])
        self.assertEqual(_factor_group_values("factor:MIN60,inverse_factor:KLEN", config), ["factor:MIN60", "inverse_factor:KLEN"])

    def test_liquidity_filter_state_supports_disabled_side(self) -> None:
        self.assertEqual(_liquidity_filter_state("none", 0.35), (False, "none", 0.0))
        self.assertEqual(_liquidity_filter_state("OFF", 0.20), (False, "none", 0.0))
        self.assertEqual(_liquidity_filter_state("high", 0.20), (True, "high", 0.20))

    def test_requested_factor_columns_for_groups_unions_exact_factor_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "factors.parquet"
            pd.DataFrame(
                {
                    "datetime": [pd.Timestamp("2024-01-01")],
                    "instrument": ["A"],
                    "MIN60": [1.0],
                    "KLEN": [2.0],
                    "ROC20": [3.0],
                }
            ).to_parquet(path)

            columns = _requested_factor_columns_for_groups(
                str(path),
                {
                    "strategy": {"factor_group": "momentum"},
                    "dynamic_ic_selector": {},
                    "ml_strategy": {},
                    "regime_score_blend": {},
                    "regime_score_filter": {},
                },
                ["factor:MIN60", "inverse_factor:KLEN"],
            )

        self.assertEqual(columns, ["KLEN", "MIN60"])

    def test_with_timing_overrides_applies_exposure_and_market_drawdown_trigger(self) -> None:
        config = {"defensive_timing": {"enabled": False}, "market_regime": {}}

        result = _with_timing_overrides(config, "enabled", 1.0, 0.5, 0.1, 0.08)

        self.assertTrue(result["defensive_timing"]["enabled"])
        self.assertEqual(result["defensive_timing"]["sideways_exposure"], 0.5)
        self.assertEqual(result["defensive_timing"]["bear_exposure"], 0.1)
        self.assertEqual(result["market_regime"]["bear_drawdown_threshold"], 0.08)
        self.assertFalse(config["defensive_timing"]["enabled"])

    def test_target_quality_fields_reject_total_pass_with_weak_year(self) -> None:
        yearly = pd.DataFrame(
            [
                {"year": 2024, "annual_return": 0.12, "max_drawdown": -0.10},
                {"year": 2025, "annual_return": -0.02, "max_drawdown": -0.18},
            ]
        )
        coverage = pd.DataFrame(
            [
                {"year": 2024, "passes_min_days": True},
                {"year": 2025, "passes_min_days": True},
            ]
        )

        fields = _target_quality_fields(
            {"annual_return": 0.30, "max_drawdown": -0.10},
            yearly,
            coverage,
            {"min_yearly_annual_return": 0.10, "max_yearly_drawdown_limit": -0.20},
            0.20,
            -0.20,
        )

        self.assertGreater(fields["annual_return_gap"], 0)
        self.assertGreater(fields["drawdown_buffer"], 0)
        self.assertEqual(fields["year_ann_pass"], 1)
        self.assertEqual(fields["year_dd_pass"], 2)
        self.assertFalse(fields["yearly_annual_return_pass"])
        self.assertFalse(fields["meets_target"])

    def test_target_quality_fields_reject_missing_year_coverage(self) -> None:
        yearly = pd.DataFrame([{"year": 2024, "annual_return": 0.12, "max_drawdown": -0.10}])
        coverage = pd.DataFrame(
            [
                {"year": 2024, "passes_min_days": True},
                {"year": 2025, "passes_min_days": True},
            ]
        )

        fields = _target_quality_fields(
            {"annual_return": 0.30, "max_drawdown": -0.10},
            yearly,
            coverage,
            {"min_yearly_annual_return": 0.10, "max_yearly_drawdown_limit": -0.20},
            0.20,
            -0.20,
        )

        self.assertFalse(fields["year_coverage_pass"])
        self.assertEqual(fields["missing_years"], "2025")
        self.assertFalse(fields["meets_target"])

    def test_best_rows_prioritizes_yearly_shortfalls(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "name": "high_total_bad_year",
                    "annual_return": 0.50,
                    "max_drawdown": -0.10,
                    "min_year_annual_return": -0.05,
                    "worst_year_drawdown": -0.10,
                    "min_yearly_annual_return": 0.10,
                    "max_yearly_drawdown_limit": -0.20,
                    "year_coverage_pass": True,
                    "meets_target": False,
                    "sharpe": 2.0,
                    "calmar": 5.0,
                },
                {
                    "name": "lower_total_better_year",
                    "annual_return": 0.18,
                    "max_drawdown": -0.18,
                    "min_year_annual_return": 0.09,
                    "worst_year_drawdown": -0.19,
                    "min_yearly_annual_return": 0.10,
                    "max_yearly_drawdown_limit": -0.20,
                    "year_coverage_pass": True,
                    "meets_target": False,
                    "sharpe": 1.0,
                    "calmar": 1.0,
                },
            ]
        )

        best = _best_rows(rows, 0.20, -0.20)

        self.assertEqual(best.iloc[0]["name"], "lower_total_better_year")


if __name__ == "__main__":
    unittest.main()
