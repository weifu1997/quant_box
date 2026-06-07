from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.run_risk_refine import _combo_key, _completed_keys, _with_timing_overrides


class RunRiskRefineTests(unittest.TestCase):
    def test_combo_key_includes_timing_exposure_and_drawdown_trigger(self) -> None:
        base = _combo_key("low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        changed = _combo_key("low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.0, 0.08, 0.0, 0.5, 1.0)

        self.assertNotEqual(base, changed)

    def test_combo_key_includes_score_blend_weights(self) -> None:
        base = _combo_key("low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        changed = _combo_key("low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.0, 1.0)

        self.assertNotEqual(base, changed)

    def test_combo_key_includes_industry_weight_cap(self) -> None:
        base = _combo_key("low", 0.35, 15, 20, None, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        changed = _combo_key("low", 0.35, 15, 20, 0.25, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)

        self.assertNotEqual(base, changed)

    def test_completed_keys_reads_extended_timing_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk_refine.csv"
            pd.DataFrame(
                [
                    {
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

        expected = _combo_key("low", 0.35, 15, 20, 0.25, None, 0.65, 0.12, 60, 0.3, 0.02, "enabled", 1.0, 0.6, 0.3, 0.08, 0.0, 0.5, 1.0)
        self.assertIn(expected, keys)

    def test_with_timing_overrides_applies_exposure_and_market_drawdown_trigger(self) -> None:
        config = {"defensive_timing": {"enabled": False}, "market_regime": {}}

        result = _with_timing_overrides(config, "enabled", 1.0, 0.5, 0.1, 0.08)

        self.assertTrue(result["defensive_timing"]["enabled"])
        self.assertEqual(result["defensive_timing"]["sideways_exposure"], 0.5)
        self.assertEqual(result["defensive_timing"]["bear_exposure"], 0.1)
        self.assertEqual(result["market_regime"]["bear_drawdown_threshold"], 0.08)
        self.assertFalse(config["defensive_timing"]["enabled"])


if __name__ == "__main__":
    unittest.main()
