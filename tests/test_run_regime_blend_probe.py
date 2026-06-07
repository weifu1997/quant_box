from __future__ import annotations

import unittest

from scripts.run_regime_blend_probe import _score_key, _with_probe_overrides, _with_timing_probe


class RunRegimeBlendProbeTests(unittest.TestCase):
    def test_score_key_includes_regime_blend_weights(self) -> None:
        base = _score_key("low", 0.35, 0.08, 0.0, 0.5, 1.0)
        changed = _score_key("low", 0.35, 0.08, 0.0, 0.25, 1.0)

        self.assertNotEqual(base, changed)

    def test_with_probe_overrides_applies_liquidity_regime_and_blend(self) -> None:
        config = {
            "liquidity_filter": {"enabled": False, "side": "high", "quantile": 0.2},
            "market_regime": {},
            "regime_score_blend": {"enabled": False},
        }

        result = _with_probe_overrides(
            config,
            liquidity_side="low",
            liquidity_quantile=0.35,
            bear_drawdown_threshold=0.08,
            bull_defensive_weight=0.1,
            sideways_defensive_weight=0.6,
            bear_defensive_weight=0.9,
        )

        self.assertTrue(result["liquidity_filter"]["enabled"])
        self.assertEqual(result["liquidity_filter"]["side"], "low")
        self.assertEqual(result["liquidity_filter"]["quantile"], 0.35)
        self.assertEqual(result["market_regime"]["bear_drawdown_threshold"], 0.08)
        self.assertTrue(result["regime_score_blend"]["enabled"])
        self.assertEqual(result["regime_score_blend"]["bull_defensive_weight"], 0.1)
        self.assertEqual(result["regime_score_blend"]["sideways_defensive_weight"], 0.6)
        self.assertEqual(result["regime_score_blend"]["bear_defensive_weight"], 0.9)
        self.assertFalse(config["liquidity_filter"]["enabled"])

    def test_with_timing_probe_applies_exposures_without_mutating_source(self) -> None:
        config = {"defensive_timing": {"sideways_exposure": 0.6, "bear_exposure": 0.3}}

        result = _with_timing_probe(config, sideways_exposure=1.0, bear_exposure=0.1)

        self.assertEqual(result["defensive_timing"]["sideways_exposure"], 1.0)
        self.assertEqual(result["defensive_timing"]["bear_exposure"], 0.1)
        self.assertEqual(config["defensive_timing"]["sideways_exposure"], 0.6)


if __name__ == "__main__":
    unittest.main()
