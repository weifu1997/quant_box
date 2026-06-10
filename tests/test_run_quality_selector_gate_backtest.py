"""Tests for quality/selector regime gate helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_quality_selector_gate_backtest import (
    annual_market_state_gated_scores,
    annual_market_state_source,
    annual_volatility_gated_scores,
    annual_volatility_source,
    gated_scores,
    normalize_score_panel,
    parse_regimes,
)


class RunQualitySelectorGateBacktestTests(unittest.TestCase):
    def test_normalize_score_panel_zscores_each_date(self) -> None:
        scores = pd.Series(
            [1.0, 3.0, 10.0, 10.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-31"), "A"),
                    (pd.Timestamp("2024-01-31"), "B"),
                    (pd.Timestamp("2024-02-29"), "A"),
                    (pd.Timestamp("2024-02-29"), "B"),
                ],
                names=["date", "instrument"],
            ),
            name="score",
        )

        normalized = normalize_score_panel(scores)

        jan = normalized.xs(pd.Timestamp("2024-01-31"), level=0)
        feb = normalized.xs(pd.Timestamp("2024-02-29"), level=0)
        self.assertAlmostEqual(float(jan.mean()), 0.0)
        self.assertAlmostEqual(float(feb.sum()), 0.0)

    def test_gated_scores_choose_source_by_regime(self) -> None:
        date_a = pd.Timestamp("2024-01-31")
        date_b = pd.Timestamp("2024-02-29")
        quality = pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_product([[date_a, date_b], ["Q"]], names=["date", "instrument"]),
            name="score",
        )
        selector = pd.Series(
            [3.0, 4.0],
            index=pd.MultiIndex.from_product([[date_a, date_b], ["S"]], names=["date", "instrument"]),
            name="score",
        )
        regimes = pd.Series(["bull", "bear"], index=[date_a, date_b])

        scores, rows = gated_scores(
            quality_scores=quality,
            selector_scores=selector,
            regimes=regimes,
            quality_regimes={"bull", "sideways"},
            selector_regimes={"bear"},
        )

        self.assertEqual(scores.xs(date_a, level=0).index.tolist(), ["Q"])
        self.assertEqual(scores.xs(date_b, level=0).index.tolist(), ["S"])
        self.assertEqual([row["source"] for row in rows], ["quality", "selector"])

    def test_parse_regimes_trims_values(self) -> None:
        self.assertEqual(parse_regimes("bull, sideways"), {"bull", "sideways"})

    def test_annual_volatility_source_uses_declared_threshold_direction(self) -> None:
        self.assertEqual(annual_volatility_source(0.12, threshold=0.18, selector_if="below"), "selector")
        self.assertEqual(annual_volatility_source(0.24, threshold=0.18, selector_if="below"), "quality")
        self.assertEqual(annual_volatility_source(0.24, threshold=0.18, selector_if="above"), "selector")
        self.assertEqual(annual_volatility_source(None, threshold=0.18, selector_if="below"), "quality")

    def test_annual_volatility_gate_locks_source_for_calendar_year(self) -> None:
        date_a = pd.Timestamp("2024-01-31")
        date_b = pd.Timestamp("2024-02-29")
        quality = pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_product([[date_a, date_b], ["Q"]], names=["date", "instrument"]),
            name="score",
        )
        selector = pd.Series(
            [3.0, 4.0],
            index=pd.MultiIndex.from_product([[date_a, date_b], ["S"]], names=["date", "instrument"]),
            name="score",
        )
        annual_volatility = pd.Series(
            [0.12, 0.30],
            index=[pd.Timestamp("2024-01-30"), pd.Timestamp("2024-02-28")],
            name="annual_volatility",
        )

        scores, rows = annual_volatility_gated_scores(
            quality_scores=quality,
            selector_scores=selector,
            annual_volatility=annual_volatility,
            threshold=0.18,
            selector_if="below",
        )

        self.assertEqual(scores.xs(date_a, level=0).index.tolist(), ["S"])
        self.assertEqual(scores.xs(date_b, level=0).index.tolist(), ["S"])
        self.assertEqual([row["source"] for row in rows], ["selector", "selector"])
        self.assertEqual({row["volatility_date"] for row in rows}, {"2024-01-30"})

    def test_annual_market_state_source_prefers_quality_only_in_weak_high_vol_state(self) -> None:
        self.assertEqual(
            annual_market_state_source(
                0.24,
                0.03,
                0.12,
                volatility_min=0.20,
                quality_momentum_max=0.08,
                quality_ret252_min=0.0,
            ),
            "quality",
        )
        self.assertEqual(
            annual_market_state_source(
                0.24,
                0.12,
                0.12,
                volatility_min=0.20,
                quality_momentum_max=0.08,
                quality_ret252_min=0.0,
            ),
            "selector",
        )

    def test_annual_market_state_gate_locks_source_for_calendar_year(self) -> None:
        date_a = pd.Timestamp("2024-01-31")
        date_b = pd.Timestamp("2024-02-29")
        quality = pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_product([[date_a, date_b], ["Q"]], names=["date", "instrument"]),
            name="score",
        )
        selector = pd.Series(
            [3.0, 4.0],
            index=pd.MultiIndex.from_product([[date_a, date_b], ["S"]], names=["date", "instrument"]),
            name="score",
        )
        annual_state = pd.DataFrame(
            {
                "annual_volatility": [0.24, 0.10],
                "momentum": [0.03, 0.20],
                "ret252": [0.12, 0.15],
            },
            index=[pd.Timestamp("2024-01-30"), pd.Timestamp("2024-02-28")],
        )

        scores, rows = annual_market_state_gated_scores(
            quality_scores=quality,
            selector_scores=selector,
            annual_state=annual_state,
            volatility_min=0.20,
            quality_momentum_max=0.08,
            quality_ret252_min=0.0,
        )

        self.assertEqual(scores.xs(date_a, level=0).index.tolist(), ["Q"])
        self.assertEqual(scores.xs(date_b, level=0).index.tolist(), ["Q"])
        self.assertEqual([row["source"] for row in rows], ["quality", "quality"])
        self.assertEqual({row["state_date"] for row in rows}, {"2024-01-30"})


if __name__ == "__main__":
    unittest.main()
