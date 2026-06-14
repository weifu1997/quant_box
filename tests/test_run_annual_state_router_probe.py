"""Tests for the annual state router research probe."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_annual_state_router_probe import (
    parse_source_args,
    route_source,
    run_annual_state_router,
)


class RunAnnualStateRouterProbeTests(unittest.TestCase):
    def test_parse_source_args_requires_name_path_pairs(self) -> None:
        self.assertEqual(parse_source_args(["alpha=outputs/a.csv"]), {"alpha": "outputs/a.csv"})
        with self.assertRaisesRegex(ValueError, "name=path"):
            parse_source_args(["outputs/a.csv"])

    def test_route_source_maps_market_states_to_sources(self) -> None:
        self.assertEqual(route_source(ret126=0.0, ret252=float("nan"), vol252=float("nan")), ("db_size", "ret252_missing"))
        self.assertEqual(route_source(ret126=0.02, ret252=-0.10, vol252=0.30), ("quality", "negative_high_vol"))
        self.assertEqual(route_source(ret126=-0.02, ret252=-0.03, vol252=0.12), ("beta", "flat_with_negative_half_year"))
        self.assertEqual(route_source(ret126=0.20, ret252=0.35, vol252=0.20), ("selector", "strong_trailing_market"))

    def test_router_locks_source_for_calendar_year_and_applies_exposure(self) -> None:
        dates = pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-03",
                "2024-12-31",
                "2025-01-02",
                "2025-01-03",
            ]
        )
        source_returns = pd.DataFrame(
            {
                "beta": [0.10, 0.10, 0.10, 0.02, 0.02],
                "db_size": [0.01, 0.01, 0.01, 0.20, 0.20],
                "quality": [0.01, 0.01, 0.01, 0.01, 0.01],
                "selector": [0.01, 0.01, 0.01, 0.01, 0.01],
                "industry": [0.01, 0.01, 0.01, 0.01, 0.01],
            },
            index=dates,
        )
        benchmark = pd.Series(
            [100.0, 110.0, 120.0],
            index=pd.to_datetime(["2024-01-01", "2024-06-30", "2024-12-31"]),
        )

        run = run_annual_state_router(
            source_returns=source_returns,
            benchmark=benchmark,
            initial_source="beta",
            missing_ret252_exposure=0.5,
            flat_negative_exposure=0.9,
        )

        self.assertEqual(run.routes["source"].tolist(), ["beta", "db_size"])
        self.assertEqual(run.routes["reason"].tolist(), ["insufficient_history", "ret252_missing"])
        self.assertEqual(float(run.routes.iloc[1]["exposure"]), 0.5)
        # 2025 uses db_size at half exposure: 1 + 20% * 0.5 per day.
        self.assertAlmostEqual(float(run.equity.iloc[-1]), (1.1**3) * 1.1 * 1.1)


if __name__ == "__main__":
    unittest.main()
