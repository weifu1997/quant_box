"""Tests for annual state-router grid helpers."""

from __future__ import annotations

from argparse import Namespace
import unittest

from scripts.run_annual_state_router_grid import iter_grid, parse_reason_set, parse_reason_set_list


class RunAnnualStateRouterGridTests(unittest.TestCase):
    def test_parse_reason_set_list_uses_semicolon_sets_and_plus_members(self) -> None:
        parsed = parse_reason_set_list("none; low_vol_moderate_uptrend+moderate_positive_roc60 ")

        self.assertEqual(parsed, ["none", "low_vol_moderate_uptrend+moderate_positive_roc60"])
        self.assertEqual(
            parse_reason_set(parsed[1]),
            {"low_vol_moderate_uptrend", "moderate_positive_roc60"},
        )

    def test_iter_grid_enumerates_moderate_low_exposures_for_real_low_source(self) -> None:
        args = _grid_args(moderate_low_sources="beta20", moderate_low_exposures="0.4,0.6")

        combos = iter_grid(args)

        self.assertEqual([combo["moderate_low_exposure"] for combo in combos], [0.4, 0.6])
        self.assertEqual({combo["moderate_low_source"] for combo in combos}, {"beta20"})

    def test_iter_grid_collapses_moderate_low_exposure_when_low_source_is_none(self) -> None:
        args = _grid_args(moderate_low_sources="none", moderate_low_exposures="0.4,0.6")

        combos = iter_grid(args)

        self.assertEqual(len(combos), 1)
        self.assertIsNone(combos[0]["moderate_low_source"])
        self.assertEqual(combos[0]["moderate_low_exposure"], 1.0)


def _grid_args(**overrides: str) -> Namespace:
    values = {
        "missing_ret252_exposures": "0.7",
        "strong_trailing_exposures": "0.8",
        "moderate_positive_sources": "roc60",
        "moderate_positive_ret252_mins": "0.2",
        "moderate_low_sources": "beta20",
        "moderate_low_ret252_mins": "0.18",
        "moderate_low_ret252_maxs": "0.2",
        "moderate_low_exposures": "0.4",
        "turnover_modes": "rank10",
        "turnover_boost_reason_sets": "none",
        "turnover_boost_max_turnovers": "2",
        "turnover_boost_rank_buffers": "10",
        "equity_overlay_sideways_exposures": "none",
        "equity_overlay_bear_exposures": "none",
        "defensive_bear_exposures": "none",
    }
    values.update(overrides)
    return Namespace(**values)


if __name__ == "__main__":
    unittest.main()
