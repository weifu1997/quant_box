"""Tests for annual state-router grid helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.run_annual_state_router_grid import (
    append_row,
    combo_key,
    completed_keys,
    grid_exposure_fields,
    iter_grid,
    parse_bool_list,
    parse_reason_set,
    parse_reason_set_list,
)


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

    def test_iter_grid_enumerates_moderate_positive_exposures(self) -> None:
        args = _grid_args(moderate_positive_exposures="0.7,1.0")

        combos = iter_grid(args)

        self.assertEqual([combo["moderate_positive_exposure"] for combo in combos], [0.7, 1.0])

    def test_iter_grid_collapses_moderate_low_exposure_when_low_source_is_none(self) -> None:
        args = _grid_args(moderate_low_sources="none", moderate_low_exposures="0.4,0.6")

        combos = iter_grid(args)

        self.assertEqual(len(combos), 1)
        self.assertIsNone(combos[0]["moderate_low_source"])
        self.assertEqual(combos[0]["moderate_low_exposure"], 1.0)

    def test_iter_grid_enumerates_moderate_lower_band_sources(self) -> None:
        args = _grid_args(moderate_lower_sources="none,rsqr20", moderate_lower_exposures="0.4,0.6")

        combos = iter_grid(args)

        self.assertEqual([combo["moderate_lower_source"] for combo in combos], [None, "rsqr20", "rsqr20"])
        self.assertEqual([combo["moderate_lower_exposure"] for combo in combos], [1.0, 0.4, 0.6])

    def test_iter_grid_enumerates_max_industry_weight_overrides(self) -> None:
        args = _grid_args(max_industry_weights="none,0.35")

        combos = iter_grid(args)

        self.assertEqual([combo["max_industry_weight"] for combo in combos], [None, 0.35])

    def test_iter_grid_enumerates_source_top_n_overrides(self) -> None:
        args = _grid_args(beta_top_ns="none,6,7")

        combos = iter_grid(args)

        self.assertEqual([combo["beta_top_n"] for combo in combos], [None, 6, 7])
        self.assertEqual({combo["beta20_top_n"] for combo in combos}, {None})

    def test_iter_grid_enumerates_risk_exit_min_position_options(self) -> None:
        args = _grid_args(risk_exit_min_positions_options="none,5")

        combos = iter_grid(args)

        self.assertEqual([combo["risk_exit_min_positions"] for combo in combos], [None, 5])

    def test_iter_grid_enumerates_risk_exit_min_position_reason_sets(self) -> None:
        args = _grid_args(risk_exit_min_positions_options="5", risk_exit_min_positions_reason_sets="none;default_beta")

        combos = iter_grid(args)

        self.assertEqual([combo["risk_exit_min_positions_reasons"] for combo in combos], [None, "default_beta"])

    def test_parse_bool_list_accepts_true_false_values(self) -> None:
        self.assertEqual(parse_bool_list("false,true,1,0"), [False, True, True, False])

    def test_append_row_rewrites_with_column_union(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.csv"

            append_row(path, {"key": "old", "annual_return": 0.2})
            append_row(path, {"key": "new", "annual_return": 0.3, "rebalance_after_risk_exit": True})

            frame = pd.read_csv(path)

        self.assertEqual(frame["key"].tolist(), ["old", "new"])
        self.assertIn("rebalance_after_risk_exit", frame.columns)
        self.assertTrue(bool(frame.iloc[1]["rebalance_after_risk_exit"]))

    def test_completed_keys_rebuilds_current_key_when_optional_columns_were_added(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.csv"
            combo = iter_grid(_grid_args(beta_top_ns="6"))[0]
            legacy_combo = dict(combo)
            for field in [
                "moderate_lower_exposure",
                "moderate_lower_ret252_max",
                "moderate_lower_ret252_min",
                "moderate_lower_source",
                "moderate_positive_exposure",
                "risk_exit_min_positions",
                "risk_exit_min_positions_reasons",
            ]:
                legacy_combo.pop(field)
            legacy_key = "|".join(f"{key}={legacy_combo[key]}" for key in sorted(legacy_combo))

            append_row(path, {**legacy_combo, "key": legacy_key, "annual_return": 0.2})

            keys = completed_keys(path)

        self.assertIn(combo_key(combo), keys)

    def test_grid_exposure_fields_reports_latest_position_count(self) -> None:
        holdings = pd.DataFrame(
            [
                {"date": "2024-01-02", "instrument": "A", "value": 100.0},
                {"date": "2024-01-03", "instrument": "A", "value": 100.0},
                {"date": "2024-01-03", "instrument": "B", "value": 50.0},
            ]
        )

        fields = grid_exposure_fields(holdings, {"research": {"exposure": {"daily_basic_file": "missing.parquet"}}})

        self.assertEqual(fields["latest_position_count"], 2)
        self.assertAlmostEqual(fields["latest_top_position_weight"], 2 / 3)


def _grid_args(**overrides: str) -> Namespace:
    values = {
        "missing_ret252_exposures": "0.7",
        "strong_trailing_exposures": "0.8",
        "moderate_positive_sources": "roc60",
        "moderate_positive_ret252_mins": "0.2",
        "moderate_positive_exposures": "1.0",
        "moderate_low_sources": "beta20",
        "moderate_low_ret252_mins": "0.18",
        "moderate_low_ret252_maxs": "0.2",
        "moderate_low_exposures": "0.4",
        "moderate_lower_sources": "none",
        "moderate_lower_ret252_mins": "0.16",
        "moderate_lower_ret252_maxs": "0.18",
        "moderate_lower_exposures": "1.0",
        "turnover_modes": "rank10",
        "turnover_boost_reason_sets": "none",
        "turnover_boost_max_turnovers": "2",
        "turnover_boost_rank_buffers": "10",
        "equity_overlay_sideways_exposures": "none",
        "equity_overlay_bear_exposures": "none",
        "defensive_bear_exposures": "none",
        "max_industry_weights": "none",
        "rebalance_after_risk_exit_options": "false",
        "risk_exit_min_positions_options": "none",
        "risk_exit_min_positions_reason_sets": "none",
        "beta_top_ns": "none",
        "beta20_top_ns": "none",
    }
    values.update(overrides)
    return Namespace(**values)


if __name__ == "__main__":
    unittest.main()
