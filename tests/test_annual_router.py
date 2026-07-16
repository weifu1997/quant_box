"""Characterization tests for the production annual-router module boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import pandas as pd

from src import annual_router


class AnnualRouterModuleTests(unittest.TestCase):
    def test_production_module_does_not_import_cli_scripts(self) -> None:
        module_path = Path(annual_router.__file__).resolve()
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )

        self.assertFalse(
            any(name == "scripts" or name.startswith("scripts.") for name in imported_modules),
            imported_modules,
        )

    def test_legacy_script_exports_reference_production_implementations(self) -> None:
        from scripts import run_annual_state_router_backtest as backtest_script
        from scripts import run_annual_state_router_grid as grid_script
        from scripts import run_annual_state_router_probe as probe_script

        self.assertIs(backtest_script.ScoreSourceDefinition, annual_router.ScoreSourceDefinition)
        self.assertIs(backtest_script.run_annual_state_score_router, annual_router.run_annual_state_score_router)
        self.assertIs(grid_script.definitions_for_turnover_mode, annual_router.definitions_for_turnover_mode)
        self.assertIs(probe_script.route_for_date, annual_router.route_for_date)
        self.assertIs(probe_script.run_annual_state_router, annual_router.run_annual_state_router)

    def test_latest_score_lookup_preserves_point_in_time_cross_section(self) -> None:
        scores = pd.Series(
            [1.0, float("nan"), 3.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-31"), "A"),
                    (pd.Timestamp("2024-02-15"), "B"),
                    (pd.Timestamp("2024-02-15"), "C"),
                ],
                names=["date", "instrument"],
            ),
            name="score",
        )

        result = annual_router.latest_score_on_or_before(scores, pd.Timestamp("2024-02-29"))

        self.assertEqual(result.index.tolist(), ["C"])
        self.assertEqual(result.name, "score")
        self.assertEqual(float(result.iloc[0]), 3.0)

    def test_turnover_modes_return_new_definition_mapping(self) -> None:
        definition = annual_router.ScoreSourceDefinition(
            name="beta",
            kind="factor",
            top_n=5,
            max_turnover=1,
            rank_buffer=20,
        )

        definitions = {"beta": definition}

        result = annual_router.definitions_for_turnover_mode(definitions, "turnover2")

        self.assertIsNot(result, definitions)
        self.assertEqual(result["beta"].max_turnover, 2)
        self.assertEqual(result["beta"].rank_buffer, 10)
        self.assertEqual(definition.max_turnover, 1)


if __name__ == "__main__":
    unittest.main()
