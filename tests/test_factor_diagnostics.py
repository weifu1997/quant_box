"""Tests for factor diagnostic table builders."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.factor_diagnostics import build_factor_diagnostics, write_factor_diagnostics


class FactorDiagnosticsTests(unittest.TestCase):
    def test_build_factor_diagnostics_writes_yearly_ic_and_group_returns(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        instruments = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
        index = pd.MultiIndex.from_product([dates[:3], instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "alpha_a": [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
                "alpha_b": [4, 3, 2, 1, 5, 4, 3, 2, 6, 5, 4, 3],
            },
            index=index,
            dtype=float,
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame(
                    {
                        "000001.SZ": [10.0, 10.0, 10.0, 10.0],
                        "000002.SZ": [10.0, 10.5, 11.0, 11.5],
                        "000003.SZ": [10.0, 11.0, 12.0, 13.0],
                        "000004.SZ": [10.0, 11.5, 13.0, 14.5],
                    },
                    index=dates,
                )
            },
            axis=1,
        )

        tables = build_factor_diagnostics(factors, prices, min_obs=2, quantiles=2)

        self.assertIn("alpha_a", tables["ic_summary"].index)
        self.assertIn("year", tables["yearly_ic"].columns)
        self.assertIn("top_minus_bottom", tables["group_returns"].columns)
        self.assertGreater(len(tables["group_returns"]), 0)

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("src.factor_diagnostics.resolve_path", side_effect=lambda value: Path(value)):
                paths = write_factor_diagnostics(tables, out_dir)

            self.assertTrue(Path(paths["factor_ic_summary"]).exists())
            self.assertTrue(Path(paths["factor_ic_yearly"]).exists())
            self.assertTrue(Path(paths["factor_group_returns"]).exists())


if __name__ == "__main__":
    unittest.main()
