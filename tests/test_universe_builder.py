"""Tests for historical universe snapshot building."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.universe_builder import (
    apply_configured_historical_universe,
    build_historical_universe,
    filter_scores_by_historical_universe,
    write_historical_universe,
)


class UniverseBuilderTests(unittest.TestCase):
    def test_build_historical_universe_keeps_core_and_top_satellite_members(self) -> None:
        rows = pd.DataFrame(
            [
                {"index_code": "000300.SH", "con_code": "000001.SZ", "trade_date": "20240131", "weight": 5.0},
                {"index_code": "000905.SH", "con_code": "000002.SZ", "trade_date": "20240131", "weight": 4.0},
                {"index_code": "000852.SH", "con_code": "000003.SZ", "trade_date": "20240131", "weight": 3.0},
                {"index_code": "000852.SH", "con_code": "000004.SZ", "trade_date": "20240131", "weight": 2.0},
                {"index_code": "000852.SH", "con_code": "000005.SZ", "trade_date": "20240131", "weight": 1.0},
            ]
        )

        universe = build_historical_universe(rows, satellite_top_n=2)

        self.assertEqual(universe["instrument"].tolist(), ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"])
        self.assertEqual(universe.loc[universe["instrument"] == "000001.SZ", "sources"].iloc[0], "hs300")
        self.assertEqual(int(universe.loc[universe["instrument"] == "000003.SZ", "csi1000_rank"].iloc[0]), 1)
        self.assertNotIn("000005.SZ", set(universe["instrument"]))

    def test_filter_scores_uses_latest_prior_snapshot_without_future_leakage(self) -> None:
        universe = pd.DataFrame(
            [
                {"trade_date": "2024-01-31", "instrument": "000001.SZ"},
                {"trade_date": "2024-02-29", "instrument": "000002.SZ"},
            ]
        )
        scores = pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-15"), "000001.SZ"),
                    (pd.Timestamp("2024-02-01"), "000001.SZ"),
                    (pd.Timestamp("2024-02-01"), "000002.SZ"),
                    (pd.Timestamp("2024-03-01"), "000001.SZ"),
                    (pd.Timestamp("2024-03-01"), "000002.SZ"),
                ],
                names=["datetime", "instrument"],
            ),
            name="score",
        )

        filtered = filter_scores_by_historical_universe(scores, universe)

        self.assertEqual(
            filtered.index.tolist(),
            [
                (pd.Timestamp("2024-02-01"), "000001.SZ"),
                (pd.Timestamp("2024-03-01"), "000002.SZ"),
            ],
        )

    def test_apply_configured_historical_universe_reads_configured_file(self) -> None:
        scores = pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-02-01"), "000001.SZ"), (pd.Timestamp("2024-02-01"), "000002.SZ")],
                names=["datetime", "instrument"],
            ),
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "historical_universe.csv"
            write_historical_universe(pd.DataFrame([{"trade_date": "2024-01-31", "instrument": "000002.SZ"}]), path)

            filtered = apply_configured_historical_universe(
                scores,
                {
                    "universe_builder": {
                        "enabled": True,
                        "output_file": str(path),
                    }
                },
            )

        self.assertEqual(filtered.index.get_level_values("instrument").tolist(), ["000002.SZ"])


if __name__ == "__main__":
    unittest.main()
