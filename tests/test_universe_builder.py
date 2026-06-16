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

    def test_build_historical_universe_keeps_hs300_fallback_index_code(self) -> None:
        rows = pd.DataFrame(
            [
                {"index_code": "399300.SZ", "con_code": "600000.SH", "trade_date": "20150130", "weight": 2.0},
                {"index_code": "000905.SH", "con_code": "000001.SZ", "trade_date": "20150130", "weight": 1.0},
            ]
        )

        universe = build_historical_universe(rows)

        self.assertEqual(set(universe["instrument"]), {"600000.SH", "000001.SZ"})
        hs300 = universe[universe["instrument"] == "600000.SH"].iloc[0]
        self.assertEqual(hs300["sources"], "hs300")
        self.assertEqual(hs300["index_codes"], "399300.SZ")
        self.assertAlmostEqual(float(hs300["hs300_weight"]), 2.0)

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

    def test_filter_scores_carries_each_source_forward_independently(self) -> None:
        rows = pd.DataFrame(
            [
                {"index_code": "000300.SH", "con_code": "600000.SH", "trade_date": "20240131", "weight": 3.0},
                {"index_code": "000905.SH", "con_code": "000001.SZ", "trade_date": "20240131", "weight": 2.0},
                {"index_code": "000852.SH", "con_code": "300001.SZ", "trade_date": "20240215", "weight": 1.0},
            ]
        )
        universe = build_historical_universe(rows, satellite_top_n=300)
        scores = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-02-16"), "600000.SH"),
                    (pd.Timestamp("2024-02-16"), "000001.SZ"),
                    (pd.Timestamp("2024-02-16"), "300001.SZ"),
                ],
                names=["datetime", "instrument"],
            ),
        )

        filtered = filter_scores_by_historical_universe(scores, universe)

        self.assertEqual(
            filtered.index.get_level_values("instrument").tolist(),
            ["600000.SH", "000001.SZ", "300001.SZ"],
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

    def test_apply_configured_historical_universe_requires_file_by_default_when_enabled(self) -> None:
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-02-01"), "000001.SZ")],
                names=["datetime", "instrument"],
            ),
        )
        with TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing_historical_universe.csv"

            with self.assertRaisesRegex(FileNotFoundError, "Run scripts/run_build_universe.py first"):
                apply_configured_historical_universe(
                    scores,
                    {
                        "universe_builder": {
                            "enabled": True,
                            "output_file": str(missing_path),
                        }
                    },
                )


if __name__ == "__main__":
    unittest.main()
