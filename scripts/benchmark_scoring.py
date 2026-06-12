"""Run a lightweight scoring benchmark with synthetic factor data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import build_strategy_scores
from src.strategy import _NORMALIZED_FACTOR_FRAME_CACHE


def main() -> None:
    """Run the benchmark and print JSON timing results."""
    parser = argparse.ArgumentParser(description="Benchmark build_strategy_scores on synthetic factors.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--instruments", type=int, default=300)
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clear-cache-between", action="store_true")
    args = parser.parse_args()

    factors = _synthetic_factors(args.days, args.instruments, args.factors, args.seed)
    config = {"strategy": {"factor_group": "momentum", "min_cross_section_obs": 5}}
    timings: list[float] = []
    rows = 0
    for _repeat in range(max(args.repeats, 1)):
        if args.clear_cache_between:
            _NORMALIZED_FACTOR_FRAME_CACHE.clear()
        start = time.perf_counter()
        scores = build_strategy_scores(factors, config)
        timings.append(time.perf_counter() - start)
        rows = len(scores)

    result = {
        "days": args.days,
        "instruments": args.instruments,
        "factors": args.factors,
        "repeats": max(args.repeats, 1),
        "score_rows": rows,
        "seconds": timings,
        "first_seconds": timings[0],
        "best_seconds": min(timings),
        "cache_entries": len(_NORMALIZED_FACTOR_FRAME_CACHE),
        "clear_cache_between": bool(args.clear_cache_between),
    }
    print(json.dumps(result, indent=2))


def _synthetic_factors(days: int, instruments: int, factors: int, seed: int) -> pd.DataFrame:
    """Build a deterministic factor frame shaped like Alpha-style scoring input."""
    dates = pd.bdate_range("2024-01-02", periods=max(days, 1))
    symbols = [f"{idx:06d}.SZ" for idx in range(max(instruments, 1))]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "instrument"])
    columns = [f"ROC{idx + 1}" for idx in range(max(factors, 1))]
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(index), len(columns))).astype("float32")
    return pd.DataFrame(values, index=index, columns=columns)


if __name__ == "__main__":
    main()
