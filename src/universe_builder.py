"""Build and apply point-in-time stock-universe snapshots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.data_fetcher_frames import normalize_index_constituents_frame
from src.tushare_client import _normalize_symbol_series


logger = logging.getLogger(__name__)

HISTORICAL_UNIVERSE_COLUMNS = [
    "trade_date",
    "instrument",
    "sources",
    "index_codes",
    "source_count",
    "weight",
    "hs300_weight",
    "csi500_weight",
    "csi1000_weight",
    "csi1000_rank",
]
DEFAULT_SOURCE_LABELS = {
    "000300.SH": "hs300",
    "399300.SZ": "hs300",
    "000905.SH": "csi500",
    "399905.SZ": "csi500",
    "000852.SH": "csi1000",
}


def configured_universe_builder(config: dict | None = None) -> dict:
    """Return the universe-builder config block."""
    cfg = config or load_config()
    return cfg.get("universe_builder", {})


def build_historical_universe(
    index_constituents: pd.DataFrame,
    *,
    core_index_codes: Iterable[str] | None = None,
    satellite_index_code: str = "000852.SH",
    satellite_top_n: int = 300,
) -> pd.DataFrame:
    """Build date-stamped universe snapshots from index_weight rows."""
    frame = normalize_index_constituents_frame(index_constituents)
    if frame.empty:
        return pd.DataFrame(columns=HISTORICAL_UNIVERSE_COLUMNS)

    core_codes = set(_normalize_codes(core_index_codes or ["000300.SH", "000905.SH"]))
    satellite_code = _normalize_codes([satellite_index_code])[0]
    top_n = max(0, int(satellite_top_n))

    rows = frame.copy()
    rows["source"] = rows["index_code"].map(lambda value: DEFAULT_SOURCE_LABELS.get(str(value), str(value)))
    rows["satellite_rank"] = pd.NA

    core = rows[rows["index_code"].isin(core_codes)].copy()
    satellite = rows[rows["index_code"] == satellite_code].copy()
    if not satellite.empty and top_n > 0:
        satellite = satellite.sort_values(["trade_date", "weight", "con_code"], ascending=[True, False, True]).copy()
        satellite["satellite_rank"] = satellite.groupby("trade_date", sort=False).cumcount() + 1
        satellite = satellite[satellite["satellite_rank"] <= top_n].copy()
    else:
        satellite = satellite.iloc[0:0].copy()

    selected = pd.concat([core, satellite], ignore_index=True)
    if selected.empty:
        return pd.DataFrame(columns=HISTORICAL_UNIVERSE_COLUMNS)

    records: list[dict[str, object]] = []
    for (trade_date, instrument), group in selected.groupby(["trade_date", "con_code"], sort=True):
        sources = sorted(set(group["source"].astype(str)))
        index_codes = sorted(set(group["index_code"].astype(str)))
        weights = {
            str(row.index_code): float(row.weight) if pd.notna(row.weight) else pd.NA
            for row in group.itertuples(index=False)
        }
        satellite_ranks = pd.to_numeric(group["satellite_rank"], errors="coerce").dropna()
        records.append(
            {
                "trade_date": pd.Timestamp(trade_date).normalize(),
                "instrument": str(instrument),
                "sources": "|".join(sources),
                "index_codes": "|".join(index_codes),
                "source_count": len(sources),
                "weight": _max_numeric(group["weight"]),
                "hs300_weight": weights.get("000300.SH", weights.get("399300.SZ", pd.NA)),
                "csi500_weight": weights.get("000905.SH", weights.get("399905.SZ", pd.NA)),
                "csi1000_weight": weights.get(satellite_code, pd.NA),
                "csi1000_rank": int(satellite_ranks.min()) if not satellite_ranks.empty else pd.NA,
            }
        )

    result = pd.DataFrame(records, columns=HISTORICAL_UNIVERSE_COLUMNS)
    result["instrument"] = _normalize_symbol_series(result["instrument"])
    return result.sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def build_historical_universe_from_file(
    index_constituents_file: str | Path | None = None,
    output_file: str | Path | None = None,
    config: dict | None = None,
) -> Path:
    """Build configured historical-universe snapshots from a cached index_weight CSV."""
    cfg = config or load_config()
    universe_cfg = configured_universe_builder(cfg)
    index_path = resolve_path(
        index_constituents_file
        or universe_cfg.get("index_constituents_file")
        or cfg.get("data_governance", {}).get("index_constituents_file")
        or cfg.get("data", {}).get("hs300_constituents_file", "data/raw/hs300_constituents.csv")
    )
    if not index_path.exists():
        raise FileNotFoundError(f"Index constituents file not found: {index_path}. Run scripts/run_build_universe.py first.")
    rows = pd.read_csv(index_path)
    universe = build_historical_universe(
        rows,
        core_index_codes=universe_cfg.get("core_index_codes", ["000300.SH", "000905.SH"]),
        satellite_index_code=str(universe_cfg.get("satellite_index_code", "000852.SH")),
        satellite_top_n=int(universe_cfg.get("satellite_top_n", 300)),
    )
    out_path = resolve_path(output_file or universe_cfg.get("output_file", "data/raw/historical_universe.csv"))
    write_historical_universe(universe, out_path)
    return out_path


def write_historical_universe(universe: pd.DataFrame, path: str | Path) -> Path:
    """Write a historical universe CSV using the project CSV encoding."""
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = normalize_historical_universe_frame(universe)
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def load_historical_universe(path: str | Path) -> pd.DataFrame:
    """Load and normalize a historical universe CSV."""
    universe_path = resolve_path(path)
    if not universe_path.exists():
        raise FileNotFoundError(f"Historical universe file not found: {universe_path}")
    return normalize_historical_universe_frame(pd.read_csv(universe_path))


def normalize_historical_universe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a saved or in-memory historical universe frame."""
    if frame.empty:
        return pd.DataFrame(columns=HISTORICAL_UNIVERSE_COLUMNS)
    renamed = frame.rename(columns={"date": "trade_date", "con_code": "instrument", "ts_code": "instrument"}).copy()
    missing = [column for column in ["trade_date", "instrument"] if column not in renamed.columns]
    if missing:
        raise ValueError(f"Historical universe data is missing columns: {missing}")
    for column in HISTORICAL_UNIVERSE_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    renamed = renamed[HISTORICAL_UNIVERSE_COLUMNS]
    renamed["trade_date"] = pd.to_datetime(renamed["trade_date"], errors="coerce").dt.normalize()
    renamed["instrument"] = _normalize_symbol_series(renamed["instrument"])
    renamed["source_count"] = pd.to_numeric(renamed["source_count"], errors="coerce").fillna(0).astype(int)
    for column in ["weight", "hs300_weight", "csi500_weight", "csi1000_weight", "csi1000_rank"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed = renamed.dropna(subset=["trade_date", "instrument"])
    renamed = renamed[renamed["instrument"] != ""]
    return renamed.drop_duplicates(["trade_date", "instrument"], keep="last").sort_values(["trade_date", "instrument"]).reset_index(drop=True)


def apply_configured_historical_universe(scores: pd.Series, config: dict | None = None) -> pd.Series:
    """Filter a score panel when historical universe filtering is enabled."""
    cfg = config or load_config()
    universe_cfg = configured_universe_builder(cfg)
    if not bool(universe_cfg.get("enabled", False)):
        return scores
    path = resolve_path(universe_cfg.get("output_file", "data/raw/historical_universe.csv"))
    if not path.exists():
        if bool(universe_cfg.get("require_file", False)):
            raise FileNotFoundError(f"Historical universe file not found: {path}. Run scripts/run_build_universe.py first.")
        logger.warning("Historical universe filtering is enabled but file is missing: %s", path)
        return scores
    universe = load_historical_universe(path)
    filtered = filter_scores_by_historical_universe(scores, universe)
    logger.info("Historical universe filter kept %d/%d score rows using %s.", len(filtered), len(scores), path)
    return filtered


def filter_scores_by_historical_universe(scores: pd.Series, universe: pd.DataFrame) -> pd.Series:
    """Keep score rows whose instruments are in the latest available universe snapshot."""
    if scores.empty:
        return scores
    if not isinstance(scores.index, pd.MultiIndex) or scores.index.nlevels < 2:
        raise ValueError("scores must use MultiIndex: datetime/instrument.")
    snapshots = normalize_historical_universe_frame(universe)
    if snapshots.empty:
        return scores.iloc[0:0].copy()

    snapshot_dates = pd.DatetimeIndex(snapshots["trade_date"]).dropna().unique().sort_values()
    members = {
        pd.Timestamp(date).normalize(): set(group["instrument"].astype(str))
        for date, group in snapshots.groupby("trade_date", sort=True)
    }
    score_dates = pd.to_datetime(scores.index.get_level_values(0), errors="coerce").normalize()
    instruments = _normalize_symbol_series(pd.Series(scores.index.get_level_values(1))).to_numpy()
    keep = [False] * len(scores)

    position_by_date = pd.Series(range(len(scores))).groupby(score_dates, sort=True)
    for score_date, positions in position_by_date:
        if pd.isna(score_date):
            continue
        pos = snapshot_dates.searchsorted(pd.Timestamp(score_date).normalize(), side="right") - 1
        if pos < 0:
            continue
        allowed = members.get(pd.Timestamp(snapshot_dates[pos]).normalize(), set())
        if not allowed:
            continue
        row_positions = positions.to_numpy(dtype=int)
        matches = pd.Index(instruments[row_positions]).isin(allowed)
        for row_position, matched in zip(row_positions, matches):
            keep[int(row_position)] = bool(matched)

    result = scores.iloc[keep].copy()
    result.attrs = dict(getattr(scores, "attrs", {}))
    return result


def _normalize_codes(values: Iterable[object]) -> list[str]:
    codes = [str(value).strip().upper() for value in values if str(value).strip()]
    if not codes:
        raise ValueError("At least one index code is required.")
    return codes


def _max_numeric(values: pd.Series) -> float | object:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else pd.NA
