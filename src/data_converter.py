from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.data_fetcher import normalize_daily_frame


FEATURE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "vwap"]


def convert_to_qlib_format(
    raw_dir: str | Path | None = None,
    qlib_dir: str | Path | None = None,
) -> dict[str, int | Path]:
    config = load_config()
    source_dir = resolve_path(raw_dir or config["data"].get("raw_dir", "data/raw"))
    target_dir = resolve_path(qlib_dir or config["qlib"]["provider_uri"])
    calendar_dir = target_dir / "calendars"
    feature_dir = target_dir / "features"
    instrument_dir = target_dir / "instruments"

    calendar_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    instrument_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(source_dir.glob("*.csv"))
    csv_files = [path for path in csv_files if path.name != "hs300_constituents.csv"]
    if not csv_files:
        raise FileNotFoundError(f"No raw stock csv files found in {source_dir}")

    all_dates: set[str] = set()
    prepared: list[tuple[str, pd.DataFrame]] = []

    for csv_file in csv_files:
        raw = pd.read_csv(csv_file)
        code = _instrument_from_filename(csv_file)
        df = normalize_daily_frame(raw, default_ts_code=code)
        df = df.sort_values("trade_date")
        if df.empty:
            continue

        all_dates.update(df["trade_date"].dt.strftime("%Y-%m-%d"))
        prepared.append((code, df))

    calendar = sorted(all_dates)
    calendar_index = {date: idx for idx, date in enumerate(calendar)}
    pd.Series(calendar).to_csv(calendar_dir / "day.txt", index=False, header=False, encoding="utf-8")

    instruments: list[tuple[str, str, str]] = []
    close_frames: list[pd.Series] = []
    for code, df in prepared:
        start = df["trade_date"].min().strftime("%Y-%m-%d")
        end = df["trade_date"].max().strftime("%Y-%m-%d")
        instruments.append((code, start, end))

        feature_df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        feature_df["vwap"] = np.where(
            feature_df["volume"].astype(float) > 0,
            feature_df["amount"].astype(float) * 10 / feature_df["volume"].astype(float),
            np.nan,
        )
        feature_df = feature_df[FEATURE_COLUMNS]
        stock_dir = feature_dir / code.lower()
        stock_dir.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(stock_dir / "day.parquet", index=False)
        _write_qlib_bin_features(stock_dir, feature_df, calendar, calendar_index)

        close = feature_df.set_index("date")["close"].rename(code.lower())
        close_frames.append(close)

    _write_instruments(instrument_dir / "all.txt", instruments)
    _write_instruments(instrument_dir / "csi300.txt", instruments)

    prices_dir = resolve_path("data/prices")
    prices_dir.mkdir(parents=True, exist_ok=True)
    if close_frames:
        pd.concat(close_frames, axis=1).sort_index().to_parquet(prices_dir / "close.parquet")

    return {
        "calendar_days": len(calendar),
        "instruments": len(instruments),
        "provider_uri": target_dir,
        "close_price_file": prices_dir / "close.parquet",
    }


def _instrument_from_filename(path: Path) -> str:
    return path.stem.upper()


def _write_instruments(path: Path, instruments: list[tuple[str, str, str]]) -> None:
    rows = [f"{code.lower()}\t{start}\t{end}" for code, start, end in instruments]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_qlib_bin_features(
    stock_dir: Path,
    feature_df: pd.DataFrame,
    calendar: list[str],
    calendar_index: dict[str, int],
) -> None:
    indexed = feature_df.copy()
    indexed["date"] = pd.to_datetime(indexed["date"]).dt.strftime("%Y-%m-%d")
    indexed = indexed.set_index("date").sort_index()
    if indexed.empty:
        return

    start_idx = calendar_index[indexed.index.min()]
    end_idx = calendar_index[indexed.index.max()]
    date_slice = calendar[start_idx : end_idx + 1]
    for field in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
        values = indexed[field].reindex(date_slice).astype("float32").to_numpy()
        payload = np.hstack([[start_idx], values]).astype("<f")
        payload.tofile(stock_dir / f"{field}.day.bin")
