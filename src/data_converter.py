from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.data_fetcher import normalize_daily_frame


FEATURE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "vwap"]
PRICE_COLUMNS = ["open", "high", "low", "close", "vwap"]
PANEL_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "vwap"]


def convert_to_qlib_format(
    raw_dir: str | Path | None = None,
    qlib_dir: str | Path | None = None,
) -> dict[str, int | Path]:
    config = load_config()
    data_cfg = config.get("data", {})
    source_dir = resolve_path(raw_dir or data_cfg.get("raw_dir", "data/raw"))
    target_dir = resolve_path(qlib_dir or config["qlib"]["provider_uri"])
    missing_value = config.get("qlib", {}).get("missing_value", np.nan)
    missing_value = float(missing_value) if missing_value is not None else np.nan
    calendar_dir = target_dir / "calendars"
    feature_dir = target_dir / "features"
    instrument_dir = target_dir / "instruments"
    prices_dir = resolve_path("data/prices")

    calendar_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    instrument_dir.mkdir(parents=True, exist_ok=True)
    prices_dir.mkdir(parents=True, exist_ok=True)

    universe_file = Path(data_cfg.get("constituents_file", "")).name
    hs300_universe_file = Path(data_cfg.get("hs300_constituents_file", "")).name
    csv_files = sorted(source_dir.glob("*.csv"))
    metadata_files = {
        name
        for name in [
            universe_file,
            hs300_universe_file,
            "hs300_constituents.csv",
            "mainboard_a_stocks.csv",
            "failed_fetches.csv",
        ]
        if name
    }
    csv_files = [path for path in csv_files if path.name not in metadata_files]
    if not csv_files:
        _remove_price_outputs(prices_dir)
        raise FileNotFoundError(f"No raw stock csv files found in {source_dir}")

    calendar_file = data_cfg.get("calendar_file")
    tradable_file = data_cfg.get("tradable_file")
    tradable_codes = _load_tradable_codes(tradable_file) if tradable_file else None
    all_dates: set[str] = set()
    prepared: list[tuple[str, pd.DataFrame]] = []

    for csv_file in csv_files:
        raw = pd.read_csv(csv_file)
        code = _instrument_from_filename(csv_file)
        if tradable_codes is not None and code.upper() not in tradable_codes:
            continue
        df = normalize_daily_frame(raw, default_ts_code=code)
        df = df.sort_values("trade_date")
        if df.empty:
            continue

        all_dates.update(df["trade_date"].dt.strftime("%Y-%m-%d"))
        prepared.append((code, df))

    if not prepared:
        _remove_price_outputs(prices_dir)
        raise ValueError("No valid stock data remained after filtering raw CSV files.")

    if calendar_file:
        calendar_path = resolve_path(calendar_file)
        calendar = _read_calendar_dates(calendar_path)
    else:
        calendar = sorted(all_dates)
    calendar_index = {date: idx for idx, date in enumerate(calendar)}
    pd.Series(calendar).to_csv(calendar_dir / "day.txt", index=False, header=False, encoding="utf-8")

    instruments: list[tuple[str, str, str]] = []
    close_frames: list[pd.Series] = []
    panel_frames: list[pd.DataFrame] = []
    adjusted_close_frames: list[pd.Series] = []
    adjusted_panel_frames: list[pd.DataFrame] = []
    for code, df in prepared:
        start = df["trade_date"].min().strftime("%Y-%m-%d")
        end = df["trade_date"].max().strftime("%Y-%m-%d")
        instruments.append((code, start, end))

        raw_feature_df = _prepare_feature_frame(df, adjusted=False)
        feature_df = _prepare_feature_frame(df, adjusted=True)
        stock_dir = feature_dir / code.lower()
        stock_dir.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(stock_dir / "day.parquet", index=False)
        _write_qlib_bin_features(stock_dir, feature_df, calendar, calendar_index, missing_value=missing_value)

        close_frames.append(_close_panel(raw_feature_df, code))
        panel_frames.append(_ohlcv_panel(raw_feature_df, code))
        adjusted_close_frames.append(_close_panel(feature_df, code))
        adjusted_panel_frames.append(_ohlcv_panel(feature_df, code))

    instrument_name = str(config.get("qlib", {}).get("instruments", "all"))
    _write_instruments(instrument_dir / "all.txt", instruments)
    _write_instruments(instrument_dir / f"{instrument_name}.txt", instruments)

    if close_frames:
        pd.concat(close_frames, axis=1).sort_index().to_parquet(prices_dir / "close.parquet")
    if panel_frames:
        pd.concat(panel_frames, axis=1).sort_index().to_parquet(prices_dir / "ohlcv.parquet")
    if adjusted_close_frames:
        pd.concat(adjusted_close_frames, axis=1).sort_index().to_parquet(prices_dir / "close_adjusted.parquet")
    if adjusted_panel_frames:
        pd.concat(adjusted_panel_frames, axis=1).sort_index().to_parquet(prices_dir / "ohlcv_adjusted.parquet")

    return {
        "calendar_days": len(calendar),
        "instruments": len(instruments),
        "provider_uri": target_dir,
        "close_price_file": prices_dir / "close.parquet",
        "ohlcv_price_file": prices_dir / "ohlcv.parquet",
        "adjusted_close_price_file": prices_dir / "close_adjusted.parquet",
        "adjusted_ohlcv_price_file": prices_dir / "ohlcv_adjusted.parquet",
    }


def _prepare_feature_frame(df: pd.DataFrame, adjusted: bool) -> pd.DataFrame:
    feature_df = df.rename(columns={"trade_date": "date", "vol": "volume"}).copy()
    feature_df = _sanitize_market_values(feature_df)
    feature_df["vwap"] = np.where(
        feature_df["volume"].astype(float) > 0,
        feature_df["amount"].astype(float) * 10 / feature_df["volume"].astype(float),
        np.nan,
    )
    if adjusted:
        feature_df = _apply_adjustment(feature_df)
    return feature_df[FEATURE_COLUMNS]


def _sanitize_market_values(feature_df: pd.DataFrame) -> pd.DataFrame:
    sanitized = feature_df.copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in sanitized.columns:
            continue
        values = pd.to_numeric(sanitized[col], errors="coerce")
        sanitized[col] = values.where(values > 0)
    return sanitized


def _load_tradable_codes(path_value: str | Path) -> set[str]:
    path = resolve_path(path_value)
    df = pd.read_csv(path)
    col = next((name for name in ["ts_code", "instrument", "code", "con_code"] if name in df.columns), None)
    if col is None:
        raise ValueError(f"{path} must contain one of: ts_code, instrument, code, con_code.")
    symbols: set[str] = set()
    for value in df[col].dropna():
        symbol = _normalize_symbol(value)
        if symbol:
            symbols.add(symbol)
    return symbols


def _normalize_symbol(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _read_calendar_dates(path: Path) -> list[str]:
    frame = pd.read_csv(path, header=None)
    if frame.empty:
        return []
    first_row = [str(value).strip().lower() for value in frame.iloc[0].tolist()]
    known_headers = {"cal_date", "trade_date", "date", "datetime", "is_open"}
    if set(first_row) & known_headers:
        frame.columns = first_row
        frame = frame.iloc[1:].reset_index(drop=True)
    lower_columns = {str(col).strip().lower(): col for col in frame.columns}
    if "is_open" in lower_columns:
        frame = frame[_open_day_mask(frame[lower_columns["is_open"]])]
    date_col = next(
        (lower_columns[name] for name in ["cal_date", "trade_date", "date", "datetime"] if name in lower_columns),
        frame.columns[0],
    )
    raw = frame[date_col].dropna().astype(str).str.strip()
    raw = raw[~raw.str.lower().isin({"cal_date", "trade_date", "date", "datetime"})]
    compact = raw.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(raw, errors="coerce")
    dates = pd.Series(parsed).where(pd.Series(parsed).notna(), pd.Series(fallback))
    return sorted({pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates.dropna()})


def _open_day_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y", "open"})


def _close_panel(feature_df: pd.DataFrame, code: str) -> pd.Series:
    return feature_df.set_index("date")["close"].rename(code.lower())


def _ohlcv_panel(feature_df: pd.DataFrame, code: str) -> pd.DataFrame:
    panel = feature_df.set_index("date")[PANEL_COLUMNS].copy()
    panel.columns = pd.MultiIndex.from_product([panel.columns, [code.lower()]], names=["field", "instrument"])
    return panel


def _apply_adjustment(feature_df: pd.DataFrame) -> pd.DataFrame:
    if "adj_factor" not in feature_df.columns or feature_df["adj_factor"].dropna().empty:
        return feature_df

    adjusted = feature_df.copy()
    if adjusted["adj_factor"].isna().any():
        raise ValueError("adj_factor has missing values. Re-run data update to backfill adjusted-price data.")
    latest_factor = float(adjusted["adj_factor"].dropna().iloc[-1])
    if latest_factor <= 0:
        raise ValueError("latest adj_factor must be positive.")
    multiplier = adjusted["adj_factor"].astype(float) / latest_factor
    for col in PRICE_COLUMNS:
        adjusted[col] = adjusted[col].astype(float) * multiplier
    if "volume" in adjusted.columns:
        adjusted["volume"] = adjusted["volume"].astype(float) / multiplier
    return adjusted


def _instrument_from_filename(path: Path) -> str:
    return path.stem.upper()


def _write_instruments(path: Path, instruments: list[tuple[str, str, str]]) -> None:
    rows = [f"{code.lower()}\t{start}\t{end}" for code, start, end in instruments]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _remove_price_outputs(prices_dir: Path) -> None:
    for name in ["close.parquet", "ohlcv.parquet", "close_adjusted.parquet", "ohlcv_adjusted.parquet"]:
        path = prices_dir / name
        if path.exists():
            path.unlink()


def _write_qlib_bin_features(
    stock_dir: Path,
    feature_df: pd.DataFrame,
    calendar: list[str],
    calendar_index: dict[str, int],
    missing_value: float,
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
        values = np.where(np.isfinite(values), values, np.nan)
        if not pd.isna(missing_value):
            values = np.nan_to_num(values, nan=missing_value, posinf=missing_value, neginf=missing_value)
        payload = np.hstack([[start_idx], values]).astype("<f")
        payload.tofile(stock_dir / f"{field}.day.bin")
