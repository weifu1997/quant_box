"""模块说明：覆盖 real_data 相关行为的测试用例。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = Path(__file__).resolve().parent / "data_snapshot"
FULL_DATA_ROOT = ROOT / "data"

DEFAULT_INSTRUMENTS = (
    "000001.SZ",
    "000002.SZ",
    "000006.SZ",
    "000007.SZ",
    "000008.SZ",
    "000009.SZ",
    "000011.SZ",
    "000012.SZ",
)
DEFAULT_START = "2024-01-02"
DEFAULT_END = "2024-04-30"
DEFAULT_FACTOR_COLUMNS = ("LOW0", "ROC5", "ROC20", "STD20", "BETA20")
DEFAULT_PRICE_FIELDS = ("open", "close", "low", "high", "volume", "amount")


@dataclass(frozen=True)
class RealMarketData:
    """类说明：封装 RealMarketData 相关数据和行为。"""
    factors: pd.DataFrame
    prices: pd.DataFrame
    close: pd.DataFrame
    daily_basic: pd.DataFrame
    instruments: list[str]
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class _MarketDataPaths:
    """类说明：封装 MarketDataPaths 相关数据和行为。"""
    name: str
    root: Path
    price_panel: Path
    close_panel: Path
    factor_panel: Path
    daily_basic: Path


def require_real_market_data(
    instruments: Iterable[str] = DEFAULT_INSTRUMENTS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    factor_columns: Iterable[str] = DEFAULT_FACTOR_COLUMNS,
    price_fields: Iterable[str] = DEFAULT_PRICE_FIELDS,
    require_daily_basic: bool = False,
) -> RealMarketData:
    """Load a deterministic real A-share slice from the committed snapshot."""
    instrument_tuple = tuple(_normalize_instrument(value) for value in instruments)
    factor_column_tuple = tuple(str(value) for value in factor_columns)
    price_field_tuple = tuple(str(value) for value in price_fields)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    paths = _select_data_paths(start_ts, end_ts, instrument_tuple, factor_column_tuple, price_field_tuple)
    return _load_real_market_data(
        paths.name,
        paths.price_panel,
        paths.close_panel,
        paths.factor_panel,
        paths.daily_basic,
        instrument_tuple,
        start,
        end,
        factor_column_tuple,
        price_field_tuple,
        require_daily_basic,
    )


@lru_cache(maxsize=16)
def _load_real_market_data(
    source_name: str,
    price_panel: Path,
    close_panel: Path,
    factor_panel: Path,
    daily_basic_panel: Path,
    instruments: tuple[str, ...],
    start: str,
    end: str,
    factor_columns: tuple[str, ...],
    price_fields: tuple[str, ...],
    require_daily_basic: bool,
) -> RealMarketData:
    """函数说明：加载 load_real_market_data 的内部辅助逻辑。"""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    storage_instruments = tuple(_storage_instrument(value) for value in instruments)
    try:
        close = pd.read_parquet(close_panel, columns=list(storage_instruments))
    except Exception as exc:  # pragma: no cover - exercised when the committed snapshot is corrupted.
        pytest.fail(f"Unable to read real close price panel from {source_name}: {exc}")
    close = close.loc[(close.index >= start_ts) & (close.index <= end_ts)].copy()
    close.columns = [_normalize_instrument(value) for value in close.columns]
    close = close.dropna(axis=1, how="all")

    available = [instrument for instrument in instruments if instrument in close.columns]
    if set(available) != set(instruments):
        missing = sorted(set(instruments) - set(available))
        pytest.fail(f"Real close price panel from {source_name} is missing requested instruments: {missing}")

    price_columns = [str((field, _storage_instrument(instrument))) for field in price_fields for instrument in available]
    prices = _read_price_panel(price_panel, price_columns, source_name)
    prices = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)].copy()
    prices.columns = pd.MultiIndex.from_tuples(
        [(str(field), _normalize_instrument(instrument)) for field, instrument in prices.columns],
        names=["field", "instrument"],
    )
    prices = prices.loc[:, prices.columns.get_level_values("instrument").isin(available)]
    _require_price_columns(prices, price_fields, available, source_name)

    try:
        factors = pd.read_parquet(factor_panel, columns=list(factor_columns))
    except Exception as exc:  # pragma: no cover - exercised when the committed snapshot is corrupted.
        pytest.fail(f"Unable to read real factor panel from {source_name}: {exc}")
    dates = pd.to_datetime(factors.index.get_level_values("datetime")).normalize()
    factor_instruments = pd.Index([_normalize_instrument(value) for value in factors.index.get_level_values("instrument")])
    factors = factors[(dates >= start_ts) & (dates <= end_ts) & factor_instruments.isin(available)].copy()
    if factors.empty:
        pytest.fail(f"Real factor panel from {source_name} has no rows for the requested slice.")
    factors = _normalize_factor_index(factors)

    active = sorted(set(factors.index.get_level_values("instrument")) & set(available))
    if set(active) != set(available):
        missing = sorted(set(available) - set(active))
        pytest.fail(f"Real factor panel from {source_name} is missing requested instruments: {missing}")
    factors = factors[factors.index.get_level_values("instrument").isin(active)].sort_index()
    prices = prices.loc[:, prices.columns.get_level_values("instrument").isin(active)].sort_index()
    close = close.loc[:, active].sort_index()

    try:
        daily_basic = pd.read_parquet(daily_basic_panel)
    except Exception as exc:  # pragma: no cover - exercised when the committed snapshot is corrupted.
        pytest.fail(f"Unable to read real daily_basic panel from {source_name}: {exc}")
    if {"trade_date", "ts_code"}.issubset(daily_basic.columns):
        daily_dates = pd.to_datetime(daily_basic["trade_date"], errors="coerce").dt.normalize()
        daily_codes = pd.Index([_normalize_instrument(value) for value in daily_basic["ts_code"]])
        daily_basic = daily_basic[(daily_dates >= start_ts) & (daily_dates <= end_ts) & daily_codes.isin(active)].copy()
        daily_basic["ts_code"] = [_normalize_instrument(value) for value in daily_basic["ts_code"]]
    else:
        daily_basic = daily_basic.iloc[0:0].copy()
    if require_daily_basic and daily_basic.empty:
        pytest.fail(f"Real daily_basic panel from {source_name} has no rows for the requested slice.")

    return RealMarketData(
        factors=factors,
        prices=prices,
        close=close,
        daily_basic=daily_basic,
        instruments=active,
        start=start_ts,
        end=end_ts,
    )


def _select_data_paths(
    start: pd.Timestamp,
    end: pd.Timestamp,
    instruments: tuple[str, ...],
    factor_columns: tuple[str, ...],
    price_fields: tuple[str, ...],
) -> _MarketDataPaths:
    """函数说明：选择 select_data_paths 的内部辅助逻辑。"""
    snapshot_paths = _paths_for("snapshot", SNAPSHOT_ROOT)
    _require_files(snapshot_paths)
    if _snapshot_covers(start, end, instruments, factor_columns, price_fields):
        return snapshot_paths

    full_paths = _paths_for("full data cache", FULL_DATA_ROOT)
    if _files_exist(full_paths):
        return full_paths

    pytest.fail(
        "Requested real data slice is outside tests/fixtures/data_snapshot and the full data cache is missing."
    )


def _paths_for(name: str, root: Path) -> _MarketDataPaths:
    """函数说明：处理 paths_for 的内部辅助逻辑。"""
    return _MarketDataPaths(
        name=name,
        root=root,
        price_panel=root / "prices" / "ohlcv_adjusted.parquet",
        close_panel=root / "prices" / "close_adjusted.parquet",
        factor_panel=root / "factors" / "alpha158.parquet",
        daily_basic=root / "factors" / "daily_basic.parquet",
    )


def _require_files(paths: _MarketDataPaths) -> None:
    """函数说明：检查 require_files 的内部辅助逻辑。"""
    missing = [str(path.relative_to(ROOT)) for path in _path_values(paths) if not path.exists()]
    if missing:
        pytest.fail("Committed real market data snapshot is missing: " + ", ".join(missing))


def _files_exist(paths: _MarketDataPaths) -> bool:
    """函数说明：处理 files_exist 的内部辅助逻辑。"""
    return all(path.exists() for path in _path_values(paths))


def _path_values(paths: _MarketDataPaths) -> tuple[Path, ...]:
    """函数说明：处理 path_values 的内部辅助逻辑。"""
    return (paths.price_panel, paths.close_panel, paths.factor_panel, paths.daily_basic)


def _snapshot_covers(
    start: pd.Timestamp,
    end: pd.Timestamp,
    instruments: tuple[str, ...],
    factor_columns: tuple[str, ...],
    price_fields: tuple[str, ...],
) -> bool:
    """函数说明：处理 snapshot_covers 的内部辅助逻辑。"""
    manifest_path = SNAPSHOT_ROOT / "manifest.json"
    if not manifest_path.exists():
        pytest.fail("Committed real market data snapshot is missing: tests/fixtures/data_snapshot/manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        pytest.fail(f"Committed real market data snapshot manifest is invalid JSON: {exc}")

    snapshot_start = pd.Timestamp(manifest["start"]).normalize()
    snapshot_end = pd.Timestamp(manifest["end"]).normalize()
    snapshot_instruments = {_normalize_instrument(value) for value in manifest.get("instruments", [])}
    snapshot_factors = {str(value) for value in manifest.get("factor_columns", [])}
    snapshot_fields = {str(value) for value in manifest.get("price_fields", [])}

    return (
        start >= snapshot_start
        and end <= snapshot_end
        and set(instruments).issubset(snapshot_instruments)
        and set(factor_columns).issubset(snapshot_factors)
        and set(price_fields).issubset(snapshot_fields)
    )


def _read_price_panel(path: Path, columns: list[str], source_name: str) -> pd.DataFrame:
    """函数说明：读取 read_price_panel 的内部辅助逻辑。"""
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        try:
            prices = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - exercised when the committed snapshot is corrupted.
            pytest.fail(f"Unable to read real OHLCV price panel from {source_name}: {exc}")
        return prices.loc[:, [column for column in prices.columns if str(column) in set(columns)]]


def _require_price_columns(
    prices: pd.DataFrame,
    price_fields: tuple[str, ...],
    instruments: list[str],
    source_name: str,
) -> None:
    """函数说明：检查 require_price_columns 的内部辅助逻辑。"""
    available = set(prices.columns)
    expected = {(field, instrument) for field in price_fields for instrument in instruments}
    if not expected.issubset(available):
        missing = sorted(expected - available)
        pytest.fail(f"Real OHLCV price panel from {source_name} is missing requested columns: {missing}")


def _normalize_instrument(value: object) -> str:
    """函数说明：规范化 normalize_instrument 的内部辅助逻辑。"""
    return str(value).strip().upper()


def _storage_instrument(value: object) -> str:
    """函数说明：处理 storage_instrument 的内部辅助逻辑。"""
    return str(value).strip().lower()


def _normalize_factor_index(factors: pd.DataFrame) -> pd.DataFrame:
    """函数说明：规范化 normalize_factor_index 的内部辅助逻辑。"""
    frame = factors.copy()
    datetime_values = pd.to_datetime(frame.index.get_level_values("datetime"))
    instrument_values = [_normalize_instrument(value) for value in frame.index.get_level_values("instrument")]
    frame.index = pd.MultiIndex.from_arrays([datetime_values, instrument_values], names=["datetime", "instrument"])
    return frame
