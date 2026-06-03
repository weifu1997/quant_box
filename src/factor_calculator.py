from __future__ import annotations

import json
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.trading_calendar import resolve_target_date_value

logger = logging.getLogger(__name__)
_QLIB_INIT_STATE: tuple[str, str] | None = None
warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)


def compute_alpha158_factors(
    start_date: str,
    end_date: str,
    instruments: str | list[str] | None = None,
    provider_uri: str | Path | None = None,
) -> pd.DataFrame:
    try:
        import qlib
        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH
    except ImportError as exc:
        raise RuntimeError("pyqlib is required to compute Alpha158 factors. Install requirements first.") from exc

    config = load_config()
    end = resolve_target_date_value(end_date, config=config)
    qlib_cfg = config.get("qlib", {})
    provider = resolve_path(provider_uri or qlib_cfg["provider_uri"])
    region = qlib_cfg.get("region", "cn")
    instruments = instruments or qlib_cfg.get("instruments", "csi300")

    _ensure_qlib_initialized(qlib, provider, region)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        # Alpha158 may emit noisy log(0) RuntimeWarnings while producing NaN feature values.
        warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
        handler = Alpha158(
            instruments=instruments,
            start_time=start_date,
            end_time=end,
            fit_start_time=start_date,
            fit_end_time=end,
        )
        dataset = DatasetH(handler, segments={"full": (start_date, end)})
        factors = dataset.prepare("full", col_set="feature")
    if not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("Expected Alpha158 result to use a MultiIndex of datetime/instrument.")
    return factors.replace([np.inf, -np.inf], np.nan).sort_index()


def _ensure_qlib_initialized(qlib_module, provider: Path, region: str) -> None:
    global _QLIB_INIT_STATE
    state = (str(provider), str(region))
    if _QLIB_INIT_STATE == state:
        return
    if _QLIB_INIT_STATE is not None and _QLIB_INIT_STATE != state:
        logger.warning(
            "Reinitializing qlib from provider=%s region=%s to provider=%s region=%s.",
            _QLIB_INIT_STATE[0],
            _QLIB_INIT_STATE[1],
            state[0],
            state[1],
        )
    try:
        qlib_module.init(provider_uri=state[0], region=state[1])
    except Exception as exc:
        if "already" not in str(exc).lower() and "initialized" not in str(exc).lower():
            raise
        logger.warning("qlib appears to be already initialized; reusing existing global state: %s", exc)
    _QLIB_INIT_STATE = state


def load_or_compute_factors(
    start_date: str,
    end_date: str,
    cache_file: str | Path | None = None,
    force: bool = False,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    config = load_config()
    end = resolve_target_date_value(end_date, config=config)
    path = resolve_path(cache_file or config["factors"]["cache_file"])
    if path.exists() and not force:
        cached = _read_factor_cache(path, columns=columns)
        if _factor_cache_matches_request(cached, start_date, end, config, cache_file=path):
            return cached

    path.parent.mkdir(parents=True, exist_ok=True)
    factors = compute_alpha158_factors(start_date, end)
    factors.to_parquet(path)
    _write_factor_cache_meta(path, factors, start_date, end, config)
    if columns is not None:
        factors = factors[[column for column in columns if column in factors.columns]]
    return factors


def factor_cache_columns(cache_file: str | Path | None = None) -> list[str]:
    config = load_config()
    path = resolve_path(cache_file or config["factors"]["cache_file"])
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq

        names = pq.ParquetFile(path).schema.names
    except Exception:
        try:
            names = list(pd.read_parquet(path).columns)
        except Exception:
            return []
    return [str(name) for name in names if str(name) not in {"datetime", "instrument"}]


def _read_factor_cache(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    requested = [str(column) for column in columns]
    parquet_columns = [*requested, "datetime", "instrument"]
    try:
        return pd.read_parquet(path, columns=parquet_columns)
    except (KeyError, ValueError):
        cached = pd.read_parquet(path)
        return cached[[column for column in requested if column in cached.columns]]


def _factor_cache_matches_request(
    factors: pd.DataFrame,
    start_date: str,
    end_date: str,
    config: dict,
    cache_file: str | Path | None = None,
) -> bool:
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return False

    if not _factor_cache_meta_matches(config, start_date, end_date, cache_file=cache_file):
        return False

    factor_dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    latest_factor_date = factor_dates.max()
    price_dates, price_symbols = _price_cache_state(config, start_date, end_date)
    if not price_dates.empty and latest_factor_date < price_dates.max():
        return False

    if price_symbols:
        factor_symbols = set(factors.index.get_level_values(1).astype(str).str.upper())
        if not price_symbols.issubset(factor_symbols):
            return False

    if price_dates.empty:
        requested_end = pd.Timestamp(end_date).normalize()
        if latest_factor_date < requested_end:
            return False
    return True


def _price_cache_state(config: dict, start_date: str, end_date: str) -> tuple[pd.DatetimeIndex, set[str]]:
    price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    if not price_path.exists() and price_path.name in {"ohlcv.parquet", "ohlcv_adjusted.parquet"}:
        fallback_name = "close_adjusted.parquet" if price_path.name == "ohlcv_adjusted.parquet" else "close.parquet"
        fallback = price_path.with_name(fallback_name)
        if fallback.exists():
            price_path = fallback
    if not price_path.exists():
        return pd.DatetimeIndex([]), set()

    prices = pd.read_parquet(price_path)
    dates = pd.to_datetime(prices.index).normalize()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = pd.DatetimeIndex(dates[(dates >= start) & (dates <= end)]).unique()
    if isinstance(prices.columns, pd.MultiIndex):
        symbols = set(prices.columns.get_level_values(-1).astype(str).str.upper())
    else:
        symbols = set(prices.columns.astype(str).str.upper())
    return dates, symbols


def _factor_cache_meta_path(cache_path: str | Path | None, config: dict) -> Path:
    path = resolve_path(cache_path or config["factors"]["cache_file"])
    return path.with_name(f"{path.name}.meta.json")


def _factor_cache_meta_matches(config: dict, start_date: str, end_date: str, cache_file: str | Path | None = None) -> bool:
    meta_path = _factor_cache_meta_path(cache_file or config["factors"]["cache_file"], config)
    if not meta_path.exists():
        return "qlib" not in config
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if "qlib" not in config:
        return False

    expected = _factor_cache_meta_payload(None, start_date, end_date, config)
    for key in ["provider_uri", "region", "instruments", "start_date", "end_date"]:
        if meta.get(key) != expected.get(key):
            return False
    return True


def _write_factor_cache_meta(path: Path, factors: pd.DataFrame, start_date: str, end_date: str, config: dict) -> None:
    payload = _factor_cache_meta_payload(factors, start_date, end_date, config)
    path.with_name(f"{path.name}.meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _factor_cache_meta_payload(factors: pd.DataFrame | None, start_date: str, end_date: str, config: dict) -> dict[str, object]:
    qlib_cfg = config.get("qlib", {})
    payload: dict[str, object] = {
        "provider_uri": str(resolve_path(qlib_cfg.get("provider_uri", "data/qlib_data"))),
        "region": str(qlib_cfg.get("region", "cn")),
        "instruments": qlib_cfg.get("instruments", "csi300"),
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
    }
    if factors is not None and isinstance(factors.index, pd.MultiIndex):
        payload["columns"] = list(map(str, factors.columns))
        payload["rows"] = int(len(factors))
        payload["symbols"] = sorted(set(factors.index.get_level_values(1).astype(str).str.upper()))
    return payload
