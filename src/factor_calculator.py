"""模块说明：计算、缓存和读取策略因子面板。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.common import looks_like_field_table as _looks_like_field_table
from src.config_loader import load_config, resolve_path
from src.trading_calendar import resolve_target_date_value

logger = logging.getLogger(__name__)
_QLIB_INIT_STATE: tuple[str, str] | None = None


def compute_alpha158_factors(
    start_date: str,
    end_date: str,
    instruments: str | list[str] | None = None,
    provider_uri: str | Path | None = None,
) -> pd.DataFrame:
    """函数说明：计算 compute_alpha158_factors 主要逻辑。"""
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
    """函数说明：确保 ensure_qlib_initialized 的内部辅助逻辑。"""
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
    """函数说明：加载 load_or_compute_factors 主要逻辑。"""
    config = load_config()
    end = resolve_target_date_value(end_date, config=config)
    path = resolve_path(cache_file or config["factors"]["cache_file"])
    is_default_cache = _is_default_factor_cache(path, config)
    if path.exists() and not force:
        cached = _read_factor_cache(path, columns=columns)
        if not is_default_cache:
            return _slice_custom_factor_cache(cached, path, start_date, end, config, columns=columns)
        if _factor_cache_matches_request(cached, start_date, end, config, cache_file=path, columns=columns):
            return _slice_factor_cache(cached, start_date, end)

    path.parent.mkdir(parents=True, exist_ok=True)
    factors = compute_alpha158_factors(start_date, end)
    if _should_write_factor_cache(path, start_date, end, config):
        _write_factor_cache(path, factors, start_date, end, config)
    else:
        logger.warning(
            "Computed factors for %s to %s but did not overwrite default cache %s because the request is a partial date range.",
            start_date,
            end,
            path,
        )
    if columns is not None:
        factors = factors[[column for column in columns if column in factors.columns]]
    return factors


def factor_cache_columns(cache_file: str | Path | None = None) -> list[str]:
    """函数说明：处理 factor_cache_columns 主要逻辑。"""
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
    """函数说明：读取 read_factor_cache 的内部辅助逻辑。"""
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
    columns: list[str] | None = None,
) -> bool:
    """函数说明：处理 factor_cache_matches_request 的内部辅助逻辑。"""
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return False
    if columns is not None:
        requested_columns = {str(column) for column in columns}
        cached_columns = {str(column) for column in factors.columns}
        if not requested_columns.issubset(cached_columns):
            return False

    if not _factor_cache_meta_matches(config, start_date, end_date, cache_file=cache_file):
        return False

    return _factor_cache_data_matches_request(factors, start_date, end_date, config)


def _slice_custom_factor_cache(
    factors: pd.DataFrame,
    path: Path,
    start_date: str,
    end_date: str,
    config: dict,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Validate and slice a user-supplied factor cache without Alpha158 metadata."""
    if factors.empty:
        raise ValueError(f"Custom factor cache is empty: {path}")
    if not isinstance(factors.index, pd.MultiIndex):
        raise ValueError(f"Custom factor cache must use a MultiIndex of datetime/instrument: {path}")
    if columns is not None:
        requested_columns = [str(column) for column in columns]
        cached_columns = {str(column) for column in factors.columns}
        missing = [column for column in requested_columns if column not in cached_columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Custom factor cache {path} is missing requested columns: {missing_text}")
    if not _factor_cache_data_matches_request(factors, start_date, end_date, config):
        raise ValueError(f"Custom factor cache {path} does not cover requested dates or symbols from {start_date} to {end_date}.")
    return _slice_factor_cache(factors, start_date, end_date)


def _factor_cache_data_matches_request(
    factors: pd.DataFrame,
    start_date: str,
    end_date: str,
    config: dict,
) -> bool:
    """Validate cache frame shape, date coverage, and symbol coverage against price data."""
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return False
    factor_dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    latest_factor_date = factor_dates.max()
    price_dates, price_symbols = _price_cache_state(config, start_date, end_date)
    if not price_dates.empty:
        first_required_date = pd.Timestamp(price_dates.min()).normalize()
    else:
        first_required_date = requested_start
    if factor_dates.min() > first_required_date:
        return False
    if not price_dates.empty and latest_factor_date < price_dates.max():
        return False

    if price_symbols:
        factor_symbols = _normalize_symbols(factors.index.get_level_values(1))
        if not price_symbols.issubset(factor_symbols):
            return False

    if price_dates.empty:
        if latest_factor_date < requested_end:
            return False
    return True


def _slice_factor_cache(factors: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """函数说明：处理 slice_factor_cache 的内部辅助逻辑。"""
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return factors
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return factors[(dates >= start) & (dates <= end)]


def _price_cache_state(config: dict, start_date: str, end_date: str) -> tuple[pd.DatetimeIndex, set[str]]:
    """函数说明：处理 price_cache_state 的内部辅助逻辑。"""
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
        symbols = _normalize_symbols(prices.columns.get_level_values(-1))
    else:
        if _looks_like_field_table(prices.columns):
            raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
        symbols = _normalize_symbols(prices.columns)
    return dates, symbols


def _factor_cache_meta_path(cache_path: str | Path | None, config: dict) -> Path:
    """函数说明：处理 factor_cache_meta_path 的内部辅助逻辑。"""
    path = resolve_path(cache_path or config["factors"]["cache_file"])
    return path.with_name(f"{path.name}.meta.json")


def _factor_cache_meta_matches(config: dict, start_date: str, end_date: str, cache_file: str | Path | None = None) -> bool:
    """函数说明：处理 factor_cache_meta_matches 的内部辅助逻辑。"""
    meta_path = _factor_cache_meta_path(cache_file or config["factors"]["cache_file"], config)
    if not meta_path.exists():
        return "qlib" not in config
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = _factor_cache_meta_payload(None, start_date, end_date, config)
    keys = []
    if "qlib" in config:
        keys = ["provider_uri", "region", "instruments"]
    for key in keys:
        if meta.get(key) != expected.get(key):
            return False
    try:
        cached_start = pd.Timestamp(meta.get("start_date")).normalize()
        cached_end = pd.Timestamp(meta.get("end_date")).normalize()
        requested_start = pd.Timestamp(expected.get("start_date")).normalize()
        requested_end = pd.Timestamp(expected.get("end_date")).normalize()
    except (TypeError, ValueError):
        return False
    if cached_start > requested_start or cached_end < requested_end:
        return False
    return True


def _write_factor_cache(path: Path, factors: pd.DataFrame, start_date: str, end_date: str, config: dict) -> None:
    """函数说明：写入 write_factor_cache 的内部辅助逻辑。"""
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_meta_path = path.with_name(f"{path.name}.tmp.meta.json")
    meta_path = path.with_name(f"{path.name}.meta.json")
    try:
        factors.to_parquet(tmp_path)
        payload = _factor_cache_meta_payload(factors, start_date, end_date, config)
        tmp_meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
        tmp_meta_path.replace(meta_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        tmp_meta_path.unlink(missing_ok=True)


def _factor_cache_meta_payload(factors: pd.DataFrame | None, start_date: str, end_date: str, config: dict) -> dict[str, object]:
    """函数说明：处理 factor_cache_meta_payload 的内部辅助逻辑。"""
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
        payload["symbols"] = sorted(_normalize_symbols(factors.index.get_level_values(1)))
    return payload


def _normalize_symbols(values: object) -> set[str]:
    """函数说明：规范化 normalize_symbols 的内部辅助逻辑。"""
    symbols = pd.Index(values).dropna().astype(str).str.strip().str.upper()
    return set(symbol for symbol in symbols if symbol)


def _should_write_factor_cache(path: Path, start_date: str, end_date: str, config: dict) -> bool:
    """函数说明：处理 should_write_factor_cache 的内部辅助逻辑。"""
    if not _is_default_factor_cache(path, config):
        return True
    data_cfg = config.get("data", {})
    default_start = data_cfg.get("history_start_date", data_cfg.get("start_date"))
    if default_start is None:
        return True

    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    required_start = _default_data_start_date(config, default_start)
    if requested_start > required_start:
        return False

    default_end = _default_data_end_date(config, default_start)
    if default_end is None:
        return True
    return requested_end >= default_end


def _is_default_factor_cache(path: Path, config: dict) -> bool:
    """函数说明：判断 is_default_factor_cache 是否成立。"""
    default_value = config.get("factors", {}).get("cache_file")
    if default_value is None:
        return False
    try:
        return path.resolve() == resolve_path(default_value).resolve()
    except OSError:
        return str(path) == str(resolve_path(default_value))


def _default_data_end_date(config: dict, default_start: str) -> pd.Timestamp | None:
    """函数说明：处理 default_data_end_date 的内部辅助逻辑。"""
    data_cfg = config.get("data", {})
    configured_end = data_cfg.get("end_date")
    if configured_end not in {None, "", "auto"}:
        return pd.Timestamp(resolve_target_date_value(str(configured_end), config=config)).normalize()

    auto_target = _auto_target_end_date(config)
    price_dates, _symbols = _price_cache_state(config, default_start, "2100-01-01")
    if price_dates.empty:
        return auto_target
    price_end = pd.Timestamp(price_dates.max()).normalize()
    if auto_target is None:
        return price_end
    return min(price_end, auto_target)


def _default_data_start_date(config: dict, default_start: str) -> pd.Timestamp:
    """函数说明：返回默认因子缓存需要覆盖的实际可用起点。"""
    configured_start = pd.Timestamp(default_start).normalize()
    price_dates, _symbols = _price_cache_state(config, default_start, "2100-01-01")
    if price_dates.empty:
        return configured_start
    return max(configured_start, pd.Timestamp(price_dates.min()).normalize())


def _auto_target_end_date(config: dict) -> pd.Timestamp | None:
    """函数说明：解析自动目标日，失败时返回空值。"""
    try:
        return pd.Timestamp(resolve_target_date_value("auto", config=config)).normalize()
    except Exception as exc:
        logger.debug("Unable to resolve auto factor cache end date; falling back to price cache max date: %s", exc)
        return None
