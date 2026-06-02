from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import load_config, resolve_path


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
    qlib_cfg = config.get("qlib", {})
    provider = resolve_path(provider_uri or qlib_cfg["provider_uri"])
    region = qlib_cfg.get("region", "cn")
    instruments = instruments or qlib_cfg.get("instruments", "csi300")

    qlib.init(provider_uri=str(provider), region=region)
    handler = Alpha158(
        instruments=instruments,
        start_time=start_date,
        end_time=end_date,
        fit_start_time=start_date,
        fit_end_time=end_date,
    )
    dataset = DatasetH(handler, segments={"full": (start_date, end_date)})
    factors = dataset.prepare("full", col_set="feature")
    if not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("Expected Alpha158 result to use a MultiIndex of datetime/instrument.")
    return factors.sort_index()


def load_or_compute_factors(
    start_date: str,
    end_date: str,
    cache_file: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    config = load_config()
    path = resolve_path(cache_file or config["factors"]["cache_file"])
    if path.exists() and not force:
        cached = pd.read_parquet(path)
        if _factor_cache_matches_request(cached, start_date, end_date, config):
            return cached

    path.parent.mkdir(parents=True, exist_ok=True)
    factors = compute_alpha158_factors(start_date, end_date)
    factors.to_parquet(path)
    return factors


def _factor_cache_matches_request(factors: pd.DataFrame, start_date: str, end_date: str, config: dict) -> bool:
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
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
    price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv.parquet"))
    if not price_path.exists() and price_path.name == "ohlcv.parquet":
        fallback = price_path.with_name("close.parquet")
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
