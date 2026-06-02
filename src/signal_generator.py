from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.factor_ic import calculate_rolling_ic, make_rolling_ic_weights
from src.strategy import composite_factor, select_stocks


def read_previous_holdings(path: str | Path | None = None) -> list[str]:
    config = load_config()
    holdings_path = resolve_path(path or config["outputs"]["holdings_file"])
    if not holdings_path.exists():
        return []
    df = pd.read_csv(holdings_path)
    col = "instrument" if "instrument" in df.columns else "ticker"
    if col not in df.columns:
        return []
    return df[col].dropna().astype(str).tolist()


def generate_signal(
    signal_date: str,
    previous_holdings: list[str] | None = None,
    factor_file: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    config = load_config()
    data_cfg = config["data"]
    strategy_cfg = config["strategy"]
    use_latest_date = str(signal_date).lower() == "latest"
    factor_end_date = data_cfg["end_date"] if use_latest_date else signal_date

    factors = load_or_compute_factors(
        start_date=data_cfg["start_date"],
        end_date=factor_end_date,
        cache_file=factor_file or config["factors"]["cache_file"],
    )
    factor_group = strategy_cfg.get("factor_group", "momentum")
    dynamic_weights = None
    if factor_group == "ic_weighted":
        ic_cfg = config.get("ic", {})
        price_file = resolve_path(ic_cfg.get("price_file", "data/prices/ohlcv.parquet"))
        if not price_file.exists() and price_file.name == "ohlcv.parquet":
            fallback_price_file = price_file.with_name("close.parquet")
            if fallback_price_file.exists():
                price_file = fallback_price_file
        if not price_file.exists():
            raise FileNotFoundError(f"Price file not found for rolling IC weights: {price_file}")
        prices = pd.read_parquet(price_file)
        rolling_ic = calculate_rolling_ic(
            factors,
            prices,
            window=int(ic_cfg.get("window", 252)),
            min_periods=int(ic_cfg.get("min_periods", 60)),
        )
        dynamic_weights = make_rolling_ic_weights(
            rolling_ic,
            top_k=int(ic_cfg.get("top_k", 30)),
            min_abs_ic=float(ic_cfg.get("min_abs_ic", 0.02)),
            min_periods=int(ic_cfg.get("min_periods", 60)),
            correlation_threshold=float(ic_cfg.get("corr_threshold", 0.7)),
        )
    scores = composite_factor(factors, method=factor_group, factor_weights_dynamic=dynamic_weights)
    latest_date = pd.Timestamp(scores.index.get_level_values(0).max()).normalize()
    if use_latest_date:
        signal_date = latest_date.strftime("%Y-%m-%d")
    else:
        requested_date = pd.Timestamp(signal_date).normalize()
        if latest_date != requested_date:
            raise ValueError(f"Factor cache latest date {latest_date.date()} does not match signal_date {requested_date.date()}.")
    latest_scores = scores.xs(latest_date, level=0, drop_level=True)
    previous_holdings = previous_holdings if previous_holdings is not None else read_previous_holdings()
    holdings = select_stocks(
        latest_scores,
        top_n=int(strategy_cfg.get("top_n", 7)),
        previous_holdings=previous_holdings or None,
        max_turnover=int(strategy_cfg.get("max_turnover", 1)),
        rank_buffer=int(strategy_cfg.get("rank_buffer", 0)),
    )

    old_set = set(previous_holdings or [])
    new_set = set(holdings)
    rows = []
    for code in holdings:
        rows.append({"date": signal_date, "instrument": code, "action": "HOLD" if code in old_set else "BUY"})
    for code in sorted(old_set - new_set):
        rows.append({"date": signal_date, "instrument": code, "action": "SELL"})
    return pd.DataFrame(rows), holdings


def save_signal(signal_df: pd.DataFrame, holdings: list[str], signal_date: str) -> tuple[Path, Path]:
    config = load_config()
    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"signal_{signal_date}.csv"
    holdings_path = resolve_path(config["outputs"]["holdings_file"])
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"instrument": holdings}).to_csv(holdings_path, index=False, encoding="utf-8-sig")
    return signal_path, holdings_path
