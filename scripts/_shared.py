"""模块说明：提供脚本间复用的参数解析和质量门槛工具。"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config_loader import resolve_path
from src.factor_calculator import factor_cache_columns
from src.scoring import DEFAULT_DYNAMIC_IC_CANDIDATES, DYNAMIC_IC_SELECTOR_GROUPS
from src.strategy import factor_columns_for_method


def dated_output_path(prefix: str, suffix: str = ".csv", output_dir: str = "outputs", today: date | None = None) -> str:
    """函数说明：处理 dated_output_path 主要逻辑。"""
    run_date = today or date.today()
    return f"{output_dir}/{prefix}_{run_date:%Y%m%d}{suffix}"


def requested_factor_columns(
    factor_file: str | Path,
    strategy_cfg: dict,
    dynamic_cfg: dict | None = None,
    ml_cfg: dict | None = None,
    score_blend_cfg: dict | None = None,
    score_filter_cfg: dict | None = None,
) -> list[str] | None:
    """函数说明：处理 requested_factor_columns 主要逻辑。"""
    if bool((ml_cfg or {}).get("enabled", False)):
        available_columns = factor_cache_columns(factor_file)
        if not available_columns:
            return None
        configured = (ml_cfg or {}).get("feature_columns")
        if configured:
            requested = [str(column) for column in configured]
            return _with_regime_component_columns(
                [column for column in requested if column in available_columns],
                available_columns,
                score_blend_cfg,
                score_filter_cfg,
            )
        feature_limit = (ml_cfg or {}).get("feature_limit")
        if feature_limit is not None:
            return _with_regime_component_columns(
                available_columns[: max(1, int(feature_limit))],
                available_columns,
                score_blend_cfg,
                score_filter_cfg,
            )
        return None

    group = str(strategy_cfg.get("factor_group", "momentum")).strip().lower()
    if group in {"all", "ic_weighted"}:
        return None
    available_columns = factor_cache_columns(factor_file)
    if not available_columns:
        return None
    if group in DYNAMIC_IC_SELECTOR_GROUPS:
        candidates = (dynamic_cfg or {}).get("candidates", DEFAULT_DYNAMIC_IC_CANDIDATES)
        requested: set[str] = set()
        for candidate in candidates:
            method = strip_direction_prefix(str(candidate))
            requested.update(str(column) for column in factor_columns_for_method(available_columns, method))
        return _with_regime_component_columns(sorted(requested), available_columns, score_blend_cfg, score_filter_cfg) if requested else None
    requested = [str(column) for column in factor_columns_for_method(available_columns, group)]
    return _with_regime_component_columns(sorted(requested), available_columns, score_blend_cfg, score_filter_cfg) if requested else None


def strip_direction_prefix(value: str) -> str:
    """函数说明：去除 strip_direction_prefix 主要逻辑。"""
    lowered = value.strip().lower()
    for prefix in ("low_", "inverse_", "short_"):
        if lowered.startswith(prefix):
            return value.strip()[len(prefix) :]
    return value


def yearly_stats(equity_curve: pd.Series, config: dict) -> pd.DataFrame:
    """函数说明：处理 yearly_stats 主要逻辑。"""
    if equity_curve.empty:
        return pd.DataFrame(columns=["year", "start", "end", "days", "total_return", "annual_return", "max_drawdown"])
    annual_days = int(config.get("annual_trading_days", 252))
    equity = equity_curve.sort_index().astype(float)
    rows: list[dict[str, object]] = []
    for year, segment in equity.groupby(equity.index.year):
        segment = segment.dropna()
        if segment.empty:
            continue
        total_return = float(segment.iloc[-1] / segment.iloc[0] - 1) if segment.iloc[0] else 0.0
        periods = max(len(segment) - 1, 1)
        annual_return = float((1 + total_return) ** (annual_days / periods) - 1) if total_return > -1 else -1.0
        drawdown = segment / segment.cummax() - 1
        rows.append(
            {
                "year": int(year),
                "start": segment.index.min().date().isoformat(),
                "end": segment.index.max().date().isoformat(),
                "days": int(len(segment)),
                "total_return": total_return,
                "annual_return": annual_return,
                "max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def yearly_quality_gate(yearly: pd.DataFrame, config: dict) -> dict[str, object]:
    """函数说明：处理 yearly_quality_gate 主要逻辑。"""
    ml_cfg = config.get("ml_strategy", {})
    min_return = float(ml_cfg.get("min_yearly_annual_return", ml_cfg.get("target_annual_return", 0.20)))
    drawdown_limit = float(ml_cfg.get("max_drawdown_limit", -0.20))
    if yearly.empty:
        return {
            "min_yearly_annual_return": min_return,
            "max_drawdown_limit": drawdown_limit,
            "years_below_return_target": [],
            "years_breaching_drawdown_limit": [],
        }
    years = pd.to_numeric(yearly["year"], errors="coerce").astype("Int64")
    annual = pd.to_numeric(yearly["annual_return"], errors="coerce")
    drawdown = pd.to_numeric(yearly["max_drawdown"], errors="coerce")
    return {
        "min_yearly_annual_return": min_return,
        "max_drawdown_limit": drawdown_limit,
        "years_below_return_target": [int(year) for year in years[annual < min_return].dropna().to_list()],
        "years_breaching_drawdown_limit": [int(year) for year in years[drawdown < drawdown_limit].dropna().to_list()],
    }


def read_selected_params(path_value: str | Path) -> dict[str, object]:
    """函数说明：读取 read_selected_params 主要逻辑。"""
    path = resolve_path(path_value)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def probe_symbols(source: str, max_symbols: int) -> list[str]:
    """函数说明：处理 probe_symbols 主要逻辑。"""
    if source == "all":
        return []
    paths = [resolve_path("outputs/auto_backtest_trades.csv"), resolve_path("outputs/auto_backtest_holdings.csv")]
    symbols: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda column: column == "instrument")
        if "instrument" in frame.columns:
            symbols.update(_normalize_symbol(value) for value in frame["instrument"].dropna().astype(str))
    clean = sorted(symbol for symbol in symbols if symbol)
    if max_symbols > 0:
        clean = clean[:max_symbols]
    return clean


def probe_factor_columns(factor_file: str | Path, config: dict) -> list[str] | None:
    """函数说明：处理 probe_factor_columns 主要逻辑。"""
    requested = requested_factor_columns(
        factor_file,
        config.get("strategy", {}),
        config.get("dynamic_ic_selector", {}),
        config.get("ml_strategy", {}),
        config.get("regime_score_blend", {}),
        config.get("regime_score_filter", {}),
    )
    columns = set(requested or [])
    if bool(config.get("regime_score_blend", {}).get("enabled", False)):
        available = set(factor_cache_columns(factor_file))
        for item in config.get("regime_score_blend", {}).get("defensive_components", []):
            column = str(item.get("column", ""))
            if column in available:
                columns.add(column)
    return sorted(columns) if columns else requested


def read_factor_subset(
    factor_file: str | Path,
    factor_columns: list[str] | None,
    start_date: str,
    end_date: str,
    symbols: list[str],
) -> pd.DataFrame:
    """函数说明：读取 read_factor_subset 主要逻辑。"""
    path = resolve_path(factor_file)
    columns = [*(factor_columns or []), "datetime", "instrument"] if factor_columns else None
    factors = pd.read_parquet(path, columns=columns)
    if not isinstance(factors.index, pd.MultiIndex):
        factors = factors.set_index(["datetime", "instrument"])
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    mask = (dates >= pd.Timestamp(start_date).normalize()) & (dates <= pd.Timestamp(end_date).normalize())
    if symbols:
        wanted = set(_normalize_symbol(symbol) for symbol in symbols)
        instruments = factors.index.get_level_values(1).map(_normalize_symbol)
        mask &= instruments.isin(wanted)
    return factors[mask].sort_index()


def read_price_subset(
    price_file: str | Path,
    fields: Iterable[str],
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """函数说明：读取 read_price_subset 主要逻辑。"""
    path = resolve_path(price_file)
    columns = None
    if symbols:
        columns = _price_column_names(path, fields, symbols)
    prices = pd.read_parquet(path, columns=columns)
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices[(prices.index >= pd.Timestamp(start_date).normalize()) & (prices.index <= pd.Timestamp(end_date).normalize())]
    return prices.sort_index()


def _with_regime_component_columns(
    columns: list[str],
    available_columns: list[str],
    score_blend_cfg: dict | None = None,
    score_filter_cfg: dict | None = None,
) -> list[str]:
    """函数说明：处理 with_regime_component_columns 的内部辅助逻辑。"""
    if not bool((score_blend_cfg or {}).get("enabled", False)) and not bool((score_filter_cfg or {}).get("enabled", False)):
        return columns
    available = {str(column) for column in available_columns}
    requested = {str(column) for column in columns}
    for item in (score_blend_cfg or {}).get("defensive_components", []):
        column = str(item.get("column", ""))
        if column in available:
            requested.add(column)
    for item in _score_filter_components(score_filter_cfg):
        column = str(item.get("column", ""))
        if column in available:
            requested.add(column)
    return sorted(requested)


def _score_filter_components(score_filter_cfg: dict | None) -> list[dict]:
    """函数说明：处理 score_filter_components 的内部辅助逻辑。"""
    cfg = score_filter_cfg or {}
    components: list[dict] = []
    components.extend(cfg.get("components") or cfg.get("defensive_components") or [])
    for rule in cfg.get("rules", []):
        components.extend(rule.get("components") or [])
    return components


def _price_column_names(path: Path, fields: Iterable[str], symbols: list[str]) -> list[str]:
    """函数说明：处理 price_column_names 的内部辅助逻辑。"""
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    result: list[str] = []
    for field in fields:
        for symbol in symbols:
            name = str((str(field).lower(), _normalize_symbol(symbol)))
            if name in available:
                result.append(name)
    if not result:
        raise ValueError("No matching price columns found for the requested fields/symbols.")
    return result


def _normalize_symbol(value: object) -> str:
    """函数说明：规范化 normalize_symbol 的内部辅助逻辑。"""
    return str(value).strip().lower()
