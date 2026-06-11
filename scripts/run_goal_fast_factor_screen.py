"""模块说明：提供 run_goal_fast_factor_screen 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import weakref

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import normalize_instrument
from src.config_loader import load_config, resolve_path
from src.factor_calculator import factor_cache_columns
from src.fast_monthly_backtest import prepare_fast_period_data, run_fast_prepared_backtest
from src.scoring import _apply_liquidity_filter, build_strategy_scores
from src.selection_risk import _base_limit_down_threshold_for_stock, _normalize_price_frame, _price_field
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value
from scripts._shared import dated_output_path, yearly_stats

_SELECTION_RISK_ELIGIBILITY_CACHE: dict[tuple[int, str], tuple[weakref.ReferenceType[pd.DataFrame], pd.DataFrame]] = {}


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    parser = argparse.ArgumentParser(description="Fast screen exact single-factor candidates for the active goal.")
    parser.add_argument("--output", default=dated_output_path("goal_fast_factor_screen"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--top-n", default="7,10,15,20")
    parser.add_argument("--liquidity-modes", default="none,high:0.80,high:0.65")
    parser.add_argument("--limit-columns", type=int, default=0)
    parser.add_argument("--factor-file", default="", help="Factor parquet file to screen.")
    parser.add_argument("--columns", default="", help="Comma-separated factor columns to screen; defaults to all cache columns.")
    parser.add_argument("--directions", default="long_high,long_low", help="Comma-separated directions: long_high,long_low.")
    args = parser.parse_args()

    config = load_config()
    start_date = config["data"]["start_date"]
    end_date = resolve_target_date_value(config["data"]["end_date"], config=config)
    factor_file = args.factor_file or config["factors"]["cache_file"]
    available_columns = factor_cache_columns(factor_file)
    columns = _requested_screen_columns(args.columns, available_columns)
    if args.limit_columns:
        columns = columns[: args.limit_columns]
    component_columns = _score_component_columns(config, available_columns)
    top_ns = [int(value.strip()) for value in args.top_n.split(",") if value.strip()]
    liquidity_modes = [_parse_liquidity_mode(value) for value in args.liquidity_modes.split(",") if value.strip()]
    directions = _selected_directions(args.directions)
    trade_price_field = str(config.get("backtest", {}).get("trade_price_field", "close")).lower()
    price_fields = {"close", trade_price_field}
    if any(bool(mode.get("enabled", False)) for mode in liquidity_modes):
        price_fields.add(str(config.get("liquidity_filter", {}).get("field", "amount")).lower())
    price_fields.update(_selection_risk_price_fields(config))
    prices = _read_price_fields(
        config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"),
        sorted(price_fields),
        start_date,
        end_date,
    )

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    for batch_start in range(0, len(columns), args.batch_size):
        batch = columns[batch_start : batch_start + args.batch_size]
        read_columns = list(dict.fromkeys([*batch, *component_columns]))
        factors = _read_factor_subset(factor_file, read_columns, start_date, end_date)
        for column in batch:
            score_columns = [col for col in [column, *component_columns] if col in factors.columns]
            for direction_name, factor_group in [
                ("long_high", f"factor:{column}"),
                ("long_low", f"inverse_factor:{column}"),
            ]:
                if direction_name not in directions:
                    continue
                try:
                    directional_scores = build_strategy_scores(
                        factors[score_columns],
                        _screen_config(config, factor_group, {"enabled": False}),
                        price_df=prices,
                    )
                except Exception as exc:
                    for liquidity_mode in liquidity_modes:
                        rows.append(_error_row(column, direction_name, liquidity_mode, str(exc)))
                    continue
                for liquidity_mode in liquidity_modes:
                    try:
                        scores = _filter_scores(directional_scores, prices, config, liquidity_mode)
                    except Exception as exc:
                        rows.append(_error_row(column, direction_name, liquidity_mode, str(exc)))
                        continue
                    scores = resample_signals(scores, "monthly")
                    scores = _filter_scores_by_selection_risk(scores, prices, config)
                    prepared = prepare_fast_period_data(scores, prices, start_date, end_date, trade_price_field=trade_price_field)
                    for top_n in top_ns:
                        bt_config = {
                            **config["backtest"],
                            "top_n": top_n,
                            "max_turnover": 1,
                            "rank_buffer": 20,
                            "rebalance_freq": "monthly",
                            "rebalance_drift_threshold": 0.02,
                        }
                        result = run_fast_prepared_backtest(prepared, bt_config)
                        yearly = _fast_yearly_stats(result.equity_curve)
                        row = {
                            "factor_group": factor_group,
                            "factor": column,
                            "direction": direction_name,
                            **_liquidity_row(liquidity_mode),
                            "top_n": top_n,
                            **result.metrics,
                        }
                        row.update(_screen_quality_fields(row, config, yearly=yearly))
                        rows.append(row)
        frame = _sorted(rows)
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(
            f"screened {min(batch_start + args.batch_size, len(columns))}/{len(columns)} columns in {time.monotonic() - started:.1f}s; "
            f"best annual={frame.iloc[0].get('annual_return', 0.0):.4f} dd={frame.iloc[0].get('max_drawdown', 0.0):.4f}",
            flush=True,
        )
    print(f"Saved screen to {out_path}")


def _requested_screen_columns(value: str, available_columns: list[str]) -> list[str]:
    if not str(value or "").strip():
        return available_columns
    requested = [item.strip() for item in str(value).split(",") if item.strip()]
    available_by_lower = {str(column).lower(): str(column) for column in available_columns}
    selected: list[str] = []
    missing: list[str] = []
    for column in requested:
        match = available_by_lower.get(column.lower())
        if match is None:
            missing.append(column)
        elif match not in selected:
            selected.append(match)
    if missing:
        raise ValueError(f"Requested screen columns are missing from factor cache: {', '.join(missing)}")
    return selected


def _screen_config(config: dict, factor_group: str, liquidity_mode: dict[str, object]) -> dict:
    """函数说明：处理 screen_config 的内部辅助逻辑。"""
    result = dict(config)
    result["strategy"] = {**config.get("strategy", {}), "factor_group": factor_group, "rebalance_freq": "monthly"}
    result["liquidity_filter"] = {**config.get("liquidity_filter", {}), **liquidity_mode}
    return result


def _filter_scores(
    scores: pd.Series,
    prices: pd.DataFrame,
    config: dict,
    liquidity_mode: dict[str, object],
) -> pd.Series:
    """函数说明：过滤 filter_scores 的内部辅助逻辑。"""
    filter_config = {**config.get("liquidity_filter", {}), **liquidity_mode}
    if not bool(filter_config.get("enabled", False)):
        return scores
    return _apply_liquidity_filter(scores, prices, filter_config)


def _filter_scores_by_selection_risk(scores: pd.Series, prices: pd.DataFrame, config: dict) -> pd.Series:
    """鍑芥暟璇存槑锛氳繃婊?filter_scores_by_selection_risk 鐨勫唴閮ㄨ緟鍔╅€昏緫銆?"""
    risk_cfg = config.get("selection_risk_filter", {})
    if not isinstance(risk_cfg, dict) or not bool(risk_cfg.get("enabled", False)) or scores.empty:
        return scores
    if not isinstance(scores.index, pd.MultiIndex):
        return scores
    eligibility = _selection_risk_eligibility(prices, config)
    if eligibility.empty:
        return scores.where(False)
    score_dates = pd.DatetimeIndex(pd.to_datetime(scores.index.get_level_values(0), errors="coerce")).normalize()
    signal_dates = pd.DatetimeIndex(score_dates.dropna().unique()).sort_values()
    eligible_by_signal = eligibility.reindex(signal_dates, method="ffill").fillna(False)
    pieces = []
    for date, daily in scores.groupby(level=0, sort=True):
        signal_date = pd.Timestamp(date).normalize()
        daily_scores = daily.droplevel(0)
        if signal_date not in eligible_by_signal.index:
            filtered = daily_scores.where(False)
        else:
            eligible_row = eligible_by_signal.loc[signal_date]
            normalized_instruments = [normalize_instrument(value) for value in daily_scores.index]
            mask = pd.Series(
                [bool(eligible_row.get(instrument, False)) for instrument in normalized_instruments],
                index=daily_scores.index,
            )
            filtered = daily_scores.where(mask)
        filtered.index = pd.MultiIndex.from_arrays([[signal_date] * len(filtered), filtered.index.astype(str)], names=scores.index.names)
        pieces.append(filtered)
    return pd.concat(pieces).sort_index().rename(scores.name) if pieces else scores


def _selection_risk_eligibility(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    cache_key = (id(prices), repr((config.get("selection_risk_filter", {}), config.get("backtest", {}))))
    cached = _SELECTION_RISK_ELIGIBILITY_CACHE.get(cache_key)
    if cached is not None and cached[0]() is prices:
        return cached[1]
    eligibility = _build_selection_risk_eligibility(prices, config)
    _SELECTION_RISK_ELIGIBILITY_CACHE[cache_key] = (weakref.ref(prices), eligibility)
    return eligibility


def _build_selection_risk_eligibility(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    risk_cfg = config.get("selection_risk_filter", {})
    if not isinstance(risk_cfg, dict) or not bool(risk_cfg.get("enabled", False)):
        return pd.DataFrame()
    normalized_prices = _normalize_price_frame(prices)
    close = _price_field(normalized_prices, "close")
    if close.empty:
        raise ValueError("selection_risk_filter requires a close field in the price panel.")
    dates = pd.DatetimeIndex(normalized_prices.index).unique().sort_values()
    instruments = pd.Index(close.columns.astype(str)).drop_duplicates()
    instruments = instruments[instruments != ""]
    if dates.empty or instruments.empty:
        return pd.DataFrame(index=dates, columns=instruments, dtype=bool)

    required_fields = [str(value).strip().lower() for value in risk_cfg.get("required_price_fields", ["open", "close"])]
    required_fields = [field for field in required_fields if field]
    require_positive_volume = bool(risk_cfg.get("require_positive_volume", True))
    max_missing = max(0, int(risk_cfg.get("max_missing_price_sessions", 0)))
    lookback = max(1, int(risk_cfg.get("lookback_sessions", 5)))

    missing_sessions = pd.DataFrame(False, index=dates, columns=instruments)
    for field in required_fields:
        missing_sessions |= _missing_selection_risk_field(normalized_prices, field, dates, instruments)
    if require_positive_volume:
        missing_sessions |= _missing_selection_risk_field(normalized_prices, "volume", dates, instruments)
    missing_count = missing_sessions.rolling(lookback, min_periods=1).sum()
    eligible = missing_count <= max_missing

    max_limit_down_days = risk_cfg.get("max_limit_down_days", 0)
    if max_limit_down_days is not None:
        limit_down = _selection_risk_limit_down_days(normalized_prices, dates, instruments, config, risk_cfg)
        limit_down_count = limit_down.rolling(lookback, min_periods=1).sum()
        eligible &= limit_down_count <= max(0, int(max_limit_down_days))
    return eligible.fillna(False).astype(bool)


def _missing_selection_risk_field(
    prices: pd.DataFrame,
    field: str,
    dates: pd.DatetimeIndex,
    instruments: pd.Index,
) -> pd.DataFrame:
    frame = _price_field(prices, field)
    if frame.empty:
        return pd.DataFrame(True, index=dates, columns=instruments)
    values = frame.reindex(dates).reindex(columns=instruments).astype("float64")
    return values.isna() | (values <= 0)


def _selection_risk_limit_down_days(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    instruments: pd.Index,
    config: dict,
    risk_cfg: dict,
) -> pd.DataFrame:
    close = _price_field(prices, "close").reindex(dates).reindex(columns=instruments).astype("float64")
    low = _price_field(prices, "low")
    probe = low.reindex(dates).reindex(columns=instruments).astype("float64") if not low.empty else close
    previous_close = close.shift(1)
    thresholds = pd.Series(
        [_base_limit_down_threshold_for_stock(str(stock), config) for stock in instruments],
        index=instruments,
        dtype="float64",
    )
    threshold_frame = pd.DataFrame(np.tile(thresholds.to_numpy(), (len(dates), 1)), index=dates, columns=instruments)
    is_st = _price_field(prices, "is_st")
    if not is_st.empty:
        st_flags = is_st.reindex(dates).reindex(columns=instruments).fillna(False).astype(bool)
        threshold_frame = threshold_frame.where(~st_flags, float(config.get("backtest", {}).get("st_limit_down_threshold", 0.049)))
    buffer = max(float(risk_cfg.get("limit_down_buffer", 0.0)), 0.0)
    threshold_frame = threshold_frame.sub(buffer).clip(lower=0.0)
    limit_down = probe <= previous_close * (1 - threshold_frame)
    limit_down &= previous_close.notna() & probe.notna() & (previous_close > 0)
    return limit_down.fillna(False)


def _selection_risk_price_fields(config: dict) -> set[str]:
    """鍑芥暟璇存槑锛氬鐞?selection_risk_price_fields 鐨勫唴閮ㄨ緟鍔╅€昏緫銆?"""
    risk_cfg = config.get("selection_risk_filter", {})
    if not isinstance(risk_cfg, dict) or not bool(risk_cfg.get("enabled", False)):
        return set()
    fields = {str(field).strip().lower() for field in risk_cfg.get("required_price_fields", ["open", "close"]) if str(field).strip()}
    fields.add("close")
    if bool(risk_cfg.get("require_positive_volume", True)):
        fields.add("volume")
    if risk_cfg.get("max_limit_down_days", 0) is not None:
        fields.add("low")
    return fields


def _parse_liquidity_mode(value: str) -> dict[str, object]:
    """函数说明：解析 parse_liquidity_mode 的内部辅助逻辑。"""
    mode = value.strip().lower()
    if mode in {"none", "off", "false"}:
        return {"enabled": False}
    side, _, quantile = mode.partition(":")
    return {"enabled": True, "side": side, "quantile": float(quantile or 0.20)}


def _selected_columns(available_columns: list[str], raw_value: str) -> list[str]:
    """鍑芥暟璇存槑锛氬鐞?selected_columns 鐨勫唴閮ㄨ緟鍔╅€昏緫銆?"""
    available_by_lower = {str(column).lower(): str(column) for column in available_columns}
    selected: list[str] = []
    missing: list[str] = []
    for value in raw_value.split(","):
        name = value.strip()
        if not name:
            continue
        column = available_by_lower.get(name.lower())
        if column is None:
            missing.append(name)
            continue
        if column not in selected:
            selected.append(column)
    if missing:
        raise ValueError(f"Selected factor columns not found in cache: {', '.join(missing)}")
    if not selected:
        raise ValueError("--columns did not contain any usable factor column names.")
    return selected


def _selected_directions(raw_value: str) -> set[str]:
    selected = {value.strip() for value in raw_value.split(",") if value.strip()}
    allowed = {"long_high", "long_low"}
    invalid = sorted(selected - allowed)
    if invalid:
        raise ValueError(f"Unsupported directions: {', '.join(invalid)}")
    return selected or allowed


def _liquidity_row(liquidity_mode: dict[str, object]) -> dict[str, object]:
    """函数说明：处理 liquidity_row 的内部辅助逻辑。"""
    side = str(liquidity_mode.get("side", ""))
    kept_side = {"low": "higher_liquidity", "high": "lower_liquidity"}.get(side, "")
    return {
        "liquidity_enabled": bool(liquidity_mode.get("enabled", False)),
        "liquidity_side": side,
        "liquidity_rejected_side": side,
        "liquidity_kept_side": kept_side,
        "liquidity_quantile": liquidity_mode.get("quantile", ""),
    }


def _error_row(column: str, direction_name: str, liquidity_mode: dict[str, object], message: str) -> dict[str, object]:
    """函数说明：处理 error_row 的内部辅助逻辑。"""
    return {
        "factor_group": "",
        "factor": column,
        "direction": direction_name,
        **_liquidity_row(liquidity_mode),
        "top_n": "",
        "error": message,
        "annual_return": 0.0,
        "max_drawdown": -1.0,
        "year_count": 0,
        "year_ann_pass": 0,
        "year_dd_pass": 0,
        "min_yearly_annual_return": 0.0,
        "yearly_all_pass": False,
        "yearly_target_gap": 1.0,
        "min_yearly_annual_return_observed": 0.0,
        "worst_yearly_drawdown": -1.0,
        "years_below_return_target": "",
        "years_breaching_drawdown_limit": "",
        "target_gap": 1.0,
        "meets_full_target": False,
    }


def _screen_quality_fields(metrics: dict[str, object], config: dict, yearly: pd.DataFrame | None = None) -> dict[str, object]:
    """函数说明：处理 screen_quality_fields 的内部辅助逻辑。"""
    return_threshold, drawdown_limit, turnover_limit = _quality_thresholds(config)
    annual_return = float(metrics.get("annual_return", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    annual_turnover = float(metrics.get("annual_turnover", metrics.get("annual_weight_turnover", 0.0)) or 0.0)
    yearly_fields = _yearly_quality_fields(yearly, return_threshold, drawdown_limit)
    year_count = int(yearly_fields["year_count"])
    yearly_pass = year_count <= 0 or (
        int(yearly_fields["year_ann_pass"]) >= year_count and int(yearly_fields["year_dd_pass"]) >= year_count
    )
    turnover_pass = annual_turnover <= turnover_limit
    global_gap = max(0.0, return_threshold - annual_return) + max(0.0, drawdown_limit - max_drawdown)
    if turnover_limit > 0:
        global_gap += max(0.0, annual_turnover - turnover_limit) / turnover_limit
    yearly_gap = 0.0
    if year_count > 0:
        yearly_gap = max(0.0, return_threshold - float(yearly_fields["min_yearly_annual_return"])) + max(
            0.0,
            drawdown_limit - float(yearly_fields["worst_yearly_drawdown"]),
        )
    return {
        **yearly_fields,
        "turnover_pass": bool(turnover_pass),
        "formal_confirmation_required": True,
        "approximation_notes": "fast_screen_ignores_formal_tradability_capacity_and_risk_exits",
        "meets_full_target": bool(
            annual_return >= return_threshold and max_drawdown >= drawdown_limit and yearly_pass and turnover_pass
        ),
        "target_gap": global_gap + yearly_gap,
    }


def _yearly_quality_fields(yearly: pd.DataFrame | None, return_threshold: float, drawdown_limit: float) -> dict[str, object]:
    if yearly is None or yearly.empty:
        return {
            "year_count": 0,
            "year_ann_pass": 0,
            "year_dd_pass": 0,
            "yearly_all_pass": False,
            "yearly_target_gap": 1.0,
            "min_yearly_annual_return": 0.0,
            "min_yearly_annual_return_observed": 0.0,
            "worst_yearly_drawdown": 0.0,
            "years_below_return_target": "",
            "years_breaching_drawdown_limit": "",
        }
    annual = pd.to_numeric(yearly.get("annual_return"), errors="coerce")
    drawdown = pd.to_numeric(yearly.get("max_drawdown"), errors="coerce")
    years = pd.to_numeric(yearly.get("year"), errors="coerce").astype("Int64")
    return_fail = (annual < return_threshold).fillna(True)
    drawdown_fail = (drawdown < drawdown_limit).fillna(True)
    return_shortfall = (return_threshold - annual).clip(lower=0.0).fillna(return_threshold)
    drawdown_shortfall = (drawdown_limit - drawdown).clip(lower=0.0).fillna(abs(drawdown_limit))
    min_annual = float(annual.min()) if not annual.dropna().empty else 0.0
    worst_drawdown = float(drawdown.min()) if not drawdown.dropna().empty else 0.0
    year_count = int(len(yearly))
    return {
        "year_count": year_count,
        "year_ann_pass": int((~return_fail).sum()),
        "year_dd_pass": int((~drawdown_fail).sum()),
        "yearly_all_pass": bool(year_count > 0 and not bool(return_fail.any()) and not bool(drawdown_fail.any())),
        "yearly_target_gap": float(return_shortfall.sum() + drawdown_shortfall.sum()),
        "min_yearly_annual_return": min_annual,
        "min_yearly_annual_return_observed": min_annual,
        "worst_yearly_drawdown": worst_drawdown,
        "years_below_return_target": ";".join(str(int(year)) for year in years[return_fail].dropna().to_list()),
        "years_breaching_drawdown_limit": ";".join(str(int(year)) for year in years[drawdown_fail].dropna().to_list()),
    }


def _fast_yearly_stats(equity_curve: pd.Series) -> pd.DataFrame:
    """Calculate yearly stats for sparse fast-backtest equity points."""
    if equity_curve.empty:
        return pd.DataFrame(columns=["year", "start", "end", "days", "total_return", "annual_return", "max_drawdown"])
    equity = equity_curve.sort_index().astype(float)
    rows: list[dict[str, object]] = []
    for year, segment in equity.groupby(equity.index.year):
        segment = segment.dropna()
        if len(segment) < 2:
            continue
        total_return = float(segment.iloc[-1] / segment.iloc[0] - 1.0) if segment.iloc[0] else 0.0
        years = max((segment.index[-1] - segment.index[0]).days / 365.25, 1 / 252)
        annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
        drawdown = segment / segment.cummax() - 1.0
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


def _quality_thresholds(config: dict) -> tuple[float, float, float]:
    """函数说明：处理 quality_thresholds 的内部辅助逻辑。"""
    quality = config.get("quality", {})
    return_threshold = float(quality.get("min_backtest_annual_return", quality.get("target_annual_return", 0.20)))
    drawdown_limit = float(quality.get("max_backtest_drawdown_limit", quality.get("max_drawdown_limit", -0.20)))
    turnover_limit = float(quality.get("max_annual_turnover", float("inf")))
    return return_threshold, drawdown_limit, turnover_limit


def _screen_yearly_quality_fields(yearly: pd.DataFrame, config: dict) -> dict[str, object]:
    """鍑芥暟璇存槑锛氬鐞?screen_yearly_quality_fields 鐨勫唴閮ㄨ緟鍔╅€昏緫銆?"""
    return_threshold, drawdown_limit, _turnover_limit = _quality_thresholds(config)
    if yearly.empty:
        return {
            "year_count": 0,
            "year_ann_pass": 0,
            "year_dd_pass": 0,
            "yearly_all_pass": False,
            "yearly_target_gap": 1.0,
            "min_yearly_annual_return_observed": 0.0,
            "worst_yearly_drawdown": 0.0,
            "years_below_return_target": "",
            "years_breaching_drawdown_limit": "",
        }

    rows = yearly.copy()
    years = pd.to_numeric(rows["year"], errors="coerce").astype("Int64")
    annual = pd.to_numeric(rows["annual_return"], errors="coerce")
    drawdown = pd.to_numeric(rows["max_drawdown"], errors="coerce")
    return_fail = (annual < return_threshold).fillna(True)
    drawdown_fail = (drawdown < drawdown_limit).fillna(True)
    return_shortfall = (return_threshold - annual).clip(lower=0.0).fillna(return_threshold)
    drawdown_shortfall = (drawdown_limit - drawdown).clip(lower=0.0).fillna(abs(drawdown_limit))
    year_count = int(len(rows))
    return {
        "year_count": year_count,
        "year_ann_pass": int((~return_fail).sum()),
        "year_dd_pass": int((~drawdown_fail).sum()),
        "yearly_all_pass": bool(year_count > 0 and not bool(return_fail.any()) and not bool(drawdown_fail.any())),
        "yearly_target_gap": float(return_shortfall.sum() + drawdown_shortfall.sum()),
        "min_yearly_annual_return_observed": float(annual.min()) if annual.notna().any() else 0.0,
        "worst_yearly_drawdown": float(drawdown.min()) if drawdown.notna().any() else 0.0,
        "years_below_return_target": ";".join(str(int(year)) for year in years[return_fail].dropna().to_list()),
        "years_breaching_drawdown_limit": ";".join(str(int(year)) for year in years[drawdown_fail].dropna().to_list()),
    }


def _sorted(rows: list[dict[str, object]]) -> pd.DataFrame:
    """函数说明：处理 sorted 的内部辅助逻辑。"""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in [
        "annual_return",
        "max_drawdown",
        "sharpe",
        "target_gap",
        "min_yearly_annual_return",
        "year_count",
        "year_ann_pass",
        "year_dd_pass",
        "yearly_target_gap",
        "min_yearly_annual_return_observed",
        "worst_yearly_drawdown",
    ]:
        frame[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce")
    return frame.sort_values(
        [
            "yearly_all_pass",
            "yearly_target_gap",
            "year_ann_pass",
            "year_dd_pass",
            "meets_full_target",
            "target_gap",
            "annual_return",
            "max_drawdown",
        ],
        ascending=[False, True, False, False, False, True, False, False],
    )


def _read_factor_subset(path_value: str | Path, columns: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """函数说明：读取 read_factor_subset 的内部辅助逻辑。"""
    path = resolve_path(path_value)
    requested = [*columns, "datetime", "instrument"]
    factors = pd.read_parquet(path, columns=requested)
    if not isinstance(factors.index, pd.MultiIndex):
        factors = factors.set_index(["datetime", "instrument"])
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    mask = (dates >= pd.Timestamp(start_date).normalize()) & (dates <= pd.Timestamp(end_date).normalize())
    return factors.loc[mask, [column for column in columns if column in factors.columns]].sort_index()


def _read_price_fields(path_value: str | Path, fields: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """函数说明：读取 read_price_fields 的内部辅助逻辑。"""
    path = resolve_path(path_value)
    columns = _price_columns_for_fields(path, fields)
    prices = pd.read_parquet(path, columns=columns)
    prices.index = pd.to_datetime(prices.index).normalize()
    mask = (prices.index >= pd.Timestamp(start_date).normalize()) & (prices.index <= pd.Timestamp(end_date).normalize())
    return prices.loc[mask].sort_index()


def _price_columns_for_fields(path: Path, fields: list[str]) -> list[str] | None:
    """函数说明：处理 price_columns_for_fields 的内部辅助逻辑。"""
    wanted = {str(field).strip().lower() for field in fields if str(field).strip()}
    if not wanted:
        return None
    try:
        import pyarrow.parquet as pq

        names = pq.ParquetFile(path).schema.names
    except Exception:
        return None
    selected = []
    for name in names:
        lowered = str(name).lower()
        if lowered in wanted or any(lowered.startswith(f"('{field}',") for field in wanted):
            selected.append(str(name))
    return selected or None


def _score_component_columns(config: dict, available_columns: list[str]) -> list[str]:
    """函数说明：处理 score_component_columns 的内部辅助逻辑。"""
    if not bool(config.get("regime_score_blend", {}).get("enabled", False)):
        return []
    available = {str(column) for column in available_columns}
    columns: list[str] = []
    for item in config.get("regime_score_blend", {}).get("defensive_components", []):
        column = str(item.get("column", ""))
        if column in available and column not in columns:
            columns.append(column)
    return columns


if __name__ == "__main__":
    main()
