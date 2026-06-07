from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.factor_calculator import factor_cache_columns
from src.fast_monthly_backtest import prepare_fast_period_data, run_fast_prepared_backtest
from src.scoring import _apply_liquidity_filter, build_strategy_scores
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast screen exact single-factor candidates for the active goal.")
    parser.add_argument("--output", default="outputs/goal_fast_factor_screen_20260607.csv")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--top-n", default="7,10,15,20")
    parser.add_argument("--liquidity-modes", default="none,low:0.20,low:0.35,high:0.20")
    parser.add_argument("--limit-columns", type=int, default=0)
    args = parser.parse_args()

    config = load_config()
    start_date = config["data"]["start_date"]
    end_date = resolve_target_date_value(config["data"]["end_date"], config=config)
    columns = factor_cache_columns(config["factors"]["cache_file"])
    if args.limit_columns:
        columns = columns[: args.limit_columns]
    component_columns = _score_component_columns(config, factor_cache_columns(config["factors"]["cache_file"]))
    top_ns = [int(value.strip()) for value in args.top_n.split(",") if value.strip()]
    liquidity_modes = [_parse_liquidity_mode(value) for value in args.liquidity_modes.split(",") if value.strip()]
    trade_price_field = str(config.get("backtest", {}).get("trade_price_field", "close")).lower()
    price_fields = {"close", trade_price_field}
    if any(bool(mode.get("enabled", False)) for mode in liquidity_modes):
        price_fields.add(str(config.get("liquidity_filter", {}).get("field", "amount")).lower())
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
        factors = _read_factor_subset(config["factors"]["cache_file"], read_columns, start_date, end_date)
        for column in batch:
            score_columns = [col for col in [column, *component_columns] if col in factors.columns]
            for direction_name, factor_group in [
                ("long_high", f"factor:{column}"),
                ("long_low", f"inverse_factor:{column}"),
            ]:
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
                        row = {
                            "factor_group": factor_group,
                            "factor": column,
                            "direction": direction_name,
                            **_liquidity_row(liquidity_mode),
                            "top_n": top_n,
                            **result.metrics,
                        }
                        row["meets_full_target"] = bool(row.get("annual_return", 0.0) >= 0.20 and row.get("max_drawdown", 0.0) >= -0.20)
                        row["target_gap"] = max(0.0, 0.20 - float(row.get("annual_return", 0.0))) + max(
                            0.0,
                            -0.20 - float(row.get("max_drawdown", 0.0)),
                        )
                        rows.append(row)
        frame = _sorted(rows)
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(
            f"screened {min(batch_start + args.batch_size, len(columns))}/{len(columns)} columns in {time.monotonic() - started:.1f}s; "
            f"best annual={frame.iloc[0].get('annual_return', 0.0):.4f} dd={frame.iloc[0].get('max_drawdown', 0.0):.4f}",
            flush=True,
        )
    print(f"Saved screen to {out_path}")


def _screen_config(config: dict, factor_group: str, liquidity_mode: dict[str, object]) -> dict:
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
    filter_config = {**config.get("liquidity_filter", {}), **liquidity_mode}
    if not bool(filter_config.get("enabled", False)):
        return scores
    return _apply_liquidity_filter(scores, prices, filter_config)


def _parse_liquidity_mode(value: str) -> dict[str, object]:
    mode = value.strip().lower()
    if mode in {"none", "off", "false"}:
        return {"enabled": False}
    side, _, quantile = mode.partition(":")
    return {"enabled": True, "side": side, "quantile": float(quantile or 0.20)}


def _liquidity_row(liquidity_mode: dict[str, object]) -> dict[str, object]:
    return {
        "liquidity_enabled": bool(liquidity_mode.get("enabled", False)),
        "liquidity_side": liquidity_mode.get("side", ""),
        "liquidity_quantile": liquidity_mode.get("quantile", ""),
    }


def _error_row(column: str, direction_name: str, liquidity_mode: dict[str, object], message: str) -> dict[str, object]:
    return {
        "factor_group": "",
        "factor": column,
        "direction": direction_name,
        **_liquidity_row(liquidity_mode),
        "top_n": "",
        "error": message,
        "annual_return": 0.0,
        "max_drawdown": -1.0,
        "target_gap": 1.0,
        "meets_full_target": False,
    }


def _sorted(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ["annual_return", "max_drawdown", "sharpe", "target_gap"]:
        frame[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce")
    return frame.sort_values(["meets_full_target", "target_gap", "annual_return", "max_drawdown"], ascending=[False, True, False, False])


def _read_factor_subset(path_value: str | Path, columns: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    path = resolve_path(path_value)
    requested = [*columns, "datetime", "instrument"]
    factors = pd.read_parquet(path, columns=requested)
    if not isinstance(factors.index, pd.MultiIndex):
        factors = factors.set_index(["datetime", "instrument"])
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    mask = (dates >= pd.Timestamp(start_date).normalize()) & (dates <= pd.Timestamp(end_date).normalize())
    return factors.loc[mask, [column for column in columns if column in factors.columns]].sort_index()


def _read_price_fields(path_value: str | Path, fields: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    path = resolve_path(path_value)
    columns = _price_columns_for_fields(path, fields)
    prices = pd.read_parquet(path, columns=columns)
    prices.index = pd.to_datetime(prices.index).normalize()
    mask = (prices.index >= pd.Timestamp(start_date).normalize()) & (prices.index <= pd.Timestamp(end_date).normalize())
    return prices.loc[mask].sort_index()


def _price_columns_for_fields(path: Path, fields: list[str]) -> list[str] | None:
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
