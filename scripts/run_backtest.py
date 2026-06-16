"""模块说明：提供 run_backtest 命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.risk_policy import RiskPolicy
from src.scoring import build_strategy_scores
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value
from src.universe_builder import apply_configured_historical_universe
from src.universe_coverage import summarize_universe_coverage
from scripts._shared import requested_factor_columns, strip_direction_prefix, yearly_quality_gate, yearly_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    config = load_config()
    parser = argparse.ArgumentParser(description="Run a lightweight ranking-strategy backtest.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--benchmark-file", help="Optional benchmark close parquet/csv for alpha, beta and IR.")
    args = parser.parse_args()
    end_date = resolve_target_date_value(args.end_date, config=config)
    config["data"]["end_date"] = end_date
    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_run_logging(out_dir)
    logger.info("Backtest run log: %s", log_path)
    logger.info(
        "Run context: %s",
        json.dumps(_run_context(args, config, end_date, log_path), ensure_ascii=False, default=str),
    )
    logger.info(
        "Backtest requested: start=%s end=%s factor_file=%s price_file=%s benchmark_file=%s",
        args.start_date,
        end_date,
        args.factor_file,
        args.price_file,
        args.benchmark_file or "",
    )
    logger.info("Quality targets: %s", json.dumps(_quality_targets(config), ensure_ascii=False, default=str))

    try:
        price_file = resolve_path(args.price_file)
        if not price_file.exists():
            raise FileNotFoundError(f"Price file not found: {price_file}. Run scripts/run_convert_data.py first.")
        logger.info("Price file: %s", json.dumps(_path_summary(price_file), ensure_ascii=False, default=str))
        logger.info("Factor cache file: %s", json.dumps(_path_summary(resolve_path(args.factor_file)), ensure_ascii=False, default=str))
        prices = pd.read_parquet(price_file)
        price_summary = _frame_summary(prices)
        logger.info("Price input: %s", json.dumps(price_summary, ensure_ascii=False, default=str))
        factor_columns = _requested_factor_columns(
            args.factor_file,
            config.get("strategy", {}),
            config.get("dynamic_ic_selector", {}),
            config.get("ml_strategy", {}),
            config.get("regime_score_blend", {}),
            config.get("regime_score_filter", {}),
        )
        if factor_columns is None:
            logger.info("Factor columns requested: all available columns.")
        else:
            logger.info(
                "Factor columns requested: count=%s sample=%s factor_group=%s",
                len(factor_columns),
                factor_columns[:12],
                config.get("strategy", {}).get("factor_group"),
            )
        factors = load_or_compute_factors(args.start_date, end_date, cache_file=args.factor_file, columns=factor_columns)
        factor_summary = _frame_summary(factors)
        logger.info("Factor input: %s", json.dumps(factor_summary, ensure_ascii=False, default=str))
        logger.info(
            "Input alignment: %s",
            json.dumps(_input_alignment_summary(prices, factors), ensure_ascii=False, default=str),
        )
        scores = build_strategy_scores(factors, config, price_df=prices)
        score_raw_summary = _score_summary(scores)
        logger.info("Score panel before resample: %s", json.dumps(score_raw_summary, ensure_ascii=False, default=str))
        scores = apply_configured_historical_universe(scores, config)
        score_universe_summary = _score_summary(scores)
        logger.info("Score panel after historical universe filter: %s", json.dumps(score_universe_summary, ensure_ascii=False, default=str))
        scores = resample_signals(scores, config["strategy"].get("rebalance_freq", "daily"))
        score_summary = _score_summary(scores)
        logger.info("Score panel after resample: %s", json.dumps(score_summary, ensure_ascii=False, default=str))

        bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **config["strategy"]}, prices, config)
        bt_config = RiskPolicy(config).apply_to_backtest_config(bt_config)
        logger.info("Backtest config snapshot: %s", json.dumps(_backtest_config_snapshot(bt_config), ensure_ascii=False, default=str))
        if args.benchmark_file:
            benchmark_path = resolve_path(args.benchmark_file)
            logger.info("Benchmark file: %s", json.dumps(_path_summary(benchmark_path), ensure_ascii=False, default=str))
            if benchmark_path.suffix.lower() == ".csv":
                benchmark = pd.read_csv(benchmark_path, index_col=0).iloc[:, 0]
                benchmark.index = pd.to_datetime(benchmark.index)
            else:
                benchmark_df = pd.read_parquet(benchmark_path)
                benchmark = benchmark_df.iloc[:, 0] if isinstance(benchmark_df, pd.DataFrame) else benchmark_df
                benchmark.index = pd.to_datetime(benchmark.index)
            bt_config["benchmark_curve"] = benchmark
            logger.info("Benchmark input: %s", json.dumps(_series_summary(benchmark), ensure_ascii=False, default=str))
        result = run_backtest(scores, prices, args.start_date, end_date, bt_config)

        equity_path = out_dir / "backtest_equity.csv"
        holdings_path = out_dir / "backtest_holdings.csv"
        trades_path = out_dir / "backtest_trades.csv"
        metrics_path = out_dir / "backtest_metrics.json"
        yearly_path = out_dir / "backtest_yearly.csv"
        summary_path = out_dir / "backtest_run_summary.json"
        result.equity_curve.to_csv(equity_path, encoding="utf-8-sig")
        result.holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")
        result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
        metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        yearly = yearly_stats(result.equity_curve, bt_config)
        yearly_gate = yearly_quality_gate(yearly, config)
        yearly = _annotate_yearly_quality(yearly, yearly_gate)
        yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
        coverage = summarize_universe_coverage(config, price_df=prices)
        (out_dir / "universe_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        run_summary = _build_run_summary(
            result,
            yearly,
            coverage,
            {
                "start_date": args.start_date,
                "end_date": end_date,
                "factor_file": args.factor_file,
                "price_file": str(price_file),
                "benchmark_file": args.benchmark_file or "",
                "log_file": str(log_path),
            },
            yearly_gate=yearly_gate,
            data={
                "prices": price_summary,
                "factors": factor_summary,
                "scores_before_resample": score_raw_summary,
                "scores_after_universe_filter": score_universe_summary,
                "scores_after_resample": score_summary,
                "alignment": _input_alignment_summary(prices, factors),
            },
        )
        summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        logger.info("Backtest finished.")
        for key, value in result.metrics.items():
            logger.info("%s: %.6f", key, value)
        logger.info("Equity summary: %s", json.dumps(run_summary["equity"], ensure_ascii=False, default=str))
        logger.info("Drawdown summary: %s", json.dumps(run_summary["drawdown"], ensure_ascii=False, default=str))
        logger.info("Holding summary: %s", json.dumps(run_summary["holdings"], ensure_ascii=False, default=str))
        logger.info("Trade summary: %s", json.dumps(run_summary["trades"], ensure_ascii=False, default=str))
        logger.info("Rebalance summary: %s", json.dumps(run_summary["rebalances"], ensure_ascii=False, default=str))
        logger.info("Cost summary: %s", json.dumps(run_summary["costs"], ensure_ascii=False, default=str))
        logger.info("Yearly trade summary: %s", json.dumps(run_summary["yearly_trades"], ensure_ascii=False, default=str))
        logger.info("Top traded instruments: %s", json.dumps(run_summary["top_traded_instruments"], ensure_ascii=False, default=str))
        logger.info("Yearly target gate: %s", json.dumps(yearly_gate, ensure_ascii=False, default=str))
        _log_yearly_rows(yearly, yearly_gate)
        logger.info(
            "Universe coverage: %d/%d target symbols in price panel (%.2f%%).",
            coverage["price_target_symbols"],
            coverage["target_symbols"],
            coverage["price_target_coverage"] * 100,
        )
        logger.info(
            "Output files: equity=%s holdings=%s trades=%s metrics=%s yearly=%s summary=%s coverage=%s",
            equity_path,
            holdings_path,
            trades_path,
            metrics_path,
            yearly_path,
            summary_path,
            out_dir / "universe_coverage.json",
        )
    except Exception:
        logger.exception("Backtest failed. See the traceback and prior input summaries in this log.")
        raise


def _requested_factor_columns(
    factor_file: str,
    strategy_cfg: dict,
    dynamic_cfg: dict | None = None,
    ml_cfg: dict | None = None,
    score_blend_cfg: dict | None = None,
    score_filter_cfg: dict | None = None,
) -> list[str] | None:
    """函数说明：处理 requested_factor_columns 的内部辅助逻辑。"""
    return requested_factor_columns(factor_file, strategy_cfg, dynamic_cfg, ml_cfg, score_blend_cfg, score_filter_cfg)


def _strip_direction_prefix(value: str) -> str:
    """函数说明：去除 strip_direction_prefix 的内部辅助逻辑。"""
    return strip_direction_prefix(value)


def _configure_run_logging(out_dir: Path) -> Path:
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_stem = f"backtest_{datetime.now():%Y%m%d_%H%M%S_%f}"
    path = logs_dir / f"{log_stem}.log"
    counter = 1
    while path.exists():
        path = logs_dir / f"{log_stem}_{counter}.log"
        counter += 1
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if getattr(existing, "_quant_box_backtest_file", False):
            root_logger.removeHandler(existing)
            existing.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    handler._quant_box_backtest_file = True  # type: ignore[attr-defined]
    root_logger.addHandler(handler)
    return path


def _run_context(args: argparse.Namespace, config: dict[str, Any], end_date: str, log_path: Path) -> dict[str, Any]:
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(Path.cwd()),
        "root": str(ROOT),
        "python": sys.executable,
        "pid": os.getpid(),
        "argv": sys.argv,
        "git": _git_context(),
        "inputs": {
            "start_date": args.start_date,
            "end_date": end_date,
            "factor_file": args.factor_file,
            "price_file": args.price_file,
            "benchmark_file": args.benchmark_file or "",
            "log_file": str(log_path),
        },
        "config": _safe_config_snapshot(config),
    }


def _git_context() -> dict[str, Any]:
    context: dict[str, Any] = {}
    commands = {
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "commit": ["git", "rev-parse", "--short", "HEAD"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            context[key] = completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            context[key] = None
    try:
        completed = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        context["dirty"] = bool(lines)
        context["dirty_file_count"] = len(lines)
    except (OSError, subprocess.CalledProcessError):
        context["dirty"] = None
    return context


def _safe_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": _select_keys(
            config.get("data", {}),
            ["start_date", "history_start_date", "end_date", "universe", "freq", "exclude_st"],
        ),
        "strategy": _select_keys(
            config.get("strategy", {}),
            [
                "factor_group",
                "rebalance_freq",
                "top_n",
                "max_turnover",
                "rank_buffer",
                "stop_loss_pct",
                "take_profit_pct",
                "circuit_breaker_drawdown",
                "rebalance_drift_threshold",
                "max_industry_weight",
                "min_cross_section_obs",
            ],
        ),
        "backtest": _select_keys(
            config.get("backtest", {}),
            [
                "initial_capital",
                "commission",
                "stamp_tax",
                "transfer_fee",
                "slippage",
                "dynamic_slippage_enabled",
                "max_participation_rate",
                "capacity_window",
                "trade_price_field",
                "valuation_price_field",
                "stale_price_exit_days",
                "stale_price_exit_policy",
            ],
        ),
        "quality": _quality_targets(config),
        "ml_strategy": _select_keys(
            config.get("ml_strategy", {}),
            ["enabled", "model_type", "rebalance_freq", "top_n", "target_annual_return", "min_yearly_annual_return", "max_drawdown_limit"],
        ),
        "filters": {
            "liquidity_filter": _select_keys(config.get("liquidity_filter", {}), ["enabled", "field", "window", "quantile", "side"]),
            "selection_risk_filter": _select_keys(
                config.get("selection_risk_filter", {}),
                ["enabled", "lookback_sessions", "max_missing_price_sessions", "max_limit_down_days", "require_positive_volume"],
            ),
            "market_regime": _select_keys(
                config.get("market_regime", {}),
                ["enabled", "ma_window", "momentum_window", "volatility_window", "lag_days"],
            ),
            "defensive_timing": _select_keys(
                config.get("defensive_timing", {}),
                ["enabled", "bull_exposure", "sideways_exposure", "bear_exposure", "exposure_rebalance_threshold"],
            ),
        },
    }


def _quality_targets(config: dict[str, Any]) -> dict[str, Any]:
    quality = config.get("quality", {})
    ml_cfg = config.get("ml_strategy", {})
    return {
        "target_annual_return": quality.get("target_annual_return", ml_cfg.get("target_annual_return", 0.20)),
        "min_backtest_annual_return": quality.get("min_backtest_annual_return", quality.get("target_annual_return", 0.20)),
        "min_yearly_annual_return": ml_cfg.get(
            "min_yearly_annual_return",
            quality.get("min_backtest_annual_return", quality.get("target_annual_return", 0.20)),
        ),
        "max_backtest_drawdown_limit": quality.get("max_backtest_drawdown_limit", quality.get("max_drawdown_limit", -0.20)),
        "max_yearly_drawdown_limit": ml_cfg.get(
            "max_drawdown_limit",
            quality.get("max_backtest_drawdown_limit", quality.get("max_drawdown_limit", -0.20)),
        ),
        "max_annual_turnover": quality.get("max_annual_turnover"),
        "max_annual_trade_cost_ratio": quality.get("max_annual_trade_cost_ratio"),
    }


def _select_keys(values: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: values.get(key) for key in keys if key in values}


def _path_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        summary["size_bytes"] = int(stat.st_size)
        summary["modified_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    return summary


def _frame_summary(frame: pd.DataFrame | pd.Series) -> dict[str, Any]:
    if isinstance(frame, pd.Series):
        return _series_summary(frame)
    summary: dict[str, Any] = {"rows": int(len(frame)), "columns": int(len(frame.columns))}
    summary["memory_mb"] = round(float(frame.memory_usage(index=True, deep=True).sum()) / 1_000_000, 3)
    summary["duplicate_index_rows"] = int(frame.index.duplicated().sum())
    start, end = _index_date_bounds(frame.index)
    if start is not None:
        summary["start"] = start
        summary["end"] = end
    if isinstance(frame.columns, pd.MultiIndex):
        summary["fields"] = sorted({str(value) for value in frame.columns.get_level_values(0)})
        if frame.columns.nlevels > 1:
            summary["instruments"] = int(frame.columns.get_level_values(1).nunique())
    else:
        summary["sample_columns"] = [str(column) for column in list(frame.columns[:8])]
    return summary


def _series_summary(series: pd.Series) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(series))}
    summary["non_null"] = int(series.notna().sum())
    summary["duplicate_index_rows"] = int(series.index.duplicated().sum())
    start, end = _index_date_bounds(series.index)
    if start is not None:
        summary["start"] = start
        summary["end"] = end
    return summary


def _score_summary(scores: pd.Series | pd.DataFrame) -> dict[str, Any]:
    summary = _frame_summary(scores) if isinstance(scores, pd.DataFrame) else _series_summary(scores)
    if isinstance(scores, pd.Series):
        numeric = pd.to_numeric(scores, errors="coerce").dropna()
        summary["non_null_scores"] = int(len(numeric))
        if not numeric.empty:
            summary["score_min"] = float(numeric.min())
            summary["score_max"] = float(numeric.max())
            summary["score_mean"] = float(numeric.mean())
    index = scores.index
    if isinstance(index, pd.MultiIndex) and index.nlevels >= 2:
        dates = pd.to_datetime(index.get_level_values(0), errors="coerce")
        dates = dates[~pd.isna(dates)]
        summary["signal_dates"] = int(pd.Index(dates.normalize()).nunique()) if len(dates) else 0
        summary["instruments"] = int(index.get_level_values(1).nunique())
    return summary


def _input_alignment_summary(prices: pd.DataFrame, factors: pd.DataFrame) -> dict[str, Any]:
    price_dates = _normalized_dates_from_index(prices.index)
    factor_dates = _normalized_dates_from_index(factors.index)
    common_dates = price_dates.intersection(factor_dates)
    price_symbols = _price_symbols(prices)
    factor_symbols = _factor_symbols(factors)
    common_symbols = price_symbols.intersection(factor_symbols)
    return {
        "price_dates": int(len(price_dates)),
        "factor_dates": int(len(factor_dates)),
        "common_dates": int(len(common_dates)),
        "first_common_date": common_dates.min().date().isoformat() if len(common_dates) else None,
        "last_common_date": common_dates.max().date().isoformat() if len(common_dates) else None,
        "price_symbols": int(len(price_symbols)),
        "factor_symbols": int(len(factor_symbols)),
        "common_symbols": int(len(common_symbols)),
        "price_only_symbols": int(len(price_symbols - factor_symbols)),
        "factor_only_symbols": int(len(factor_symbols - price_symbols)),
    }


def _normalized_dates_from_index(index: pd.Index) -> pd.DatetimeIndex:
    values = index.get_level_values(0) if isinstance(index, pd.MultiIndex) and index.nlevels else index
    dates = pd.to_datetime(values, errors="coerce")
    dates = dates[~pd.isna(dates)]
    if not len(dates):
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(dates).normalize().unique().sort_values()


def _price_symbols(prices: pd.DataFrame) -> set[str]:
    if isinstance(prices.columns, pd.MultiIndex) and prices.columns.nlevels > 1:
        return {str(value).strip().lower() for value in prices.columns.get_level_values(1).dropna().unique()}
    return set()


def _factor_symbols(factors: pd.DataFrame) -> set[str]:
    if isinstance(factors.index, pd.MultiIndex) and factors.index.nlevels > 1:
        return {str(value).strip().lower() for value in factors.index.get_level_values(1).dropna().unique()}
    if "instrument" in factors.columns:
        return {str(value).strip().lower() for value in factors["instrument"].dropna().unique()}
    return set()


def _backtest_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "initial_capital",
        "annual_trading_days",
        "top_n",
        "max_turnover",
        "rebalance_freq",
        "trade_price_field",
        "valuation_price_field",
        "commission",
        "stamp_tax",
        "transfer_fee",
        "slippage",
        "dynamic_slippage_enabled",
        "max_participation_rate",
        "capacity_window",
        "limit_up_threshold",
        "limit_down_threshold",
        "stop_loss_pct",
        "take_profit_pct",
        "circuit_breaker_drawdown",
        "circuit_breaker_cooldown_days",
        "circuit_breaker_target_exposure",
        "rebalance_drift_threshold",
        "score_weighted",
        "max_weight_per_stock",
        "max_industry_weight",
    ]
    snapshot = {key: config.get(key) for key in keys if key in config}
    if "annual_drawdown_guard" in config:
        snapshot["annual_drawdown_guard"] = config.get("annual_drawdown_guard")
    if "equity_overlay" in config:
        snapshot["equity_overlay"] = config.get("equity_overlay")
    if "exposure_schedule" in config:
        exposure = config.get("exposure_schedule")
        snapshot["exposure_schedule"] = _series_summary(exposure) if isinstance(exposure, pd.Series) else bool(exposure)
    return snapshot


def _annotate_yearly_quality(yearly: pd.DataFrame, yearly_gate: dict[str, Any]) -> pd.DataFrame:
    if yearly.empty:
        return yearly.copy()
    result = yearly.copy()
    min_return = float(yearly_gate.get("min_yearly_annual_return", 0.20))
    drawdown_limit = float(yearly_gate.get("max_drawdown_limit", -0.20))
    result["annual_return_target"] = min_return
    result["max_drawdown_limit"] = drawdown_limit
    result["annual_return_pass"] = pd.to_numeric(result["annual_return"], errors="coerce") >= min_return
    result["drawdown_pass"] = pd.to_numeric(result["max_drawdown"], errors="coerce") >= drawdown_limit
    result["year_pass"] = result["annual_return_pass"] & result["drawdown_pass"]
    return result


def _build_run_summary(
    result: Any,
    yearly: pd.DataFrame,
    coverage: dict[str, Any],
    inputs: dict[str, Any],
    yearly_gate: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "inputs": inputs,
        "equity": _equity_summary(result.equity_curve),
        "drawdown": _drawdown_summary(result.equity_curve),
        "holdings": _holding_summary(getattr(result, "holdings", pd.DataFrame()), result.equity_curve),
        "trades": _trade_summary(result.trades),
        "rebalances": _rebalance_summary(result.trades),
        "costs": _cost_summary(result.metrics, result.trades),
        "yearly_trades": _yearly_trade_summary(result.trades),
        "top_traded_instruments": _top_traded_instruments(result.trades),
        "metrics": result.metrics,
        "yearly": yearly.to_dict(orient="records"),
        "universe_coverage": coverage,
    }
    if yearly_gate is not None:
        summary["yearly_quality_gate"] = yearly_gate
        summary["yearly_failures"] = _yearly_failures(yearly, yearly_gate)
    if data is not None:
        summary["data"] = data
    return summary


def _yearly_failures(yearly: pd.DataFrame, yearly_gate: dict[str, Any]) -> dict[str, Any]:
    if yearly.empty:
        return {"return_failures": [], "drawdown_failures": [], "failed_years": []}
    min_return = float(yearly_gate.get("min_yearly_annual_return", 0.20))
    drawdown_limit = float(yearly_gate.get("max_drawdown_limit", -0.20))
    rows = yearly.copy()
    annual_return = pd.to_numeric(rows.get("annual_return"), errors="coerce")
    max_drawdown = pd.to_numeric(rows.get("max_drawdown"), errors="coerce")
    return_failures = rows[annual_return < min_return]
    drawdown_failures = rows[max_drawdown < drawdown_limit]
    failed_years = sorted(
        set(pd.to_numeric(return_failures["year"], errors="coerce").dropna().astype(int).to_list())
        | set(pd.to_numeric(drawdown_failures["year"], errors="coerce").dropna().astype(int).to_list())
    )
    return {
        "return_failures": _yearly_rows_for_log(return_failures),
        "drawdown_failures": _yearly_rows_for_log(drawdown_failures),
        "failed_years": failed_years,
        "passed_year_count": int(len(rows) - len(failed_years)),
        "total_year_count": int(len(rows)),
    }


def _yearly_rows_for_log(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["year", "start", "end", "days", "annual_return", "max_drawdown", "annual_return_pass", "drawdown_pass", "year_pass"]
    result: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        result.append({key: row.get(key) for key in columns if key in row})
    return result


def _log_yearly_rows(yearly: pd.DataFrame, yearly_gate: dict[str, Any]) -> None:
    if yearly.empty:
        logger.warning("Yearly summary is empty; no yearly target validation is available.")
        return
    min_return = float(yearly_gate.get("min_yearly_annual_return", 0.20))
    drawdown_limit = float(yearly_gate.get("max_drawdown_limit", -0.20))
    for row in yearly.to_dict(orient="records"):
        logger.info(
            "Year %s: annual_return=%.6f target=%.6f pass=%s max_drawdown=%.6f limit=%.6f pass=%s days=%s start=%s end=%s",
            row.get("year"),
            float(row.get("annual_return", 0.0) or 0.0),
            min_return,
            bool(row.get("annual_return_pass", False)),
            float(row.get("max_drawdown", 0.0) or 0.0),
            drawdown_limit,
            bool(row.get("drawdown_pass", False)),
            row.get("days"),
            row.get("start"),
            row.get("end"),
        )
    failures = _yearly_failures(yearly, yearly_gate)
    if failures["failed_years"]:
        logger.warning("Yearly target failed years: %s", failures["failed_years"])
        logger.warning("Yearly return failures: %s", json.dumps(failures["return_failures"], ensure_ascii=False, default=str))
        logger.warning("Yearly drawdown failures: %s", json.dumps(failures["drawdown_failures"], ensure_ascii=False, default=str))
    else:
        logger.info("All yearly targets passed.")


def _equity_summary(equity_curve: pd.Series) -> dict[str, Any]:
    if equity_curve.empty:
        return {"days": 0}
    equity = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if equity.empty:
        return {"days": 0}
    return {
        "days": int(len(equity)),
        "start": pd.Timestamp(equity.index.min()).date().isoformat(),
        "end": pd.Timestamp(equity.index.max()).date().isoformat(),
        "initial_equity": float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
        "min_equity": float(equity.min()),
        "max_equity": float(equity.max()),
    }


def _drawdown_summary(equity_curve: pd.Series) -> dict[str, Any]:
    if equity_curve.empty:
        return {"max_drawdown": 0.0}
    equity = pd.to_numeric(equity_curve, errors="coerce").dropna().sort_index()
    if equity.empty:
        return {"max_drawdown": 0.0}
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    peak_value = float(equity.loc[peak])
    trough_value = float(equity.loc[trough])
    recovery_date = None
    if peak_value > 0:
        recovered = equity.loc[trough:][equity.loc[trough:] >= peak_value]
        if not recovered.empty:
            recovery_date = pd.Timestamp(recovered.index[0]).date().isoformat()
    return {
        "max_drawdown": float(drawdown.loc[trough]),
        "peak_date": pd.Timestamp(peak).date().isoformat(),
        "trough_date": pd.Timestamp(trough).date().isoformat(),
        "recovery_date": recovery_date,
        "peak_equity": peak_value,
        "trough_equity": trough_value,
        "days_to_trough": int(max((pd.Timestamp(trough) - pd.Timestamp(peak)).days, 0)),
        "worst_periods": _drawdown_periods(equity, limit=5),
    }


def _trade_summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"rows": 0, "executable_rows": 0, "blocked_rows": 0}
    status_values = _normalized_column(trades, "status")
    blocked = trades[status_values == "blocked"]
    executable = trades[status_values != "blocked"].copy()
    notional = _trade_notional(executable)
    dates = _trade_dates(trades)
    return {
        "rows": int(len(trades)),
        "executable_rows": int((status_values != "blocked").sum()),
        "blocked_rows": int((status_values == "blocked").sum()),
        "partial_rows": int((status_values == "partial").sum()),
        "risk_exit_rows": int((status_values == "risk_exit").sum()),
        "status_counts": _value_counts(trades, "status"),
        "side_counts": _value_counts(trades, "side"),
        "blocked_reason_counts": _value_counts(blocked, "reason"),
        "instruments": int(trades["instrument"].nunique()) if "instrument" in trades.columns else 0,
        "trade_dates": int(pd.Index(dates.dropna().dt.normalize()).nunique()) if not dates.empty else 0,
        "first_trade_date": dates.min().date().isoformat() if not dates.dropna().empty else None,
        "last_trade_date": dates.max().date().isoformat() if not dates.dropna().empty else None,
        "gross_notional": float(notional.sum()) if not notional.empty else 0.0,
        "avg_order_notional": float(notional.mean()) if not notional.empty else 0.0,
        "net_cash_flow": _sum_numeric(executable, "cash"),
    }


def _cost_summary(metrics: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "commission_cost": float(metrics.get("commission_cost", 0.0)),
        "tax_cost": float(metrics.get("tax_cost", 0.0)),
        "transfer_fee_cost": float(metrics.get("transfer_fee_cost", 0.0)),
        "slippage_cost": float(metrics.get("slippage_cost", 0.0)),
        "trade_cost": float(metrics.get("trade_cost", 0.0)),
        "trade_cost_ratio": float(metrics.get("trade_cost_ratio", 0.0)),
        "annual_trade_cost_ratio": float(metrics.get("annual_trade_cost_ratio", 0.0)),
    }
    if not trades.empty and "capacity_warning" in trades.columns:
        summary["capacity_warning_count"] = int(pd.Series(trades["capacity_warning"]).fillna(False).astype(bool).sum())
    if not trades.empty:
        summary["cost_by_side"] = _cost_by_group(trades, "side")
        summary["cost_by_status"] = _cost_by_group(trades, "status")
    return summary


def _holding_summary(holdings: pd.DataFrame, equity_curve: pd.Series | None = None) -> dict[str, Any]:
    if holdings.empty:
        return {"rows": 0, "holding_dates": 0, "instruments": 0}
    rows = holdings.copy()
    if "date" not in rows.columns:
        return {"rows": int(len(rows)), "holding_dates": 0, "instruments": int(rows["instrument"].nunique()) if "instrument" in rows.columns else 0}
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["value"] = _numeric_column(rows, "value")
    dated = rows.dropna(subset=["date"])
    position_counts = dated.groupby(dated["date"].dt.normalize())["instrument"].nunique() if "instrument" in dated.columns else pd.Series(dtype=float)
    gross_value = dated.groupby(dated["date"].dt.normalize())["value"].sum()
    summary: dict[str, Any] = {
        "rows": int(len(holdings)),
        "holding_dates": int(len(gross_value)),
        "instruments": int(dated["instrument"].nunique()) if "instrument" in dated.columns else 0,
        "avg_positions": float(position_counts.mean()) if not position_counts.empty else 0.0,
        "min_positions": int(position_counts.min()) if not position_counts.empty else 0,
        "max_positions": int(position_counts.max()) if not position_counts.empty else 0,
        "avg_gross_holding_value": float(gross_value.mean()) if not gross_value.empty else 0.0,
        "max_gross_holding_value": float(gross_value.max()) if not gross_value.empty else 0.0,
        "top_held_instruments_by_days": _top_held_instruments(dated),
    }
    if not dated.empty and "instrument" in dated.columns:
        weights = dated.copy()
        weights["date"] = weights["date"].dt.normalize()
        daily_total = weights.groupby("date")["value"].transform("sum")
        concentration = (weights["value"] / daily_total.where(daily_total != 0.0)).dropna()
        summary["max_single_name_holding_weight"] = float(concentration.max()) if not concentration.empty else 0.0
    if equity_curve is not None and not equity_curve.empty and not gross_value.empty:
        equity = pd.to_numeric(equity_curve, errors="coerce").dropna()
        equity.index = pd.to_datetime(equity.index, errors="coerce").normalize()
        aligned_equity = equity.reindex(gross_value.index)
        exposure = (gross_value / aligned_equity.where(aligned_equity != 0.0)).dropna()
        summary["avg_gross_exposure"] = float(exposure.mean()) if not exposure.empty else 0.0
        summary["max_gross_exposure"] = float(exposure.max()) if not exposure.empty else 0.0
    return summary


def _rebalance_summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "signal_date" not in trades.columns:
        return {"signal_dates": 0}
    signal_dates = pd.to_datetime(trades["signal_date"], errors="coerce").dropna().dt.normalize()
    if signal_dates.empty:
        return {"signal_dates": 0}
    trade_dates = _trade_dates(trades).dropna().dt.normalize()
    return {
        "signal_dates": int(signal_dates.nunique()),
        "first_signal_date": signal_dates.min().date().isoformat(),
        "last_signal_date": signal_dates.max().date().isoformat(),
        "trade_dates": int(trade_dates.nunique()) if not trade_dates.empty else 0,
        "orders_per_signal_date_avg": float(len(trades) / signal_dates.nunique()),
    }


def _yearly_trade_summary(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty or "date" not in trades.columns:
        return []
    rows = trades.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.dropna(subset=["date"])
    if rows.empty:
        return []
    rows["year"] = rows["date"].dt.year.astype(int)
    status_values = _normalized_column(rows, "status")
    rows["_notional"] = _trade_notional(rows)
    summaries: list[dict[str, Any]] = []
    for year, group in rows.groupby("year", sort=True):
        group_status = status_values.loc[group.index]
        blocked = group[group_status == "blocked"]
        summaries.append(
            {
                "year": int(year),
                "rows": int(len(group)),
                "executable_rows": int((group_status != "blocked").sum()),
                "blocked_rows": int((group_status == "blocked").sum()),
                "status_counts": _value_counts(group, "status"),
                "side_counts": _value_counts(group, "side"),
                "blocked_reason_counts": _value_counts(blocked, "reason"),
                "instruments": int(group["instrument"].nunique()) if "instrument" in group.columns else 0,
                "gross_notional": float(group["_notional"].sum()),
                "commission_cost": _sum_numeric(group, "commission_cost"),
                "tax_cost": _sum_numeric(group, "tax_cost"),
                "transfer_fee_cost": _sum_numeric(group, "transfer_fee_cost"),
                "slippage_cost": _sum_numeric(group, "slippage_cost"),
                "trade_cost": float(
                    _sum_numeric(group, "commission_cost")
                    + _sum_numeric(group, "tax_cost")
                    + _sum_numeric(group, "transfer_fee_cost")
                    + _sum_numeric(group, "slippage_cost")
                ),
                "capacity_warning_count": _capacity_warning_count(group),
            }
        )
    return summaries


def _top_traded_instruments(trades: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if trades.empty or "instrument" not in trades.columns:
        return []
    rows = trades.copy()
    rows["_notional"] = _trade_notional(rows)
    status_values = _normalized_column(rows, "status")
    rows["_blocked"] = status_values == "blocked"
    grouped = rows.groupby("instrument", dropna=True)
    result = pd.DataFrame(
        {
            "orders": grouped.size(),
            "blocked_orders": grouped["_blocked"].sum(),
            "gross_notional": grouped["_notional"].sum(),
            "trade_cost": grouped.apply(
                lambda group: _sum_numeric(group, "commission_cost")
                + _sum_numeric(group, "tax_cost")
                + _sum_numeric(group, "transfer_fee_cost")
                + _sum_numeric(group, "slippage_cost")
            ),
        }
    )
    result = result.sort_values(["gross_notional", "orders"], ascending=False).head(limit)
    return [
        {
            "instrument": str(index),
            "orders": int(row["orders"]),
            "blocked_orders": int(row["blocked_orders"]),
            "gross_notional": float(row["gross_notional"]),
            "trade_cost": float(row["trade_cost"]),
        }
        for index, row in result.iterrows()
    ]


def _drawdown_periods(equity: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    if equity.empty:
        return []
    values = pd.to_numeric(equity, errors="coerce").dropna().sort_index()
    if values.empty:
        return []
    drawdown = values / values.cummax() - 1.0
    underwater = drawdown < 0
    periods: list[dict[str, Any]] = []
    start_pos: int | None = None
    for pos, is_underwater in enumerate(underwater.to_list()):
        if is_underwater and start_pos is None:
            start_pos = pos
        if start_pos is not None and (not is_underwater or pos == len(underwater) - 1):
            end_pos = pos - 1 if not is_underwater else pos
            segment = drawdown.iloc[start_pos : end_pos + 1]
            if not segment.empty:
                trough = segment.idxmin()
                peak_window = values.loc[:trough]
                peak = peak_window.idxmax()
                recovery_date = None if is_underwater and pos == len(underwater) - 1 else values.index[pos]
                periods.append(
                    {
                        "peak_date": pd.Timestamp(peak).date().isoformat(),
                        "trough_date": pd.Timestamp(trough).date().isoformat(),
                        "recovery_date": pd.Timestamp(recovery_date).date().isoformat() if recovery_date is not None else None,
                        "max_drawdown": float(segment.loc[trough]),
                        "peak_equity": float(values.loc[peak]),
                        "trough_equity": float(values.loc[trough]),
                        "days_to_trough": int(max((pd.Timestamp(trough) - pd.Timestamp(peak)).days, 0)),
                        "days_to_recovery": (
                            int(max((pd.Timestamp(recovery_date) - pd.Timestamp(peak)).days, 0)) if recovery_date is not None else None
                        ),
                    }
                )
            start_pos = None
    return sorted(periods, key=lambda item: item["max_drawdown"])[:limit]


def _top_held_instruments(holdings: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if holdings.empty or "instrument" not in holdings.columns or "date" not in holdings.columns:
        return []
    rows = holdings.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["date"])
    if rows.empty:
        return []
    grouped = rows.groupby("instrument")
    held_days = grouped["date"].nunique().sort_values(ascending=False).head(limit)
    return [{"instrument": str(index), "holding_days": int(value)} for index, value in held_days.items()]


def _cost_by_group(trades: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    if trades.empty or column not in trades.columns:
        return {}
    rows = trades.copy()
    rows[column] = rows[column].fillna("").astype(str).str.strip().str.lower()
    result: dict[str, dict[str, float]] = {}
    for key, group in rows.groupby(column):
        if not key:
            continue
        result[str(key)] = {
            "commission_cost": _sum_numeric(group, "commission_cost"),
            "tax_cost": _sum_numeric(group, "tax_cost"),
            "transfer_fee_cost": _sum_numeric(group, "transfer_fee_cost"),
            "slippage_cost": _sum_numeric(group, "slippage_cost"),
            "trade_cost": float(
                _sum_numeric(group, "commission_cost")
                + _sum_numeric(group, "tax_cost")
                + _sum_numeric(group, "transfer_fee_cost")
                + _sum_numeric(group, "slippage_cost")
            ),
        }
    return result


def _trade_notional(trades: pd.DataFrame) -> pd.Series:
    if trades.empty or "shares" not in trades.columns or "price" not in trades.columns:
        return pd.Series(dtype=float)
    shares = pd.to_numeric(trades["shares"], errors="coerce").fillna(0.0).abs()
    prices = pd.to_numeric(trades["price"], errors="coerce").fillna(0.0).abs()
    return shares * prices


def _trade_dates(trades: pd.DataFrame) -> pd.Series:
    if trades.empty or "date" not in trades.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(trades["date"], errors="coerce")


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(_numeric_column(frame, column).sum())


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _capacity_warning_count(trades: pd.DataFrame) -> int:
    if trades.empty or "capacity_warning" not in trades.columns:
        return 0
    return int(pd.Series(trades["capacity_warning"]).fillna(False).astype(bool).sum())


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = _normalized_column(frame, column).value_counts()
    return {str(key): int(value) for key, value in counts.items() if str(key)}


def _normalized_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame[column].fillna("").astype(str).str.strip().str.lower()


def _index_date_bounds(index: pd.Index) -> tuple[str | None, str | None]:
    values = index.get_level_values(0) if isinstance(index, pd.MultiIndex) and index.nlevels else index
    dates = pd.to_datetime(values, errors="coerce")
    dates = dates[~pd.isna(dates)]
    if not len(dates):
        return None, None
    return pd.Timestamp(dates.min()).date().isoformat(), pd.Timestamp(dates.max()).date().isoformat()


if __name__ == "__main__":
    main()
