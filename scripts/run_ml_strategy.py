from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import factor_cache_columns, load_or_compute_factors
from src.market_regime import (
    aggregate_regime_performance,
    defensive_exposure_schedule,
    detect_market_regime,
    summarize_regime_performance,
)
from src.ml_strategy import build_ml_scores
from src.trading_calendar import resolve_target_date_value


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run rolling Alpha158 ML stock-selection backtest.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--model-type", choices=["auto", "lightgbm", "xgboost", "sklearn_gbdt", "ridge_numpy"], help="Override model type.")
    parser.add_argument("--train-years", type=int, help="Override rolling training window in years.")
    parser.add_argument("--max-train-rows", type=int, help="Override sampled rows per monthly training window.")
    parser.add_argument("--min-train-rows", type=int, help="Override minimum clean training rows.")
    parser.add_argument("--feature-limit", type=int, help="Use the first N Alpha158 columns for a faster run.")
    parser.add_argument("--score-weighted", action="store_true", help="Allocate selected stocks by score instead of equal weight.")
    parser.add_argument("--disable-defensive", action="store_true", help="Disable real-time defensive exposure timing.")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    config["data"]["end_date"] = end_date
    ml_cfg = config.setdefault("ml_strategy", {})
    ml_cfg["enabled"] = True
    if args.model_type:
        ml_cfg["model_type"] = args.model_type
    if args.train_years is not None:
        ml_cfg["train_years"] = args.train_years
    if args.max_train_rows is not None:
        ml_cfg["max_train_rows"] = args.max_train_rows
    if args.min_train_rows is not None:
        ml_cfg["min_train_rows"] = args.min_train_rows
    if args.feature_limit is not None:
        ml_cfg["feature_limit"] = args.feature_limit
    if args.score_weighted:
        ml_cfg["score_weighted"] = True
    if args.disable_defensive:
        config.setdefault("defensive_timing", {})["enabled"] = False

    price_file = resolve_path(args.price_file)
    if not price_file.exists():
        raise FileNotFoundError(f"Price file not found: {price_file}. Run scripts/run_convert_data.py first.")
    prices = pd.read_parquet(price_file)
    factor_columns = _requested_ml_factor_columns(args.factor_file, ml_cfg)
    logger.info("Loading %s factor columns for rolling ML.", "all" if factor_columns is None else len(factor_columns))
    factors = _load_cached_or_compute_factors(args.start_date, end_date, args.factor_file, factor_columns)

    ml_result = build_ml_scores(factors, prices, config)
    if ml_result.scores.empty:
        raise RuntimeError("Rolling ML strategy produced no usable scores.")

    regimes = detect_market_regime(prices, config)
    exposure = defensive_exposure_schedule(regimes, config, pd.Index(pd.to_datetime(prices.index)))
    bt_config = {**config["backtest"], **config["strategy"]}
    bt_config["top_n"] = int(ml_cfg.get("top_n", bt_config.get("top_n", 15)))
    bt_config["score_weighted"] = bool(ml_cfg.get("score_weighted", False))
    bt_config["exposure_schedule"] = exposure
    bt_config["exposure_rebalance_threshold"] = float(config.get("defensive_timing", {}).get("exposure_rebalance_threshold", 0.05))

    result = run_backtest(ml_result.scores, prices, args.start_date, end_date, bt_config)
    yearly = _yearly_stats(result.equity_curve, bt_config)
    regime_stats = summarize_regime_performance(result.equity_curve, regimes, bt_config)
    regime_summary = aggregate_regime_performance(regime_stats)

    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = out_dir / "ml_strategy_equity.csv"
    holdings_path = out_dir / "ml_strategy_holdings.csv"
    trades_path = out_dir / "ml_strategy_trades.csv"
    metrics_path = out_dir / "ml_strategy_metrics.json"
    diagnostics_path = out_dir / "ml_strategy_training_diagnostics.csv"
    yearly_path = out_dir / "ml_strategy_yearly.csv"
    regime_path = out_dir / "ml_strategy_regime_stats.csv"
    regime_summary_path = out_dir / "ml_strategy_regime_summary.csv"
    report_path = out_dir / "ml_strategy_report.md"
    svg_path = out_dir / "ml_strategy_equity_curve.svg"

    result.equity_curve.to_csv(equity_path, encoding="utf-8-sig")
    result.holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(result.metrics, indent=2, default=str), encoding="utf-8")
    ml_result.diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    regime_stats.to_csv(regime_path, index=False, encoding="utf-8-sig")
    regime_summary.to_csv(regime_summary_path, index=False, encoding="utf-8-sig")
    _write_equity_svg(result.equity_curve, svg_path)
    report_path.write_text(
        _markdown_report(result.metrics, ml_result.diagnostics, yearly, regime_summary, config, svg_path.name),
        encoding="utf-8",
    )

    logger.info("ML strategy backtest finished.")
    for key, value in result.metrics.items():
        logger.info("%s: %.6f", key, value)
    logger.info("Report written to %s", report_path)


def _requested_ml_factor_columns(factor_file: str, ml_cfg: dict) -> list[str] | None:
    available = factor_cache_columns(factor_file)
    if not available:
        return None
    configured = ml_cfg.get("feature_columns")
    if configured:
        requested = [str(column) for column in configured]
        return [column for column in requested if column in available]
    feature_limit = ml_cfg.get("feature_limit")
    if feature_limit is None:
        return None
    return available[: max(1, int(feature_limit))]


def _load_cached_or_compute_factors(
    start_date: str,
    end_date: str,
    factor_file: str,
    factor_columns: list[str] | None,
) -> pd.DataFrame:
    factor_path = resolve_path(factor_file)
    if factor_path.exists():
        columns = None if factor_columns is None else [*factor_columns, "datetime", "instrument"]
        try:
            factors = pd.read_parquet(factor_path, columns=columns)
        except (KeyError, ValueError):
            factors = pd.read_parquet(factor_path)
            if factor_columns is not None:
                factors = factors[[column for column in factor_columns if column in factors.columns]]
        return _slice_factor_dates(factors, start_date, end_date)
    return load_or_compute_factors(start_date, end_date, cache_file=factor_file, columns=factor_columns)


def _slice_factor_dates(factors: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return factors
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    mask = (dates >= pd.Timestamp(start_date).normalize()) & (dates <= pd.Timestamp(end_date).normalize())
    return factors[mask]


def _yearly_stats(equity_curve: pd.Series, config: dict) -> pd.DataFrame:
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


def _write_equity_svg(equity_curve: pd.Series, path: Path) -> None:
    width, height = 960, 360
    margin = 36
    if equity_curve.empty or len(equity_curve) < 2:
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>', encoding="utf-8")
        return
    values = equity_curve.astype(float).to_numpy()
    values = values / values[0]
    x = np.linspace(margin, width - margin, len(values))
    y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
    y_span = max(y_max - y_min, 1e-9)
    y = height - margin - (values - y_min) / y_span * (height - 2 * margin)
    points = " ".join(f"{xv:.2f},{yv:.2f}" for xv, yv in zip(x, y))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#cccccc"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#cccccc"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{points}"/>
<text x="{margin}" y="24" font-family="Arial" font-size="16">ML strategy equity curve, normalized</text>
<text x="{margin}" y="{height - 10}" font-family="Arial" font-size="12">{equity_curve.index.min().date()} to {equity_curve.index.max().date()}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _markdown_report(
    metrics: dict[str, float],
    diagnostics: pd.DataFrame,
    yearly: pd.DataFrame,
    regime_summary: pd.DataFrame,
    config: dict,
    equity_svg_name: str,
) -> str:
    completed = diagnostics[pd.to_numeric(diagnostics.get("train_rows_used", 0), errors="coerce").fillna(0) > 0]
    no_lookahead = bool(completed["no_lookahead"].all()) if not completed.empty and "no_lookahead" in completed else False
    model_counts = completed["model_used"].value_counts().to_dict() if "model_used" in completed else {}
    ml_cfg = config.get("ml_strategy", {})
    lines = [
        "# Rolling Alpha158 ML Strategy Report",
        "",
        f"![Equity curve]({equity_svg_name})",
        "",
        "## Backtest Metrics",
        "",
        f"- Annual return: {metrics.get('annual_return', 0.0):.2%}",
        f"- Max drawdown: {metrics.get('max_drawdown', 0.0):.2%}",
        f"- Sharpe: {metrics.get('sharpe', 0.0):.2f}",
        f"- Calmar: {metrics.get('calmar', 0.0):.2f}",
        "",
        "## No-Lookahead Audit",
        "",
        f"- Completed monthly model fits: {len(completed)}",
        f"- All completed fits satisfy max_label_end < signal_date: {no_lookahead}",
        f"- Label horizon sessions: {ml_cfg.get('label_horizon_sessions', 20)}",
        f"- Rolling train years: {ml_cfg.get('train_years', 3)}",
        f"- Feature limit from Alpha158: {ml_cfg.get('feature_limit')}",
        f"- Fundamental data lag setting for future fundamental factors: {ml_cfg.get('fundamental_lag_days', 90)} days",
        f"- Model usage: {model_counts}",
        "",
        "## Yearly Returns",
        "",
        _markdown_table(yearly[["year", "total_return", "max_drawdown"]]) if not yearly.empty else "No yearly rows.",
        "",
        "## Regime Summary",
        "",
        _markdown_table(regime_summary) if not regime_summary.empty else "No regime rows.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    headers = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        rows.append(values)
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header_line, divider, *body])


if __name__ == "__main__":
    main()
