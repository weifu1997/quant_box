"""模块说明：提供 run_ml_strategy 命令行入口。"""

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
from src.data_coverage import (
    build_price_data_gaps,
    build_skipped_months,
    build_yearly_equity_coverage,
    price_coverage_summary,
)
from src.factor_calculator import factor_cache_columns, load_or_compute_factors
from src.feature_extension import append_daily_basic_features, append_price_derived_features
from src.market_regime import (
    aggregate_regime_performance,
    defensive_exposure_schedule,
    detect_market_regime,
    detect_reporting_regime,
    summarize_regime_performance,
)
from src.ml_strategy import build_ml_scores
from src.neutralization import load_daily_basic, load_industry_map, neutralize_score_panel
from src.score_blending import apply_regime_score_blend
from src.selection_constraints import apply_selection_constraints_to_backtest_config
from src.trading_calendar import resolve_target_date_value


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    config = load_config()
    parser = argparse.ArgumentParser(description="Run rolling Alpha158 ML stock-selection backtest.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--model-type", choices=["auto", "lightgbm", "xgboost", "sklearn_gbdt", "ridge_numpy"], help="Override model type.")
    parser.add_argument(
        "--model-objective",
        choices=["regression", "classification", "ranking"],
        help="Override model training objective.",
    )
    parser.add_argument("--train-years", type=int, help="Override rolling training window in years.")
    parser.add_argument(
        "--training-start-date",
        help="Override ML training history start date. Use 'auto' to derive start-date minus train-years.",
    )
    parser.add_argument(
        "--label-mode",
        choices=[
            "raw_return",
            "cross_sectional_rank",
            "cross_sectional_zscore",
            "cross_sectional_demean",
            "cross_sectional_long_short",
            "cross_sectional_top_quantile",
        ],
        help="Override ML training label transformation.",
    )
    parser.add_argument(
        "--label-return-adjustment",
        choices=["raw", "vol_adjusted"],
        help="Override forward-return adjustment before label transformation.",
    )
    parser.add_argument("--ensemble-window", type=int, help="Override recent-model ensemble window.")
    parser.add_argument("--max-train-rows", type=int, help="Override sampled rows per monthly training window.")
    parser.add_argument("--min-train-rows", type=int, help="Override minimum clean training rows.")
    parser.add_argument("--feature-limit", type=int, help="Use the first N Alpha158 columns for a faster run.")
    parser.add_argument("--enable-feature-ic-evolution", action="store_true", help="Select and weight ML features by recent rolling IC.")
    parser.add_argument("--score-weighted", action="store_true", help="Allocate selected stocks by score instead of equal weight.")
    parser.add_argument("--disable-defensive", action="store_true", help="Disable real-time defensive exposure timing.")
    parser.add_argument("--enable-neutralization", action="store_true", help="Neutralize ML scores by industry and market cap when data is available.")
    parser.add_argument("--enable-feature-extensions", action="store_true", help="Append lagged Tushare daily_basic fields to ML features when cached data is available.")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    config["data"]["end_date"] = end_date
    ml_cfg = config.setdefault("ml_strategy", {})
    ml_cfg["enabled"] = True
    if args.model_type:
        ml_cfg["model_type"] = args.model_type
    if args.model_objective:
        ml_cfg["model_objective"] = args.model_objective
    if args.train_years is not None:
        ml_cfg["train_years"] = args.train_years
    if args.training_start_date:
        ml_cfg["training_start_date"] = args.training_start_date
    if args.label_mode:
        ml_cfg["label_mode"] = args.label_mode
    if args.label_return_adjustment:
        ml_cfg["label_return_adjustment"] = args.label_return_adjustment
    if args.ensemble_window is not None:
        ml_cfg["ensemble_window"] = args.ensemble_window
    if args.max_train_rows is not None:
        ml_cfg["max_train_rows"] = args.max_train_rows
    if args.min_train_rows is not None:
        ml_cfg["min_train_rows"] = args.min_train_rows
    if args.feature_limit is not None:
        ml_cfg["feature_limit"] = args.feature_limit
    if args.enable_feature_ic_evolution:
        ml_cfg["feature_ic_evolution"] = True
    if args.score_weighted:
        ml_cfg["score_weighted"] = True
    if args.disable_defensive:
        config.setdefault("defensive_timing", {})["enabled"] = False
    if args.enable_neutralization:
        config.setdefault("neutralization", {})["enabled"] = True
    if args.enable_feature_extensions:
        config.setdefault("feature_extensions", {})["enabled"] = True

    training_start_date = _resolve_training_start_date(args.start_date, ml_cfg)
    ml_cfg["training_data_start_date"] = training_start_date
    ml_cfg["signal_start_date"] = args.start_date
    ml_cfg["signal_end_date"] = end_date

    price_file = resolve_path(args.price_file)
    if not price_file.exists():
        raise FileNotFoundError(f"Price file not found: {price_file}. Run scripts/run_convert_data.py first.")
    prices = pd.read_parquet(price_file)
    coverage_summary = price_coverage_summary(prices, args.start_date, end_date)
    coverage_summary["training_data_start_date"] = training_start_date
    coverage_summary["actual_training_price_start"] = _actual_price_start(prices, training_start_date, end_date)
    data_gaps = build_price_data_gaps(prices, args.start_date, end_date)
    factor_columns = _requested_ml_factor_columns(args.factor_file, ml_cfg)
    logger.info("Loading %s factor columns for rolling ML.", "all" if factor_columns is None else len(factor_columns))
    factors = _load_cached_or_compute_factors(training_start_date, end_date, args.factor_file, factor_columns)
    coverage_summary["actual_factor_start"] = _actual_factor_start(factors)
    daily_basic = load_daily_basic(config.get("data", {}).get("daily_basic_file", "data/factors/daily_basic.parquet"))
    industry_map = load_industry_map(config.get("data", {}).get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    factors, daily_basic_extension_summary = append_daily_basic_features(
        factors,
        daily_basic,
        config.get("feature_extensions", {}),
    )
    factors, price_extension_summary = append_price_derived_features(
        factors,
        prices,
        config.get("feature_extensions", {}),
    )
    feature_extension_summary = _feature_extension_summary(daily_basic_extension_summary, price_extension_summary)

    ml_result = build_ml_scores(
        factors,
        prices,
        config,
        industry_map=industry_map,
        daily_basic=daily_basic,
    )
    if ml_result.scores.empty:
        raise RuntimeError("Rolling ML strategy produced no usable scores.")
    neutralization_summary = {"enabled": False, "dates_neutralized": 0, "industry_dates": 0, "market_cap_dates": 0}
    if bool(config.get("neutralization", {}).get("enabled", False)):
        neutralized_scores, neutralization_summary = neutralize_score_panel(
            ml_result.scores,
            industry_map=industry_map,
            daily_basic=daily_basic,
            config=config.get("neutralization", {}),
        )
        ml_result.scores = neutralized_scores

    timing_regimes = detect_market_regime(prices, config)
    reporting_regimes = detect_reporting_regime(prices, config)
    ml_result.scores, score_blend_summary = apply_regime_score_blend(
        ml_result.scores,
        factors,
        timing_regimes,
        config.get("regime_score_blend", {}),
    )
    exposure = defensive_exposure_schedule(timing_regimes, config, pd.Index(pd.to_datetime(prices.index)))
    bt_config = {**config["backtest"], **config["strategy"]}
    bt_config["top_n"] = int(ml_cfg.get("top_n", bt_config.get("top_n", 15)))
    bt_config["score_weighted"] = bool(ml_cfg.get("score_weighted", False))
    bt_config["exposure_schedule"] = exposure
    bt_config["exposure_rebalance_threshold"] = float(config.get("defensive_timing", {}).get("exposure_rebalance_threshold", 0.05))
    bt_config = apply_selection_constraints_to_backtest_config(bt_config, config)

    result = run_backtest(ml_result.scores, prices, args.start_date, end_date, bt_config)
    yearly = _yearly_stats(result.equity_curve, bt_config)
    regime_stats = summarize_regime_performance(result.equity_curve, reporting_regimes, bt_config)
    regime_summary = aggregate_regime_performance(regime_stats)
    skipped_months = build_skipped_months(ml_result.diagnostics)
    yearly_coverage = build_yearly_equity_coverage(result.equity_curve, args.start_date, end_date)

    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = out_dir / "ml_strategy_equity.csv"
    holdings_path = out_dir / "ml_strategy_holdings.csv"
    trades_path = out_dir / "ml_strategy_trades.csv"
    metrics_path = out_dir / "ml_strategy_metrics.json"
    scores_path = out_dir / "ml_strategy_scores.parquet"
    diagnostics_path = out_dir / "ml_strategy_training_diagnostics.csv"
    yearly_path = out_dir / "ml_strategy_yearly.csv"
    regime_path = out_dir / "ml_strategy_regime_stats.csv"
    regime_summary_path = out_dir / "ml_strategy_regime_summary.csv"
    data_gaps_path = out_dir / "ml_strategy_data_gaps.csv"
    skipped_months_path = out_dir / "ml_strategy_skipped_months.csv"
    yearly_coverage_path = out_dir / "ml_strategy_yearly_coverage.csv"
    report_path = out_dir / "ml_strategy_report.md"
    svg_path = out_dir / "ml_strategy_equity_curve.svg"

    result.equity_curve.to_csv(equity_path, encoding="utf-8-sig")
    result.holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(result.metrics, indent=2, default=str), encoding="utf-8")
    scores_frame = ml_result.scores.rename("score").to_frame()
    scores_frame.attrs = {}
    scores_frame.to_parquet(scores_path)
    ml_result.diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    regime_stats.to_csv(regime_path, index=False, encoding="utf-8-sig")
    regime_summary.to_csv(regime_summary_path, index=False, encoding="utf-8-sig")
    data_gaps.to_csv(data_gaps_path, index=False, encoding="utf-8-sig")
    skipped_months.to_csv(skipped_months_path, index=False, encoding="utf-8-sig")
    yearly_coverage.to_csv(yearly_coverage_path, index=False, encoding="utf-8-sig")
    _write_equity_svg(result.equity_curve, svg_path)
    report_path.write_text(
        _markdown_report(
            result.metrics,
            ml_result.diagnostics,
            yearly,
            regime_summary,
            coverage_summary,
            data_gaps,
            skipped_months,
            yearly_coverage,
            neutralization_summary,
            feature_extension_summary,
            score_blend_summary,
            config,
            svg_path.name,
        ),
        encoding="utf-8",
    )

    logger.info("ML strategy backtest finished.")
    for key, value in result.metrics.items():
        logger.info("%s: %.6f", key, value)
    logger.info("Report written to %s", report_path)


def _requested_ml_factor_columns(factor_file: str, ml_cfg: dict) -> list[str] | None:
    """函数说明：处理 requested_ml_factor_columns 的内部辅助逻辑。"""
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
    """函数说明：加载 load_cached_or_compute_factors 的内部辅助逻辑。"""
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


def _resolve_training_start_date(backtest_start_date: str, ml_cfg: dict) -> str:
    """函数说明：解析 resolve_training_start_date 的内部辅助逻辑。"""
    configured = str(ml_cfg.get("training_start_date", "auto")).strip()
    if configured and configured.lower() not in {"auto", "none", "null"}:
        return pd.Timestamp(configured).date().isoformat()
    train_years = int(ml_cfg.get("train_years", 3))
    return (pd.Timestamp(backtest_start_date).normalize() - pd.DateOffset(years=train_years)).date().isoformat()


def _actual_price_start(prices: pd.DataFrame, start_date: str, end_date: str) -> str:
    """函数说明：处理 actual_price_start 的内部辅助逻辑。"""
    if prices.empty:
        return ""
    dates = pd.to_datetime(prices.index).normalize()
    mask = (dates >= pd.Timestamp(start_date).normalize()) & (dates <= pd.Timestamp(end_date).normalize())
    selected = pd.DatetimeIndex(dates[mask]).dropna()
    if selected.empty:
        return ""
    return selected.min().date().isoformat()


def _actual_factor_start(factors: pd.DataFrame) -> str:
    """函数说明：处理 actual_factor_start 的内部辅助逻辑。"""
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return ""
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    if len(dates) == 0:
        return ""
    return pd.DatetimeIndex(dates).min().date().isoformat()


def _slice_factor_dates(factors: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """函数说明：处理 slice_factor_dates 的内部辅助逻辑。"""
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return factors
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    mask = (dates >= pd.Timestamp(start_date).normalize()) & (dates <= pd.Timestamp(end_date).normalize())
    return factors[mask]


def _yearly_stats(equity_curve: pd.Series, config: dict) -> pd.DataFrame:
    """函数说明：处理 yearly_stats 的内部辅助逻辑。"""
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
    """函数说明：写入 write_equity_svg 的内部辅助逻辑。"""
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
    coverage_summary: dict[str, object],
    data_gaps: pd.DataFrame,
    skipped_months: pd.DataFrame,
    yearly_coverage: pd.DataFrame,
    neutralization_summary: dict[str, object],
    feature_extension_summary: dict[str, object],
    score_blend_summary: dict[str, object],
    config: dict,
    equity_svg_name: str,
) -> str:
    """函数说明：处理 markdown_report 的内部辅助逻辑。"""
    completed = diagnostics[pd.to_numeric(diagnostics.get("train_rows_used", 0), errors="coerce").fillna(0) > 0]
    no_lookahead = bool(completed["no_lookahead"].all()) if not completed.empty and "no_lookahead" in completed else False
    model_counts = completed["model_used"].value_counts().to_dict() if "model_used" in completed else {}
    ml_cfg = config.get("ml_strategy", {})
    fundamental_enabled = bool(ml_cfg.get("fundamental_factors_enabled", False))
    skip_reason_counts = _skip_reason_counts(skipped_months)
    yearly_gate = _yearly_quality_gate(yearly, config)
    quality_gate = _quality_gate(metrics, yearly_gate, config)
    ensemble_summary = _ensemble_summary(completed)
    feature_summary = _feature_evolution_summary(completed)
    warmup_summary = _warmup_summary(diagnostics, coverage_summary, config)
    lines = [
        "# Rolling Alpha158 ML Strategy Report",
        "",
        f"![Equity curve]({equity_svg_name})",
        "",
        "## Backtest Metrics",
        "",
        f"- Total return: {metrics.get('total_return', 0.0):.2%}",
        f"- Annual return: {metrics.get('annual_return', 0.0):.2%}",
        f"- Max drawdown: {metrics.get('max_drawdown', 0.0):.2%}",
        f"- Sharpe: {metrics.get('sharpe', 0.0):.2f}",
        f"- Calmar: {metrics.get('calmar', 0.0):.2f}",
        f"- Annual turnover: {metrics.get('annual_turnover', 0.0):.2f}",
        f"- Annual trade cost ratio: {metrics.get('annual_trade_cost_ratio', 0.0):.2%}",
        "",
        "## Strategy Quality Gate",
        "",
        f"- Status: {'PASS' if quality_gate['passed'] else 'FAIL'}",
        f"- Annual return target: {quality_gate['target_annual_return']:.2%}",
        f"- Max drawdown limit: {quality_gate['max_drawdown_limit']:.2%}",
        f"- Annual return actual: {quality_gate['annual_return']:.2%}",
        f"- Max drawdown actual: {quality_gate['max_drawdown']:.2%}",
        f"- Issues: {quality_gate['issues'] if quality_gate['issues'] else 'none'}",
        f"- Min yearly annual return target: {yearly_gate['min_yearly_annual_return']:.2%}",
        f"- Yearly max drawdown limit: {yearly_gate['max_drawdown_limit']:.2%}",
        f"- Years below return target: {yearly_gate['years_below_return_target']}",
        f"- Years breaching drawdown limit: {yearly_gate['years_breaching_drawdown_limit']}",
        "",
        "## Training Warmup",
        "",
        f"- Requested start date: {warmup_summary['requested_start']}",
        f"- Required warmup start date: {warmup_summary['required_warmup_start']}",
        f"- Training data load start date: {warmup_summary['training_data_start']}",
        f"- Actual price start date: {warmup_summary['actual_price_start']}",
        f"- Actual factor start date: {warmup_summary['actual_factor_start']}",
        f"- First completed ML signal date: {warmup_summary['first_completed_signal_date']}",
        f"- Rolling train years: {warmup_summary['train_years']}",
        f"- Warmup data starts early enough for first requested year: {str(warmup_summary['has_full_train_warmup']).lower()}",
        f"- Monthly fits skipped before first completed signal: {warmup_summary['skipped_before_first_completed']}",
        "",
        "## Model Ensemble",
        "",
        f"- Ensemble window: {ml_cfg.get('ensemble_window', 3)}",
        f"- Average ensemble size: {ensemble_summary['average_size']:.2f}",
        f"- Models in ensemble: {ensemble_summary['model_counts']}",
        "",
        "## Feature Evolution",
        "",
        f"- Feature IC evolution enabled: {str(bool(ml_cfg.get('feature_ic_evolution', False))).lower()}",
        f"- Feature IC top K: {ml_cfg.get('feature_ic_top_k', 30)}",
        f"- Average feature count: {feature_summary['average_feature_count']:.2f}",
        f"- Evolved monthly fits: {feature_summary['evolved_fits']}",
        "",
        "## Feature Extensions",
        "",
        f"- Enabled: {str(bool(feature_extension_summary.get('enabled', False))).lower()}",
        f"- Daily basic features added: {feature_extension_summary.get('daily_basic_features_added', feature_extension_summary.get('features_added', 0))}",
        f"- Daily basic dates matched: {feature_extension_summary.get('daily_basic_dates_matched', feature_extension_summary.get('dates_matched', 0))}",
        f"- Daily basic lag days: {feature_extension_summary.get('lag_days', config.get('feature_extensions', {}).get('daily_basic_lag_days', 1))}",
        f"- Fields: {feature_extension_summary.get('fields', [])}",
        f"- Price-derived features added: {feature_extension_summary.get('price_features_added', 0)}",
        f"- Price-derived dates matched: {feature_extension_summary.get('price_dates_matched', 0)}",
        f"- Price feature lag sessions: {feature_extension_summary.get('price_lag_sessions', config.get('feature_extensions', {}).get('price_feature_lag_sessions', 1))}",
        f"- Price fields: {feature_extension_summary.get('price_fields', [])}",
        "",
        "## Regime Score Blend",
        "",
        f"- Enabled: {str(bool(score_blend_summary.get('enabled', False))).lower()}",
        f"- Dates blended: {score_blend_summary.get('dates_blended', 0)}",
        f"- Average defensive components: {float(score_blend_summary.get('average_components', 0.0)):.2f}",
        f"- Bull defensive weight: {float(score_blend_summary.get('bull_defensive_weight', config.get('regime_score_blend', {}).get('bull_defensive_weight', 0.0))):.2f}",
        f"- Sideways defensive weight: {float(score_blend_summary.get('sideways_defensive_weight', config.get('regime_score_blend', {}).get('sideways_defensive_weight', 0.5))):.2f}",
        f"- Bear defensive weight: {float(score_blend_summary.get('bear_defensive_weight', config.get('regime_score_blend', {}).get('bear_defensive_weight', 1.0))):.2f}",
        "",
        "## Neutralization",
        "",
        f"- Enabled: {str(bool(neutralization_summary.get('enabled', False))).lower()}",
        f"- Dates neutralized: {neutralization_summary.get('dates_neutralized', 0)}",
        f"- Industry dates: {neutralization_summary.get('industry_dates', 0)}",
        f"- Market-cap dates: {neutralization_summary.get('market_cap_dates', 0)}",
        f"- Market-cap field: {neutralization_summary.get('market_cap_field', config.get('neutralization', {}).get('market_cap_field', 'circ_mv'))}",
        "",
        "## Factor Source",
        "",
        "- Factor source: Alpha158 price-volume features",
        f"- Fundamental factors used: {str(fundamental_enabled).lower()}",
        "- Fundamental lag applied: not applicable" if not fundamental_enabled else f"- Fundamental lag applied: {ml_cfg.get('fundamental_lag_days', 90)} days",
        "",
        "## Regime Definitions",
        "",
        f"- Timing regime: realtime lagged, lag_days={config.get('market_regime', {}).get('lag_days', 1)}",
        f"- Reporting regime: objective reporting only, lag_days={config.get('reporting_regime', {}).get('lag_days', 0)}",
        "",
        "## No-Lookahead Audit",
        "",
        f"- Completed monthly model fits: {len(completed)}",
        f"- All completed fits satisfy max_label_end < signal_date: {no_lookahead}",
        f"- Label horizon sessions: {ml_cfg.get('label_horizon_sessions', 20)}",
        f"- Label mode: {ml_cfg.get('label_mode', 'raw_return')}",
        f"- Label return adjustment: {ml_cfg.get('label_return_adjustment', 'raw')}",
        f"- Label volatility window: {ml_cfg.get('label_volatility_window', 20)}",
        f"- Label min cross-section obs: {ml_cfg.get('label_min_cross_section_obs', 20)}",
        f"- Label top quantile: {ml_cfg.get('label_top_quantile', 0.20)}",
        f"- Label bottom quantile: {ml_cfg.get('label_bottom_quantile', 0.20)}",
        f"- Training neutralization enabled: {str(bool(ml_cfg.get('training_neutralization', {}).get('enabled', False))).lower()}",
        f"- Training neutralization industry: {str(bool(ml_cfg.get('training_neutralization', {}).get('industry', False))).lower()}",
        f"- Training neutralization market cap: {str(bool(ml_cfg.get('training_neutralization', {}).get('market_cap', False))).lower()}",
        f"- Rolling train years: {ml_cfg.get('train_years', 3)}",
        f"- Feature limit from Alpha158: {ml_cfg.get('feature_limit')}",
        f"- Model usage: {model_counts}",
        f"- Model objective: {ml_cfg.get('model_objective', 'regression')}",
        f"- Skipped monthly model fits: {len(skipped_months)}",
        "",
        "## Data Coverage",
        "",
        f"- Requested date range: {coverage_summary.get('start_date')} to {coverage_summary.get('end_date')}",
        f"- Actual price date range: {coverage_summary.get('actual_start', '')} to {coverage_summary.get('actual_end', '')}",
        f"- Price dates: {coverage_summary.get('price_dates', 0)}",
        f"- Symbols: {coverage_summary.get('symbols', 0)}",
        f"- Gap dates: {coverage_summary.get('gap_dates', 0)}",
        f"- Mean price coverage: {float(coverage_summary.get('mean_coverage', 0.0)):.2%}",
        f"- Min price coverage: {float(coverage_summary.get('min_coverage', 0.0)):.2%}",
        "",
        "## Skipped Months",
        "",
        _markdown_table(skip_reason_counts) if not skip_reason_counts.empty else "No skipped monthly model fits.",
        "",
        "## Year Coverage",
        "",
        _markdown_table(yearly_coverage) if not yearly_coverage.empty else "No year coverage rows.",
        "",
        "## Yearly Returns",
        "",
        _markdown_table(yearly[["year", "total_return", "annual_return", "max_drawdown"]]) if not yearly.empty else "No yearly rows.",
        "",
        "## Regime Summary",
        "",
        _markdown_table(regime_summary) if not regime_summary.empty else "No regime rows.",
        "",
    ]
    return "\n".join(lines)


def _ensemble_summary(completed: pd.DataFrame) -> dict[str, object]:
    """函数说明：处理 ensemble_summary 的内部辅助逻辑。"""
    if completed.empty:
        return {"average_size": 0.0, "model_counts": {}}
    average_size = float(pd.to_numeric(completed.get("ensemble_size", 0), errors="coerce").fillna(0).mean())
    counts: dict[str, int] = {}
    if "ensemble_models" in completed.columns:
        for value in completed["ensemble_models"].fillna("").astype(str):
            for name in [item.strip() for item in value.split(",") if item.strip()]:
                counts[name] = counts.get(name, 0) + 1
    return {"average_size": average_size, "model_counts": counts}


def _feature_evolution_summary(completed: pd.DataFrame) -> dict[str, object]:
    """函数说明：处理 feature_evolution_summary 的内部辅助逻辑。"""
    if completed.empty:
        return {"average_feature_count": 0.0, "evolved_fits": 0}
    feature_count = pd.to_numeric(completed.get("feature_count", 0), errors="coerce").fillna(0)
    evolved = completed.get("feature_ic_evolved", pd.Series(False, index=completed.index)).astype(bool)
    return {"average_feature_count": float(feature_count.mean()), "evolved_fits": int(evolved.sum())}


def _feature_extension_summary(daily_basic: dict[str, object], price_derived: dict[str, object]) -> dict[str, object]:
    """函数说明：处理 feature_extension_summary 的内部辅助逻辑。"""
    return {
        "enabled": bool(daily_basic.get("enabled", False) or price_derived.get("enabled", False)),
        "daily_basic_features_added": int(daily_basic.get("features_added", 0) or 0),
        "daily_basic_dates_matched": int(daily_basic.get("dates_matched", 0) or 0),
        "lag_days": daily_basic.get("lag_days", 0),
        "fields": daily_basic.get("fields", []),
        "price_features_added": int(price_derived.get("features_added", 0) or 0),
        "price_dates_matched": int(price_derived.get("dates_matched", 0) or 0),
        "price_lag_sessions": price_derived.get("lag_sessions", 0),
        "price_fields": price_derived.get("fields", []),
    }


def _warmup_summary(diagnostics: pd.DataFrame, coverage_summary: dict[str, object], config: dict) -> dict[str, object]:
    """函数说明：处理 warmup_summary 的内部辅助逻辑。"""
    ml_cfg = config.get("ml_strategy", {})
    requested_start = pd.Timestamp(coverage_summary.get("start_date", config.get("data", {}).get("start_date", ""))).normalize()
    actual_price_start_raw = (
        coverage_summary.get("actual_training_price_start")
        or coverage_summary.get("actual_start")
        or coverage_summary.get("start_date")
        or requested_start
    )
    actual_price_start = pd.Timestamp(actual_price_start_raw).normalize()
    train_years = int(ml_cfg.get("train_years", 3))
    required_warmup_start = requested_start - pd.DateOffset(years=train_years)
    training_data_start = coverage_summary.get("training_data_start_date") or ml_cfg.get("training_data_start_date") or required_warmup_start
    actual_factor_start = coverage_summary.get("actual_factor_start") or ""
    completed = pd.DataFrame()
    if not diagnostics.empty and "train_rows_used" in diagnostics.columns:
        used = pd.to_numeric(diagnostics.get("train_rows_used", 0), errors="coerce").fillna(0)
        completed = diagnostics[used > 0]
    first_completed = None
    skipped_before_first = 0
    if not completed.empty and "signal_date" in completed.columns:
        first_completed_ts = pd.to_datetime(completed["signal_date"], errors="coerce").dropna().min()
        if pd.notna(first_completed_ts):
            first_completed = pd.Timestamp(first_completed_ts).date().isoformat()
            if "signal_date" in diagnostics.columns:
                signal_dates = pd.to_datetime(diagnostics["signal_date"], errors="coerce")
                skipped_before_first = int((signal_dates < pd.Timestamp(first_completed_ts)).sum())
    return {
        "requested_start": requested_start.date().isoformat(),
        "actual_price_start": actual_price_start.date().isoformat(),
        "training_data_start": str(pd.Timestamp(training_data_start).date()),
        "actual_factor_start": str(actual_factor_start),
        "required_warmup_start": required_warmup_start.date().isoformat(),
        "first_completed_signal_date": first_completed or "",
        "train_years": train_years,
        "has_full_train_warmup": bool(actual_price_start <= required_warmup_start),
        "skipped_before_first_completed": skipped_before_first,
    }


def _quality_gate(metrics: dict[str, float], yearly_gate: dict[str, object], config: dict) -> dict[str, object]:
    """函数说明：处理 quality_gate 的内部辅助逻辑。"""
    ml_cfg = config.get("ml_strategy", {})
    quality_cfg = config.get("quality", {})
    target_return = float(ml_cfg.get("target_annual_return", quality_cfg.get("target_annual_return", 0.20)))
    drawdown_limit = float(ml_cfg.get("max_drawdown_limit", -0.20))
    annual_return = float(metrics.get("annual_return", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    issues: list[str] = []
    if annual_return < target_return:
        issues.append(f"annual_return_below_target:{annual_return:.4f}<{target_return:.4f}")
    if max_drawdown < drawdown_limit:
        issues.append(f"max_drawdown_breaches_limit:{max_drawdown:.4f}<{drawdown_limit:.4f}")
    if yearly_gate.get("years_below_return_target"):
        issues.append(f"yearly_annual_return_below_target:{yearly_gate['years_below_return_target']}")
    if yearly_gate.get("years_breaching_drawdown_limit"):
        issues.append(f"yearly_max_drawdown_breaches_limit:{yearly_gate['years_breaching_drawdown_limit']}")
    return {
        "passed": not issues,
        "target_annual_return": target_return,
        "max_drawdown_limit": drawdown_limit,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "issues": issues,
    }


def _yearly_quality_gate(yearly: pd.DataFrame, config: dict) -> dict[str, object]:
    """函数说明：处理 yearly_quality_gate 的内部辅助逻辑。"""
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


def _skip_reason_counts(skipped_months: pd.DataFrame) -> pd.DataFrame:
    """函数说明：处理 skip_reason_counts 的内部辅助逻辑。"""
    if skipped_months.empty or "skip_reason" not in skipped_months.columns:
        return pd.DataFrame(columns=["skip_reason", "months"])
    counts = skipped_months["skip_reason"].fillna("").astype(str).str.strip()
    counts = counts[counts != ""].value_counts().rename_axis("skip_reason").reset_index(name="months")
    return counts


def _markdown_table(frame: pd.DataFrame) -> str:
    """函数说明：处理 markdown_table 的内部辅助逻辑。"""
    if frame.empty:
        return ""
    headers = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        values = []
        for header, value in zip(headers, row.tolist()):
            if header in {"year", "days", "segments", "months"} and pd.notna(value):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
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
