"""Run a single-framework fundamental quality backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import requested_factor_columns, yearly_stats
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds, write_audit_outputs
from src.backtest import run_backtest
from src.common import normalize_instrument
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.fundamental_data import build_fundamental_screen, normalize_dividend_frame, normalize_fina_indicator_frame
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.risk_policy import RiskPolicy
from src.scoring import build_strategy_scores
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a monthly fundamental quality ranking strategy.")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--max-turnover", type=int, default=5)
    parser.add_argument("--rank-buffer", type=int, default=20)
    parser.add_argument("--min-total-score", type=float, default=4.0)
    parser.add_argument("--status", default="PASS,WATCH", help="Comma-separated review statuses to keep, or all.")
    parser.add_argument("--combine-mode", choices=["quality_only", "filter_price", "blend"], default="quality_only")
    parser.add_argument("--price-factor-group", default="momentum")
    parser.add_argument("--quality-weight", type=float, default=0.35)
    parser.add_argument("--output-prefix", default="outputs/fundamental_quality_backtest")
    parser.add_argument("--no-defensive-timing", action="store_true")
    args = parser.parse_args()

    config = load_config()
    start_date = args.start_date or config["data"]["start_date"]
    end_date = resolve_target_date_value(args.end_date or config["data"]["end_date"], config=config)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))
    daily_basic = pd.read_parquet(resolve_path(config["data"]["daily_basic_file"]))
    fina_indicator = normalize_fina_indicator_frame(
        pd.read_parquet(resolve_path(config["fundamentals"]["fina_indicator_file"]))
    )
    dividend = normalize_dividend_frame(pd.read_parquet(resolve_path(config["fundamentals"]["dividend_file"])))
    stock_basic = _load_optional_stock_basic(config)

    covered_symbols = sorted(
        set(fina_indicator["ts_code"].dropna().astype(str))
        | set(dividend["ts_code"].dropna().astype(str))
    )
    daily_basic = daily_basic[daily_basic["ts_code"].astype(str).str.upper().isin(covered_symbols)].copy()
    signal_dates = month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
    statuses = None if args.status.strip().lower() == "all" else {item.strip().upper() for item in args.status.split(",") if item.strip()}
    scores, diagnostics = build_fundamental_quality_score_panel(
        config=config,
        signal_dates=signal_dates,
        daily_basic=daily_basic,
        fina_indicator=fina_indicator,
        dividend=dividend,
        stock_basic=stock_basic,
        min_total_score=args.min_total_score,
        statuses=statuses,
    )
    if scores.empty:
        raise ValueError("No fundamental quality scores were generated. Expand fundamental coverage first.")
    if args.combine_mode != "quality_only":
        price_scores = build_price_score_panel(
            config=config,
            prices=prices,
            start_date=start_date,
            end_date=end_date,
            factor_group=args.price_factor_group,
            rebalance_freq="monthly",
        )
        scores = combine_quality_and_price_scores(
            price_scores=price_scores,
            quality_scores=scores,
            mode=args.combine_mode,
            quality_weight=args.quality_weight,
        )
        if scores.empty:
            raise ValueError("No combined price/fundamental scores were generated.")

    bt_config = {
        **config.get("backtest", {}),
        **config.get("strategy", {}),
        "top_n": args.top_n,
        "max_turnover": args.max_turnover,
        "rank_buffer": args.rank_buffer,
        "rebalance_freq": "monthly",
    }
    timing_config = dict(config)
    if args.no_defensive_timing:
        timing_config.setdefault("defensive_timing", {})["enabled"] = False
    bt_config = apply_defensive_timing_to_backtest_config(bt_config, prices, timing_config)
    bt_config = RiskPolicy(config).apply_to_backtest_config(bt_config)
    result = run_backtest(scores, prices, start_date, end_date, bt_config)
    yearly = yearly_stats(result.equity_curve, bt_config)
    return_target, drawdown_limit = goal_thresholds(config)
    audited_yearly, audit_summary = audit_yearly_goal(
        yearly,
        return_target=return_target,
        drawdown_limit=drawdown_limit,
    )

    result.equity_curve.to_csv(Path(str(output_prefix) + "_equity.csv"), encoding="utf-8-sig")
    result.holdings.to_csv(Path(str(output_prefix) + "_holdings.csv"), index=False, encoding="utf-8-sig")
    result.trades.to_csv(Path(str(output_prefix) + "_trades.csv"), index=False, encoding="utf-8-sig")
    audited_yearly.to_csv(Path(str(output_prefix) + "_years.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(Path(str(output_prefix) + "_diagnostics.csv"), index=False, encoding="utf-8-sig")
    metrics_payload = {
        "metrics": result.metrics,
        "audit": audit_summary,
        "score_rows": int(len(scores)),
        "covered_symbols": int(len(covered_symbols)),
        "combine_mode": args.combine_mode,
        "price_factor_group": args.price_factor_group if args.combine_mode != "quality_only" else None,
        "quality_weight": args.quality_weight if args.combine_mode == "blend" else None,
    }
    Path(str(output_prefix) + "_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_audit_outputs(
        output_prefix=Path(str(output_prefix) + "_audit"),
        yearly=audited_yearly,
        summary=audit_summary,
        metrics=result.metrics,
    )
    print(
        f"fundamental_quality annual={result.metrics.get('annual_return', 0.0):.4f} "
        f"dd={result.metrics.get('max_drawdown', 0.0):.4f} "
        f"yearly={audit_summary['year_return_pass_count']}/{audit_summary['year_drawdown_pass_count']} "
        f"goal={audit_summary['is_goal_met']}"
    )
    print(f"wrote prefix: {output_prefix}")


def month_end_signal_dates(
    price_index: pd.Index,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(pd.to_datetime(price_index, errors="coerce")).dropna().normalize().unique().sort_values()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = dates[(dates >= start) & (dates <= end)]
    if dates.empty:
        return []
    by_month = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()
    return [pd.Timestamp(value).normalize() for value in by_month.tolist()]


def build_fundamental_quality_score_panel(
    *,
    config: dict[str, Any],
    signal_dates: list[pd.Timestamp],
    daily_basic: pd.DataFrame,
    fina_indicator: pd.DataFrame,
    dividend: pd.DataFrame,
    stock_basic: pd.DataFrame,
    min_total_score: float,
    statuses: set[str] | None,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    parts: list[pd.Series] = []
    diagnostics: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        screen = build_fundamental_screen(
            config=config,
            as_of=signal_date,
            daily_basic=daily_basic,
            fina_indicator=fina_indicator,
            dividend=dividend,
            prices=pd.DataFrame(),
            stock_basic=stock_basic,
        )
        frame = screen.frame
        score = fundamental_quality_scores(frame, min_total_score=min_total_score, statuses=statuses)
        diagnostics.append(
            {
                "date": signal_date.date().isoformat(),
                "rows": int(len(frame)),
                "scored": int(score.notna().sum()),
                "pass": int(frame["review_status"].eq("PASS").sum()) if "review_status" in frame.columns else 0,
                "watch": int(frame["review_status"].eq("WATCH").sum()) if "review_status" in frame.columns else 0,
            }
        )
        if score.dropna().empty:
            continue
        score.index = pd.MultiIndex.from_product([[signal_date], score.index.astype(str)], names=["date", "instrument"])
        parts.append(score)
    if not parts:
        return pd.Series(dtype=float, name="score"), diagnostics
    return pd.concat(parts).sort_index().rename("score"), diagnostics


def build_price_score_panel(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    factor_group: str,
    rebalance_freq: str,
) -> pd.Series:
    strategy = {**config.get("strategy", {}), "factor_group": factor_group, "rebalance_freq": rebalance_freq}
    scoring_config = dict(config)
    scoring_config["strategy"] = strategy
    factor_file = config["factors"]["cache_file"]
    factor_columns = requested_factor_columns(
        factor_file,
        strategy,
        config.get("dynamic_ic_selector", {}),
        config.get("ml_strategy", {}),
        config.get("regime_score_blend", {}),
        config.get("regime_score_filter", {}),
    )
    factors = load_or_compute_factors(start_date, end_date, cache_file=factor_file, columns=factor_columns)
    price_scores = build_strategy_scores(factors, scoring_config, price_df=prices)
    return resample_signals(price_scores, rebalance_freq)


def combine_quality_and_price_scores(
    *,
    price_scores: pd.Series,
    quality_scores: pd.Series,
    mode: str,
    quality_weight: float = 0.35,
) -> pd.Series:
    if price_scores.empty or quality_scores.empty:
        return pd.Series(dtype=float, name="score")
    price_scores = _normalized_score_panel(price_scores)
    quality_scores = _normalized_score_panel(quality_scores)
    parts: list[pd.Series] = []
    for signal_date, quality_daily in quality_scores.groupby(level=0, sort=True):
        date = pd.Timestamp(signal_date).normalize()
        if date not in price_scores.index.get_level_values(0):
            continue
        price_daily = price_scores.xs(date, level=0, drop_level=True)
        quality_daily = quality_daily.droplevel(0)
        common = price_daily.dropna().index.intersection(quality_daily.dropna().index)
        if common.empty:
            continue
        if mode == "filter_price":
            combined = price_daily.loc[common].astype(float)
        elif mode == "blend":
            combined = _zscore(price_daily.loc[common]) + float(quality_weight) * _zscore(quality_daily.loc[common])
        else:
            raise ValueError(f"Unsupported combine mode: {mode}")
        combined.index = pd.MultiIndex.from_product([[date], combined.index.astype(str)], names=["date", "instrument"])
        parts.append(combined.rename("score"))
    if not parts:
        return pd.Series(dtype=float, name="score")
    return pd.concat(parts).sort_index().rename("score")


def fundamental_quality_scores(
    frame: pd.DataFrame,
    *,
    min_total_score: float = 4.0,
    statuses: set[str] | None = None,
) -> pd.Series:
    if frame.empty or "ts_code" not in frame.columns:
        return pd.Series(dtype=float, name="score")
    data = frame.copy()
    if statuses is not None and "review_status" in data.columns:
        data = data[data["review_status"].astype(str).str.upper().isin(statuses)]
    if "total_score" in data.columns:
        data = data[pd.to_numeric(data["total_score"], errors="coerce") >= float(min_total_score)]
    if data.empty:
        return pd.Series(dtype=float, name="score")

    score = pd.to_numeric(data.get("total_score"), errors="coerce").fillna(0.0) * 10.0
    score = score + _pct_rank(data, "roe", ascending=True)
    score = score + _pct_rank(data, "dividend_yield_ttm", ascending=True)
    score = score + _pct_rank(data, "fcf_yield", ascending=True)
    score = score + _pct_rank(data, "debt_to_assets", ascending=False)
    score.index = data["ts_code"].astype(str).str.upper()
    return score.sort_values(ascending=False).rename("score")


def _pct_rank(frame: pd.DataFrame, column: str, *, ascending: bool) -> pd.Series:
    values = pd.to_numeric(frame.get(column), errors="coerce")
    if not isinstance(values, pd.Series) or values.dropna().empty:
        return pd.Series(0.0, index=frame.index)
    return values.rank(pct=True, ascending=ascending).fillna(0.0)


def _normalized_score_panel(scores: pd.Series) -> pd.Series:
    if not isinstance(scores.index, pd.MultiIndex) or scores.index.nlevels < 2:
        raise ValueError("score panel must use a MultiIndex with date and instrument levels.")
    frame = scores.rename("score").reset_index()
    frame.iloc[:, 0] = pd.to_datetime(frame.iloc[:, 0], errors="coerce").dt.normalize()
    frame.iloc[:, 1] = frame.iloc[:, 1].map(normalize_instrument)
    frame = frame.dropna(subset=[frame.columns[0], frame.columns[1], "score"])
    result = pd.Series(
        pd.to_numeric(frame["score"], errors="coerce").to_numpy(),
        index=pd.MultiIndex.from_frame(frame.iloc[:, :2], names=["date", "instrument"]),
        name="score",
    ).dropna()
    return result[~result.index.duplicated(keep="last")].sort_index()


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not std or pd.isna(std):
        return pd.Series(0.0, index=values.index)
    return ((numeric - numeric.mean()) / std).fillna(0.0)


def _load_optional_stock_basic(config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config.get("data", {}).get("constituents_file", ""))
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
