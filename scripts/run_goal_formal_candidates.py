"""模块说明：提供 run_goal_formal_candidates 命令行入口。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import dated_output_path, requested_factor_columns
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.research_diagnostics import build_research_diagnostics, write_research_diagnostics
from src.risk_policy import RiskPolicy
from src.scoring import build_strategy_scores
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    parser = argparse.ArgumentParser(description="Run selected full-history formal candidates for the active goal.")
    parser.add_argument("--output", default=dated_output_path("goal_formal_candidate_summary"))
    parser.add_argument("--start-index", type=int, default=1, help="1-based candidate index to start from.")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip candidate-level research diagnostics.")
    parser.add_argument("--resume", action="store_true", help="Skip candidates already present in the output CSV.")
    parser.add_argument("--candidates-file", default="", help="Optional JSON file with candidate specs to run.")
    parser.add_argument("--factor-file", default="", help="Factor parquet file to use for scoring.")
    args = parser.parse_args()

    config = load_config()
    factor_file = args.factor_file or config["factors"]["cache_file"]
    start_date = config["data"]["start_date"]
    end_date = resolve_target_date_value(config["data"]["end_date"], config=config)
    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_specs(args.candidates_file or None)
    candidates = candidates[max(0, args.start_index - 1) :]
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]

    factor_cache: dict[tuple[str, ...] | None, pd.DataFrame] = {}
    score_cache: dict[str, pd.Series] = {}
    rows, completed_candidates = _load_existing_candidate_rows(out_path) if args.resume else ([], set())
    for idx, candidate in enumerate(candidates, start=1):
        if candidate["name"] in completed_candidates:
            print(f"{idx}/{len(candidates)} {candidate['name']}: skipped existing result", flush=True)
            continue
        started = time.monotonic()
        print(f"{idx}/{len(candidates)} {candidate['name']}: preparing config", flush=True)
        strategy = _strategy_config(config, candidate)
        factor_config = _scoring_config(config, strategy, candidate)
        factor_columns = requested_factor_columns(
            factor_file,
            strategy,
            factor_config.get("dynamic_ic_selector", {}),
            factor_config.get("ml_strategy", {}),
            factor_config.get("regime_score_blend", {}),
            factor_config.get("regime_score_filter", {}),
        )
        factor_key = None if factor_columns is None else tuple(sorted(factor_columns))
        if factor_key not in factor_cache:
            print(
                f"{idx}/{len(candidates)} {candidate['name']}: loading factors columns="
                f"{'all' if factor_columns is None else len(factor_columns)}",
                flush=True,
            )
            try:
                factor_cache[factor_key] = load_or_compute_factors(
                    start_date,
                    end_date,
                    cache_file=factor_file,
                    columns=factor_columns,
                )
            except Exception as exc:
                row = _candidate_error_row(candidate, started, exc, config.get("quality", {}))
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"{idx}/{len(candidates)} {candidate['name']}: error={exc}", flush=True)
                continue
        scoring_key = _score_key(candidate)
        if scoring_key not in score_cache:
            print(f"{idx}/{len(candidates)} {candidate['name']}: building scores", flush=True)
            scoring_config = factor_config
            try:
                scores = build_strategy_scores(factor_cache[factor_key], scoring_config, price_df=prices)
            except Exception as exc:
                row = _candidate_error_row(candidate, started, exc, config.get("quality", {}))
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"{idx}/{len(candidates)} {candidate['name']}: error={exc}", flush=True)
                continue
            score_cache[scoring_key] = resample_signals(scores, strategy.get("rebalance_freq", "daily"))

        timing_config = _timing_config(config, candidate)
        bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **strategy}, prices, timing_config)
        for key, value in candidate.get("backtest", {}).items():
            if value == "__remove__":
                bt_config.pop(key, None)
            else:
                bt_config[key] = value
        bt_config = RiskPolicy(config).apply_to_backtest_config(bt_config)

        print(f"{idx}/{len(candidates)} {candidate['name']}: running formal backtest", flush=True)
        try:
            result = run_backtest(score_cache[scoring_key], prices, start_date, end_date, bt_config)
        except Exception as exc:
            row = _candidate_error_row(candidate, started, exc, config.get("quality", {}))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"{idx}/{len(candidates)} {candidate['name']}: error={exc}", flush=True)
            continue
        yearly = _yearly_stats(result.equity_curve)
        year_ann_pass, year_dd_pass = _yearly_pass_counts(yearly, config.get("quality", {}))
        year_count = int(len(yearly))
        row = {
            "candidate": candidate["name"],
            "seconds": time.monotonic() - started,
            **candidate.get("recorded_hint", {}),
            **result.metrics,
            "year_count": year_count,
            "year_ann_pass": year_ann_pass,
            "year_dd_pass": year_dd_pass,
        }
        row.update(_quality_flags(row, config.get("quality", {})))
        prefix = out_path.with_name(f"{out_path.stem}_{candidate['name']}")
        print(f"{idx}/{len(candidates)} {candidate['name']}: writing artifacts", flush=True)
        row.update(_write_candidate_artifacts(prefix, result, yearly, prices, config, write_diagnostics=not args.skip_diagnostics))
        Path(str(prefix) + "_metrics.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(
            f"{idx}/{len(candidates)} {candidate['name']}: annual={row['annual_return']:.4f} "
            f"dd={row['max_drawdown']:.4f} acceptable={row['is_acceptable']} yearly={row['year_ann_pass']}/{row['year_dd_pass']}",
            flush=True,
        )

    print(f"Saved summary to {out_path}")


def _load_existing_candidate_rows(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """函数说明：加载 load_existing_candidate_rows 的内部辅助逻辑。"""
    if not path.exists() or path.stat().st_size == 0:
        return [], set()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return [], set()
    if frame.empty or "candidate" not in frame.columns:
        return [], set()
    rows = frame.to_dict(orient="records")
    candidates = {str(value) for value in frame["candidate"].dropna().astype(str)}
    return rows, candidates


CANDIDATE_SPECS_PATH = ROOT / "config" / "goal_formal_candidates.json"


def _candidate_specs(path: str | Path | None = None) -> list[dict[str, Any]]:
    """函数说明：处理 candidate_specs 的内部辅助逻辑。"""
    specs_path = Path(path) if path is not None else CANDIDATE_SPECS_PATH
    if not specs_path.is_absolute():
        specs_path = ROOT / specs_path
    try:
        specs = json.loads(specs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Candidate specs file is invalid JSON: {specs_path}") from exc
    _validate_candidate_specs(specs, specs_path)
    return specs


def _validate_candidate_specs(specs: Any, path: Path) -> None:
    """函数说明：校验 validate_candidate_specs 的内部辅助逻辑。"""
    if not isinstance(specs, list):
        raise ValueError(f"Candidate specs file must contain a list: {path}")
    seen: set[str] = set()
    for idx, candidate in enumerate(specs, start=1):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate #{idx} in {path} must be an object.")
        name = candidate.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Candidate #{idx} in {path} is missing a non-empty name.")
        if name in seen:
            raise ValueError(f"Duplicate candidate name in {path}: {name}")
        seen.add(name)
        strategy = candidate.get("strategy")
        if strategy is not None and not isinstance(strategy, dict):
            raise ValueError(f"Candidate {name} strategy must be an object.")
        backtest = candidate.get("backtest")
        if backtest is not None and not isinstance(backtest, dict):
            raise ValueError(f"Candidate {name} backtest must be an object.")


def _strategy_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 strategy_config 的内部辅助逻辑。"""
    strategy = deepcopy(config["strategy"])
    for key, value in candidate.get("strategy", {}).items():
        if value == "__remove__":
            strategy.pop(key, None)
        else:
            strategy[key] = value
    return strategy


def _scoring_config(config: dict[str, Any], strategy: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 scoring_config 的内部辅助逻辑。"""
    result = deepcopy(config)
    result["strategy"] = strategy
    for key in ["liquidity_filter", "market_regime", "regime_score_blend", "regime_score_filter", "dynamic_ic_selector"]:
        if key in candidate:
            result[key] = {**result.get(key, {}), **candidate[key]}
    return result


def _timing_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 timing_config 的内部辅助逻辑。"""
    result = deepcopy(config)
    if "market_regime" in candidate:
        result["market_regime"] = {**result.get("market_regime", {}), **candidate["market_regime"]}
    result.setdefault("defensive_timing", {})["enabled"] = bool(candidate.get("defensive_timing", True))
    if "defensive_timing_config" in candidate:
        result["defensive_timing"] = {**result.get("defensive_timing", {}), **candidate["defensive_timing_config"]}
    return result


def _score_key(candidate: dict[str, Any]) -> str:
    """函数说明：处理 score_key 的内部辅助逻辑。"""
    strategy_items = tuple(sorted(candidate.get("strategy", {}).items()))
    liquidity_items = tuple(sorted(candidate.get("liquidity_filter", {}).items()))
    market_items = json.dumps(candidate.get("market_regime", {}), sort_keys=True, default=str)
    blend_items = json.dumps(candidate.get("regime_score_blend", {}), sort_keys=True, default=str)
    filter_items = json.dumps(candidate.get("regime_score_filter", {}), sort_keys=True, default=str)
    dynamic_items = json.dumps(candidate.get("dynamic_ic_selector", {}), sort_keys=True, default=str)
    return repr((strategy_items, liquidity_items, market_items, blend_items, filter_items, dynamic_items))


def _quality_flags(metrics: dict[str, Any], quality_cfg: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 quality_flags 的内部辅助逻辑。"""
    annual_return = float(metrics.get("annual_return", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    annual_turnover = float(metrics.get("annual_turnover", 0.0) or 0.0)
    annual_cost = float(metrics.get("annual_trade_cost_ratio", metrics.get("trade_cost_ratio", 0.0)) or 0.0)
    year_count = int(metrics.get("year_count", 0) or 0)
    year_ann_pass = int(metrics.get("year_ann_pass", 0) or 0)
    year_dd_pass = int(metrics.get("year_dd_pass", 0) or 0)
    return_threshold, drawdown_limit = _quality_return_drawdown_thresholds(quality_cfg)
    turnover_limit = float(quality_cfg.get("max_annual_turnover", 20.0))
    cost_limit = float(quality_cfg.get("max_annual_trade_cost_ratio", 0.2))
    flags = {
        "annual_return_pass": annual_return >= return_threshold,
        "drawdown_pass": max_drawdown >= drawdown_limit,
        "annual_turnover_pass": annual_turnover <= turnover_limit,
        "annual_trade_cost_ratio_pass": annual_cost <= cost_limit,
        "yearly_annual_return_pass": bool(year_count <= 0 or year_ann_pass >= year_count),
        "yearly_drawdown_pass": bool(year_count <= 0 or year_dd_pass >= year_count),
        "annual_return_threshold": return_threshold,
        "drawdown_limit": drawdown_limit,
        "annual_turnover_limit": turnover_limit,
        "annual_trade_cost_ratio_limit": cost_limit,
    }
    year_count = int(metrics.get("year_count", 0) or 0)
    year_ann_pass = int(metrics.get("year_ann_pass", 0) or 0)
    year_dd_pass = int(metrics.get("year_dd_pass", 0) or 0)
    flags["yearly_return_pass"] = bool(year_count > 0 and year_ann_pass >= year_count)
    flags["yearly_drawdown_pass"] = bool(year_count > 0 and year_dd_pass >= year_count)
    flags["yearly_all_pass"] = bool(flags["yearly_return_pass"] and flags["yearly_drawdown_pass"])
    flags["is_acceptable"] = bool(
        flags["annual_return_pass"]
        and flags["drawdown_pass"]
        and flags["yearly_all_pass"]
        and flags["annual_turnover_pass"]
        and flags["annual_trade_cost_ratio_pass"]
        and flags["yearly_annual_return_pass"]
        and flags["yearly_drawdown_pass"]
    )
    return flags


def _candidate_error_row(candidate: dict[str, Any], started: float, exc: Exception, quality_cfg: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate.get("name", ""),
        "seconds": time.monotonic() - started,
        "error": f"{type(exc).__name__}: {exc}",
        "annual_return": 0.0,
        "max_drawdown": -1.0,
        "annual_turnover": 0.0,
        "annual_trade_cost_ratio": 0.0,
        "year_count": 0,
        "year_ann_pass": 0,
        "year_dd_pass": 0,
    }
    row.update(_quality_flags(row, quality_cfg))
    return row


def _quality_return_drawdown_thresholds(quality_cfg: dict[str, Any]) -> tuple[float, float]:
    """函数说明：处理 quality_return_drawdown_thresholds 的内部辅助逻辑。"""
    return_threshold = float(quality_cfg.get("min_backtest_annual_return", quality_cfg.get("target_annual_return", 0.20)))
    drawdown_limit = float(quality_cfg.get("max_backtest_drawdown_limit", quality_cfg.get("max_drawdown_limit", -0.20)))
    return return_threshold, drawdown_limit


def _yearly_pass_counts(yearly: pd.DataFrame, quality_cfg: dict[str, Any]) -> tuple[int, int]:
    """函数说明：处理 yearly_pass_counts 的内部辅助逻辑。"""
    if yearly.empty:
        return 0, 0
    return_threshold, drawdown_limit = _quality_return_drawdown_thresholds(quality_cfg)
    annual_passes = int((yearly["annual_return"] >= return_threshold).sum())
    drawdown_passes = int((yearly["max_drawdown"] >= drawdown_limit).sum())
    return annual_passes, drawdown_passes


def _write_candidate_artifacts(
    prefix: Path,
    result,
    yearly: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict[str, Any],
    write_diagnostics: bool,
) -> dict[str, str]:
    """函数说明：写入 write_candidate_artifacts 的内部辅助逻辑。"""
    paths = {
        "equity_path": str(prefix) + "_equity.csv",
        "years_path": str(prefix) + "_years.csv",
        "trades_path": str(prefix) + "_trades.csv",
        "holdings_path": str(prefix) + "_holdings.csv",
    }
    result.equity_curve.to_csv(paths["equity_path"], encoding="utf-8-sig")
    yearly.to_csv(paths["years_path"], index=False, encoding="utf-8-sig")
    result.trades.to_csv(paths["trades_path"], index=False, encoding="utf-8-sig")
    result.holdings.to_csv(paths["holdings_path"], index=False, encoding="utf-8-sig")
    if write_diagnostics:
        diagnostics, tables = build_research_diagnostics(result.equity_curve, result.holdings, result.trades, prices, config)
        research_paths = write_research_diagnostics(
            diagnostics,
            tables,
            prefix.parent,
            prefix=f"{prefix.name}_research",
        )
        paths["research_diagnostics_path"] = research_paths.get("research_diagnostics", "")
    return paths


def _yearly_stats(equity_curve: pd.Series) -> pd.DataFrame:
    """函数说明：处理 yearly_stats 的内部辅助逻辑。"""
    if equity_curve.empty:
        return pd.DataFrame(columns=["year", "days", "annual_return", "max_drawdown"])
    rows: list[dict[str, Any]] = []
    curve = equity_curve.sort_index()
    for year, group in curve.groupby(curve.index.year):
        if len(group) <= 1 or float(group.iloc[0]) <= 0:
            continue
        total_return = float(group.iloc[-1] / group.iloc[0] - 1.0)
        years = max((group.index.max() - group.index.min()).days / 365.25, 1 / 252)
        annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
        max_drawdown = float((group / group.cummax() - 1.0).min())
        rows.append({"year": int(year), "days": int(len(group)), "annual_return": annual_return, "max_drawdown": max_drawdown})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
