from __future__ import annotations

import argparse
import copy
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.fast_monthly_backtest import _close_frame as _fast_close_frame
from src.fast_monthly_backtest import prepare_fast_period_data
from src.fast_monthly_backtest import run_fast_prepared_backtest
from src.market_regime import defensive_exposure_schedule, detect_market_regime
from src.trading_calendar import resolve_target_date_value
from scripts.run_ml_strategy import _yearly_quality_gate, _yearly_stats


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run fast ML score portfolio experiments without retraining models.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--scores-file", default="outputs/ml_strategy_scores.parquet")
    parser.add_argument("--out-file", default="outputs/ml_strategy_experiments.csv")
    parser.add_argument("--engine", choices=["formal", "fast"], default="formal", help="formal uses daily execution backtest; fast is approximate monthly screening.")
    parser.add_argument("--top-n", default="5,10,15,20,30")
    parser.add_argument("--max-turnover", default="1,3,5,10,15")
    parser.add_argument("--rank-buffer", default="0,5,10,20")
    parser.add_argument("--score-weighted", default="false,true")
    parser.add_argument("--score-direction", default="1,-1", help="Use 1 for high-score long, -1 for low-score long.")
    parser.add_argument("--max-weight", default="none,0.20,0.15,0.10")
    parser.add_argument("--bull-exposure", default="1.0")
    parser.add_argument("--sideways-exposure", default="0.3,0.5,0.7,0.85")
    parser.add_argument("--bear-exposure", default="0.0,0.15,0.3,0.55")
    parser.add_argument("--stop-loss", default="none,0.06,0.10")
    parser.add_argument("--take-profit", default="none,0.25,0.50")
    parser.add_argument("--circuit-breaker-drawdown", default="none,0.06,0.10,0.12")
    parser.add_argument("--circuit-breaker-cooldown-days", default="0,20,60")
    parser.add_argument("--target-vol", default="none,0.10,0.15")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit combinations for quick smoke runs.")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    prices = pd.read_parquet(resolve_path(args.price_file))
    fast_prices = _fast_close_frame(prices) if args.engine == "fast" else prices
    scores = _load_scores(resolve_path(args.scores_file))
    timing_regimes = detect_market_regime(prices, config)
    out_path = resolve_path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exposure_cache: dict[tuple[float, float, float], pd.Series] = {}
    fast_data_cache = {}

    rows: list[dict[str, object]] = []
    combos = product(
        _csv_ints(args.top_n),
        _csv_ints(args.max_turnover),
        _csv_ints(args.rank_buffer),
        _csv_bools(args.score_weighted),
        _csv_floats(args.score_direction),
        _csv_optional_floats(args.max_weight),
        _csv_floats(args.bull_exposure),
        _csv_floats(args.sideways_exposure),
        _csv_floats(args.bear_exposure),
        _csv_optional_floats(args.stop_loss),
        _csv_optional_floats(args.take_profit),
        _csv_optional_floats(args.circuit_breaker_drawdown),
        _csv_ints(args.circuit_breaker_cooldown_days),
        _csv_optional_floats(args.target_vol),
    )
    for idx, (
        top_n,
        max_turnover,
        rank_buffer,
        score_weighted,
        score_direction,
        max_weight,
        bull_exposure,
        sideways_exposure,
        bear_exposure,
        stop_loss,
        take_profit,
        circuit_breaker_drawdown,
        circuit_breaker_cooldown_days,
        target_vol,
    ) in enumerate(combos, start=1):
        if args.max_rows and idx > args.max_rows:
            break
        run_config = copy.deepcopy(config)
        run_config.setdefault("defensive_timing", {}).update(
            {
                "bull_exposure": bull_exposure,
                "sideways_exposure": sideways_exposure,
                "bear_exposure": bear_exposure,
            }
        )
        exposure_key = (bull_exposure, sideways_exposure, bear_exposure)
        if exposure_key not in exposure_cache:
            exposure_cache[exposure_key] = defensive_exposure_schedule(
                timing_regimes,
                run_config,
                pd.Index(pd.to_datetime(prices.index)),
            )
        exposure = exposure_cache[exposure_key]
        bt_config = {**run_config["backtest"], **run_config["strategy"]}
        bt_config.update(
            {
                "top_n": top_n,
                "max_turnover": max_turnover,
                "rank_buffer": rank_buffer,
                "score_weighted": score_weighted,
                "exposure_schedule": exposure,
                "exposure_rebalance_threshold": float(run_config.get("defensive_timing", {}).get("exposure_rebalance_threshold", 0.05)),
            }
        )
        if max_weight is not None:
            bt_config["max_weight_per_stock"] = max_weight
        if stop_loss is not None:
            bt_config["stop_loss_pct"] = stop_loss
        else:
            bt_config.pop("stop_loss_pct", None)
        if take_profit is not None:
            bt_config["take_profit_pct"] = take_profit
        else:
            bt_config.pop("take_profit_pct", None)
        if circuit_breaker_drawdown is not None:
            bt_config["circuit_breaker_drawdown"] = circuit_breaker_drawdown
            bt_config["circuit_breaker_cooldown_days"] = circuit_breaker_cooldown_days
        else:
            bt_config.pop("circuit_breaker_drawdown", None)
            bt_config.pop("circuit_breaker_cooldown_days", None)
        if target_vol is not None:
            bt_config["target_vol"] = target_vol
            bt_config["max_leverage"] = 1.0

        experiment_scores = scores if score_direction >= 0 else (scores * -1.0).rename("score")
        if args.engine == "fast":
            if score_direction not in fast_data_cache:
                fast_data_cache[score_direction] = prepare_fast_period_data(experiment_scores, fast_prices, args.start_date, end_date)
            result = run_fast_prepared_backtest(fast_data_cache[score_direction], bt_config)
            yearly = _fast_yearly_stats(result.equity_curve)
        else:
            result = run_backtest(experiment_scores, prices, args.start_date, end_date, bt_config)
            yearly = _yearly_stats(result.equity_curve, bt_config)
        quality = _quality(result.metrics, yearly, run_config)
        row = {
            "combo_id": idx,
            "engine": args.engine,
            "top_n": top_n,
            "max_turnover": max_turnover,
            "rank_buffer": rank_buffer,
            "score_weighted": score_weighted,
            "score_direction": score_direction,
            "max_weight_per_stock": "" if max_weight is None else max_weight,
            "bull_exposure": bull_exposure,
            "sideways_exposure": sideways_exposure,
            "bear_exposure": bear_exposure,
            "stop_loss_pct": "" if stop_loss is None else stop_loss,
            "take_profit_pct": "" if take_profit is None else take_profit,
            "circuit_breaker_drawdown": "" if circuit_breaker_drawdown is None else circuit_breaker_drawdown,
            "circuit_breaker_cooldown_days": "" if circuit_breaker_drawdown is None else circuit_breaker_cooldown_days,
            "target_vol": "" if target_vol is None else target_vol,
            **result.metrics,
            **quality,
        }
        rows.append(row)
        _write_results(rows, out_path)
        print(
            f"{idx}: pass={quality['quality_pass']} annual={result.metrics.get('annual_return', 0.0):.4f} "
            f"dd={result.metrics.get('max_drawdown', 0.0):.4f} top_n={top_n} weighted={score_weighted} "
            f"turnover={max_turnover} buffer={rank_buffer} dir={score_direction} "
            f"max_w={max_weight} exp=({bull_exposure},{sideways_exposure},{bear_exposure}) "
            f"stop={stop_loss} take={take_profit} cb={circuit_breaker_drawdown}/{circuit_breaker_cooldown_days} vol={target_vol}",
            flush=True,
        )

    result_df = _write_results(rows, out_path)
    print(f"Saved experiments to {out_path}")
    if not result_df.empty:
        print(json.dumps(result_df.head(10).to_dict(orient="records"), indent=2, default=str))


def _write_results(rows: list[dict[str, object]], out_path: Path) -> pd.DataFrame:
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(["quality_pass", "annual_return", "max_drawdown"], ascending=[False, False, False])
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result_df


def _load_scores(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}. Run scripts/run_ml_strategy.py first.")
    frame = pd.read_parquet(path)
    if isinstance(frame.index, pd.MultiIndex):
        score = frame["score"] if "score" in frame.columns else frame.iloc[:, 0]
    else:
        date_col = "datetime" if "datetime" in frame.columns else "date"
        if date_col not in frame.columns or "instrument" not in frame.columns or "score" not in frame.columns:
            raise ValueError("Scores file must have MultiIndex or datetime/instrument/score columns.")
        index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(frame[date_col]).dt.normalize(), frame["instrument"].astype(str)],
            names=["datetime", "instrument"],
        )
        score = pd.Series(frame["score"].to_numpy(dtype=float), index=index, name="score")
    score.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(score.index.get_level_values(0)).normalize(),
            score.index.get_level_values(1).astype(str),
        ],
        names=["datetime", "instrument"],
    )
    return score.sort_index().rename("score")


def _quality(metrics: dict[str, float], yearly: pd.DataFrame, config: dict) -> dict[str, object]:
    ml_cfg = config.get("ml_strategy", {})
    target_return = float(ml_cfg.get("target_annual_return", 0.20))
    min_yearly_return = float(ml_cfg.get("min_yearly_annual_return", target_return))
    drawdown_limit = float(ml_cfg.get("max_drawdown_limit", -0.20))
    annual_return = float(metrics.get("annual_return", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    yearly_gate = _yearly_quality_gate(yearly, config)
    issues = []
    if annual_return < target_return:
        issues.append(f"annual_return_below_target:{annual_return:.4f}<{target_return:.4f}")
    if max_drawdown < drawdown_limit:
        issues.append(f"max_drawdown_breaches_limit:{max_drawdown:.4f}<{drawdown_limit:.4f}")
    if yearly_gate.get("years_below_return_target"):
        issues.append(f"yearly_annual_return_below_target:{yearly_gate['years_below_return_target']}")
    if yearly_gate.get("years_breaching_drawdown_limit"):
        issues.append(f"yearly_max_drawdown_breaches_limit:{yearly_gate['years_breaching_drawdown_limit']}")
    return {
        "quality_pass": not issues,
        "quality_issues": ";".join(issues),
        "min_yearly_annual_return": min_yearly_return,
        "years_below_return_target": ",".join(str(year) for year in yearly_gate.get("years_below_return_target", [])),
        "years_breaching_drawdown_limit": ",".join(str(year) for year in yearly_gate.get("years_breaching_drawdown_limit", [])),
    }


def _fast_yearly_stats(equity_curve: pd.Series) -> pd.DataFrame:
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


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _csv_bools(value: str) -> list[bool]:
    return [item.strip().lower() in {"1", "true", "yes", "y"} for item in value.split(",") if item.strip()]


def _csv_optional_floats(value: str) -> list[float | None]:
    result: list[float | None] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        result.append(None if text in {"none", "null", "na"} else float(text))
    return result


if __name__ == "__main__":
    main()
