"""Historical backtest, quality, and research-diagnostics stage."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.auto_signal.models import AnnualStateRouterRuntime, BacktestStageResult
from src.auto_signal.status import stage, write_json
from src.auto_tuning import assess_backtest_quality
from src.backtest import BacktestResult, run_backtest
from src.failure_analysis import build_yearly_breakdown
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.research_diagnostics import build_research_diagnostics, write_research_diagnostics
from src.risk_policy import RiskPolicy
from src.scoring import build_strategy_scores
from src.strategy import resample_signals


@dataclass(frozen=True)
class BacktestStageServices:
    annual_state_router_enabled: Callable[[dict[str, Any]], bool]
    build_annual_state_router_runtime: Callable[..., AnnualStateRouterRuntime]
    build_strategy_scores: Callable[..., pd.Series] = build_strategy_scores
    resample_signals: Callable[..., pd.Series] = resample_signals
    apply_defensive_timing_to_backtest_config: Callable[..., dict[str, Any]] = apply_defensive_timing_to_backtest_config
    risk_policy_factory: Callable[[dict[str, Any]], RiskPolicy] = RiskPolicy
    run_backtest: Callable[..., BacktestResult] = run_backtest
    build_yearly_breakdown: Callable[..., pd.DataFrame] = build_yearly_breakdown
    assess_backtest_quality: Callable[..., Any] = assess_backtest_quality
    build_research_diagnostics: Callable[..., tuple[dict[str, Any], dict[str, pd.DataFrame]]] = build_research_diagnostics
    write_research_diagnostics: Callable[..., dict[str, str]] = write_research_diagnostics


def run_backtest_stage(
    args: Namespace,
    selected_config: dict[str, Any],
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
    *,
    services: BacktestStageServices,
) -> BacktestStageResult:
    """Run or skip the historical backtest and research diagnostics."""
    equity_path = out_dir / "auto_backtest_equity.csv"
    holdings_bt_path = out_dir / "auto_backtest_holdings.csv"
    trades_path = out_dir / "auto_backtest_trades.csv"
    metrics_path = out_dir / "auto_backtest_metrics.json"
    backtest_runtime_config: dict[str, Any] = {**selected_config["backtest"], **selected_config["strategy"]}
    annual_state_router: AnnualStateRouterRuntime | None = None
    if args.skip_backtest:
        stage(status, out_dir, "backtest", "skipped")
        result = BacktestResult(
            equity_curve=pd.Series(dtype=float, name="equity"),
            holdings=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={"backtest_skipped": True},
        )
    else:
        if services.annual_state_router_enabled(selected_config):
            stage(status, out_dir, "backtest", "running", "building annual state router scores")
            annual_state_router = services.build_annual_state_router_runtime(
                config=selected_config,
                prices=prices,
                start_date=args.start_date,
                end_date=end_date,
                out_dir=out_dir,
                artifacts=artifacts,
                status=status,
            )
            scores = annual_state_router.routed.scores
            bt_config = annual_state_router.backtest_config
        else:
            stage(status, out_dir, "backtest", "running", "building strategy scores")
            scores = services.build_strategy_scores(factors, selected_config, price_df=prices)
            stage(status, out_dir, "backtest", "running", "resampling signals")
            scores = services.resample_signals(scores, selected_config["strategy"].get("rebalance_freq", "daily"))
            stage(status, out_dir, "backtest", "running", "preparing backtest config")
            bt_config = services.apply_defensive_timing_to_backtest_config(
                backtest_runtime_config,
                prices,
                selected_config,
            )
            bt_config = services.risk_policy_factory(selected_config).apply_to_backtest_config(bt_config)
        backtest_runtime_config = bt_config
        stage(status, out_dir, "backtest", "running", "running historical backtest")
        result = services.run_backtest(scores, prices, args.start_date, end_date, bt_config)
        result.equity_curve.to_csv(equity_path, encoding="utf-8-sig")
        result.holdings.to_csv(holdings_bt_path, index=False, encoding="utf-8-sig")
        result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
        stage(status, out_dir, "backtest", "complete")

    write_json(metrics_path, result.metrics)
    artifacts.append(metrics_path)
    if not args.skip_backtest:
        artifacts.extend([equity_path, holdings_bt_path, trades_path])

    backtest_yearly_quality = (
        services.build_yearly_breakdown(result, backtest_runtime_config) if not args.skip_backtest else pd.DataFrame()
    )
    backtest_quality = services.assess_backtest_quality(
        result.metrics,
        selected_config.get("quality", {}),
        yearly=backtest_yearly_quality,
    )
    backtest_quality_path = out_dir / "auto_backtest_quality.json"
    write_json(backtest_quality_path, backtest_quality.to_dict())
    artifacts.append(backtest_quality_path)
    backtest_quality_gate = backtest_quality.is_acceptable or args.allow_low_quality

    research_diagnostics: dict[str, Any] = {
        "enabled": False,
        "issues": ["backtest_skipped"] if args.skip_backtest else [],
    }
    research_tables: dict[str, pd.DataFrame] = {}
    research_files: dict[str, str] = {}
    if not args.skip_backtest:
        stage(status, out_dir, "research_diagnostics", "running")
        research_diagnostics, research_tables = services.build_research_diagnostics(
            result.equity_curve,
            result.holdings,
            result.trades,
            prices,
            selected_config,
        )
        research_files = services.write_research_diagnostics(research_diagnostics, research_tables, out_dir)
        artifacts.extend(Path(path) for path in research_files.values())
        stage(
            status,
            out_dir,
            "research_diagnostics",
            "complete",
            ",".join(map(str, research_diagnostics.get("issues", []))) or "ok",
        )
    else:
        stage(status, out_dir, "research_diagnostics", "skipped")

    return BacktestStageResult(
        result=result,
        backtest_runtime_config=backtest_runtime_config,
        annual_state_router=annual_state_router,
        backtest_quality=backtest_quality,
        backtest_quality_gate=backtest_quality_gate,
        backtest_quality_path=backtest_quality_path,
        metrics_path=metrics_path,
        research_diagnostics=research_diagnostics,
        research_tables=research_tables,
        research_files=research_files,
    )
