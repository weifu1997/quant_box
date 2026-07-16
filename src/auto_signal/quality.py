"""Parameter-quality adapters used by the auto-signal orchestration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.auto_tuning import ParameterQualityReport
from src.config_loader import resolve_path


def validated_strategy_quality(config: dict[str, Any], quality_config: dict) -> ParameterQualityReport | None:
    evidence_cfg = config.get("validated_strategy", {})
    if not isinstance(evidence_cfg, dict) or not bool(evidence_cfg.get("enabled", False)):
        return None

    summary_file = evidence_cfg.get("summary_file")
    candidate = str(evidence_cfg.get("candidate", "")).strip()
    if not summary_file:
        return formal_candidate_quality_report(quality_config, ["validated_strategy_summary_file_missing"])

    summary_path = resolve_path(summary_file)
    if not summary_path.exists():
        return formal_candidate_quality_report(
            quality_config,
            [f"validated_strategy_summary_file_not_found:{summary_path}"],
        )

    frame = pd.read_csv(summary_path)
    if frame.empty:
        return formal_candidate_quality_report(quality_config, ["validated_strategy_summary_empty"])
    if "candidate" not in frame.columns:
        return formal_candidate_quality_report(quality_config, ["validated_strategy_candidate_column_missing"])
    if candidate:
        rows = frame[frame["candidate"].astype(str) == candidate]
        if rows.empty:
            return formal_candidate_quality_report(quality_config, [f"validated_strategy_candidate_not_found:{candidate}"])
    elif len(frame) == 1:
        rows = frame
    else:
        return formal_candidate_quality_report(quality_config, ["validated_strategy_candidate_missing"])

    row = rows.iloc[-1]
    annual_return = quality_number(row.get("annual_return"), 0.0)
    max_drawdown = quality_number(row.get("max_drawdown"), 0.0)
    sharpe = quality_number(row.get("sharpe"), 0.0)
    annual_turnover = quality_number(row.get("annual_turnover"), 0.0)
    annual_trade_cost_ratio = quality_number(row.get("annual_trade_cost_ratio"), 0.0)
    windows = 1
    positive_return_rate = 1.0 if annual_return > 0 else 0.0
    annual_return_mean = annual_return
    annual_return_min = annual_return
    max_drawdown_worst = max_drawdown

    years_path_value = row.get("years_path")
    if pd.notna(years_path_value) and str(years_path_value).strip():
        years_path = resolve_path(str(years_path_value))
        if years_path.exists():
            yearly = pd.read_csv(years_path)
            if {"annual_return", "max_drawdown"}.issubset(yearly.columns) and not yearly.empty:
                yearly_returns = pd.to_numeric(yearly["annual_return"], errors="coerce").dropna()
                yearly_drawdowns = pd.to_numeric(yearly["max_drawdown"], errors="coerce").dropna()
                if not yearly_returns.empty:
                    windows = int(len(yearly_returns))
                    positive_return_rate = float((yearly_returns > 0).mean())
                    annual_return_mean = float(yearly_returns.mean())
                    annual_return_min = float(yearly_returns.min())
                if not yearly_drawdowns.empty:
                    max_drawdown_worst = float(yearly_drawdowns.min())

    min_windows = int(quality_config.get("min_validation_windows", 3))
    min_positive = float(quality_config.get("min_positive_return_rate", 0.5))
    min_return = float(quality_config.get("min_optimizer_annual_return", quality_config.get("target_annual_return", 0.20)))
    min_sharpe = float(quality_config.get("min_sharpe_mean", 0.0))
    max_drawdown_limit = float(quality_config.get("max_drawdown_limit", -0.20))
    max_turnover = float(quality_config.get("max_annual_turnover", 20.0))
    max_cost = float(quality_config.get("max_annual_trade_cost_ratio", 0.2))

    issues: list[str] = []
    if bool(evidence_cfg.get("require_is_acceptable", True)) and not truthy(row.get("is_acceptable")):
        issues.append("validated_strategy_not_formally_acceptable")
    if windows < min_windows:
        issues.append(f"validation_windows_below_threshold:{windows}<{min_windows}")
    if positive_return_rate < min_positive:
        issues.append(f"positive_return_rate_below_threshold:{positive_return_rate:.4f}<{min_positive:.4f}")
    if annual_return < min_return:
        issues.append(f"validated_strategy_annual_return_below_threshold:{annual_return:.4f}<{min_return:.4f}")
    if annual_return_mean < min_return:
        issues.append(f"annual_return_mean_below_threshold:{annual_return_mean:.4f}<{min_return:.4f}")
    if sharpe < min_sharpe:
        issues.append(f"sharpe_mean_below_threshold:{sharpe:.4f}<{min_sharpe:.4f}")
    if max_drawdown < max_drawdown_limit:
        issues.append(f"validated_strategy_max_drawdown_worse_than_limit:{max_drawdown:.4f}<{max_drawdown_limit:.4f}")
    if max_drawdown_worst < max_drawdown_limit:
        issues.append(f"max_drawdown_worse_than_limit:{max_drawdown_worst:.4f}<{max_drawdown_limit:.4f}")
    if annual_turnover > max_turnover:
        issues.append(f"annual_turnover_above_threshold:{annual_turnover:.4f}>{max_turnover:.4f}")
    if annual_trade_cost_ratio > max_cost:
        issues.append(f"annual_trade_cost_ratio_above_threshold:{annual_trade_cost_ratio:.4f}>{max_cost:.4f}")

    return ParameterQualityReport(
        is_acceptable=not issues,
        issues=issues,
        windows=windows,
        positive_return_rate=positive_return_rate,
        annual_return_mean=annual_return_mean,
        annual_return_min=annual_return_min,
        sharpe_mean=sharpe,
        max_drawdown_worst=max_drawdown_worst,
        annual_turnover_mean=annual_turnover,
        annual_trade_cost_ratio_mean=annual_trade_cost_ratio,
        min_validation_windows=min_windows,
        min_positive_return_rate=min_positive,
        min_optimizer_annual_return=min_return,
        min_sharpe_mean=min_sharpe,
        max_drawdown_limit=max_drawdown_limit,
        max_annual_turnover=max_turnover,
        max_annual_trade_cost_ratio=max_cost,
    )


def formal_candidate_quality_report(quality_config: dict, issues: list[str]) -> ParameterQualityReport:
    min_windows = int(quality_config.get("min_validation_windows", 3))
    min_positive = float(quality_config.get("min_positive_return_rate", 0.5))
    min_return = float(quality_config.get("min_optimizer_annual_return", quality_config.get("target_annual_return", 0.20)))
    min_sharpe = float(quality_config.get("min_sharpe_mean", 0.0))
    max_drawdown = float(quality_config.get("max_drawdown_limit", -0.20))
    max_turnover = float(quality_config.get("max_annual_turnover", 20.0))
    max_cost = float(quality_config.get("max_annual_trade_cost_ratio", 0.2))
    return ParameterQualityReport(
        is_acceptable=False,
        issues=issues,
        windows=0,
        positive_return_rate=0.0,
        annual_return_mean=0.0,
        annual_return_min=0.0,
        sharpe_mean=0.0,
        max_drawdown_worst=0.0,
        annual_turnover_mean=0.0,
        annual_trade_cost_ratio_mean=0.0,
        min_validation_windows=min_windows,
        min_positive_return_rate=min_positive,
        min_optimizer_annual_return=min_return,
        min_sharpe_mean=min_sharpe,
        max_drawdown_limit=max_drawdown,
        max_annual_turnover=max_turnover,
        max_annual_trade_cost_ratio=max_cost,
    )


def quality_number(value: Any, default: float) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def skipped_quality(quality_config: dict) -> ParameterQualityReport:
    report = formal_candidate_quality_report(quality_config, ["parameter_validation_skipped"])
    return report
