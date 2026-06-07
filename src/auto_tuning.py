from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd


STRATEGY_PARAM_COLUMNS = ["factor_group", "top_n", "max_turnover", "rank_buffer", "rebalance_freq"]
RISK_PARAM_COLUMNS = [
    "max_weight_per_stock",
    "stop_loss_pct",
    "take_profit_pct",
    "circuit_breaker_drawdown",
    "circuit_breaker_cooldown_days",
    "circuit_breaker_target_exposure",
    "target_vol",
    "max_industry_weight",
    "rebalance_drift_threshold",
]
PARAM_COLUMNS = STRATEGY_PARAM_COLUMNS
OPTIMIZABLE_PARAM_COLUMNS = [*STRATEGY_PARAM_COLUMNS, *RISK_PARAM_COLUMNS]
METRIC_DEFAULTS = {
    "optimization_score": 0.0,
    "annual_return": 0.0,
    "sharpe": 0.0,
    "max_drawdown": 0.0,
    "annual_turnover": 0.0,
    "annual_trade_cost_ratio": 0.0,
    "win_rate": 0.0,
}


@dataclass
class ParameterQualityReport:
    is_acceptable: bool
    issues: list[str]
    windows: int
    positive_return_rate: float
    annual_return_mean: float
    annual_return_min: float
    sharpe_mean: float
    max_drawdown_worst: float
    annual_turnover_mean: float
    annual_trade_cost_ratio_mean: float
    min_validation_windows: int
    min_positive_return_rate: float
    min_optimizer_annual_return: float
    min_sharpe_mean: float
    max_drawdown_limit: float
    max_annual_turnover: float
    max_annual_trade_cost_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestQualityReport:
    is_acceptable: bool
    issues: list[str]
    annual_return: float
    max_drawdown: float
    calmar: float
    year_count: int
    expected_year_count: int
    year_ann_pass: int
    year_dd_pass: int
    min_year_annual_return: float
    worst_year_drawdown: float
    min_yearly_annual_return: float
    max_yearly_drawdown_limit: float
    year_coverage_pass: bool
    min_backtest_annual_return: float
    max_backtest_drawdown_limit: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_parameter_validation(
    validation: pd.DataFrame,
    param_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    param_columns = _parameter_columns(validation, param_columns)
    if validation.empty:
        return pd.DataFrame(columns=[*param_columns, "auto_score"])

    frame = validation.copy()
    missing_params = [column for column in param_columns if column not in frame.columns]
    if missing_params:
        raise ValueError(f"Validation results missing parameter columns: {missing_params}")

    for column, default in METRIC_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)

    grouped = frame.groupby(param_columns, dropna=False)
    summary = grouped.agg(
        windows=("optimization_score", "count"),
        optimization_score_mean=("optimization_score", "mean"),
        optimization_score_median=("optimization_score", "median"),
        optimization_score_std=("optimization_score", "std"),
        annual_return_mean=("annual_return", "mean"),
        annual_return_median=("annual_return", "median"),
        annual_return_min=("annual_return", "min"),
        sharpe_mean=("sharpe", "mean"),
        sharpe_median=("sharpe", "median"),
        sharpe_min=("sharpe", "min"),
        max_drawdown_worst=("max_drawdown", "min"),
        annual_turnover_mean=("annual_turnover", "mean"),
        annual_trade_cost_ratio_mean=("annual_trade_cost_ratio", "mean"),
        win_rate_mean=("win_rate", "mean"),
    ).reset_index()
    summary["positive_return_rate"] = grouped["annual_return"].apply(lambda series: float((series > 0).mean())).to_numpy()
    summary["positive_score_rate"] = grouped["optimization_score"].apply(lambda series: float((series > 0).mean())).to_numpy()
    summary["optimization_score_std"] = summary["optimization_score_std"].fillna(0.0)
    summary["auto_score"] = (
        summary["optimization_score_mean"]
        + 0.5 * summary["optimization_score_median"]
        - 0.5 * summary["optimization_score_std"]
        + summary["positive_return_rate"]
        + 0.5 * summary["positive_score_rate"]
        + 0.25 * summary["sharpe_mean"]
        + summary["max_drawdown_worst"]
        - summary["annual_trade_cost_ratio_mean"]
    )
    return summary.sort_values(
        [
            "auto_score",
            "positive_return_rate",
            "optimization_score_mean",
            "max_drawdown_worst",
            "annual_turnover_mean",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def select_stable_params(
    summary: pd.DataFrame,
    quality_config: dict | None = None,
    param_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    param_columns = _parameter_columns(summary, param_columns)
    if summary.empty:
        raise ValueError("Cannot select parameters from an empty validation summary.")
    candidates = _target_filtered_summary(summary, quality_config)
    best = candidates.iloc[0]
    return {column: _python_scalar(best[column]) for column in param_columns}


def assess_parameter_quality(summary: pd.DataFrame, quality_config: dict | None = None) -> ParameterQualityReport:
    cfg = quality_config or {}
    min_windows = int(cfg.get("min_validation_windows", 3))
    min_positive = float(cfg.get("min_positive_return_rate", 0.5))
    min_optimizer_return = float(cfg.get("min_optimizer_annual_return", cfg.get("target_annual_return", 0.20)))
    min_yearly_return = float(cfg.get("min_yearly_annual_return", min_optimizer_return))
    min_sharpe = float(cfg.get("min_sharpe_mean", 0.0))
    max_drawdown = float(cfg.get("max_drawdown_limit", -0.20))
    max_yearly_drawdown = float(cfg.get("max_yearly_drawdown_limit", max_drawdown))
    max_turnover = float(cfg.get("max_annual_turnover", 20.0))
    max_cost = float(cfg.get("max_annual_trade_cost_ratio", 0.2))
    issues: list[str] = []

    if summary.empty:
        return ParameterQualityReport(
            is_acceptable=False,
            issues=["parameter_summary_empty"],
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
            min_optimizer_annual_return=min_optimizer_return,
            min_sharpe_mean=min_sharpe,
            max_drawdown_limit=max_drawdown,
            max_annual_turnover=max_turnover,
            max_annual_trade_cost_ratio=max_cost,
        )

    best = _target_filtered_summary(summary, quality_config).iloc[0]
    windows = int(_number(best.get("windows"), 0))
    positive_return_rate = _number(best.get("positive_return_rate"), 0.0)
    annual_return_mean = _number(best.get("annual_return_mean"), 0.0)
    annual_return_min = _number(best.get("annual_return_min"), 0.0)
    sharpe_mean = _number(best.get("sharpe_mean"), 0.0)
    max_drawdown_worst = _number(best.get("max_drawdown_worst"), 0.0)
    annual_turnover_mean = _number(best.get("annual_turnover_mean"), 0.0)
    annual_trade_cost_ratio_mean = _number(best.get("annual_trade_cost_ratio_mean"), 0.0)

    if windows < min_windows:
        issues.append(f"validation_windows_below_threshold:{windows}<{min_windows}")
    if positive_return_rate < min_positive:
        issues.append(f"positive_return_rate_below_threshold:{positive_return_rate:.4f}<{min_positive:.4f}")
    if annual_return_mean < min_optimizer_return:
        issues.append(f"annual_return_mean_below_threshold:{annual_return_mean:.4f}<{min_optimizer_return:.4f}")
    if annual_return_min < min_yearly_return:
        issues.append(f"annual_return_min_below_threshold:{annual_return_min:.4f}<{min_yearly_return:.4f}")
    if sharpe_mean < min_sharpe:
        issues.append(f"sharpe_mean_below_threshold:{sharpe_mean:.4f}<{min_sharpe:.4f}")
    if max_drawdown_worst < max_yearly_drawdown:
        issues.append(f"max_drawdown_worse_than_limit:{max_drawdown_worst:.4f}<{max_yearly_drawdown:.4f}")
    if annual_turnover_mean > max_turnover:
        issues.append(f"annual_turnover_above_threshold:{annual_turnover_mean:.4f}>{max_turnover:.4f}")
    if annual_trade_cost_ratio_mean > max_cost:
        issues.append(f"annual_trade_cost_ratio_above_threshold:{annual_trade_cost_ratio_mean:.4f}>{max_cost:.4f}")

    return ParameterQualityReport(
        is_acceptable=not issues,
        issues=issues,
        windows=windows,
        positive_return_rate=positive_return_rate,
        annual_return_mean=annual_return_mean,
        annual_return_min=annual_return_min,
        sharpe_mean=sharpe_mean,
        max_drawdown_worst=max_drawdown_worst,
        annual_turnover_mean=annual_turnover_mean,
        annual_trade_cost_ratio_mean=annual_trade_cost_ratio_mean,
        min_validation_windows=min_windows,
        min_positive_return_rate=min_positive,
        min_optimizer_annual_return=min_optimizer_return,
        min_sharpe_mean=min_sharpe,
        max_drawdown_limit=max_drawdown,
        max_annual_turnover=max_turnover,
        max_annual_trade_cost_ratio=max_cost,
    )


def assess_backtest_quality(
    metrics: dict[str, Any],
    quality_config: dict | None = None,
    yearly: pd.DataFrame | None = None,
    yearly_coverage: pd.DataFrame | None = None,
) -> BacktestQualityReport:
    cfg = quality_config or {}
    min_return = float(cfg.get("min_backtest_annual_return", cfg.get("target_annual_return", 0.20)))
    max_drawdown = float(cfg.get("max_backtest_drawdown_limit", -0.20))
    min_yearly_return = float(cfg.get("min_yearly_annual_return", min_return))
    max_yearly_drawdown = float(cfg.get("max_yearly_drawdown_limit", max_drawdown))
    issues: list[str] = []

    if not metrics or bool(metrics.get("backtest_skipped", False)):
        return BacktestQualityReport(
            is_acceptable=False,
            issues=["backtest_skipped"],
            annual_return=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            year_count=0,
            expected_year_count=0,
            year_ann_pass=0,
            year_dd_pass=0,
            min_year_annual_return=0.0,
            worst_year_drawdown=0.0,
            min_yearly_annual_return=min_yearly_return,
            max_yearly_drawdown_limit=max_yearly_drawdown,
            year_coverage_pass=False,
            min_backtest_annual_return=min_return,
            max_backtest_drawdown_limit=max_drawdown,
        )

    annual_return = _number(metrics.get("annual_return"), 0.0)
    observed_drawdown = _number(metrics.get("max_drawdown"), 0.0)
    calmar = _number(metrics.get("calmar"), 0.0)
    yearly_fields = _backtest_yearly_quality_fields(yearly, yearly_coverage, min_yearly_return, max_yearly_drawdown)

    if annual_return < min_return:
        issues.append(f"backtest_annual_return_below_threshold:{annual_return:.4f}<{min_return:.4f}")
    if observed_drawdown < max_drawdown:
        issues.append(f"backtest_max_drawdown_worse_than_limit:{observed_drawdown:.4f}<{max_drawdown:.4f}")
    if yearly is not None:
        if not yearly_fields["year_coverage_pass"]:
            missing = str(yearly_fields.get("missing_years", "")).strip()
            suffix = f":{missing}" if missing else ""
            issues.append(f"backtest_year_coverage_incomplete{suffix}")
        if yearly_fields["year_count"] <= 0:
            issues.append("backtest_yearly_stats_missing")
        elif yearly_fields["year_ann_pass"] < yearly_fields["year_count"]:
            issues.append(
                "backtest_yearly_annual_return_below_threshold:"
                f"{yearly_fields['min_year_annual_return']:.4f}<{min_yearly_return:.4f}"
            )
        if yearly_fields["year_count"] > 0 and yearly_fields["year_dd_pass"] < yearly_fields["year_count"]:
            issues.append(
                "backtest_yearly_drawdown_worse_than_limit:"
                f"{yearly_fields['worst_year_drawdown']:.4f}<{max_yearly_drawdown:.4f}"
            )

    return BacktestQualityReport(
        is_acceptable=not issues,
        issues=issues,
        annual_return=annual_return,
        max_drawdown=observed_drawdown,
        calmar=calmar,
        year_count=int(yearly_fields["year_count"]),
        expected_year_count=int(yearly_fields["expected_year_count"]),
        year_ann_pass=int(yearly_fields["year_ann_pass"]),
        year_dd_pass=int(yearly_fields["year_dd_pass"]),
        min_year_annual_return=float(yearly_fields["min_year_annual_return"]),
        worst_year_drawdown=float(yearly_fields["worst_year_drawdown"]),
        min_yearly_annual_return=min_yearly_return,
        max_yearly_drawdown_limit=max_yearly_drawdown,
        year_coverage_pass=bool(yearly_fields["year_coverage_pass"]),
        min_backtest_annual_return=min_return,
        max_backtest_drawdown_limit=max_drawdown,
    )


def apply_strategy_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    selected = deepcopy(config)
    selected.setdefault("strategy", {})
    for key, value in params.items():
        if key in OPTIMIZABLE_PARAM_COLUMNS:
            selected["strategy"][key] = _python_scalar(value)
    return selected


def _target_filtered_summary(summary: pd.DataFrame, quality_config: dict | None) -> pd.DataFrame:
    cfg = quality_config or {}
    if summary.empty:
        return summary
    required = {"annual_return_mean", "max_drawdown_worst"}
    if not required.issubset(summary.columns):
        return summary
    min_return = float(cfg.get("min_optimizer_annual_return", cfg.get("target_annual_return", 0.20)))
    drawdown_limit = float(cfg.get("max_drawdown_limit", cfg.get("max_backtest_drawdown_limit", -0.20)))
    min_yearly_return = float(cfg.get("min_yearly_annual_return", min_return))
    yearly_drawdown_limit = float(cfg.get("max_yearly_drawdown_limit", drawdown_limit))
    annual = pd.to_numeric(summary["annual_return_mean"], errors="coerce")
    drawdown = pd.to_numeric(summary["max_drawdown_worst"], errors="coerce")
    mask = (annual >= min_return) & (drawdown >= drawdown_limit)
    if "annual_return_min" in summary.columns:
        annual_min = pd.to_numeric(summary["annual_return_min"], errors="coerce")
        mask &= annual_min >= min_yearly_return
    if "max_drawdown_worst" in summary.columns:
        mask &= drawdown >= yearly_drawdown_limit
    if "annual_turnover_mean" in summary.columns:
        turnover = pd.to_numeric(summary["annual_turnover_mean"], errors="coerce")
        mask &= turnover <= float(cfg.get("max_annual_turnover", float("inf")))
    if "annual_trade_cost_ratio_mean" in summary.columns:
        cost = pd.to_numeric(summary["annual_trade_cost_ratio_mean"], errors="coerce")
        mask &= cost <= float(cfg.get("max_annual_trade_cost_ratio", float("inf")))
    filtered = summary[mask]
    return filtered if not filtered.empty else summary


def _backtest_yearly_quality_fields(
    yearly: pd.DataFrame | None,
    yearly_coverage: pd.DataFrame | None,
    min_yearly_return: float,
    max_yearly_drawdown: float,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "year_count": 0,
        "expected_year_count": 0,
        "year_ann_pass": 0,
        "year_dd_pass": 0,
        "min_year_annual_return": 0.0,
        "worst_year_drawdown": 0.0,
        "year_coverage_pass": True,
        "missing_years": "",
    }
    if yearly is not None and not yearly.empty:
        annual = pd.to_numeric(yearly.get("annual_return", pd.Series(dtype=float)), errors="coerce")
        drawdown = pd.to_numeric(yearly.get("max_drawdown", pd.Series(dtype=float)), errors="coerce")
        valid = annual.notna() & drawdown.notna()
        fields["year_count"] = int(valid.sum())
        if int(valid.sum()):
            fields["year_ann_pass"] = int((annual[valid] >= min_yearly_return).sum())
            fields["year_dd_pass"] = int((drawdown[valid] >= max_yearly_drawdown).sum())
            fields["min_year_annual_return"] = float(annual[valid].min())
            fields["worst_year_drawdown"] = float(drawdown[valid].min())
    if yearly_coverage is not None:
        fields["expected_year_count"] = int(len(yearly_coverage))
        if yearly_coverage.empty:
            fields["year_coverage_pass"] = False
            return fields
        if "passes_min_days" in yearly_coverage.columns:
            coverage = yearly_coverage["passes_min_days"].fillna(False).astype(bool)
        elif "has_equity" in yearly_coverage.columns:
            coverage = yearly_coverage["has_equity"].fillna(False).astype(bool)
        else:
            coverage = pd.Series(False, index=yearly_coverage.index)
        fields["year_coverage_pass"] = bool(coverage.all())
        missing: list[str] = []
        if not coverage.all() and "year" in yearly_coverage.columns:
            missing.extend(yearly_coverage.loc[~coverage, "year"].dropna().astype(int).astype(str).tolist())
        if "year" in yearly_coverage.columns:
            expected_years = set(yearly_coverage["year"].dropna().astype(int).astype(str))
            observed_years: set[str] = set()
            if yearly is not None and not yearly.empty and "year" in yearly.columns:
                observed_years = set(yearly["year"].dropna().astype(int).astype(str))
            missing.extend(sorted(expected_years - observed_years))
        missing = sorted(set(missing))
        if missing:
            fields["year_coverage_pass"] = False
            fields["missing_years"] = ",".join(missing)
    else:
        fields["expected_year_count"] = int(fields["year_count"])
    return fields


def _parameter_columns(frame: pd.DataFrame, param_columns: Iterable[str] | None) -> list[str]:
    if param_columns is not None:
        return list(param_columns)
    inferred = [column for column in OPTIMIZABLE_PARAM_COLUMNS if column in frame.columns]
    return inferred or list(PARAM_COLUMNS)


def _python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _number(value: Any, default: float) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)
