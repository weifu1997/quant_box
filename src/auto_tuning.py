"""模块说明：评估参数质量并选择稳定的策略参数。"""

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
REQUIRED_VALIDATION_METRICS = (
    "optimization_score",
    "annual_return",
    "sharpe",
    "max_drawdown",
    "annual_turnover",
    "annual_trade_cost_ratio",
)


@dataclass
class ParameterQualityReport:
    """类说明：封装 ParameterQualityReport 相关数据和行为。"""
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
        """函数说明：处理 to_dict 主要逻辑。"""
        return asdict(self)


@dataclass
class BacktestQualityReport:
    """类说明：封装 BacktestQualityReport 相关数据和行为。"""
    is_acceptable: bool
    issues: list[str]
    annual_return: float
    max_drawdown: float
    calmar: float
    min_backtest_annual_return: float
    max_backtest_drawdown_limit: float
    yearly_min_annual_return: float
    yearly_worst_max_drawdown: float
    min_yearly_annual_return: float
    max_yearly_drawdown_limit: float
    years_below_return_target: list[int]
    years_breaching_drawdown_limit: list[int]

    def to_dict(self) -> dict[str, Any]:
        """函数说明：处理 to_dict 主要逻辑。"""
        return asdict(self)


def summarize_parameter_validation(
    validation: pd.DataFrame,
    param_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """函数说明：汇总 summarize_parameter_validation 主要逻辑。"""
    param_columns = _parameter_columns(validation, param_columns)
    if validation.empty:
        return pd.DataFrame(columns=[*param_columns, "auto_score"])

    frame = validation.copy()
    missing_params = [column for column in param_columns if column not in frame.columns]
    if missing_params:
        raise ValueError(f"Validation results missing parameter columns: {missing_params}")

    missing_metrics = [column for column in REQUIRED_VALIDATION_METRICS if column not in frame.columns]
    if missing_metrics:
        raise ValueError(f"Validation results missing metric columns: {missing_metrics}")

    invalid_metrics: list[str] = []
    for column, default in METRIC_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
        parsed = pd.to_numeric(frame[column], errors="coerce")
        if column in REQUIRED_VALIDATION_METRICS:
            invalid = parsed.isna() | parsed.abs().eq(float("inf"))
            if invalid.any():
                rows = ",".join(str(index) for index in frame.index[invalid][:5])
                invalid_metrics.append(f"{column} rows [{rows}]")
        frame[column] = parsed.fillna(default)
    if invalid_metrics:
        raise ValueError(f"Validation results contain missing or nonnumeric metrics: {'; '.join(invalid_metrics)}")

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
        + 2.0 * summary["max_drawdown_worst"]
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
    strict: bool = False,
) -> dict[str, Any]:
    """函数说明：选择 select_stable_params 主要逻辑。"""
    param_columns = _parameter_columns(summary, param_columns)
    if summary.empty:
        raise ValueError("Cannot select parameters from an empty validation summary.")
    candidates = _target_filtered_summary(summary, quality_config, fallback=not strict)
    if candidates.empty:
        raise ValueError("no_acceptable_params")
    best = candidates.iloc[0]
    return {column: _python_scalar(best[column]) for column in param_columns}


def assess_parameter_quality(summary: pd.DataFrame, quality_config: dict | None = None) -> ParameterQualityReport:
    """函数说明：评估 assess_parameter_quality 主要逻辑。"""
    cfg = quality_config or {}
    min_windows = int(cfg.get("min_validation_windows", 3))
    min_positive = float(cfg.get("min_positive_return_rate", 0.5))
    min_optimizer_return = float(cfg.get("min_optimizer_annual_return", cfg.get("target_annual_return", 0.20)))
    min_sharpe = float(cfg.get("min_sharpe_mean", 0.0))
    max_drawdown = float(cfg.get("max_drawdown_limit", -0.20))
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

    candidates = _target_filtered_summary(summary, quality_config, fallback=False)
    if candidates.empty:
        issues.append("no_acceptable_params")
        best = _target_filtered_summary(summary, quality_config, fallback=True).iloc[0]
    else:
        best = candidates.iloc[0]
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
    if annual_return_min < min_optimizer_return:
        issues.append(f"annual_return_min_below_threshold:{annual_return_min:.4f}<{min_optimizer_return:.4f}")
    if sharpe_mean < min_sharpe:
        issues.append(f"sharpe_mean_below_threshold:{sharpe_mean:.4f}<{min_sharpe:.4f}")
    if max_drawdown_worst < max_drawdown:
        issues.append(f"max_drawdown_worse_than_limit:{max_drawdown_worst:.4f}<{max_drawdown:.4f}")
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
) -> BacktestQualityReport:
    """函数说明：评估 assess_backtest_quality 主要逻辑。"""
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
            min_backtest_annual_return=min_return,
            max_backtest_drawdown_limit=max_drawdown,
            yearly_min_annual_return=0.0,
            yearly_worst_max_drawdown=0.0,
            min_yearly_annual_return=min_yearly_return,
            max_yearly_drawdown_limit=max_yearly_drawdown,
            years_below_return_target=[],
            years_breaching_drawdown_limit=[],
        )

    annual_return = _required_quality_number(
        metrics.get("annual_return"),
        "backtest_annual_return_missing_or_invalid",
        issues,
    )
    observed_drawdown = _required_quality_number(
        metrics.get("max_drawdown"),
        "backtest_max_drawdown_missing_or_invalid",
        issues,
    )
    calmar = _number(metrics.get("calmar"), 0.0)
    yearly_quality = _assess_yearly_backtest_quality(yearly, min_yearly_return, max_yearly_drawdown)

    if annual_return < min_return:
        issues.append(f"backtest_annual_return_below_threshold:{annual_return:.4f}<{min_return:.4f}")
    if observed_drawdown < max_drawdown:
        issues.append(f"backtest_max_drawdown_worse_than_limit:{observed_drawdown:.4f}<{max_drawdown:.4f}")
    issues.extend(yearly_quality["issues"])

    return BacktestQualityReport(
        is_acceptable=not issues,
        issues=issues,
        annual_return=annual_return,
        max_drawdown=observed_drawdown,
        calmar=calmar,
        min_backtest_annual_return=min_return,
        max_backtest_drawdown_limit=max_drawdown,
        yearly_min_annual_return=float(yearly_quality["yearly_min_annual_return"]),
        yearly_worst_max_drawdown=float(yearly_quality["yearly_worst_max_drawdown"]),
        min_yearly_annual_return=min_yearly_return,
        max_yearly_drawdown_limit=max_yearly_drawdown,
        years_below_return_target=list(yearly_quality["years_below_return_target"]),
        years_breaching_drawdown_limit=list(yearly_quality["years_breaching_drawdown_limit"]),
    )


def _assess_yearly_backtest_quality(
    yearly: pd.DataFrame | None,
    min_return: float,
    drawdown_limit: float,
) -> dict[str, Any]:
    """函数说明：评估逐年收益和回撤是否满足目标。"""
    result: dict[str, Any] = {
        "issues": [],
        "yearly_min_annual_return": 0.0,
        "yearly_worst_max_drawdown": 0.0,
        "years_below_return_target": [],
        "years_breaching_drawdown_limit": [],
    }
    if yearly is None:
        return result
    if yearly.empty:
        result["issues"] = ["backtest_yearly_breakdown_empty"]
        return result
    required = {"year", "annual_return", "max_drawdown"}
    if not required.issubset(yearly.columns):
        result["issues"] = [f"backtest_yearly_breakdown_missing_columns:{','.join(sorted(required - set(yearly.columns)))}"]
        return result

    frame = yearly.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["annual_return"] = pd.to_numeric(frame["annual_return"], errors="coerce")
    frame["max_drawdown"] = pd.to_numeric(frame["max_drawdown"], errors="coerce")
    issues: list[str] = []
    invalid_year = frame["year"].isna()
    if invalid_year.any():
        rows = ",".join(str(position) for position, invalid in enumerate(invalid_year.to_list()) if invalid)
        issues.append(f"backtest_yearly_year_missing_or_invalid:rows={rows}")
    frame = frame.dropna(subset=["year"])

    invalid_annual = frame["annual_return"].isna() | frame["annual_return"].abs().eq(float("inf"))
    if invalid_annual.any():
        years = ",".join(str(int(year)) for year in dict.fromkeys(frame.loc[invalid_annual, "year"].to_list()))
        issues.append(f"backtest_yearly_annual_return_missing_or_invalid:{years}")
        frame.loc[invalid_annual, "annual_return"] = float("nan")
    invalid_drawdown = frame["max_drawdown"].isna() | frame["max_drawdown"].abs().eq(float("inf"))
    if invalid_drawdown.any():
        years = ",".join(str(int(year)) for year in dict.fromkeys(frame.loc[invalid_drawdown, "year"].to_list()))
        issues.append(f"backtest_yearly_max_drawdown_missing_or_invalid:{years}")
        frame.loc[invalid_drawdown, "max_drawdown"] = float("nan")

    annual = frame["annual_return"].dropna()
    drawdown = frame["max_drawdown"].dropna()
    result["yearly_min_annual_return"] = float(annual.min()) if not annual.empty else 0.0
    result["yearly_worst_max_drawdown"] = float(drawdown.min()) if not drawdown.empty else 0.0

    below_return = frame[frame["annual_return"] < min_return]
    drawdown_breach = frame[frame["max_drawdown"] < drawdown_limit]
    result["years_below_return_target"] = [int(year) for year in below_return["year"].dropna().to_list()]
    result["years_breaching_drawdown_limit"] = [int(year) for year in drawdown_breach["year"].dropna().to_list()]

    if not below_return.empty:
        issues.append(
            "backtest_yearly_annual_return_below_threshold:"
            f"{_year_value_summary(below_return, 'annual_return')}<{min_return:.4f}"
        )
    if not drawdown_breach.empty:
        issues.append(
            "backtest_yearly_max_drawdown_worse_than_limit:"
            f"{_year_value_summary(drawdown_breach, 'max_drawdown')}<{drawdown_limit:.4f}"
        )
    result["issues"] = issues
    return result


def _year_value_summary(frame: pd.DataFrame, column: str) -> str:
    """函数说明：格式化年度质量失败的年份和值。"""
    parts = []
    for _, row in frame.iterrows():
        year = row.get("year")
        value = row.get(column)
        if pd.isna(year) or pd.isna(value):
            continue
        parts.append(f"{int(year)}={float(value):.4f}")
    return ",".join(parts)


def apply_strategy_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """函数说明：应用 apply_strategy_params 主要逻辑。"""
    selected = deepcopy(config)
    selected.setdefault("strategy", {})
    for key, value in params.items():
        if key in OPTIMIZABLE_PARAM_COLUMNS:
            selected["strategy"][key] = _python_scalar(value)
    return selected


def _target_filtered_summary(summary: pd.DataFrame, quality_config: dict | None, fallback: bool = True) -> pd.DataFrame:
    """函数说明：处理 target_filtered_summary 的内部辅助逻辑。"""
    cfg = quality_config or {}
    if summary.empty:
        return summary
    required = {"annual_return_mean", "max_drawdown_worst"}
    if not required.issubset(summary.columns):
        return summary
    min_return = float(cfg.get("min_optimizer_annual_return", cfg.get("target_annual_return", 0.20)))
    drawdown_limit = float(cfg.get("max_drawdown_limit", cfg.get("max_backtest_drawdown_limit", -0.20)))
    annual = pd.to_numeric(summary["annual_return_mean"], errors="coerce")
    drawdown = pd.to_numeric(summary["max_drawdown_worst"], errors="coerce")
    mask = (annual >= min_return) & (drawdown >= drawdown_limit)
    if "annual_return_min" in summary.columns:
        annual_min = pd.to_numeric(summary["annual_return_min"], errors="coerce")
        mask &= annual_min >= min_return
    if "annual_turnover_mean" in summary.columns:
        turnover = pd.to_numeric(summary["annual_turnover_mean"], errors="coerce")
        mask &= turnover <= float(cfg.get("max_annual_turnover", float("inf")))
    if "annual_trade_cost_ratio_mean" in summary.columns:
        cost = pd.to_numeric(summary["annual_trade_cost_ratio_mean"], errors="coerce")
        mask &= cost <= float(cfg.get("max_annual_trade_cost_ratio", float("inf")))
    filtered = summary[mask]
    if not filtered.empty:
        return filtered
    return summary if fallback else summary.iloc[0:0]


def _parameter_columns(frame: pd.DataFrame, param_columns: Iterable[str] | None) -> list[str]:
    """函数说明：处理 parameter_columns 的内部辅助逻辑。"""
    if param_columns is not None:
        return list(param_columns)
    inferred = [column for column in OPTIMIZABLE_PARAM_COLUMNS if column in frame.columns]
    return inferred or list(PARAM_COLUMNS)


def _python_scalar(value: Any) -> Any:
    """函数说明：处理 python_scalar 的内部辅助逻辑。"""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _number(value: Any, default: float) -> float:
    """函数说明：处理 number 的内部辅助逻辑。"""
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def _required_quality_number(value: Any, issue: str, issues: list[str]) -> float:
    """Parse required quality evidence and record a fail-closed issue when unavailable."""
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed) or abs(float(parsed)) == float("inf"):
        issues.append(issue)
        return 0.0
    return float(parsed)
