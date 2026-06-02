from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import pandas as pd


PARAM_COLUMNS = ["factor_group", "top_n", "max_turnover", "rank_buffer", "rebalance_freq"]
METRIC_DEFAULTS = {
    "optimization_score": 0.0,
    "annual_return": 0.0,
    "sharpe": 0.0,
    "max_drawdown": 0.0,
    "annual_turnover": 0.0,
    "annual_trade_cost_ratio": 0.0,
    "win_rate": 0.0,
}


def summarize_parameter_validation(
    validation: pd.DataFrame,
    param_columns: Iterable[str] = PARAM_COLUMNS,
) -> pd.DataFrame:
    param_columns = list(param_columns)
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


def select_stable_params(summary: pd.DataFrame, param_columns: Iterable[str] = PARAM_COLUMNS) -> dict[str, Any]:
    param_columns = list(param_columns)
    if summary.empty:
        raise ValueError("Cannot select parameters from an empty validation summary.")
    best = summary.iloc[0]
    return {column: _python_scalar(best[column]) for column in param_columns}


def apply_strategy_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    selected = deepcopy(config)
    selected.setdefault("strategy", {})
    for key, value in params.items():
        if key in PARAM_COLUMNS:
            selected["strategy"][key] = _python_scalar(value)
    return selected


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
