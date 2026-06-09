"""模块说明：将选股约束和行业映射应用到回测配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import resolve_path


def apply_selection_constraints_to_backtest_config(
    backtest_config: dict[str, Any],
    project_config: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """函数说明：应用 apply_selection_constraints_to_backtest_config 主要逻辑。"""
    result = dict(backtest_config)
    risk_filter = project_config.get("selection_risk_filter")
    if "selection_risk_filter" not in result and isinstance(risk_filter, dict):
        result["selection_risk_filter"] = dict(risk_filter)
    if not force and result.get("max_industry_weight") is None:
        return result
    if "industry_map" not in result:
        industry_map = load_industry_group_map(project_config)
        if industry_map is not None and not industry_map.empty:
            result["industry_map"] = industry_map
    return result


def load_industry_group_map(config: dict[str, Any]) -> pd.Series:
    """函数说明：加载 load_industry_group_map 主要逻辑。"""
    path = _industry_file(config)
    if path is None or not path.exists():
        return pd.Series(dtype=object, name="industry")
    frame = pd.read_csv(path)
    if "industry" not in frame.columns:
        return pd.Series(dtype=object, name="industry")
    code_col = next((column for column in ["ts_code", "con_code", "instrument", "code"] if column in frame.columns), None)
    if code_col is None:
        return pd.Series(dtype=object, name="industry")

    clean = frame.dropna(subset=[code_col]).drop_duplicates(code_col, keep="last")
    return pd.Series(
        clean["industry"].fillna("UNKNOWN").astype(str).to_numpy(),
        index=clean[code_col].astype(str).str.strip().str.upper(),
        name="industry",
    )


def _industry_file(config: dict[str, Any]) -> Path | None:
    """函数说明：处理 industry_file 的内部辅助逻辑。"""
    strategy_cfg = config.get("strategy", {}) if isinstance(config.get("strategy"), dict) else {}
    research_cfg = config.get("research", {}).get("exposure", {}) if isinstance(config.get("research"), dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    value = (
        config.get("industry_file")
        or strategy_cfg.get("industry_file")
        or research_cfg.get("industry_file")
        or data_cfg.get("constituents_file")
        or "data/raw/mainboard_a_stocks.csv"
    )
    if value is None:
        return None
    return resolve_path(value)
