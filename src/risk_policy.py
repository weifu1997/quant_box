"""Central accessors for configured selection and execution risk controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.selection_constraints import apply_selection_constraints_to_backtest_config, load_industry_group_map
from src.selection_risk import filter_scores_by_selection_risk, selection_risk_filter_enabled


EXECUTION_RISK_KEYS = (
    "slippage",
    "dynamic_slippage_enabled",
    "dynamic_slippage_threshold",
    "dynamic_slippage_multiplier",
    "max_slippage",
    "max_participation_rate",
    "capacity_window",
    "capacity_warning_threshold",
    "amount_unit",
)

STRATEGY_RISK_KEYS = (
    "max_industry_weight",
    "stop_loss_pct",
    "take_profit_pct",
)


@dataclass(frozen=True)
class RiskPolicy:
    """Thin wrapper around existing risk controls and configuration."""

    config: dict[str, Any]

    @property
    def strategy_config(self) -> dict[str, Any]:
        """Return the strategy config section."""
        value = self.config.get("strategy", {})
        return value if isinstance(value, dict) else {}

    @property
    def backtest_config(self) -> dict[str, Any]:
        """Return the backtest config section."""
        value = self.config.get("backtest", {})
        return value if isinstance(value, dict) else {}

    @property
    def max_industry_weight(self) -> float | None:
        """Return the configured single-industry weight cap."""
        return self._configured_value("max_industry_weight")

    @property
    def stop_loss_pct(self) -> object:
        """Return the configured stop-loss percentage."""
        return self._configured_value("stop_loss_pct")

    @property
    def take_profit_pct(self) -> object:
        """Return the configured take-profit percentage."""
        return self._configured_value("take_profit_pct")

    def execution_config(self) -> dict[str, Any]:
        """Return configured execution-risk parameters such as slippage and capacity."""
        values: dict[str, Any] = {}
        for key in EXECUTION_RISK_KEYS:
            value = self._configured_value(key)
            if value is not None:
                values[key] = value
        return values

    def selection_risk_enabled(self) -> bool:
        """Return whether pre-selection risk filtering is enabled."""
        return selection_risk_filter_enabled(self.config)

    def filter_selection_scores(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        signal_date: str | pd.Timestamp,
    ) -> pd.Series:
        """Apply configured pre-selection risk filters to candidate scores."""
        return filter_scores_by_selection_risk(scores, prices, signal_date, self.config)

    def industry_group_map(self) -> pd.Series | None:
        """Load the industry map only when an industry cap is active."""
        if self.max_industry_weight is None:
            return None
        return load_industry_group_map(self.config)

    def apply_to_backtest_config(
        self,
        backtest_config: dict[str, Any],
        *,
        force_industry_map: bool = False,
    ) -> dict[str, Any]:
        """Attach selection risk and group constraints to a backtest config."""
        result = dict(backtest_config)
        for key in STRATEGY_RISK_KEYS:
            value = self._configured_value(key)
            if value is not None and key not in result:
                result[key] = value
        for key, value in self.execution_config().items():
            result.setdefault(key, value)
        return apply_selection_constraints_to_backtest_config(
            result,
            self.config,
            force=force_industry_map,
        )

    def _configured_value(self, key: str) -> Any:
        if key in self.strategy_config:
            return self.strategy_config.get(key)
        if key in self.backtest_config:
            return self.backtest_config.get(key)
        return self.config.get(key)
