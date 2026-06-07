from __future__ import annotations

import unittest

import pandas as pd

from src.selection_risk import filter_scores_by_selection_risk


class SelectionRiskTests(unittest.TestCase):
    def test_filter_masks_recent_limit_down_and_missing_required_price(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = _price_panel(
            dates,
            open_values={"A": [10.0, 9.0], "B": [10.0, None], "C": [10.0, 10.2]},
            close_values={"A": [10.0, 9.0], "B": [10.0, 10.1], "C": [10.0, 10.2]},
            low_values={"A": [10.0, 9.0], "B": [10.0, 10.0], "C": [10.0, 10.1]},
        )
        scores = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"], name="score")

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", _config())

        self.assertTrue(pd.isna(filtered.loc["A"]))
        self.assertTrue(pd.isna(filtered.loc["B"]))
        self.assertEqual(float(filtered.loc["C"]), 1.0)

    def test_filter_uses_only_signal_date_and_prior_history(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        prices = _price_panel(
            dates,
            open_values={"A": [10.0, 10.2, 9.0], "B": [10.0, 10.1, 10.2]},
            close_values={"A": [10.0, 10.2, 9.0], "B": [10.0, 10.1, 10.2]},
            low_values={"A": [10.0, 10.1, 9.0], "B": [10.0, 10.0, 10.1]},
        )
        scores = pd.Series([2.0, 1.0], index=["A", "B"], name="score")

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", _config())

        self.assertEqual(float(filtered.loc["A"]), 2.0)
        self.assertEqual(float(filtered.loc["B"]), 1.0)

    def test_growth_board_limit_down_uses_growth_threshold_when_star_threshold_differs(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        stock = "300001.SZ"
        prices = _price_panel(
            dates,
            open_values={stock: [10.0, 8.8]},
            close_values={stock: [10.0, 8.8]},
            low_values={stock: [10.0, 8.8]},
        )
        scores = pd.Series([1.0], index=[stock], name="score")
        config = _config()
        config["backtest"] = {"star_limit_down_threshold": 0.099, "growth_limit_down_threshold": 0.199}

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", config)

        self.assertEqual(float(filtered.loc[stock]), 1.0)


def _config() -> dict:
    return {
        "selection_risk_filter": {
            "enabled": True,
            "lookback_sessions": 2,
            "required_price_fields": ["open", "close"],
            "max_missing_price_sessions": 0,
            "max_limit_down_days": 0,
            "require_positive_volume": True,
        },
        "backtest": {"limit_down_threshold": 0.099},
    }


def _price_panel(
    dates: pd.DatetimeIndex,
    open_values: dict[str, list[float | None]],
    close_values: dict[str, list[float | None]],
    low_values: dict[str, list[float | None]],
) -> pd.DataFrame:
    columns = sorted(close_values)
    volume = {code: [1000.0] * len(dates) for code in columns}
    return pd.concat(
        {
            "open": pd.DataFrame(open_values, index=dates),
            "close": pd.DataFrame(close_values, index=dates),
            "low": pd.DataFrame(low_values, index=dates),
            "volume": pd.DataFrame(volume, index=dates),
        },
        axis=1,
    )


if __name__ == "__main__":
    unittest.main()
