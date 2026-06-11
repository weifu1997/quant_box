"""Tests for run_backtest logging summaries."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

from scripts.run_backtest import (
    _annotate_yearly_quality,
    _build_run_summary,
    _configure_run_logging,
    _drawdown_summary,
    _frame_summary,
    _holding_summary,
    _input_alignment_summary,
    _rebalance_summary,
    _score_summary,
    _trade_summary,
    _yearly_trade_summary,
    _yearly_failures,
)


class RunBacktestLoggingTests(unittest.TestCase):
    def test_drawdown_summary_reports_peak_trough_and_recovery(self) -> None:
        equity = pd.Series(
            [100.0, 120.0, 90.0, 130.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )

        summary = _drawdown_summary(equity)

        self.assertAlmostEqual(summary["max_drawdown"], -0.25)
        self.assertEqual(summary["peak_date"], "2024-01-03")
        self.assertEqual(summary["trough_date"], "2024-01-04")
        self.assertEqual(summary["recovery_date"], "2024-01-05")

    def test_trade_summary_counts_blocked_reasons_and_executable_rows(self) -> None:
        trades = pd.DataFrame(
            [
                {"instrument": "A", "side": "BUY", "status": "filled", "reason": ""},
                {"instrument": "B", "side": "BUY", "status": "blocked", "reason": "not_buyable"},
                {"instrument": "A", "side": "SELL", "status": "partial", "reason": "capacity_limited"},
            ]
        )

        summary = _trade_summary(trades)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["executable_rows"], 2)
        self.assertEqual(summary["blocked_rows"], 1)
        self.assertEqual(summary["status_counts"], {"filled": 1, "blocked": 1, "partial": 1})
        self.assertEqual(summary["side_counts"], {"buy": 2, "sell": 1})
        self.assertEqual(summary["blocked_reason_counts"], {"not_buyable": 1})

    def test_build_run_summary_includes_core_sections(self) -> None:
        equity = pd.Series(
            [100.0, 105.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            name="equity",
        )
        trades = pd.DataFrame(
            [{"instrument": "A", "side": "BUY", "status": "filled", "commission_cost": 1.0}]
        )
        result = SimpleNamespace(
            equity_curve=equity,
            trades=trades,
            metrics={
                "total_return": 0.05,
                "commission_cost": 1.0,
                "tax_cost": 0.0,
                "transfer_fee_cost": 0.0,
                "slippage_cost": 0.0,
                "trade_cost": 1.0,
                "trade_cost_ratio": 0.01,
                "annual_trade_cost_ratio": 0.01,
            },
        )
        yearly = pd.DataFrame([{"year": 2024, "total_return": 0.05}])

        summary = _build_run_summary(result, yearly, {"target_symbols": 1}, {"start_date": "2024-01-02"})

        self.assertIn("inputs", summary)
        self.assertIn("equity", summary)
        self.assertIn("drawdown", summary)
        self.assertIn("holdings", summary)
        self.assertIn("trades", summary)
        self.assertIn("rebalances", summary)
        self.assertIn("costs", summary)
        self.assertIn("yearly_trades", summary)
        self.assertIn("top_traded_instruments", summary)
        self.assertEqual(summary["yearly"], [{"year": 2024, "total_return": 0.05}])

    def test_holding_summary_reports_concentration_and_exposure(self) -> None:
        holdings = pd.DataFrame(
            [
                {"date": "2024-01-02", "instrument": "A", "value": 100.0},
                {"date": "2024-01-02", "instrument": "B", "value": 300.0},
                {"date": "2024-01-03", "instrument": "A", "value": 200.0},
            ]
        )
        equity = pd.Series(
            [1000.0, 1000.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

        summary = _holding_summary(holdings, equity)

        self.assertEqual(summary["holding_dates"], 2)
        self.assertEqual(summary["instruments"], 2)
        self.assertAlmostEqual(summary["avg_positions"], 1.5)
        self.assertAlmostEqual(summary["max_single_name_holding_weight"], 1.0)
        self.assertAlmostEqual(summary["avg_gross_exposure"], 0.3)
        self.assertEqual(summary["top_held_instruments_by_days"][0], {"instrument": "A", "holding_days": 2})

    def test_yearly_trade_summary_reports_costs_and_blocked_reasons(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "instrument": "A",
                    "side": "BUY",
                    "shares": 100,
                    "price": 10.0,
                    "status": "filled",
                    "commission_cost": 1.0,
                    "tax_cost": 0.0,
                    "transfer_fee_cost": 0.1,
                    "slippage_cost": 0.5,
                },
                {
                    "date": "2024-01-03",
                    "instrument": "B",
                    "side": "SELL",
                    "shares": 100,
                    "price": None,
                    "status": "blocked",
                    "reason": "not_sellable",
                },
                {
                    "date": "2025-01-02",
                    "instrument": "A",
                    "side": "SELL",
                    "shares": 50,
                    "price": 12.0,
                    "status": "partial",
                    "reason": "capacity_limited",
                    "slippage_cost": 0.2,
                    "capacity_warning": True,
                },
            ]
        )

        summary = _yearly_trade_summary(trades)

        self.assertEqual([row["year"] for row in summary], [2024, 2025])
        self.assertEqual(summary[0]["blocked_rows"], 1)
        self.assertEqual(summary[0]["blocked_reason_counts"], {"not_sellable": 1})
        self.assertAlmostEqual(summary[0]["gross_notional"], 1000.0)
        self.assertAlmostEqual(summary[0]["trade_cost"], 1.6)
        self.assertEqual(summary[1]["capacity_warning_count"], 1)

    def test_rebalance_summary_counts_signal_dates_once(self) -> None:
        trades = pd.DataFrame(
            [
                {"signal_date": "2024-01-01", "date": "2024-01-02"},
                {"signal_date": "2024-01-01", "date": "2024-01-02"},
                {"signal_date": "2024-02-01", "date": "2024-02-02"},
            ]
        )

        summary = _rebalance_summary(trades)

        self.assertEqual(summary["signal_dates"], 2)
        self.assertEqual(summary["trade_dates"], 2)
        self.assertAlmostEqual(summary["orders_per_signal_date_avg"], 1.5)

    def test_configure_run_logging_replaces_previous_backtest_file_handler(self) -> None:
        root_logger = logging.getLogger()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                first = _configure_run_logging(Path(tmp))
                second = _configure_run_logging(Path(tmp))

                handlers = [handler for handler in root_logger.handlers if getattr(handler, "_quant_box_backtest_file", False)]

                self.assertNotEqual(first, second)
                self.assertEqual(len(handlers), 1)
                self.assertEqual(Path(handlers[0].baseFilename), second)
            finally:
                for handler in list(root_logger.handlers):
                    if getattr(handler, "_quant_box_backtest_file", False):
                        root_logger.removeHandler(handler)
                        handler.close()

    def test_frame_summary_describes_multiindex_price_panel(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        frame = pd.concat(
            {
                "close": pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, 2.0]}, index=dates),
                "volume": pd.DataFrame({"A": [10.0, 20.0], "B": [10.0, 20.0]}, index=dates),
            },
            axis=1,
        )

        summary = _frame_summary(frame)

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["fields"], ["close", "volume"])
        self.assertEqual(summary["instruments"], 2)
        self.assertEqual(summary["start"], "2024-01-02")
        self.assertEqual(summary["end"], "2024-01-03")

    def test_summaries_handle_factor_and_score_multiindex(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-03"), "B"),
            ],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"LOW0": [1.0, 2.0]}, index=index)
        scores = pd.Series([1.0, 2.0], index=index)

        factor_summary = _frame_summary(factors)
        score_summary = _score_summary(scores)

        self.assertEqual(factor_summary["start"], "2024-01-02")
        self.assertEqual(factor_summary["end"], "2024-01-03")
        self.assertEqual(score_summary["signal_dates"], 2)
        self.assertEqual(score_summary["instruments"], 2)

    def test_yearly_quality_annotation_marks_failed_years(self) -> None:
        yearly = pd.DataFrame(
            [
                {"year": 2023, "annual_return": 0.25, "max_drawdown": -0.10},
                {"year": 2024, "annual_return": 0.12, "max_drawdown": -0.24},
            ]
        )
        gate = {"min_yearly_annual_return": 0.20, "max_drawdown_limit": -0.20}

        annotated = _annotate_yearly_quality(yearly, gate)
        failures = _yearly_failures(annotated, gate)

        self.assertTrue(bool(annotated.loc[0, "year_pass"]))
        self.assertFalse(bool(annotated.loc[1, "annual_return_pass"]))
        self.assertFalse(bool(annotated.loc[1, "drawdown_pass"]))
        self.assertEqual(failures["failed_years"], [2024])
        self.assertEqual(failures["passed_year_count"], 1)

    def test_input_alignment_summary_counts_common_dates_and_symbols(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, 2.0]}, index=dates),
            },
            axis=1,
        )
        factor_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-03"), "a"),
                (pd.Timestamp("2024-01-04"), "c"),
            ],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"LOW0": [1.0, 2.0]}, index=factor_index)

        summary = _input_alignment_summary(prices, factors)

        self.assertEqual(summary["common_dates"], 1)
        self.assertEqual(summary["first_common_date"], "2024-01-03")
        self.assertEqual(summary["last_common_date"], "2024-01-03")
        self.assertEqual(summary["common_symbols"], 1)
        self.assertEqual(summary["price_only_symbols"], 1)
        self.assertEqual(summary["factor_only_symbols"], 1)


if __name__ == "__main__":
    unittest.main()
