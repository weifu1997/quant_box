from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.research_diagnostics import build_research_diagnostics, write_research_diagnostics


class ResearchDiagnosticsTests(unittest.TestCase):
    def test_build_research_diagnostics_reports_benchmark_cost_attribution_and_exposure(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe_file = root / "universe.csv"
            daily_basic_file = root / "daily_basic.parquet"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "600519.SH"],
                    "industry": ["Bank", "Food"],
                }
            ).to_csv(universe_file, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-03", "2024-01-03"],
                    "ts_code": [" 000001.sz ", "600519.SH"],
                    "circ_mv": [100.0, 10000.0],
                }
            ).to_parquet(daily_basic_file)
            dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
            prices = pd.concat(
                {
                    "close": pd.DataFrame(
                        {
                            "000001.SZ": [10.0, 10.5, 10.0, 11.0],
                            "600519.SH": [100.0, 101.0, 103.0, 102.0],
                        },
                        index=dates,
                    )
                },
                axis=1,
            )
            equity = pd.Series([100000.0, 101000.0, 100500.0, 103000.0], index=dates, name="equity")
            holdings = pd.DataFrame(
                [
                    {"date": "2024-01-01", "instrument": "000001.SZ", "value": 50000.0},
                    {"date": "2024-01-01", "instrument": "600519.SH", "value": 50000.0},
                    {"date": "2024-01-02", "instrument": "000001.SZ", "value": 60000.0},
                    {"date": "2024-01-02", "instrument": "600519.SH", "value": 40000.0},
                    {"date": "2024-01-03", "instrument": "000001.SZ", "value": 70000.0},
                    {"date": "2024-01-03", "instrument": "600519.SH", "value": 30000.0},
                ]
            )
            trades = pd.DataFrame(
                [
                    {
                        "date": "2024-01-03",
                        "instrument": "000001.SZ",
                        "side": " sell ",
                        "status": " FILLED ",
                        "reason": " ",
                        "shares": 100,
                        "price": 10.0,
                        "cash": 1000.0,
                        "commission_cost": 10.0,
                        "tax_cost": 5.0,
                        "transfer_fee_cost": 1.0,
                        "slippage_cost": 8.0,
                        "capacity_warning": True,
                    },
                    {
                        "date": "2024-01-03",
                        "instrument": "600519.SH",
                        "side": "SELL",
                        "status": " RISK_EXIT ",
                        "reason": "stop_loss",
                        "shares": 100,
                        "price": 9.0,
                        "cash": 900.0,
                        "commission_cost": 0.0,
                        "tax_cost": 0.0,
                        "transfer_fee_cost": 0.0,
                        "slippage_cost": 0.0,
                        "capacity_warning": False,
                    },
                    {
                        "date": "2024-01-03",
                        "instrument": "600519.SH",
                        "side": "SELL",
                        "status": " BLOCKED ",
                        "reason": "not_sellable",
                        "shares": 100,
                        "price": 0.0,
                        "cash": 0.0,
                        "commission_cost": 0.0,
                        "tax_cost": 0.0,
                        "transfer_fee_cost": 0.0,
                        "slippage_cost": 0.0,
                        "capacity_warning": False,
                    },
                ]
            )
            config = {
                "backtest": {"annual_trading_days": 252},
                "research": {
                    "benchmark": {"method": "equal_weight_universe"},
                    "exposure": {
                        "industry_file": str(universe_file),
                        "daily_basic_file": str(daily_basic_file),
                        "market_cap_field": "circ_mv",
                    },
                },
            }

            diagnostics, tables = build_research_diagnostics(equity, holdings, trades, prices, config)
            paths = write_research_diagnostics(diagnostics, tables, root)

            self.assertTrue(diagnostics["enabled"])
            self.assertIn("benchmark_total_return", diagnostics["benchmark"])
            self.assertEqual(diagnostics["cost_attribution"]["total_trade_cost"], 24.0)
            self.assertEqual(diagnostics["cost_attribution"]["capacity_warning_count"], 1)
            self.assertEqual(diagnostics["turnover_attribution"]["normal_rebalance_sell_count"], 1)
            self.assertEqual(diagnostics["turnover_attribution"]["rebalance_trim_sell_count"], 1)
            self.assertEqual(diagnostics["turnover_attribution"]["rebalance_exit_sell_count"], 0)
            self.assertEqual(diagnostics["turnover_attribution"]["rebalance_trim_trade_count"], 1)
            self.assertEqual(diagnostics["turnover_attribution"]["rebalance_trim_notional"], 1000.0)
            self.assertEqual(diagnostics["turnover_attribution"]["rebalance_trim_trade_cost"], 24.0)
            self.assertEqual(diagnostics["turnover_attribution"]["rebalance_trim_cost_share_of_total_trade_cost"], 1.0)
            self.assertEqual(diagnostics["turnover_attribution"]["annual_turnover_without_rebalance_trims_estimate"], 168.0)
            self.assertEqual(diagnostics["turnover_attribution"]["risk_exit_sell_count"], 1)
            self.assertEqual(diagnostics["turnover_attribution"]["risk_exit_notional"], 900.0)
            self.assertEqual(diagnostics["turnover_attribution"]["blocked_sell_count"], 1)
            self.assertTrue(diagnostics["holding_attribution"]["enabled"])
            self.assertTrue(diagnostics["holding_attribution"]["max_drawdown_contribution_enabled"])
            self.assertIn("max_drawdown_top_negative_industries", diagnostics["holding_attribution"])
            self.assertTrue(diagnostics["exposure"]["enabled"])
            self.assertTrue(diagnostics["drawdown"]["enabled"])
            self.assertTrue(diagnostics["regime_returns"]["enabled"])
            self.assertGreaterEqual(diagnostics["regime_returns"]["regime_count"], 1)
            self.assertTrue(diagnostics["regime_trades"]["enabled"])
            self.assertGreaterEqual(diagnostics["regime_trades"]["regime_count"], 1)
            self.assertIn("top_negative_regime_industries", diagnostics["holding_attribution"])
            self.assertEqual(diagnostics["drawdown"]["max_drawdown_peak_date"], "2024-01-02")
            self.assertEqual(diagnostics["drawdown"]["max_drawdown_trough_date"], "2024-01-03")
            self.assertEqual(diagnostics["drawdown"]["max_drawdown_recovery_date"], "2024-01-04")
            self.assertEqual(diagnostics["drawdown"]["trades_peak_to_trough"], 3)
            self.assertIn("market_cap_buckets", diagnostics["exposure"])
            self.assertEqual(diagnostics["exposure"]["market_cap_asof_date"], "2024-01-03")
            self.assertEqual(diagnostics["exposure"]["market_cap_matched_position_count"], 2)
            self.assertEqual(diagnostics["exposure"]["market_cap_unknown_position_count"], 0)
            self.assertEqual(diagnostics["exposure"]["market_cap_matched_weight"], 1.0)
            self.assertIn("position_count", tables["market_cap_exposure"].columns)
            self.assertIn("market_cap_median", tables["market_cap_exposure"].columns)
            self.assertIn("research_diagnostics", paths)
            self.assertIn("research_drawdown_periods", paths)
            self.assertIn("research_max_drawdown_trade_costs_by_status_reason", paths)
            self.assertIn("research_max_drawdown_trade_costs_by_instrument", paths)
            self.assertIn("research_max_drawdown_holding_contributions", paths)
            self.assertIn("research_max_drawdown_industry_attribution", paths)
            self.assertIn("research_max_drawdown_instrument_attribution", paths)
            self.assertIn("research_regime_returns", paths)
            self.assertIn("research_regime_trade_costs", paths)
            self.assertIn("research_regime_trade_costs_by_reason", paths)
            self.assertIn("research_regime_industry_attribution", paths)
            self.assertIn("research_regime_instrument_attribution", paths)
            self.assertIn("research_industry_attribution", paths)
            self.assertIn("research_turnover_by_category", paths)
            self.assertIn("research_turnover_by_status_reason", paths)

    def test_write_research_diagnostics_handles_unrecovered_drawdown(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
            equity = pd.Series([100.0, 90.0, 80.0], index=dates, name="equity")
            prices = pd.concat({"close": pd.DataFrame({"000001.SZ": [10.0, 9.0, 8.0]}, index=dates)}, axis=1)

            diagnostics, tables = build_research_diagnostics(
                equity,
                pd.DataFrame(),
                pd.DataFrame(),
                prices,
                {"backtest": {"annual_trading_days": 252}},
            )
            paths = write_research_diagnostics(diagnostics, tables, root)

            self.assertIsNone(diagnostics["drawdown"]["max_drawdown_recovery_date"])
            self.assertIn("research_diagnostics", paths)

    def test_benchmark_comparison_includes_first_price_return(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        equity = pd.Series([100.0, 100.0, 100.0], index=dates, name="equity")
        prices = pd.concat({"close": pd.DataFrame({"000001.SZ": [100.0, 200.0, 200.0]}, index=dates)}, axis=1)

        diagnostics, _tables = build_research_diagnostics(
            equity,
            pd.DataFrame(),
            pd.DataFrame(),
            prices,
            {"backtest": {"annual_trading_days": 252}, "research": {"benchmark": {"method": "equal_weight_universe"}}},
        )

        self.assertAlmostEqual(diagnostics["benchmark"]["benchmark_total_return"], 1.0)

    def test_benchmark_uses_last_intraday_close_per_trade_date(self) -> None:
        equity_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        equity = pd.Series([100.0, 105.0, 110.0], index=equity_dates, name="equity")
        price_dates = pd.to_datetime(["2024-01-02 15:00", "2024-01-02 09:30", "2024-01-03 15:00", "2024-01-04 15:00"])
        prices = pd.concat(
            {
                "close": pd.DataFrame(
                    {
                        "000001.SZ": [10.0, 30.0, 20.0, 20.0],
                    },
                    index=price_dates,
                )
            },
            axis=1,
        )

        diagnostics, _tables = build_research_diagnostics(
            equity,
            pd.DataFrame(),
            pd.DataFrame(),
            prices,
            {"backtest": {"annual_trading_days": 252}, "research": {"benchmark": {"method": "equal_weight_universe"}}},
        )

        self.assertAlmostEqual(diagnostics["benchmark"]["benchmark_total_return"], 1.0)

    def test_research_diagnostics_accepts_plain_close_price_panel(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        equity = pd.Series([100.0, 101.0, 103.0], index=dates, name="equity")
        prices = pd.DataFrame(
            {
                "000001.sz": [10.0, 11.0, 12.0],
                "600519.sh": [100.0, 99.0, 101.0],
            },
            index=dates,
        )
        holdings = pd.DataFrame(
            [
                {"date": "2024-01-01", "instrument": "000001.SZ", "value": 100.0},
                {"date": "2024-01-02", "instrument": "000001.SZ", "value": 100.0},
            ]
        )

        diagnostics, tables = build_research_diagnostics(
            equity,
            holdings,
            pd.DataFrame(),
            prices,
            {
                "backtest": {"annual_trading_days": 252},
                "research": {"benchmark": {"method": "equal_weight_universe"}},
            },
        )

        self.assertNotIn("benchmark_unavailable", diagnostics["benchmark"]["issues"])
        self.assertIn("benchmark_total_return", diagnostics["benchmark"])
        self.assertTrue(diagnostics["holding_attribution"]["enabled"])
        self.assertIn("holding_contributions", tables)

    def test_research_diagnostics_rejects_flat_ohlcv_price_frame(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        equity = pd.Series([100.0, 101.0, 103.0], index=dates, name="equity")
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.5, 11.0],
                "close": [10.2, 10.8, 11.2],
                "volume": [1000.0, 1200.0, 1300.0],
            },
            index=dates,
        )

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            build_research_diagnostics(
                equity,
                pd.DataFrame(),
                pd.DataFrame(),
                prices,
                {"backtest": {"annual_trading_days": 252}},
            )

    def test_hs300_equal_weight_benchmark_deduplicates_constituent_snapshots(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            constituents = root / "hs300_constituents.csv"
            pd.DataFrame(
                {
                    "con_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                    "trade_date": ["2024-01-01", "2024-02-01", "2024-02-01"],
                }
            ).to_csv(constituents, index=False)
            dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
            equity = pd.Series([100.0, 100.0, 100.0], index=dates, name="equity")
            prices = pd.concat(
                {
                    "close": pd.DataFrame(
                        {
                            "000001.SZ": [100.0, 200.0, 200.0],
                            "000002.SZ": [100.0, 100.0, 100.0],
                        },
                        index=dates,
                    )
                },
                axis=1,
            )

            diagnostics, _tables = build_research_diagnostics(
                equity,
                pd.DataFrame(),
                pd.DataFrame(),
                prices,
                {
                    "backtest": {"annual_trading_days": 252},
                    "data": {"hs300_constituents_file": str(constituents)},
                    "research": {"benchmark": {"method": "hs300_equal_weight"}},
                },
            )

        self.assertAlmostEqual(diagnostics["benchmark"]["benchmark_total_return"], 0.5)

    def test_regime_return_drawdown_counts_first_negative_return(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        equity = pd.Series([100.0, 90.0], index=dates, name="equity")
        prices = pd.concat({"close": pd.DataFrame({"000001.SZ": [10.0, 9.0]}, index=dates)}, axis=1)

        diagnostics, _tables = build_research_diagnostics(
            equity,
            pd.DataFrame(),
            pd.DataFrame(),
            prices,
            {
                "backtest": {"annual_trading_days": 252},
                "market_regime": {
                    "enabled": True,
                    "ma_window": 1,
                    "momentum_window": 1,
                    "volatility_window": 1,
                    "min_periods": 1,
                    "high_volatility_threshold": 10.0,
                    "lag_days": 0,
                },
            },
        )

        records = diagnostics["regime_returns"]["records"]
        self.assertAlmostEqual(records[0]["strategy_max_drawdown"], -0.10)

    def test_market_cap_exposure_uses_point_in_time_asof_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_basic_file = root / "daily_basic.parquet"
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-03", "2024-01-03"],
                    "ts_code": ["000001.SZ", "600519.SH"],
                    "circ_mv": [100.0, 10000.0],
                }
            ).to_parquet(daily_basic_file)

            dates = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
            prices = pd.concat(
                {
                    "close": pd.DataFrame(
                        {
                            "000001.SZ": [10.0, 10.2, 10.1],
                            "600519.SH": [100.0, 99.0, 101.0],
                            "000002.SZ": [8.0, 8.1, 8.0],
                        },
                        index=dates,
                    )
                },
                axis=1,
            )
            equity = pd.Series([100000.0, 100500.0, 100200.0], index=dates, name="equity")
            holdings = pd.DataFrame(
                [
                    {"date": "2024-01-05", "instrument": "000001.SZ", "value": 50000.0},
                    {"date": "2024-01-05", "instrument": "600519.SH", "value": 25000.0},
                    {"date": "2024-01-05", "instrument": "000002.SZ", "value": 25000.0},
                ]
            )
            config = {
                "backtest": {"annual_trading_days": 252},
                "research": {
                    "exposure": {
                        "daily_basic_file": str(daily_basic_file),
                        "market_cap_field": "circ_mv",
                        "market_cap_min_matched_weight": 0.7,
                        "market_cap_max_staleness_days": 3,
                    },
                },
            }

            diagnostics, tables = build_research_diagnostics(equity, holdings, pd.DataFrame(), prices, config)

            self.assertEqual(diagnostics["exposure"]["market_cap_asof_date"], "2024-01-03")
            self.assertEqual(diagnostics["exposure"]["market_cap_staleness_days"], 2)
            self.assertEqual(diagnostics["exposure"]["market_cap_matched_position_count"], 2)
            self.assertEqual(diagnostics["exposure"]["market_cap_unknown_position_count"], 1)
            self.assertAlmostEqual(diagnostics["exposure"]["market_cap_matched_weight"], 0.75)
            self.assertNotIn("market_cap_exposure_unavailable", diagnostics["exposure"]["issues"])
            self.assertIn("market_cap_exposure", tables)


if __name__ == "__main__":
    unittest.main()
