"""Tests for annual state router score-backtest helpers."""

from __future__ import annotations

from argparse import Namespace
import unittest

import pandas as pd

from scripts.run_annual_state_router_backtest import (
    ScoreSourceDefinition,
    RoutedScoreRun,
    adjust_route_decision,
    apply_research_config_overrides,
    apply_source_top_n_overrides,
    default_source_definitions,
    exposure_schedule_from_year_routes,
    full_gate_summary,
    research_config_overrides_payload,
    risk_exit_min_positions_schedule_from_routes,
    routed_backtest_config,
    run_annual_state_score_router,
    selection_schedule_from_routes,
    signal_trade_date_map,
)


class RunAnnualStateRouterBacktestTests(unittest.TestCase):
    def test_expanded_source_definitions_include_beta20_and_rsqr20(self) -> None:
        definitions = default_source_definitions(
            factor_file="factors.parquet",
            industry_factor_file="industry.parquet",
            selector_file="selector.csv",
            include_expanded_sources=True,
        )

        beta20 = definitions["beta20"]
        self.assertEqual(beta20.factor_group, "factor:BETA20")
        self.assertEqual(beta20.top_n, 5)
        self.assertEqual(beta20.rank_buffer, 10)
        self.assertEqual(beta20.liquidity_quantile, 0.80)

        rsqr20 = definitions["rsqr20"]
        self.assertEqual(rsqr20.factor_group, "factor:RSQR20")
        self.assertEqual(rsqr20.top_n, 7)
        self.assertEqual(rsqr20.rank_buffer, 10)
        self.assertEqual(rsqr20.liquidity_quantile, 0.80)

    def test_apply_source_top_n_overrides_updates_research_route_definitions(self) -> None:
        definitions = {
            "beta": ScoreSourceDefinition(name="beta", kind="factor", top_n=5, max_turnover=1, rank_buffer=10),
            "beta20": ScoreSourceDefinition(name="beta20", kind="factor", top_n=5, max_turnover=1, rank_buffer=10),
        }

        result = apply_source_top_n_overrides(definitions, {"beta": 7})

        self.assertEqual(result["beta"].top_n, 7)
        self.assertEqual(result["beta"].max_turnover, 1)
        self.assertEqual(result["beta20"].top_n, 5)
        self.assertEqual(definitions["beta"].top_n, 5)

    def test_apply_source_top_n_overrides_rejects_invalid_source(self) -> None:
        definitions = {
            "beta": ScoreSourceDefinition(name="beta", kind="factor", top_n=5, max_turnover=1, rank_buffer=10),
        }

        with self.assertRaisesRegex(ValueError, "unknown source"):
            apply_source_top_n_overrides(definitions, {"missing": 6})

    def test_score_router_locks_route_for_year_and_uses_source_scores(self) -> None:
        date_2024 = pd.Timestamp("2024-01-31")
        date_2025 = pd.Timestamp("2025-01-31")
        beta = pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_tuples(
                [(date_2024, "BETA_A"), (date_2025, "BETA_B")],
                names=["date", "instrument"],
            ),
            name="score",
        )
        db_size = pd.Series(
            [3.0, 4.0],
            index=pd.MultiIndex.from_tuples(
                [(date_2024, "DB_A"), (date_2025, "DB_B")],
                names=["date", "instrument"],
            ),
            name="score",
        )
        sources = {
            "beta": beta,
            "db_size": db_size,
            "quality": beta,
            "selector": beta,
            "industry": beta,
        }
        definitions = {
            name: ScoreSourceDefinition(name=name, kind="factor", top_n=5, max_turnover=1, rank_buffer=10)
            for name in sources
        }
        price_dates = pd.to_datetime(["2024-01-02", "2024-02-01", "2025-01-02", "2025-02-03"])
        benchmark = pd.Series(
            [100.0, 105.0],
            index=pd.to_datetime(["2024-01-02", "2024-12-31"]),
        )

        routed = run_annual_state_score_router(
            score_sources=sources,
            source_definitions=definitions,
            price_dates=pd.DatetimeIndex(price_dates),
            benchmark=benchmark,
            initial_source="beta",
            missing_ret252_exposure=0.5,
            flat_negative_exposure=0.9,
        )

        self.assertEqual(routed.score_routes["source"].tolist(), ["beta", "db_size"])
        self.assertEqual(routed.scores.xs(date_2024, level=0).index.tolist(), ["BETA_A"])
        self.assertEqual(routed.scores.xs(date_2025, level=0).index.tolist(), ["DB_B"])

    def test_signal_trade_date_map_uses_next_price_date(self) -> None:
        signal_dates = [pd.Timestamp("2024-12-31")]
        price_dates = pd.DatetimeIndex(pd.to_datetime(["2024-12-31", "2025-01-02"]))

        result = signal_trade_date_map(signal_dates, price_dates)

        self.assertEqual(result[pd.Timestamp("2024-12-31")], pd.Timestamp("2025-01-02"))

    def test_score_router_uses_canonical_month_end_and_latest_prior_source_score(self) -> None:
        source_date = pd.Timestamp("2024-02-15")
        signal_date = pd.Timestamp("2024-02-29")
        beta = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(source_date, "BETA")], names=["date", "instrument"]),
            name="score",
        )
        sources = {name: beta for name in ["beta", "db_size", "quality", "selector", "industry"]}
        definitions = {
            name: ScoreSourceDefinition(name=name, kind="factor", top_n=5, max_turnover=1, rank_buffer=10)
            for name in sources
        }

        routed = run_annual_state_score_router(
            score_sources=sources,
            source_definitions=definitions,
            price_dates=pd.DatetimeIndex(pd.to_datetime(["2024-02-29", "2024-03-01"])),
            benchmark=pd.Series([100.0], index=pd.to_datetime(["2024-02-01"])),
            signal_dates=[signal_date],
            initial_source="beta",
            missing_ret252_exposure=0.5,
            flat_negative_exposure=0.9,
        )

        self.assertEqual(routed.score_routes["date"].tolist(), ["2024-02-29"])
        self.assertEqual(routed.scores.xs(signal_date, level=0).index.tolist(), ["BETA"])

    def test_score_router_raises_when_routed_source_has_no_scores(self) -> None:
        date = pd.Timestamp("2025-01-31")
        sources = {
            "beta": pd.Series(
                [1.0],
                index=pd.MultiIndex.from_tuples([(date, "BETA")], names=["date", "instrument"]),
            ),
            "db_size": pd.Series(
                dtype=float,
                index=pd.MultiIndex.from_tuples([], names=["date", "instrument"]),
            ),
            "quality": pd.Series([1.0], index=pd.MultiIndex.from_tuples([(date, "Q")], names=["date", "instrument"])),
            "selector": pd.Series([1.0], index=pd.MultiIndex.from_tuples([(date, "S")], names=["date", "instrument"])),
            "industry": pd.Series([1.0], index=pd.MultiIndex.from_tuples([(date, "I")], names=["date", "instrument"])),
        }
        definitions = {
            name: ScoreSourceDefinition(name=name, kind="factor", top_n=5, max_turnover=1, rank_buffer=10)
            for name in sources
        }
        benchmark = pd.Series([100.0], index=pd.to_datetime(["2024-12-31"]))

        with self.assertRaisesRegex(ValueError, "Score source is empty: db_size"):
            run_annual_state_score_router(
                score_sources=sources,
                source_definitions=definitions,
                price_dates=pd.DatetimeIndex(pd.to_datetime(["2025-01-02"])),
                benchmark=benchmark,
                initial_source="beta",
                missing_ret252_exposure=0.5,
                flat_negative_exposure=0.9,
            )

    def test_selection_schedule_can_force_full_turnover_on_route_change(self) -> None:
        routes = pd.DataFrame(
            [
                {"date": "2024-01-31", "source": "beta", "top_n": 5, "max_turnover": 1, "rank_buffer": 10},
                {"date": "2025-01-31", "source": "industry", "top_n": 10, "max_turnover": 1, "rank_buffer": 20},
            ]
        )

        schedule = selection_schedule_from_routes(routes, full_turnover_on_route_change=True)

        self.assertEqual(schedule["2024-01-31"]["max_turnover"], 1)
        self.assertEqual(schedule["2025-01-31"]["max_turnover"], 10)
        self.assertEqual(schedule["2025-01-31"]["rank_buffer"], 0)

    def test_score_router_can_boost_turnover_for_matching_route_reason(self) -> None:
        date = pd.Timestamp("2025-01-31")
        beta = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(date, "BETA")], names=["date", "instrument"]),
            name="score",
        )
        db_size = pd.Series(
            [2.0],
            index=pd.MultiIndex.from_tuples([(date, "DB")], names=["date", "instrument"]),
            name="score",
        )
        sources = {
            "beta": beta,
            "db_size": db_size,
            "quality": beta,
            "selector": beta,
            "industry": beta,
        }
        definitions = {
            name: ScoreSourceDefinition(name=name, kind="factor", top_n=5, max_turnover=1, rank_buffer=20)
            for name in sources
        }

        routed = run_annual_state_score_router(
            score_sources=sources,
            source_definitions=definitions,
            price_dates=pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-02-03"])),
            benchmark=pd.Series([100.0], index=pd.to_datetime(["2025-01-02"])),
            initial_source="beta",
            missing_ret252_exposure=0.5,
            flat_negative_exposure=0.9,
            turnover_boost_reasons={"insufficient_history"},
            turnover_boost_max_turnover=2,
            turnover_boost_rank_buffer=7,
        )

        row = routed.score_routes.iloc[0]
        self.assertEqual(row["reason"], "insufficient_history")
        self.assertEqual(int(row["max_turnover"]), 2)
        self.assertEqual(int(row["rank_buffer"]), 7)

    def test_exposure_schedule_uses_annual_decision_dates(self) -> None:
        year_routes = pd.DataFrame(
            [
                {"decision_date": "2024-01-02", "exposure": 1.0},
                {"decision_date": "2025-01-02", "exposure": 0.65},
            ]
        )

        self.assertEqual(exposure_schedule_from_year_routes(year_routes), {"2024-01-02": 1.0, "2025-01-02": 0.65})

    def test_risk_exit_min_positions_schedule_from_routes_filters_reasons(self) -> None:
        routes = pd.DataFrame(
            [
                {"date": "2025-12-31", "reason": "moderate_low_beta20"},
                {"date": "2026-01-30", "reason": "default_beta"},
            ]
        )

        schedule = risk_exit_min_positions_schedule_from_routes(
            routes,
            min_positions=5,
            reasons={"default_beta"},
        )

        self.assertEqual(schedule, {"2026-01-30": 5})

    def test_routed_backtest_config_scopes_router_min_positions_by_reason(self) -> None:
        routed = RoutedScoreRun(
            scores=pd.Series(dtype=float),
            score_routes=pd.DataFrame(
                [
                    {"date": "2025-12-31", "source": "beta20", "reason": "moderate_low_beta20", "top_n": 5, "max_turnover": 1, "rank_buffer": 10},
                    {"date": "2026-01-30", "source": "beta", "reason": "default_beta", "top_n": 5, "max_turnover": 1, "rank_buffer": 10},
                ]
            ),
            year_routes=pd.DataFrame([{"decision_date": "2026-01-30", "exposure": 1.0}]),
        )
        prices = pd.DataFrame(index=pd.to_datetime(["2026-01-30"]))
        definitions = {"beta": ScoreSourceDefinition(name="beta", kind="factor", top_n=5, max_turnover=1, rank_buffer=10)}

        config = routed_backtest_config(
            config={
                "strategy": {"top_n": 5, "max_turnover": 1, "rank_buffer": 10},
                "backtest": {"initial_capital": 100000},
                "annual_state_router": {"risk_exit_min_positions": 5, "risk_exit_min_positions_reasons": ["default_beta"]},
            },
            prices=prices,
            routed=routed,
            source_definitions=definitions,
            full_turnover_on_route_change=False,
            use_defensive_timing=False,
        )

        self.assertEqual(config["risk_exit_min_positions"], 0)
        self.assertEqual(config["risk_exit_min_positions_schedule"], {"2026-01-30": 5})

    def test_adjust_route_decision_discounts_strong_trailing_exposure(self) -> None:
        route = {
            "year": 2020,
            "source": "selector",
            "reason": "strong_trailing_market",
            "exposure": 1.0,
            "ret252": 0.35,
        }

        adjusted = adjust_route_decision(
            route,
            moderate_positive_source=None,
            moderate_positive_ret252_min=0.20,
            strong_trailing_exposure=0.9,
        )

        self.assertEqual(adjusted["source"], "selector")
        self.assertAlmostEqual(adjusted["exposure"], 0.9)

    def test_adjust_route_decision_can_route_moderate_positive_state(self) -> None:
        route = {
            "year": 2022,
            "source": "beta",
            "reason": "default_beta",
            "exposure": 1.0,
            "ret252": 0.22,
        }

        adjusted = adjust_route_decision(
            route,
            moderate_positive_source="roc60",
            moderate_positive_ret252_min=0.20,
            strong_trailing_exposure=1.0,
        )

        self.assertEqual(adjusted["source"], "roc60")
        self.assertEqual(adjusted["reason"], "moderate_positive_roc60")

    def test_adjust_route_decision_can_scale_moderate_positive_exposure(self) -> None:
        route = {
            "year": 2022,
            "source": "beta",
            "reason": "default_beta",
            "exposure": 1.0,
            "ret252": 0.22,
        }

        adjusted = adjust_route_decision(
            route,
            moderate_positive_source="roc60",
            moderate_positive_ret252_min=0.20,
            moderate_positive_exposure=0.8,
        )

        self.assertEqual(adjusted["source"], "roc60")
        self.assertEqual(adjusted["reason"], "moderate_positive_roc60")
        self.assertAlmostEqual(adjusted["exposure"], 0.8)

    def test_adjust_route_decision_can_route_low_moderate_positive_band(self) -> None:
        route = {
            "year": 2025,
            "source": "beta",
            "reason": "default_beta",
            "exposure": 1.0,
            "ret252": 0.185,
        }

        adjusted = adjust_route_decision(
            route,
            moderate_positive_source="roc60",
            moderate_positive_ret252_min=0.20,
            moderate_low_source="db_total",
            moderate_low_ret252_min=0.18,
            moderate_low_ret252_max=0.20,
            moderate_low_exposure=0.6,
        )

        self.assertEqual(adjusted["source"], "db_total")
        self.assertEqual(adjusted["reason"], "moderate_low_db_total")
        self.assertAlmostEqual(adjusted["exposure"], 0.6)

    def test_adjust_route_decision_can_route_lower_moderate_positive_band(self) -> None:
        route = {
            "year": 2026,
            "source": "beta",
            "reason": "default_beta",
            "exposure": 1.0,
            "ret252": 0.171,
        }

        adjusted = adjust_route_decision(
            route,
            moderate_positive_source="roc60",
            moderate_positive_ret252_min=0.20,
            moderate_low_source="beta20",
            moderate_low_ret252_min=0.18,
            moderate_low_ret252_max=0.20,
            moderate_low_exposure=0.4,
            moderate_lower_source="rsqr20",
            moderate_lower_ret252_min=0.16,
            moderate_lower_ret252_max=0.18,
            moderate_lower_exposure=0.5,
        )

        self.assertEqual(adjusted["source"], "rsqr20")
        self.assertEqual(adjusted["reason"], "moderate_lower_rsqr20")
        self.assertAlmostEqual(adjusted["exposure"], 0.5)

    def test_adjust_route_decision_prefers_high_band_over_low_band(self) -> None:
        route = {
            "year": 2022,
            "source": "beta",
            "reason": "default_beta",
            "exposure": 1.0,
            "ret252": 0.216,
        }

        adjusted = adjust_route_decision(
            route,
            moderate_positive_source="roc60",
            moderate_positive_ret252_min=0.20,
            moderate_low_source="db_total",
            moderate_low_ret252_min=0.18,
            moderate_low_ret252_max=0.20,
        )

        self.assertEqual(adjusted["source"], "roc60")
        self.assertEqual(adjusted["reason"], "moderate_positive_roc60")

    def test_full_gate_summary_includes_trade_cost_gate(self) -> None:
        summary = full_gate_summary(
            metrics={
                "annual_return": 0.30,
                "max_drawdown": -0.10,
                "annual_turnover": 5.0,
                "annual_trade_cost_ratio": 0.25,
            },
            audit_summary={"is_goal_met": True},
            config={"quality": {"max_annual_turnover": 20.0, "max_annual_trade_cost_ratio": 0.20}},
            return_target=0.20,
            drawdown_limit=-0.20,
        )

        self.assertFalse(summary["annual_trade_cost_ratio_pass"])
        self.assertFalse(summary["is_full_goal_met"])

    def test_apply_research_config_overrides_copies_nested_risk_sections(self) -> None:
        config = {
            "backtest": {"equity_overlay": {"sideways_exposure": 0.5, "bear_exposure": 0.5}},
            "defensive_timing": {"sideways_exposure": 1.0, "bear_exposure": 0.6},
        }
        args = Namespace(
            max_industry_weight=0.35,
            rebalance_after_risk_exit=True,
            risk_exit_min_positions=5,
            equity_overlay_sideways_exposure=0.7,
            equity_overlay_bear_exposure=None,
            equity_overlay_drawdown_cut=0.2,
            defensive_sideways_exposure=None,
            defensive_bear_exposure=0.8,
        )

        result = apply_research_config_overrides(config, args)
        payload = research_config_overrides_payload(args)

        self.assertEqual(result["strategy"]["max_industry_weight"], 0.35)
        self.assertTrue(result["strategy"]["rebalance_after_risk_exit"])
        self.assertEqual(result["strategy"]["risk_exit_min_positions"], 5)
        self.assertEqual(result["backtest"]["equity_overlay"]["sideways_exposure"], 0.7)
        self.assertEqual(result["backtest"]["equity_overlay"]["bear_exposure"], 0.5)
        self.assertEqual(result["backtest"]["equity_overlay"]["drawdown_cut"], 0.2)
        self.assertEqual(result["defensive_timing"]["bear_exposure"], 0.8)
        self.assertEqual(config["backtest"]["equity_overlay"]["sideways_exposure"], 0.5)
        self.assertEqual(
            payload,
            {
                "equity_overlay_sideways_exposure": 0.7,
                "max_industry_weight": 0.35,
                "rebalance_after_risk_exit": True,
                "risk_exit_min_positions": 5.0,
                "equity_overlay_drawdown_cut": 0.2,
                "defensive_bear_exposure": 0.8,
            },
        )


if __name__ == "__main__":
    unittest.main()
