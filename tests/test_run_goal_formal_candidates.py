from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.run_goal_formal_candidates import (
    _candidate_specs,
    _filter_candidates,
    _load_existing_candidate_rows,
    _quality_flags,
    _score_key,
    _split_filter_values,
    _write_candidate_artifacts,
    _yearly_pass_counts,
)


class RunGoalFormalCandidatesTests(unittest.TestCase):
    def test_quality_flags_require_return_drawdown_turnover_and_cost(self) -> None:
        quality = {
            "min_backtest_annual_return": 0.20,
            "max_backtest_drawdown_limit": -0.20,
            "max_annual_turnover": 20.0,
            "max_annual_trade_cost_ratio": 0.2,
        }

        passing = _quality_flags(
            {
                "annual_return": 0.21,
                "max_drawdown": -0.19,
                "annual_turnover": 19.0,
                "annual_trade_cost_ratio": 0.1,
            },
            quality,
            pd.DataFrame([{"year": 2024, "annual_return": 0.21, "max_drawdown": -0.10}]),
            pd.DataFrame([{"year": 2024, "passes_min_days": True}]),
        )
        failing = _quality_flags(
            {
                "annual_return": 0.30,
                "max_drawdown": -0.50,
                "annual_turnover": 10.0,
                "annual_trade_cost_ratio": 0.1,
            },
            quality,
            pd.DataFrame([{"year": 2024, "annual_return": 0.30, "max_drawdown": -0.10}]),
            pd.DataFrame([{"year": 2024, "passes_min_days": True}]),
        )

        self.assertTrue(passing["is_acceptable"])
        self.assertFalse(failing["is_acceptable"])
        self.assertFalse(failing["drawdown_pass"])

    def test_quality_flags_require_yearly_return_and_drawdown_when_yearly_stats_are_provided(self) -> None:
        quality = {
            "min_backtest_annual_return": 0.20,
            "max_backtest_drawdown_limit": -0.20,
            "min_yearly_annual_return": 0.10,
            "max_yearly_drawdown_limit": -0.20,
            "max_annual_turnover": 20.0,
            "max_annual_trade_cost_ratio": 0.2,
        }
        metrics = {
            "annual_return": 0.21,
            "max_drawdown": -0.19,
            "annual_turnover": 19.0,
            "annual_trade_cost_ratio": 0.1,
        }
        weak_year = pd.DataFrame(
            [
                {"year": 2023, "annual_return": 0.12, "max_drawdown": -0.10},
                {"year": 2024, "annual_return": -0.02, "max_drawdown": -0.08},
            ]
        )
        passing_years = pd.DataFrame(
            [
                {"year": 2023, "annual_return": 0.12, "max_drawdown": -0.10},
                {"year": 2024, "annual_return": 0.11, "max_drawdown": -0.18},
            ]
        )

        weak = _quality_flags(metrics, quality, weak_year)
        passing = _quality_flags(metrics, quality, passing_years)

        self.assertFalse(weak["is_acceptable"])
        self.assertFalse(weak["yearly_annual_return_pass"])
        self.assertTrue(weak["yearly_drawdown_pass"])
        self.assertEqual(weak["year_ann_pass"], 1)
        self.assertEqual(weak["year_dd_pass"], 2)
        self.assertAlmostEqual(weak["min_year_annual_return"], -0.02)
        self.assertTrue(passing["is_acceptable"])

    def test_quality_flags_require_yearly_coverage_when_provided(self) -> None:
        quality = {
            "min_backtest_annual_return": 0.20,
            "max_backtest_drawdown_limit": -0.20,
            "min_yearly_annual_return": 0.10,
            "max_yearly_drawdown_limit": -0.20,
            "max_annual_turnover": 20.0,
            "max_annual_trade_cost_ratio": 0.2,
        }
        metrics = {
            "annual_return": 0.21,
            "max_drawdown": -0.19,
            "annual_turnover": 1.0,
            "annual_trade_cost_ratio": 0.01,
        }
        yearly = pd.DataFrame(
            [
                {"year": 2023, "annual_return": 0.12, "max_drawdown": -0.10},
                {"year": 2025, "annual_return": 0.11, "max_drawdown": -0.18},
            ]
        )
        coverage = pd.DataFrame(
            [
                {"year": 2023, "passes_min_days": True},
                {"year": 2024, "passes_min_days": True},
                {"year": 2025, "passes_min_days": True},
            ]
        )

        flags = _quality_flags(metrics, quality, yearly, coverage)

        self.assertFalse(flags["is_acceptable"])
        self.assertFalse(flags["year_coverage_pass"])
        self.assertEqual(flags["missing_years"], "2024")

    def test_yearly_pass_counts_use_quality_thresholds(self) -> None:
        yearly = pd.DataFrame(
            [
                {"year": 2022, "annual_return": 0.17, "max_drawdown": -0.16},
                {"year": 2023, "annual_return": 0.24, "max_drawdown": -0.22},
                {"year": 2024, "annual_return": 0.31, "max_drawdown": -0.10},
            ]
        )
        quality = {
            "min_backtest_annual_return": 0.25,
            "max_backtest_drawdown_limit": -0.15,
        }

        annual_passes, drawdown_passes = _yearly_pass_counts(yearly, quality)

        self.assertEqual(annual_passes, 1)
        self.assertEqual(drawdown_passes, 1)

    def test_yearly_pass_counts_use_yearly_threshold_overrides(self) -> None:
        yearly = pd.DataFrame(
            [
                {"year": 2022, "annual_return": 0.17, "max_drawdown": -0.16},
                {"year": 2023, "annual_return": 0.09, "max_drawdown": -0.22},
                {"year": 2024, "annual_return": 0.31, "max_drawdown": -0.10},
            ]
        )
        quality = {
            "min_backtest_annual_return": 0.25,
            "max_backtest_drawdown_limit": -0.15,
            "min_yearly_annual_return": 0.10,
            "max_yearly_drawdown_limit": -0.20,
        }

        annual_passes, drawdown_passes = _yearly_pass_counts(yearly, quality)

        self.assertEqual(annual_passes, 2)
        self.assertEqual(drawdown_passes, 2)

    def test_write_candidate_artifacts_persists_trades_and_holdings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
            result = SimpleNamespace(
                equity_curve=pd.Series([100.0, 101.0], index=dates, name="equity"),
                trades=pd.DataFrame([{"date": "2024-01-02", "instrument": "A", "side": "BUY"}]),
                holdings=pd.DataFrame([{"date": "2024-01-02", "instrument": "A", "value": 100.0}]),
            )

            paths = _write_candidate_artifacts(
                root / "candidate_a",
                result,
                pd.DataFrame([{"year": 2024, "annual_return": 0.1, "max_drawdown": 0.0}]),
                pd.DataFrame([{"year": 2024, "passes_min_days": True}]),
                pd.DataFrame(),
                {},
                write_diagnostics=False,
            )

            self.assertTrue(Path(paths["equity_path"]).exists())
            self.assertTrue(Path(paths["years_path"]).exists())
            self.assertTrue(Path(paths["year_coverage_path"]).exists())
            self.assertTrue(Path(paths["trades_path"]).exists())
            self.assertTrue(Path(paths["holdings_path"]).exists())
            self.assertEqual(pd.read_csv(paths["trades_path"]).iloc[0]["instrument"], "A")

    def test_split_filter_values_accepts_repeated_and_comma_separated_values(self) -> None:
        values = _split_filter_values(["a,b", " c "])

        self.assertEqual(values, ["a", "b", "c"])

    def test_filter_candidates_supports_exact_names_and_patterns(self) -> None:
        candidates = [{"name": "alpha"}, {"name": "beta_selrisk3"}, {"name": "beta_selrisk5"}]

        selected = _filter_candidates(candidates, exact_names=["alpha"], patterns=["*_selrisk3"])

        self.assertEqual([candidate["name"] for candidate in selected], ["alpha", "beta_selrisk3"])

    def test_filter_candidates_rejects_unknown_names(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Unknown candidate"):
            _filter_candidates([{"name": "alpha"}], exact_names=["missing"])

    def test_score_key_includes_regime_score_filter(self) -> None:
        base = {
            "strategy": {"factor_group": "momentum"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
        }
        filtered = {
            **base,
            "regime_score_filter": {
                "enabled": True,
                "rules": [{"regime": "bear", "components": [{"column": "ROC20", "direction": 1.0}], "min_score": 0.0}],
            },
        }

        self.assertNotEqual(_score_key(base), _score_key(filtered))

    def test_load_existing_candidate_rows_returns_completed_names(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            pd.DataFrame(
                [
                    {"candidate": "candidate_a", "annual_return": 0.21},
                    {"candidate": "candidate_b", "annual_return": 0.18},
                ]
            ).to_csv(path, index=False)

            rows, completed = _load_existing_candidate_rows(path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(completed, {"candidate_a", "candidate_b"})

    def test_candidate_specs_include_overlay_industry_cap_variants(self) -> None:
        candidates = {candidate["name"]: candidate for candidate in _candidate_specs()}

        indcap25 = candidates["momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_indcap25"]
        indcap20 = candidates["momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_indcap20"]

        self.assertEqual(indcap25["strategy"]["max_industry_weight"], 0.25)
        self.assertEqual(indcap20["strategy"]["max_industry_weight"], 0.20)
        self.assertTrue(indcap25["backtest"]["equity_overlay"]["rebalance_on_signal_only"])
        self.assertTrue(indcap20["backtest"]["equity_overlay"]["rebalance_on_signal_only"])

    def test_candidate_specs_include_market_drawdown_signal_only_schedule_variants(self) -> None:
        candidates = {candidate["name"]: candidate for candidate in _candidate_specs()}
        candidate = candidates[
            "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd08_schedsig_side06_bear02"
        ]
        milder = candidates[
            "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd12_schedsig_side1_bear05"
        ]

        self.assertTrue(candidate["defensive_timing"])
        self.assertEqual(candidate["market_regime"]["bear_drawdown_threshold"], 0.08)
        self.assertTrue(candidate["backtest"]["exposure_schedule_rebalance_on_signal_only"])
        self.assertTrue(candidate["backtest"]["equity_overlay"]["rebalance_on_signal_only"])
        self.assertEqual(milder["defensive_timing_config"]["sideways_exposure"], 1.0)
        self.assertEqual(milder["defensive_timing_config"]["bear_exposure"], 0.50)
        self.assertTrue(milder["backtest"]["exposure_schedule_rebalance_on_signal_only"])

    def test_candidate_specs_include_fast_drawdown_signal_only_schedule_variants(self) -> None:
        candidates = {candidate["name"]: candidate for candidate in _candidate_specs()}
        candidate = candidates[
            "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_fastdd20_12_schedsig_side1_bear03"
        ]

        self.assertEqual(candidate["market_regime"]["drawdown_window"], 20)
        self.assertEqual(candidate["market_regime"]["bear_drawdown_threshold"], 0.12)
        self.assertEqual(candidate["market_regime"]["bear_momentum_max"], -999.0)
        self.assertEqual(candidate["defensive_timing_config"]["sideways_exposure"], 1.0)
        self.assertEqual(candidate["defensive_timing_config"]["bear_exposure"], 0.30)
        self.assertTrue(candidate["backtest"]["exposure_schedule_rebalance_on_signal_only"])

    def test_candidate_specs_include_selection_risk_filter_variants(self) -> None:
        candidates = {candidate["name"]: candidate for candidate in _candidate_specs()}
        candidate = candidates[
            "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_selrisk5"
        ]

        risk_filter = candidate["backtest"]["selection_risk_filter"]
        self.assertTrue(risk_filter["enabled"])
        self.assertEqual(risk_filter["lookback_sessions"], 5)
        self.assertEqual(risk_filter["required_price_fields"], ["open", "close"])
        self.assertEqual(risk_filter["max_limit_down_days"], 0)


if __name__ == "__main__":
    unittest.main()
