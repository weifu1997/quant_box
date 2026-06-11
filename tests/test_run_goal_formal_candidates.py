"""模块说明：覆盖 test_run_goal_formal_candidates 相关行为的测试用例。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.run_goal_formal_candidates import (
    _candidate_specs,
    _candidate_error_row,
    _load_existing_candidate_rows,
    _quality_flags,
    _score_key,
    _scoring_config,
    _write_candidate_artifacts,
    _yearly_pass_counts,
)


class RunGoalFormalCandidatesTests(unittest.TestCase):
    """类说明：组织 RunGoalFormalCandidatesTests 测试用例。"""
    def test_quality_flags_require_return_drawdown_turnover_and_cost(self) -> None:
        """函数说明：验证 test_quality_flags_require_return_drawdown_turnover_and_cost 覆盖的行为场景。"""
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
                "year_count": 3,
                "year_ann_pass": 3,
                "year_dd_pass": 3,
            },
            quality,
        )
        failing = _quality_flags(
            {
                "annual_return": 0.30,
                "max_drawdown": -0.50,
                "annual_turnover": 10.0,
                "annual_trade_cost_ratio": 0.1,
                "year_count": 3,
                "year_ann_pass": 3,
                "year_dd_pass": 2,
            },
            quality,
        )

        self.assertTrue(passing["is_acceptable"])
        self.assertFalse(failing["is_acceptable"])
        self.assertFalse(failing["drawdown_pass"])
        self.assertFalse(failing["yearly_drawdown_pass"])

    def test_quality_flags_require_all_years_to_pass(self) -> None:
        quality = {
            "min_backtest_annual_return": 0.20,
            "max_backtest_drawdown_limit": -0.20,
            "max_annual_turnover": 20.0,
            "max_annual_trade_cost_ratio": 0.2,
        }

        flags = _quality_flags(
            {
                "annual_return": 0.30,
                "max_drawdown": -0.10,
                "annual_turnover": 5.0,
                "annual_trade_cost_ratio": 0.05,
                "year_count": 3,
                "year_ann_pass": 2,
                "year_dd_pass": 3,
            },
            quality,
        )

        self.assertFalse(flags["yearly_return_pass"])
        self.assertFalse(flags["yearly_all_pass"])
        self.assertFalse(flags["is_acceptable"])

    def test_quality_flags_require_every_year_to_pass_when_counts_are_available(self) -> None:
        quality = {
            "min_backtest_annual_return": 0.20,
            "max_backtest_drawdown_limit": -0.20,
            "max_annual_turnover": 20.0,
            "max_annual_trade_cost_ratio": 0.2,
        }

        flags = _quality_flags(
            {
                "annual_return": 0.30,
                "max_drawdown": -0.10,
                "annual_turnover": 10.0,
                "annual_trade_cost_ratio": 0.1,
                "year_count": 3,
                "year_ann_pass": 2,
                "year_dd_pass": 3,
            },
            quality,
        )

        self.assertFalse(flags["is_acceptable"])
        self.assertFalse(flags["yearly_annual_return_pass"])
        self.assertTrue(flags["yearly_drawdown_pass"])

    def test_yearly_pass_counts_use_quality_thresholds(self) -> None:
        """函数说明：验证 test_yearly_pass_counts_use_quality_thresholds 覆盖的行为场景。"""
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

    def test_write_candidate_artifacts_persists_trades_and_holdings(self) -> None:
        """函数说明：验证 test_write_candidate_artifacts_persists_trades_and_holdings 覆盖的行为场景。"""
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
                pd.DataFrame(),
                {},
                write_diagnostics=False,
            )

            self.assertTrue(Path(paths["equity_path"]).exists())
            self.assertTrue(Path(paths["years_path"]).exists())
            self.assertTrue(Path(paths["trades_path"]).exists())
            self.assertTrue(Path(paths["holdings_path"]).exists())
            self.assertEqual(pd.read_csv(paths["trades_path"]).iloc[0]["instrument"], "A")

    def test_score_key_includes_regime_score_filter(self) -> None:
        """函数说明：验证 test_score_key_includes_regime_score_filter 覆盖的行为场景。"""
        base = {
            "strategy": {"factor_group": "momentum"},
            "liquidity_filter": {"enabled": True, "side": "high", "quantile": 0.65},
        }
        filtered = {
            **base,
            "regime_score_filter": {
                "enabled": True,
                "rules": [{"regime": "bear", "components": [{"column": "ROC20", "direction": 1.0}], "min_score": 0.0}],
            },
        }

        self.assertNotEqual(_score_key(base), _score_key(filtered))

    def test_score_key_and_config_include_dynamic_ic_selector(self) -> None:
        base = {
            "strategy": {"factor_group": "dynamic_ic_selector"},
            "dynamic_ic_selector": {"candidates": ["factor:ROC60"], "top_k": 1},
        }
        changed = {
            "strategy": {"factor_group": "dynamic_ic_selector"},
            "dynamic_ic_selector": {"candidates": ["factor:ROC60", "inverse_factor:MAX60"], "top_k": 2},
        }
        config = {
            "strategy": {"factor_group": "momentum"},
            "dynamic_ic_selector": {"candidates": ["factor:LOW0"], "top_k": 1},
        }

        scoring = _scoring_config(config, base["strategy"], base)

        self.assertNotEqual(_score_key(base), _score_key(changed))
        self.assertEqual(scoring["dynamic_ic_selector"]["candidates"], ["factor:ROC60"])
        self.assertEqual(scoring["dynamic_ic_selector"]["top_k"], 1)

    def test_load_existing_candidate_rows_returns_completed_names(self) -> None:
        """函数说明：验证 test_load_existing_candidate_rows_returns_completed_names 覆盖的行为场景。"""
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

    def test_candidate_specs_loads_explicit_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_text('[{"name":"candidate_a","strategy":{"factor_group":"factor:KLEN"}}]', encoding="utf-8")

            candidates = _candidate_specs(path)

            self.assertEqual(candidates[0]["name"], "candidate_a")

    def test_candidate_error_row_is_not_acceptable(self) -> None:
        row = _candidate_error_row(
            {"name": "bad_candidate"},
            0.0,
            RuntimeError("boom"),
            {"min_backtest_annual_return": 0.20, "max_backtest_drawdown_limit": -0.20},
        )

        self.assertEqual(row["candidate"], "bad_candidate")
        self.assertIn("RuntimeError", row["error"])
        self.assertFalse(row["is_acceptable"])

    def test_candidate_specs_include_overlay_industry_cap_variants(self) -> None:
        """函数说明：验证 test_candidate_specs_include_overlay_industry_cap_variants 覆盖的行为场景。"""
        candidates = {candidate["name"]: candidate for candidate in _candidate_specs()}

        indcap25 = candidates["momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_indcap25"]
        indcap20 = candidates["momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_indcap20"]

        self.assertEqual(indcap25["strategy"]["max_industry_weight"], 0.25)
        self.assertEqual(indcap20["strategy"]["max_industry_weight"], 0.20)
        self.assertTrue(indcap25["backtest"]["equity_overlay"]["rebalance_on_signal_only"])
        self.assertTrue(indcap20["backtest"]["equity_overlay"]["rebalance_on_signal_only"])

    def test_candidate_specs_include_market_drawdown_signal_only_schedule_variants(self) -> None:
        """函数说明：验证 test_candidate_specs_include_market_drawdown_signal_only_schedule_variants 覆盖的行为场景。"""
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
        """函数说明：验证 test_candidate_specs_include_fast_drawdown_signal_only_schedule_variants 覆盖的行为场景。"""
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
        """函数说明：验证 test_candidate_specs_include_selection_risk_filter_variants 覆盖的行为场景。"""
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
