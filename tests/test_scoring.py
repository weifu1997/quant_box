from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.scoring import _dynamic_ic_selector_weights, _latest_dynamic_ic_selector_weights, build_latest_strategy_scores, build_strategy_scores
from tests.fixtures.real_data import require_real_market_data


class ScoringTests(unittest.TestCase):
    def test_build_strategy_scores_uses_configured_min_cross_section_obs(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-01-05")
        config = {"strategy": {"factor_group": "factor:LOW0", "min_cross_section_obs": 2}}

        scores = build_strategy_scores(market.factors, config)

        self.assertEqual(len(scores), len(market.factors))
        self.assertFalse(scores.isna().any())
        self.assertTrue(set(scores.index.get_level_values("instrument")).issubset(set(market.instruments)))

    def test_build_strategy_scores_uses_dynamic_ic_weights(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-04-30")
        config = {
            "strategy": {"factor_group": "ic_weighted", "min_cross_section_obs": 2},
            "ic": {"top_k": 2, "min_abs_ic": 0.0, "min_periods": 5, "window": 10, "min_obs": 2, "corr_threshold": 0.7},
        }

        scores = build_strategy_scores(market.factors, config, price_df=market.close)

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(market.factors))
        self.assertGreater(int(scores.notna().sum()), 0)
        self.assertTrue(set(scores.index.get_level_values("instrument")).issubset(set(market.instruments)))

    def test_build_strategy_scores_uses_dynamic_ic_selector(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-04-30")
        config = {
            "strategy": {"factor_group": "dynamic_ic_selector", "min_cross_section_obs": 2},
            "dynamic_ic_selector": {
                "candidates": ["factor:LOW0", "factor:ROC20"],
                "horizon": 1,
                "window": 10,
                "min_periods": 5,
                "min_obs": 2,
                "metric": "mean",
                "fallback_candidate": "factor:LOW0",
            },
        }

        scores = build_strategy_scores(market.factors, config, price_df=market.close)

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(market.factors))
        self.assertGreater(int(scores.notna().sum()), 0)
        self.assertTrue(set(scores.index.get_level_values("instrument")).issubset(set(market.instruments)))

    def test_dynamic_ic_selector_uses_configured_top_k_weights(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": range(5), "F2": range(5), "F3": range(5)}, index=index)
        prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
        rolling_ic = pd.DataFrame({"F1": [0.10], "F2": [0.05], "F3": [0.01]}, index=[pd.Timestamp("2024-01-02")])

        with patch("src.scoring.calculate_rolling_ic", return_value=rolling_ic):
            weights = _dynamic_ic_selector_weights(factors, prices, {"top_k": 2, "min_periods": 1})

        latest = weights[pd.Timestamp("2024-01-02")]
        self.assertEqual(set(latest.index), {"F1", "F2"})
        self.assertAlmostEqual(float(latest.sum()), 1.0)
        self.assertGreater(float(latest["F1"]), float(latest["F2"]))

    def test_dynamic_ic_selector_falls_back_when_top_scores_are_negative(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": range(5), "F2": range(5)}, index=index)
        prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
        rolling_ic = pd.DataFrame({"F1": [-0.01], "F2": [-0.02]}, index=[pd.Timestamp("2024-01-02")])

        with patch("src.scoring.calculate_rolling_ic", return_value=rolling_ic):
            weights = _dynamic_ic_selector_weights(
                factors,
                prices,
                {"top_k": 2, "min_periods": 1, "fallback_candidate": "factor:F2"},
            )

        latest = weights[pd.Timestamp("2024-01-02")]
        self.assertEqual(latest.to_dict(), {"F2": 1.0})

    def test_latest_dynamic_ic_selector_weights_uses_recent_history_only(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": range(len(index))}, index=index)
        prices = pd.DataFrame(10.0, index=dates, columns=["A", "B", "C", "D", "E"])
        captured_dates: list[pd.Timestamp] = []

        def fake_rolling_ic(factor_history: pd.DataFrame, *_args, **_kwargs) -> pd.DataFrame:
            captured_dates.extend(pd.to_datetime(factor_history.index.get_level_values("datetime")).unique())
            return pd.DataFrame({"F1": [0.1]}, index=[dates[-1]])

        config = {
            "dynamic_ic_selector": {
                "candidates": ["factor:F1"],
                "fallback_candidate": "factor:F1",
                "latest_weight_lookback_sessions": 4,
                "window": 3,
                "min_periods": 2,
                "horizon": 1,
            }
        }
        with patch("src.scoring.calculate_rolling_ic", side_effect=fake_rolling_ic):
            weights = _latest_dynamic_ic_selector_weights(factors, prices, config, dates[-1])

        self.assertEqual(weights.to_dict(), {"F1": 1.0})
        self.assertEqual(set(captured_dates), set(dates[-4:]))

    def test_build_strategy_scores_excludes_low_liquidity_bucket(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-04-30")
        config = {
            "strategy": {"factor_group": "factor:LOW0", "min_cross_section_obs": 2},
            "liquidity_filter": {"enabled": True, "field": "amount", "window": 5, "min_periods": 1, "quantile": 0.4, "side": "low"},
        }

        scores = build_strategy_scores(market.factors, config, price_df=market.prices)

        self.assertGreater(int(scores.isna().sum()), 0)
        self.assertGreater(int(scores.notna().sum()), 0)

    def test_build_strategy_scores_liquidity_filter_matches_price_columns_case_insensitively(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        index = pd.MultiIndex.from_product(
            [[dates[-1]], ["000001.SZ", "600519.SH"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": [1.0, 2.0]}, index=index)
        amount = pd.DataFrame(
            {
                "000001.sz": [1.0, 1.0],
                "600519.sh": [100.0, 100.0],
            },
            index=dates,
        )
        close = pd.DataFrame(10.0, index=dates, columns=amount.columns)
        prices = pd.concat({"close": close, "amount": amount}, axis=1)
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        config = {
            "strategy": {"factor_group": "factor:F1", "min_cross_section_obs": 2},
            "liquidity_filter": {
                "enabled": True,
                "field": "amount",
                "window": 2,
                "min_periods": 1,
                "quantile": 0.5,
                "side": "low",
            },
        }

        scores = build_strategy_scores(factors, config, price_df=prices)

        daily = scores.xs(pd.Timestamp("2024-01-02"), level=0)
        self.assertTrue(pd.isna(daily.loc["000001.SZ"]))
        self.assertFalse(pd.isna(daily.loc["600519.SH"]))

    def test_build_strategy_scores_liquidity_filter_uses_last_intraday_amount_per_date(self) -> None:
        price_dates = pd.to_datetime(["2024-01-01 15:00", "2024-01-02 09:30", "2024-01-02 15:00"])
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": [1.0, 2.0]}, index=index)
        amount = pd.DataFrame(
            {
                "A": [1.0, 1.0, 100.0],
                "B": [100.0, 100.0, 1.0],
            },
            index=price_dates,
        )
        close = pd.DataFrame(10.0, index=price_dates, columns=["A", "B"])
        prices = pd.concat({"close": close, "amount": amount}, axis=1)
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        config = {
            "strategy": {"factor_group": "factor:F1", "min_cross_section_obs": 2},
            "liquidity_filter": {
                "enabled": True,
                "field": "amount",
                "window": 1,
                "min_periods": 1,
                "quantile": 0.5,
                "side": "low",
            },
        }

        scores = build_strategy_scores(factors, config, price_df=prices)

        daily = scores.xs(pd.Timestamp("2024-01-02"), level=0)
        self.assertFalse(pd.isna(daily.loc["A"]))
        self.assertTrue(pd.isna(daily.loc["B"]))

    def test_build_strategy_scores_excludes_high_liquidity_bucket(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-04-30")
        config = {
            "strategy": {"factor_group": "factor:LOW0", "min_cross_section_obs": 2},
            "liquidity_filter": {"enabled": True, "field": "amount", "window": 5, "min_periods": 1, "quantile": 0.4, "side": "high"},
        }

        scores = build_strategy_scores(market.factors, config, price_df=market.prices)

        self.assertGreater(int(scores.isna().sum()), 0)
        self.assertGreater(int(scores.notna().sum()), 0)

    def test_build_strategy_scores_applies_regime_score_blend_to_real_data(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-04-30")
        dates = pd.to_datetime(market.factors.index.get_level_values("datetime")).normalize().unique()
        config = {
            "strategy": {"factor_group": "factor:LOW0", "min_cross_section_obs": 2},
            "regime_score_blend": {
                "enabled": True,
                "bear_defensive_weight": 0.5,
                "defensive_components": [{"column": "STD20", "direction": -1.0}],
            },
        }

        with patch("src.scoring.detect_market_regime", return_value=pd.Series("bear", index=dates)):
            scores = build_strategy_scores(market.factors, config, price_df=market.close)

        self.assertGreater(int(scores.notna().sum()), 0)
        self.assertEqual(scores.attrs["regime_score_blend"]["dates_blended"], len(dates))
        self.assertTrue(
            set(scores.index.get_level_values("instrument").str.lower()).issubset(set(market.instruments))
        )

    def test_build_strategy_scores_applies_regime_score_filter_to_real_data(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-04-30")
        dates = pd.to_datetime(market.factors.index.get_level_values("datetime")).normalize().unique()
        config = {
            "strategy": {"factor_group": "factor:LOW0", "min_cross_section_obs": 2},
            "regime_score_filter": {
                "enabled": True,
                "rules": [
                    {
                        "regime": "bear",
                        "components": [{"column": "ROC20", "direction": 1.0}],
                        "min_score": 0.0,
                    }
                ],
            },
        }

        with patch("src.scoring.detect_market_regime", return_value=pd.Series("bear", index=dates)):
            scores = build_strategy_scores(market.factors, config, price_df=market.close)

        self.assertGreater(int(scores.isna().sum()), 0)
        self.assertGreater(int(scores.notna().sum()), 0)
        self.assertGreater(scores.attrs["regime_score_filter"]["rows_removed"], 0)

    def test_build_strategy_scores_applies_regime_score_blend(self) -> None:
        date = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product([[date], ["A", "B"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [2.0, 1.0], "STD20": [10.0, 1.0]}, index=index)
        prices = pd.concat({"close": pd.DataFrame({"A": [10.0], "B": [10.0]}, index=[date])}, axis=1)
        config = {
            "strategy": {"factor_group": "factor:F1", "min_cross_section_obs": 2},
            "regime_score_blend": {
                "enabled": True,
                "bear_defensive_weight": 1.0,
                "defensive_components": [{"column": "STD20", "direction": -1.0}],
            },
        }

        with patch("src.scoring.detect_market_regime", return_value=pd.Series(["bear"], index=[date])):
            scores = build_strategy_scores(factors, config, price_df=prices)

        daily = scores.xs(date, level=0)
        self.assertGreater(daily.loc["B"], daily.loc["A"])
        self.assertEqual(scores.attrs["regime_score_blend"]["dates_blended"], 1)

    def test_build_strategy_scores_applies_regime_score_filter(self) -> None:
        date = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product([[date], ["A", "B", "C"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [3.0, 2.0, 1.0], "ROC20": [-0.5, 0.1, 0.5]}, index=index)
        prices = pd.concat({"close": pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0]}, index=[date])}, axis=1)
        config = {
            "strategy": {"factor_group": "factor:F1", "min_cross_section_obs": 2},
            "regime_score_filter": {
                "enabled": True,
                "rules": [
                    {
                        "regime": "bear",
                        "components": [{"column": "ROC20", "direction": 1.0}],
                        "min_score": 0.0,
                    }
                ],
            },
        }

        with patch("src.scoring.detect_market_regime", return_value=pd.Series(["bear"], index=[date])):
            scores = build_strategy_scores(factors, config, price_df=prices)

        daily = scores.xs(date, level=0)
        self.assertTrue(pd.isna(daily.loc["A"]))
        self.assertFalse(pd.isna(daily.loc["B"]))
        self.assertFalse(pd.isna(daily.loc["C"]))
        self.assertEqual(scores.attrs["regime_score_filter"]["rows_removed"], 1)

    def test_build_latest_strategy_scores_applies_regime_score_blend(self) -> None:
        date = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product([[date], ["A", "B"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [2.0, 1.0], "STD20": [10.0, 1.0]}, index=index)
        prices = pd.concat({"close": pd.DataFrame({"A": [10.0], "B": [10.0]}, index=[date])}, axis=1)
        config = {
            "strategy": {"factor_group": "factor:F1", "min_cross_section_obs": 2},
            "regime_score_blend": {
                "enabled": True,
                "bear_defensive_weight": 1.0,
                "defensive_components": [{"column": "STD20", "direction": -1.0}],
            },
        }

        with patch("src.scoring.detect_market_regime", return_value=pd.Series(["bear"], index=[date])):
            scores = build_latest_strategy_scores(factors, config, signal_date=date, price_df=prices)

        daily = scores.xs(date, level=0)
        self.assertGreater(daily.loc["B"], daily.loc["A"])

    def test_build_strategy_scores_passes_ic_stability_config(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": range(5), "F2": range(5, 0, -1)}, index=index)
        prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
        config = {
            "strategy": {"factor_group": "ic_weighted"},
            "ic": {
                "horizon": 5,
                "method": "pearson",
                "min_obs": 7,
                "top_k": 1,
                "min_abs_ic": 0.0,
                "min_periods": 1,
                "corr_threshold": 0.7,
                "weight_smoothing": 0.6,
                "max_weight_turnover": 0.5,
            },
        }

        with patch("src.scoring.calculate_rolling_ic", return_value=pd.DataFrame()) as rolling_ic, patch(
            "src.scoring.make_rolling_ic_weights",
            return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
        ) as make_weights:
            build_strategy_scores(factors, config, price_df=prices)

        rolling_kwargs = rolling_ic.call_args.kwargs
        self.assertEqual(rolling_kwargs["horizon"], 5)
        self.assertEqual(rolling_kwargs["method"], "pearson")
        self.assertEqual(rolling_kwargs["min_obs"], 7)
        kwargs = make_weights.call_args.kwargs
        self.assertEqual(kwargs["weight_smoothing"], 0.6)
        self.assertEqual(kwargs["max_weight_turnover"], 0.5)

    def test_build_strategy_scores_falls_back_to_close_price_file(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            close_path = tmp_path / "close.parquet"
            pd.DataFrame({"A": [10.0, 10.1]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])).to_parquet(close_path)
            missing_ohlcv = tmp_path / "ohlcv.parquet"
            index = pd.MultiIndex.from_product(
                [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
                names=["datetime", "instrument"],
            )
            factors = pd.DataFrame({"F1": range(5)}, index=index)
            config = {
                "strategy": {"factor_group": "ic_weighted"},
                "ic": {"price_file": str(missing_ohlcv), "top_k": 1, "min_abs_ic": 0.0, "min_periods": 1},
            }

            def fake_resolve_path(value: str | Path) -> Path:
                return Path(value)

            with patch("src.scoring.resolve_path", side_effect=fake_resolve_path), patch(
                "src.scoring.calculate_rolling_ic",
                return_value=pd.DataFrame(),
            ), patch(
                "src.scoring.make_rolling_ic_weights",
                return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
            ):
                scores = build_strategy_scores(factors, config)

        self.assertEqual(scores.name, "score")

    def test_build_strategy_scores_falls_back_to_adjusted_close_price_file(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            close_path = tmp_path / "close_adjusted.parquet"
            pd.DataFrame({"A": [10.0, 10.1]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])).to_parquet(close_path)
            missing_ohlcv = tmp_path / "ohlcv_adjusted.parquet"
            index = pd.MultiIndex.from_product(
                [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
                names=["datetime", "instrument"],
            )
            factors = pd.DataFrame({"F1": range(5)}, index=index)
            config = {
                "strategy": {"factor_group": "ic_weighted"},
                "ic": {"price_file": str(missing_ohlcv), "top_k": 1, "min_abs_ic": 0.0, "min_periods": 1},
            }

            def fake_resolve_path(value: str | Path) -> Path:
                return Path(value)

            with patch("src.scoring.resolve_path", side_effect=fake_resolve_path), patch(
                "src.scoring.calculate_rolling_ic",
                return_value=pd.DataFrame(),
            ), patch(
                "src.scoring.make_rolling_ic_weights",
                return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
            ):
                scores = build_strategy_scores(factors, config)

        self.assertEqual(scores.name, "score")

    def test_build_strategy_scores_reuses_matching_weight_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "weights.pkl"
            index = pd.MultiIndex.from_product(
                [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
                names=["datetime", "instrument"],
            )
            factors = pd.DataFrame({"F1": range(5)}, index=index)
            prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
            config = {
                "strategy": {"factor_group": "ic_weighted"},
                "ic": {
                    "weights_cache_file": str(cache_path),
                    "top_k": 1,
                    "min_abs_ic": 0.0,
                    "min_periods": 1,
                    "corr_threshold": 0.7,
                },
            }

            with patch("src.scoring.calculate_rolling_ic", return_value=pd.DataFrame()), patch(
                "src.scoring.make_rolling_ic_weights",
                return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
            ) as make_weights:
                build_strategy_scores(factors, config, price_df=prices)
                build_strategy_scores(factors, config, price_df=prices)

        make_weights.assert_called_once()

    def test_build_strategy_scores_invalidates_weight_cache_when_params_change(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "weights.pkl"
            index = pd.MultiIndex.from_product(
                [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
                names=["datetime", "instrument"],
            )
            factors = pd.DataFrame({"F1": range(5)}, index=index)
            prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
            base_config = {
                "strategy": {"factor_group": "ic_weighted"},
                "ic": {
                    "weights_cache_file": str(cache_path),
                    "top_k": 1,
                    "min_abs_ic": 0.0,
                    "min_periods": 1,
                    "corr_threshold": 0.7,
                },
            }
            changed_config = {
                **base_config,
                "ic": {**base_config["ic"], "top_k": 2},
            }

            with patch("src.scoring.calculate_rolling_ic", return_value=pd.DataFrame()), patch(
                "src.scoring.make_rolling_ic_weights",
                return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
            ) as make_weights:
                build_strategy_scores(factors, base_config, price_df=prices)
                build_strategy_scores(factors, changed_config, price_df=prices)

        self.assertEqual(make_weights.call_count, 2)

    def test_build_strategy_scores_invalidates_weight_cache_when_ic_label_config_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "weights.pkl"
            index = pd.MultiIndex.from_product(
                [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
                names=["datetime", "instrument"],
            )
            factors = pd.DataFrame({"F1": range(5)}, index=index)
            prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
            base_config = {
                "strategy": {"factor_group": "ic_weighted"},
                "ic": {
                    "weights_cache_file": str(cache_path),
                    "horizon": 1,
                    "method": "spearman",
                    "min_obs": 20,
                    "top_k": 1,
                    "min_abs_ic": 0.0,
                    "min_periods": 1,
                    "corr_threshold": 0.7,
                },
            }
            changed_config = {
                **base_config,
                "ic": {**base_config["ic"], "horizon": 2, "method": "pearson", "min_obs": 5},
            }

            with patch("src.scoring.calculate_rolling_ic", return_value=pd.DataFrame()), patch(
                "src.scoring.make_rolling_ic_weights",
                return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
            ) as make_weights:
                build_strategy_scores(factors, base_config, price_df=prices)
                build_strategy_scores(factors, changed_config, price_df=prices)

        self.assertEqual(make_weights.call_count, 2)

    def test_build_strategy_scores_invalidates_weight_cache_when_factor_values_change(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "weights.pkl"
            index = pd.MultiIndex.from_product(
                [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
                names=["datetime", "instrument"],
            )
            factors = pd.DataFrame({"F1": range(5)}, index=index)
            changed_factors = pd.DataFrame({"F1": range(10, 15)}, index=index)
            prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
            config = {
                "strategy": {"factor_group": "ic_weighted"},
                "ic": {
                    "weights_cache_file": str(cache_path),
                    "top_k": 1,
                    "min_abs_ic": 0.0,
                    "min_periods": 1,
                    "corr_threshold": 0.7,
                },
            }

            with patch("src.scoring.calculate_rolling_ic", return_value=pd.DataFrame()), patch(
                "src.scoring.make_rolling_ic_weights",
                return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
            ) as make_weights:
                build_strategy_scores(factors, config, price_df=prices)
                build_strategy_scores(changed_factors, config, price_df=prices)

        self.assertEqual(make_weights.call_count, 2)

    def test_build_latest_strategy_scores_uses_target_date_only_for_output(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "F1": list(range(15)),
                "F2": list(range(15, 0, -1)),
            },
            index=index,
        )
        prices = pd.DataFrame(
            {
                "A": [10.0, 10.1, 10.2, 10.3],
                "B": [10.0, 10.2, 10.3, 10.5],
                "C": [10.0, 10.3, 10.4, 10.6],
                "D": [10.0, 10.4, 10.5, 10.8],
                "E": [10.0, 10.5, 10.7, 11.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )
        config = {
            "strategy": {"factor_group": "ic_weighted"},
            "ic": {
                "top_k": 1,
                "min_abs_ic": 0.0,
                "min_obs": 1,
                "min_periods": 1,
                "window": 2,
                "latest_weight_lookback_sessions": 3,
            },
        }

        with patch("src.scoring.calculate_rolling_ic") as rolling_ic:
            scores = build_latest_strategy_scores(factors, config, signal_date="2024-01-04", price_df=prices)

        self.assertEqual(set(scores.index.get_level_values(0)), {pd.Timestamp("2024-01-04")})
        self.assertEqual(scores.name, "score")
        rolling_ic.assert_not_called()

    def test_build_latest_strategy_scores_passes_ic_config_to_factor_ic(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "F1": list(range(20)),
                "F2": list(range(20, 0, -1)),
            },
            index=index,
        )
        prices = pd.DataFrame(
            {
                "A": [10.0, 10.1, 10.2, 10.3],
                "B": [10.0, 10.2, 10.3, 10.5],
                "C": [10.0, 10.3, 10.4, 10.6],
                "D": [10.0, 10.4, 10.5, 10.8],
                "E": [10.0, 10.5, 10.7, 11.0],
            },
            index=dates,
        )
        config = {
            "strategy": {"factor_group": "ic_weighted"},
            "ic": {
                "horizon": 3,
                "method": "pearson",
                "min_obs": 2,
                "top_k": 1,
                "min_abs_ic": 0.0,
                "min_periods": 1,
                "window": 2,
                "latest_weight_lookback_sessions": 4,
            },
        }
        ic_frame = pd.DataFrame({"F1": [0.10, 0.20], "F2": [0.01, 0.02]}, index=dates[:2])

        with patch("src.scoring.calculate_factor_ic", return_value=ic_frame) as factor_ic:
            scores = build_latest_strategy_scores(factors, config, signal_date="2024-01-05", price_df=prices)

        kwargs = factor_ic.call_args.kwargs
        self.assertEqual(kwargs["horizon"], 3)
        self.assertEqual(kwargs["method"], "pearson")
        self.assertEqual(kwargs["min_obs"], 2)
        self.assertEqual(set(scores.index.get_level_values(0)), {pd.Timestamp("2024-01-05")})


if __name__ == "__main__":
    unittest.main()
