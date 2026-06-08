from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml_strategy import (
    _adjust_label_returns,
    _close_frame,
    _feature_columns,
    _neutralize_label_frame,
    _prepare_training_matrix,
    _transform_label_frame,
    build_ml_scores,
)
from tests.fixtures.real_data import require_real_market_data


class MLStrategyTests(unittest.TestCase):
    def test_build_ml_scores_with_real_market_data_end_to_end(self) -> None:
        market = require_real_market_data(
            start="2024-01-02",
            end="2024-04-30",
            factor_columns=("LOW0", "ROC5", "ROC20"),
        )
        signal_date = pd.to_datetime(market.factors.index.get_level_values("datetime")).normalize().max()
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 2,
                "min_train_rows": 20,
                "max_train_rows": 300,
                "feature_limit": None,
                "min_feature_fraction": 0.5,
            }
        }

        result = build_ml_scores(market.factors, market.prices, config, signal_dates=[signal_date])

        self.assertFalse(result.scores.dropna().empty)
        self.assertEqual(set(result.scores.index.get_level_values("datetime")), {signal_date})
        self.assertTrue(set(result.scores.index.get_level_values("instrument").str.lower()).issubset(set(market.instruments)))
        row = result.diagnostics.iloc[0]
        self.assertTrue(bool(row["no_lookahead"]))
        self.assertLess(pd.Timestamp(row["max_label_end"]), pd.Timestamp(row["signal_date"]))
        self.assertEqual(row["model_used"], "ridge_numpy")

    def test_cross_sectional_rank_label_mode_maps_returns_to_relative_ranks(self) -> None:
        returns = pd.DataFrame(
            {"A": [0.03, 0.01], "B": [0.01, 0.02], "C": [0.02, 0.03]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

        labels = _transform_label_frame(
            returns,
            {"label_mode": "cross_sectional_rank", "label_min_cross_section_obs": 3},
        )

        self.assertAlmostEqual(float(labels.loc["2024-01-02", "B"]), -1 / 3)
        self.assertAlmostEqual(float(labels.loc["2024-01-02", "C"]), 1 / 3)
        self.assertAlmostEqual(float(labels.loc["2024-01-02", "A"]), 1.0)

    def test_cross_sectional_long_short_label_mode_marks_top_and_bottom_quantiles(self) -> None:
        returns = pd.DataFrame(
            {"A": [0.05], "B": [0.04], "C": [0.03], "D": [0.02], "E": [0.01]},
            index=pd.to_datetime(["2024-01-02"]),
        )

        labels = _transform_label_frame(
            returns,
            {
                "label_mode": "cross_sectional_long_short",
                "label_min_cross_section_obs": 5,
                "label_top_quantile": 0.2,
                "label_bottom_quantile": 0.2,
            },
        )

        self.assertEqual(float(labels.loc["2024-01-02", "A"]), 1.0)
        self.assertEqual(float(labels.loc["2024-01-02", "E"]), -1.0)
        self.assertEqual(float(labels.loc["2024-01-02", "C"]), 0.0)

    def test_cross_sectional_top_quantile_label_mode_marks_binary_winners(self) -> None:
        returns = pd.DataFrame(
            {"A": [0.05], "B": [0.04], "C": [0.03], "D": [0.02], "E": [0.01]},
            index=pd.to_datetime(["2024-01-02"]),
        )

        labels = _transform_label_frame(
            returns,
            {
                "label_mode": "cross_sectional_top_quantile",
                "label_min_cross_section_obs": 5,
                "label_top_quantile": 0.2,
            },
        )

        self.assertEqual(float(labels.loc["2024-01-02", "A"]), 1.0)
        self.assertEqual(float(labels.loc["2024-01-02", "B"]), 0.0)
        self.assertEqual(float(labels.loc["2024-01-02", "E"]), 0.0)

    def test_label_return_adjustment_scales_by_trailing_volatility_without_future_returns(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)
        close = pd.DataFrame(
            {
                "LOW_VOL": [100.0, 101.0, 100.5, 101.0, 100.8, 101.0, 101.2, 101.4],
                "HIGH_VOL": [100.0, 110.0, 95.0, 112.0, 90.0, 115.0, 92.0, 118.0],
            },
            index=dates,
        )
        forward = pd.DataFrame({"LOW_VOL": [0.05] * len(dates), "HIGH_VOL": [0.05] * len(dates)}, index=dates)

        adjusted = _adjust_label_returns(
            forward,
            close,
            horizon=2,
            cfg={"label_return_adjustment": "vol_adjusted", "label_volatility_window": 4, "label_volatility_floor": 0.001},
        )

        last = adjusted.dropna().iloc[-1]
        self.assertGreater(float(last["LOW_VOL"]), float(last["HIGH_VOL"]))

    def test_feature_limit_keeps_extension_features(self) -> None:
        frame = pd.DataFrame(columns=["F1", "F2", "F3", "PX_LOW_AMOUNT_20", "DB_circ_mv"])

        columns = _feature_columns(frame, {"feature_limit": 2})

        self.assertEqual(columns, ["F1", "F2", "PX_LOW_AMOUNT_20", "DB_circ_mv"])

    def test_close_frame_keeps_latest_intraday_price_per_session(self) -> None:
        prices = pd.DataFrame(
            {("close", "a"): [10.0, 30.0, 20.0]},
            index=pd.to_datetime(["2024-01-02 15:00", "2024-01-02 09:30", "2024-01-03 15:00"]),
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])

        close = _close_frame(prices)

        self.assertEqual(close.index.to_list(), list(pd.to_datetime(["2024-01-02", "2024-01-03"])))
        self.assertAlmostEqual(float(close.loc[pd.Timestamp("2024-01-02"), "A"]), 10.0)

    def test_training_label_neutralization_removes_industry_means(self) -> None:
        labels = pd.DataFrame(
            {"A": [0.4], "B": [0.2], "C": [0.0], "D": [-0.1], "E": [-0.3], "F": [-0.5]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        industry = pd.Series({"A": "bank", "B": "bank", "C": "bank", "D": "tech", "E": "tech", "F": "tech"})

        neutralized = _neutralize_label_frame(
            labels,
            {"training_neutralization": {"enabled": True, "industry": True, "market_cap": False, "min_obs": 3}},
            industry_map=industry,
        )

        self.assertAlmostEqual(float(neutralized[["A", "B", "C"]].mean(axis=1).iloc[0]), 0.0)
        self.assertAlmostEqual(float(neutralized[["D", "E", "F"]].mean(axis=1).iloc[0]), 0.0)

    def test_training_label_neutralization_normalizes_symbol_inputs(self) -> None:
        labels = pd.DataFrame(
            {" a ": [0.4], "b": [0.2], " C ": [0.0], "d": [-0.1], " E ": [-0.3], "f": [-0.5]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        industry = pd.Series({"A": "bank", "B": "bank", "C": "bank", "D": "tech", "E": "tech", "F": "tech"})
        cap_values = [10.0, 20.0, 40.0]
        cap_labels = pd.DataFrame(
            {symbol: [float(np.log1p(cap))] for symbol, cap in zip([" a ", "b", " C "], cap_values, strict=True)},
            index=pd.to_datetime(["2024-01-02"]),
        )
        daily_basic = pd.DataFrame(
            {
                "trade_date": ["2024-01-02"] * 3,
                "ts_code": ["A", "B", "C"],
                "circ_mv": cap_values,
            }
        )

        industry_neutralized = _neutralize_label_frame(
            labels,
            {"training_neutralization": {"enabled": True, "industry": True, "market_cap": False, "min_obs": 3}},
            industry_map=industry,
        )
        cap_neutralized = _neutralize_label_frame(
            cap_labels,
            {"training_neutralization": {"enabled": True, "industry": False, "market_cap": True, "min_obs": 3}},
            daily_basic=daily_basic,
        )

        self.assertAlmostEqual(float(industry_neutralized[[" a ", "b", " C "]].mean(axis=1).iloc[0]), 0.0)
        self.assertAlmostEqual(float(industry_neutralized[["d", " E ", "f"]].mean(axis=1).iloc[0]), 0.0)
        self.assertTrue((cap_neutralized.abs() < 1e-12).all(axis=None))

    def test_training_label_neutralization_uses_latest_intraday_market_cap(self) -> None:
        cap_values = {"A": 10.0, "B": 20.0, "C": 40.0}
        labels = pd.DataFrame(
            {symbol: [float(np.log1p(cap))] for symbol, cap in cap_values.items()},
            index=pd.to_datetime(["2024-01-02"]),
        )
        daily_basic = pd.DataFrame(
            {
                "trade_date": ["2024-01-02 15:00", "2024-01-02 09:30", "2024-01-02", "2024-01-02"],
                "ts_code": ["A", "A", "B", "C"],
                "circ_mv": [10.0, 999.0, 20.0, 40.0],
            }
        )

        neutralized = _neutralize_label_frame(
            labels,
            {"training_neutralization": {"enabled": True, "industry": False, "market_cap": True, "min_obs": 3}},
            daily_basic=daily_basic,
        )

        self.assertTrue((neutralized.abs() < 1e-12).all(axis=None))

    def test_ranking_objective_prepares_query_groups_and_relevance_labels(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
        instruments = ["A", "B", "A", "B"]
        index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
        train_frame = pd.DataFrame({"F1": [1.0, 2.0, 3.0, 4.0]}, index=index)
        labels = pd.Series([0.1, 0.3, -0.2, 0.5], index=index).to_numpy()

        prepared = _prepare_training_matrix(
            train_frame,
            labels,
            min_feature_count=1,
            feature_weights=None,
            cfg={"model_objective": "ranking"},
        )

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared["group"], [2, 2])
        self.assertTrue((prepared["y"] >= 0).all())

    def test_ranking_objective_samples_complete_date_groups(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=20)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {"F1": range(len(index)), "F2": range(len(index), 0, -1)},
            index=index,
            dtype=float,
        )
        close = pd.DataFrame(
            {symbol: [100.0 + i + offset for i in range(len(dates))] for offset, symbol in enumerate(instruments)},
            index=dates,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "lightgbm",
                "model_objective": "ranking",
                "train_years": 1,
                "label_horizon_sessions": 1,
                "min_train_rows": 3,
                "max_train_rows": 4,
                "feature_limit": None,
            }
        }
        captured_groups: list[list[int]] = []

        class ConstantModel:
            def predict(self, X):
                return [0.0] * len(X)

        def fake_fit_lightgbm(X_train, y, cfg, seed, group=None):
            self.assertIsNotNone(group)
            self.assertEqual(sum(group), len(y))
            self.assertTrue(all(value > 1 for value in group))
            captured_groups.append(list(group))
            return ConstantModel()

        with patch("src.ml_strategy._fit_lightgbm_model", side_effect=fake_fit_lightgbm):
            result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.empty)
        self.assertEqual(result.diagnostics.iloc[0]["model_used"], "lightgbm")
        self.assertTrue(captured_groups)

    def test_build_ml_scores_uses_only_labels_known_before_signal_date(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=35)
        instruments = ["A", "B", "C", "D"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "F1": [float(i % 11) for i in range(len(index))],
                "F2": [float((i * 3) % 17) for i in range(len(index))],
            },
            index=index,
        )
        close = pd.DataFrame(
            {
                "A": range(100, 100 + len(dates)),
                "B": range(80, 80 + len(dates)),
                "C": range(60, 60 + len(dates)),
                "D": range(40, 40 + len(dates)),
            },
            index=dates,
            dtype=float,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 3,
                "min_train_rows": 8,
                "max_train_rows": 100,
                "feature_limit": None,
                "min_feature_fraction": 0.5,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.empty)
        self.assertEqual(set(result.scores.index.get_level_values("datetime")), {dates[-1]})
        row = result.diagnostics.iloc[0]
        self.assertTrue(bool(row["no_lookahead"]))
        self.assertLess(pd.Timestamp(row["max_label_end"]), pd.Timestamp(row["signal_date"]))
        self.assertEqual(row["model_used"], "ridge_numpy")
        self.assertGreater(int(row["train_rows_used"]), 0)

    def test_build_ml_scores_normalizes_price_and_factor_instruments(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=18)
        instruments = ["000001.SZ", "600519.SH", "000002.SZ"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {"F1": range(len(index)), "F2": range(len(index), 0, -1)},
            index=index,
            dtype=float,
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame(
                    {
                        instrument.lower(): [10.0 + day + offset for day in range(len(dates))]
                        for offset, instrument in enumerate(instruments)
                    },
                    index=dates,
                )
            },
            axis=1,
        )
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 1,
                "min_train_rows": 6,
                "max_train_rows": 100,
                "feature_limit": None,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.dropna().empty)
        self.assertEqual(result.diagnostics.iloc[0]["model_used"], "ridge_numpy")
        self.assertGreater(int(result.diagnostics.iloc[0]["train_rows_available"]), 0)
        self.assertEqual(set(result.scores.index.get_level_values("instrument")), set(instruments))

    def test_build_ml_scores_deduplicates_normalized_factor_instruments(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=18)
        raw_instruments = ["a", " A ", "B"]
        index = pd.MultiIndex.from_product([dates, raw_instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {"F1": range(len(index)), "F2": range(len(index), 0, -1)},
            index=index,
            dtype=float,
        )
        close = pd.DataFrame(
            {
                "A": [10.0 + day for day in range(len(dates))],
                "B": [20.0 + day for day in range(len(dates))],
            },
            index=dates,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 1,
                "min_train_rows": 6,
                "max_train_rows": 100,
                "feature_limit": None,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.empty)
        self.assertTrue(result.scores.index.is_unique)
        self.assertEqual(set(result.scores.index.get_level_values("instrument")), {"A", "B"})

    def test_build_ml_scores_rejects_flat_ohlcv_price_frame(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=18)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {"F1": range(len(index)), "F2": range(len(index), 0, -1)},
            index=index,
            dtype=float,
        )
        prices = pd.DataFrame(
            {
                "open": [10.0 + i for i in range(len(dates))],
                "close": [10.1 + i for i in range(len(dates))],
                "volume": [1000.0] * len(dates),
            },
            index=dates,
        )
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 1,
                "min_train_rows": 6,
                "max_train_rows": 100,
                "feature_limit": None,
            }
        }

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

    def test_build_ml_scores_supports_fractional_train_years(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=140)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "F1": [float(i % 13) for i in range(len(index))],
                "F2": [float((i * 5) % 17) for i in range(len(index))],
            },
            index=index,
        )
        close = pd.DataFrame(
            {
                "A": range(100, 100 + len(dates)),
                "B": range(80, 80 + len(dates)),
                "C": range(60, 60 + len(dates)),
            },
            index=dates,
            dtype=float,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 0.5,
                "label_horizon_sessions": 2,
                "min_train_rows": 12,
                "max_train_rows": 100,
                "feature_limit": None,
                "min_feature_fraction": 0.5,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.dropna().empty)
        row = result.diagnostics.iloc[0]
        self.assertEqual(row["model_used"], "ridge_numpy")
        self.assertLess(pd.Timestamp(row["train_start"]), pd.Timestamp(row["train_end"]))
        self.assertGreater(int(row["train_rows_used"]), 0)

    def test_build_ml_scores_auto_model_falls_back_to_available_model(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=28)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": range(len(index)), "F2": range(len(index), 0, -1)}, index=index, dtype=float)
        close = pd.DataFrame(10.0, index=dates, columns=instruments)
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "auto",
                "train_years": 1,
                "label_horizon_sessions": 2,
                "min_train_rows": 6,
                "max_train_rows": 50,
                "feature_limit": None,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        self.assertFalse(result.scores.empty)
        self.assertIn(result.diagnostics.iloc[0]["model_used"], {"lightgbm", "xgboost", "sklearn_gbdt", "ridge_numpy"})

    def test_build_ml_scores_ensembles_recent_models(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=45)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": range(len(index)), "F2": range(len(index), 0, -1)}, index=index, dtype=float)
        close = pd.DataFrame(
            {
                "A": range(100, 100 + len(dates)),
                "B": range(80, 80 + len(dates)),
                "C": range(60, 60 + len(dates)),
            },
            index=dates,
            dtype=float,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 2,
                "min_train_rows": 6,
                "max_train_rows": 50,
                "feature_limit": None,
                "ensemble_window": 2,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=dates[-3:])

        completed = result.diagnostics[result.diagnostics["skip_reason"] == ""]
        self.assertEqual(completed["ensemble_size"].astype(int).to_list(), [1, 2, 2])
        self.assertTrue(all(completed["ensemble_models"].astype(str).str.contains("ridge_numpy")))

    def test_build_ml_scores_filters_predictions_without_enough_price_history(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=35)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {"F1": range(len(index)), "F2": range(len(index), 0, -1)},
            index=index,
            dtype=float,
        )
        close = pd.DataFrame(
            {
                "A": range(100, 100 + len(dates)),
                "B": [float("nan")] * 32 + [10.0, 10.2, 10.3],
                "C": range(80, 80 + len(dates)),
            },
            index=dates,
            dtype=float,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 2,
                "min_train_rows": 10,
                "max_train_rows": 100,
                "feature_limit": None,
                "min_price_history_sessions": 5,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])
        valid_scores = result.scores.dropna()

        self.assertFalse(valid_scores.empty)
        self.assertNotIn("B", set(valid_scores.index.get_level_values("instrument")))

    def test_build_ml_scores_can_evolve_features_by_recent_ic(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=25)
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        per_day = []
        for _date in dates:
            per_day.extend([(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)])
        factors = pd.DataFrame(per_day, columns=["F1", "F2"], index=index)
        close = pd.DataFrame(
            {
                "A": [100 * (1.01**i) for i in range(len(dates))],
                "B": [100 * (1.02**i) for i in range(len(dates))],
                "C": [100 * (1.03**i) for i in range(len(dates))],
            },
            index=dates,
            dtype=float,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "ml_strategy": {
                "enabled": True,
                "model_type": "ridge_numpy",
                "train_years": 1,
                "label_horizon_sessions": 1,
                "min_train_rows": 6,
                "max_train_rows": 100,
                "feature_limit": None,
                "feature_ic_evolution": True,
                "feature_ic_window": 10,
                "feature_ic_top_k": 1,
                "feature_ic_min_periods": 3,
                "feature_ic_min_obs": 2,
                "feature_ic_min_abs_ic": 0.0,
            }
        }

        result = build_ml_scores(factors, prices, config, signal_dates=[dates[-1]])

        row = result.diagnostics.iloc[0]
        self.assertTrue(bool(row["feature_ic_evolved"]))
        self.assertEqual(int(row["feature_count"]), 1)


if __name__ == "__main__":
    unittest.main()
