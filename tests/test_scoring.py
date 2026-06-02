from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.scoring import build_strategy_scores


class ScoringTests(unittest.TestCase):
    def test_build_strategy_scores_uses_dynamic_ic_weights(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": range(5), "F2": range(5, 0, -1)}, index=index)
        prices = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]}, index=[pd.Timestamp("2024-01-02")])
        config = {
            "strategy": {"factor_group": "ic_weighted"},
            "ic": {"top_k": 1, "min_abs_ic": 0.0, "min_periods": 1, "corr_threshold": 0.7},
        }

        with patch("src.scoring.calculate_rolling_ic", return_value=pd.DataFrame()), patch(
            "src.scoring.make_rolling_ic_weights",
            return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
        ) as make_weights:
            scores = build_strategy_scores(factors, config, price_df=prices)

        self.assertEqual(scores.name, "score")
        self.assertGreater(scores.loc[(pd.Timestamp("2024-01-02"), "E")], scores.loc[(pd.Timestamp("2024-01-02"), "A")])
        make_weights.assert_called_once()

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
                "top_k": 1,
                "min_abs_ic": 0.0,
                "min_periods": 1,
                "corr_threshold": 0.7,
                "weight_smoothing": 0.6,
                "max_weight_turnover": 0.5,
            },
        }

        with patch("src.scoring.calculate_rolling_ic", return_value=pd.DataFrame()), patch(
            "src.scoring.make_rolling_ic_weights",
            return_value={pd.Timestamp("2024-01-02"): pd.Series({"F1": 1.0})},
        ) as make_weights:
            build_strategy_scores(factors, config, price_df=prices)

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


if __name__ == "__main__":
    unittest.main()
