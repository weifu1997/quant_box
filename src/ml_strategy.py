from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

from src.factor_ic import calculate_factor_ic, make_ic_weights


@dataclass
class MLStrategyResult:
    scores: pd.Series
    diagnostics: pd.DataFrame


@dataclass
class TrainedMLModel:
    model: Any
    model_type: str
    feature_columns: list[str]
    medians: pd.Series
    stds: pd.Series
    min_feature_count: int
    feature_weights: pd.Series | None = None


@dataclass
class RidgeNumpyModel:
    coef: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_pred_design = np.column_stack([np.ones(len(X), dtype=np.float32), X])
        return (X_pred_design @ self.coef).astype(float)


def ml_strategy_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("ml_strategy", {}).get("enabled", False))


def build_ml_scores(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict[str, Any],
    signal_dates: Iterable[pd.Timestamp | str] | None = None,
    industry_map: pd.Series | None = None,
    daily_basic: pd.DataFrame | None = None,
) -> MLStrategyResult:
    cfg = config.get("ml_strategy", {})
    if factors.empty:
        raise ValueError("factors is empty.")
    if not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("factors must use MultiIndex: datetime/instrument.")

    factors = _normalize_factor_index(factors).sort_index()
    numeric = factors.select_dtypes("number")
    if numeric.empty:
        raise ValueError("factors has no numeric columns for ML strategy.")
    feature_columns = _feature_columns(numeric, cfg)
    if not feature_columns:
        raise ValueError("ml_strategy has no usable feature columns.")
    base_feature_columns = list(feature_columns)

    close = _close_frame(prices)
    if close.empty:
        raise ValueError("prices must contain close prices for ML labels.")
    min_price_history_sessions = int(cfg.get("min_price_history_sessions", 0) or 0)
    price_history_counts = _price_history_counts(close, factors.index, min_price_history_sessions)

    horizon = max(1, int(cfg.get("label_horizon_sessions", cfg.get("label_horizon_days", 20))))
    forward_returns = close.shift(-horizon).divide(close).sub(1.0)
    forward_returns = _adjust_label_returns(forward_returns, close, horizon, cfg)
    label_frame = _transform_label_frame(forward_returns, cfg)
    label_frame = _neutralize_label_frame(label_frame, cfg, industry_map=industry_map, daily_basic=daily_basic)
    label_series = label_frame.stack(future_stack=True).rename("label")
    label_series.index = label_series.index.set_names(["datetime", "instrument"])
    labels = label_series.reindex(factors.index).to_numpy(dtype=float)
    if price_history_counts is not None:
        labels = np.where(price_history_counts.to_numpy(dtype=float) >= min_price_history_sessions, labels, np.nan)

    factor_dates = pd.DatetimeIndex(pd.to_datetime(factors.index.get_level_values(0)).normalize())
    factor_date_values = factor_dates.to_numpy(dtype="datetime64[ns]")
    price_dates = pd.DatetimeIndex(pd.to_datetime(close.index).normalize()).unique().sort_values()
    signals = _resolve_signal_dates(factor_dates, price_dates, cfg, signal_dates)

    train_years = float(cfg.get("train_years", 3.0))
    min_train_rows = int(cfg.get("min_train_rows", 20_000))
    max_train_rows = int(cfg.get("max_train_rows", 80_000))
    min_feature_fraction = float(cfg.get("min_feature_fraction", 0.5))
    seed = int(cfg.get("random_state", 42))
    ensemble_window = max(1, int(cfg.get("ensemble_window", 3)))
    ranking_objective = _is_ranking_objective(cfg)
    ic_evolution_enabled = bool(cfg.get("feature_ic_evolution", False))
    feature_ic_weights_by_date = _precompute_feature_ic_weights(
        numeric,
        close,
        base_feature_columns,
        horizon,
        cfg,
        enabled=ic_evolution_enabled,
    )

    score_parts: list[pd.Series] = []
    diagnostic_rows: list[dict[str, object]] = []
    model_buffer: list[TrainedMLModel] = []
    for signal_number, signal_date in enumerate(signals):
        signal_date = pd.Timestamp(signal_date).normalize()
        price_pos = price_dates.searchsorted(signal_date)
        if price_pos >= len(price_dates) or price_dates[price_pos] != signal_date:
            diagnostic_rows.append(_skipped_row(signal_date, "signal_date_not_in_prices"))
            continue
        train_end_pos = price_pos - horizon - 1
        if train_end_pos < 0:
            diagnostic_rows.append(_skipped_row(signal_date, "insufficient_label_history"))
            continue

        train_start_date = signal_date - pd.DateOffset(years=int(train_years))
        train_end_date = pd.Timestamp(price_dates[train_end_pos]).normalize()
        max_label_end = pd.Timestamp(price_dates[train_end_pos + horizon]).normalize()
        train_start_pos = np.searchsorted(factor_date_values, train_start_date.to_datetime64(), side="left")
        train_end_row_pos = np.searchsorted(factor_date_values, train_end_date.to_datetime64(), side="right")
        candidate_positions = np.arange(train_start_pos, train_end_row_pos)
        candidate_positions = candidate_positions[np.isfinite(labels[candidate_positions])]
        available_rows = int(len(candidate_positions))

        if available_rows < min_train_rows:
            diagnostic_rows.append(
                _diagnostic_row(
                    signal_date,
                    train_start_date,
                    train_end_date,
                    max_label_end,
                    horizon,
                    available_rows,
                    0,
                    0,
                    "skipped",
                    max_label_end >= signal_date,
                    "insufficient_train_rows",
                )
            )
            continue

        rng = np.random.default_rng(seed + signal_number)
        if max_train_rows > 0 and available_rows > max_train_rows:
            train_positions = _sample_train_positions(
                candidate_positions,
                factor_date_values,
                max_train_rows,
                rng,
                ranking_objective=ranking_objective,
            )
        else:
            train_positions = candidate_positions

        predict_start = np.searchsorted(factor_date_values, signal_date.to_datetime64(), side="left")
        predict_end = np.searchsorted(factor_date_values, signal_date.to_datetime64(), side="right")
        if predict_start == predict_end:
            diagnostic_rows.append(_skipped_row(signal_date, "missing_signal_features"))
            continue
        predict_index = numeric.iloc[predict_start:predict_end].index
        if price_history_counts is not None:
            eligible_predict = price_history_counts.iloc[predict_start:predict_end] >= min_price_history_sessions
            if not bool(eligible_predict.any()):
                diagnostic_rows.append(_skipped_row(signal_date, "insufficient_price_history"))
                continue

        active_features, feature_weights, feature_evolved = _evolve_features(
            base_feature_columns,
            train_end_date,
            feature_ic_weights_by_date,
        )
        min_feature_count = max(1, int(np.ceil(len(active_features) * min_feature_fraction)))
        train_frame = numeric.iloc[train_positions][active_features]
        y = labels[train_positions]
        prepared = _prepare_training_matrix(train_frame, y, min_feature_count, feature_weights, cfg)
        if prepared is None or len(prepared["y"]) < min_train_rows:
            diagnostic_rows.append(
                _diagnostic_row(
                    signal_date,
                    train_start_date,
                    train_end_date,
                    max_label_end,
                    horizon,
                    available_rows,
                    0 if prepared is None else len(prepared["y"]),
                    len(predict_index),
                    "skipped",
                    max_label_end >= signal_date,
                    "insufficient_clean_train_rows",
                    len(model_buffer),
                    [model.model_type for model in model_buffer],
                    len(active_features),
                    feature_evolved,
                )
            )
            continue

        model, model_used = _fit_train_model(
            prepared["X_train"],
            prepared["y"],
            cfg,
            seed + signal_number,
            group=prepared.get("group"),
        )
        model_buffer.append(
            TrainedMLModel(
                model=model,
                model_type=model_used,
                feature_columns=active_features,
                medians=prepared["medians"],
                stds=prepared["stds"],
                min_feature_count=min_feature_count,
                feature_weights=feature_weights,
            )
        )
        if len(model_buffer) > ensemble_window:
            model_buffer.pop(0)

        ensemble_predictions = _predict_ensemble(model_buffer, numeric.iloc[predict_start:predict_end])
        if ensemble_predictions.empty or not bool(ensemble_predictions.notna().any()):
            diagnostic_rows.append(
                _diagnostic_row(
                    signal_date,
                    train_start_date,
                    train_end_date,
                    max_label_end,
                    horizon,
                    available_rows,
                    len(prepared["y"]),
                    0,
                    "skipped",
                    max_label_end >= signal_date,
                    "no_valid_predict_rows",
                    len(model_buffer),
                    [model.model_type for model in model_buffer],
                    len(active_features),
                    feature_evolved,
                )
            )
            continue

        daily_scores = ensemble_predictions.reindex(predict_index).rename("score")
        if price_history_counts is not None:
            daily_scores = daily_scores.where(eligible_predict.reindex(daily_scores.index).fillna(False))
        score_parts.append(daily_scores)
        diagnostic_rows.append(
            _diagnostic_row(
                signal_date,
                train_start_date,
                train_end_date,
                max_label_end,
                horizon,
                available_rows,
                len(prepared["y"]),
                int(daily_scores.notna().sum()),
                model_used,
                max_label_end >= signal_date,
                "",
                len(model_buffer),
                [model.model_type for model in model_buffer],
                len(active_features),
                feature_evolved,
            )
        )

    scores = pd.concat(score_parts).sort_index().rename("score") if score_parts else pd.Series(dtype=float, name="score")
    diagnostics = pd.DataFrame(diagnostic_rows)
    scores.attrs["training_diagnostics"] = diagnostics
    return MLStrategyResult(scores=scores, diagnostics=diagnostics)


def _normalize_factor_index(factors: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    instruments = factors.index.get_level_values(1).astype(str)
    normalized_index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
    if factors.index.equals(normalized_index):
        return factors
    normalized = factors.copy(deep=False)
    normalized.index = normalized_index
    return normalized


def _feature_columns(numeric: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    configured = cfg.get("feature_columns")
    available = [str(column) for column in numeric.columns]
    if configured:
        requested = [str(column) for column in configured]
        columns = [column for column in requested if column in available]
    else:
        extension_columns = [column for column in available if column.startswith(("DB_", "PX_"))]
        base_columns = [column for column in available if column not in extension_columns]
        columns = base_columns
        feature_limit = cfg.get("feature_limit")
        if feature_limit is not None:
            columns = columns[: max(1, int(feature_limit))]
        columns = [*columns, *extension_columns]
    return columns


def _close_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    if isinstance(prices.columns, pd.MultiIndex):
        fields = prices.columns.get_level_values(0).astype(str).str.lower()
        if "close" not in set(fields):
            return pd.DataFrame(index=prices.index)
        close = prices.loc[:, fields == "close"].copy()
        close.columns = close.columns.get_level_values(-1).astype(str)
    elif "close" in prices.columns:
        close = prices[["close"]].copy()
    else:
        close = prices.copy()
        close.columns = close.columns.astype(str)
    close.index = pd.to_datetime(close.index).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.apply(pd.to_numeric, errors="coerce")


def _price_history_counts(close: pd.DataFrame, target_index: pd.MultiIndex, min_sessions: int) -> pd.Series | None:
    if min_sessions <= 0:
        return None
    history = close.notna().rolling(min_sessions, min_periods=1).sum()
    stacked = history.stack(future_stack=True).rename("price_history_sessions")
    stacked.index = stacked.index.set_names(["datetime", "instrument"])
    normalized_index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(target_index.get_level_values(0)).normalize(),
            target_index.get_level_values(1).astype(str),
        ],
        names=["datetime", "instrument"],
    )
    return stacked.reindex(normalized_index).fillna(0.0)


def _resolve_signal_dates(
    factor_dates: pd.DatetimeIndex,
    price_dates: pd.DatetimeIndex,
    cfg: dict[str, Any],
    signal_dates: Iterable[pd.Timestamp | str] | None,
) -> list[pd.Timestamp]:
    if signal_dates is not None:
        return sorted({pd.Timestamp(date).normalize() for date in signal_dates})

    dates = pd.DatetimeIndex(factor_dates.unique()).intersection(price_dates).sort_values()
    if dates.empty:
        return []
    start_date = cfg.get("signal_start_date")
    end_date = cfg.get("signal_end_date")
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date).normalize()]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date).normalize()]

    frequency = str(cfg.get("rebalance_freq", "monthly")).strip().lower()
    date_series = pd.Series(dates, index=dates)
    if frequency == "daily":
        return [pd.Timestamp(date).normalize() for date in dates]
    if frequency == "weekly":
        signals = date_series.resample("W-FRI").last().dropna()
    elif frequency == "monthly":
        signals = date_series.resample("ME").last().dropna()
    else:
        raise ValueError(f"Unsupported ml_strategy rebalance_freq: {frequency}")
    return [pd.Timestamp(date).normalize() for date in signals]


def _prepare_training_matrix(
    train_frame: pd.DataFrame,
    labels: np.ndarray,
    min_feature_count: int,
    feature_weights: pd.Series | None,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    train_counts = train_frame.notna().sum(axis=1).to_numpy()
    train_mask = (train_counts >= min_feature_count) & np.isfinite(labels)
    if not train_mask.any():
        return None

    y = labels[train_mask].astype(float)
    label_mode = str(cfg.get("label_mode", "raw_return")).strip().lower()
    label_clip = cfg.get("label_clip")
    if label_clip is not None:
        if label_mode in {"raw", "raw_return", "return"}:
            clip = abs(float(label_clip))
            y = np.clip(y, -clip, clip)

    train_clean = train_frame.loc[train_mask]
    group = _ranking_groups(train_clean.index, y, cfg) if _is_ranking_objective(cfg) else None
    if _is_ranking_objective(cfg):
        if group is None:
            return None
        y = _ranking_relevance_labels(train_clean.index, y, cfg)
    medians = train_clean.median(axis=0, skipna=True).fillna(0.0)
    stds = train_clean.std(axis=0, skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
    X_train = _scale_features(train_clean, medians, stds, feature_weights)
    return {"X_train": X_train, "y": y, "medians": medians, "stds": stds, "group": group}


def _is_ranking_objective(cfg: dict[str, Any]) -> bool:
    objective = str(cfg.get("model_objective", "regression")).strip().lower()
    return objective in {"ranking", "rank", "lambdarank"}


def _sample_train_positions(
    candidate_positions: np.ndarray,
    factor_date_values: np.ndarray,
    max_train_rows: int,
    rng: np.random.Generator,
    ranking_objective: bool,
) -> np.ndarray:
    if not ranking_objective:
        return np.sort(rng.choice(candidate_positions, size=max_train_rows, replace=False))

    dates = pd.DatetimeIndex(factor_date_values[candidate_positions]).normalize()
    groups = [
        candidate_positions[np.flatnonzero(dates == date)]
        for date in pd.DatetimeIndex(dates).unique().sort_values()
    ]
    groups = [group for group in groups if len(group) > 1]
    if not groups:
        return np.array([], dtype=int)

    selected: list[np.ndarray] = []
    selected_rows = 0
    for group_index in rng.permutation(len(groups)):
        selected.append(groups[int(group_index)])
        selected_rows += len(selected[-1])
        if selected_rows >= max_train_rows:
            break
    return np.sort(np.concatenate(selected))


def _ranking_groups(index: pd.MultiIndex, y: np.ndarray, cfg: dict[str, Any]) -> list[int] | None:
    if not _is_ranking_objective(cfg):
        return None
    dates = pd.to_datetime(index.get_level_values(0)).normalize()
    groups = pd.Series(1, index=dates).groupby(level=0, sort=False).sum().astype(int)
    groups = groups[groups > 1]
    if groups.empty or int(groups.sum()) != len(y):
        return None
    return groups.to_list()


def _ranking_relevance_labels(index: pd.MultiIndex, y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    dates = pd.to_datetime(index.get_level_values(0)).normalize()
    labels = pd.Series(y, index=pd.Index(dates, name="datetime"), dtype=float)
    ranked = labels.groupby(level=0, sort=False).rank(method="average", pct=True)
    bins = max(2, int(cfg.get("ranking_label_bins", 10)))
    relevance = (ranked.fillna(0.0) * bins).round().clip(lower=0, upper=bins).astype(int)
    return relevance.to_numpy(dtype=int)


def _transform_label_frame(forward_returns: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    mode = str(cfg.get("label_mode", "raw_return")).strip().lower()
    min_obs = max(1, int(cfg.get("label_min_cross_section_obs", 20)))
    if mode in {"raw", "raw_return", "return"}:
        return forward_returns

    counts = forward_returns.notna().sum(axis=1)
    valid_dates = counts >= min_obs
    if mode in {"cross_sectional_rank", "rank", "rank_pct"}:
        ranks = forward_returns.rank(axis=1, pct=True, method="average")
        labels = ranks.sub(0.5).mul(2.0)
    elif mode in {"cross_sectional_zscore", "zscore", "z_score"}:
        means = forward_returns.mean(axis=1, skipna=True)
        stds = forward_returns.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
        labels = forward_returns.sub(means, axis=0).div(stds, axis=0)
    elif mode in {"cross_sectional_demean", "demean"}:
        means = forward_returns.mean(axis=1, skipna=True)
        labels = forward_returns.sub(means, axis=0)
    elif mode in {"cross_sectional_long_short", "long_short", "top_bottom"}:
        top_q = min(max(float(cfg.get("label_top_quantile", 0.20)), 0.01), 0.50)
        bottom_q = min(max(float(cfg.get("label_bottom_quantile", top_q)), 0.01), 0.50)
        ranks = forward_returns.rank(axis=1, pct=True, method="average")
        labels = pd.DataFrame(0.0, index=forward_returns.index, columns=forward_returns.columns)
        labels = labels.where(forward_returns.notna(), np.nan)
        labels = labels.mask(ranks > 1.0 - top_q, 1.0)
        labels = labels.mask(ranks <= bottom_q, -1.0)
    elif mode in {"cross_sectional_top_quantile", "top_quantile", "binary_top"}:
        top_q = min(max(float(cfg.get("label_top_quantile", 0.20)), 0.01), 0.50)
        ranks = forward_returns.rank(axis=1, pct=True, method="average")
        labels = pd.DataFrame(0.0, index=forward_returns.index, columns=forward_returns.columns)
        labels = labels.where(forward_returns.notna(), np.nan)
        labels = labels.mask(ranks > 1.0 - top_q, 1.0)
    else:
        raise ValueError(f"Unsupported ml_strategy.label_mode: {mode}")

    labels = labels.where(valid_dates, np.nan)
    return labels.replace([np.inf, -np.inf], np.nan)


def _adjust_label_returns(
    forward_returns: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    mode = str(cfg.get("label_return_adjustment", "raw")).strip().lower()
    if mode in {"", "none", "raw"}:
        return forward_returns
    if mode not in {"vol_adjusted", "volatility_adjusted", "risk_adjusted"}:
        raise ValueError(f"Unsupported ml_strategy.label_return_adjustment: {mode}")
    window = max(2, int(cfg.get("label_volatility_window", 20)))
    min_periods = max(2, int(cfg.get("label_volatility_min_periods", max(5, window // 2))))
    min_periods = min(min_periods, window)
    floor = max(float(cfg.get("label_volatility_floor", 0.01)), 1e-9)
    trailing_vol = close.pct_change().rolling(window, min_periods=min_periods).std(ddof=0)
    trailing_vol = trailing_vol.mul(np.sqrt(max(horizon, 1))).clip(lower=floor)
    adjusted = forward_returns.divide(trailing_vol)
    return adjusted.replace([np.inf, -np.inf], np.nan)


def _neutralize_label_frame(
    labels: pd.DataFrame,
    cfg: dict[str, Any],
    industry_map: pd.Series | None = None,
    daily_basic: pd.DataFrame | None = None,
) -> pd.DataFrame:
    neutral_cfg = cfg.get("training_neutralization", {})
    if labels.empty or not bool(neutral_cfg.get("enabled", False)):
        return labels

    result = labels.copy()
    min_obs = max(3, int(neutral_cfg.get("min_obs", 20)))
    if bool(neutral_cfg.get("industry", True)) and industry_map is not None and not industry_map.empty:
        industry = industry_map.copy()
        industry.index = industry.index.astype(str).str.upper()
        groups = pd.Series(result.columns.astype(str).str.upper(), index=result.columns).map(industry).fillna("UNKNOWN")
        for group in groups.dropna().unique():
            columns = groups.index[groups == group]
            if len(columns) >= min_obs:
                result.loc[:, columns] = result.loc[:, columns].sub(result.loc[:, columns].mean(axis=1), axis=0)

    if bool(neutral_cfg.get("market_cap", True)) and daily_basic is not None and not daily_basic.empty:
        field = str(neutral_cfg.get("market_cap_field", "circ_mv"))
        if field in daily_basic.columns:
            result = _market_cap_neutralize_label_frame(result, daily_basic, field, min_obs)
    return result.replace([np.inf, -np.inf], np.nan)


def _market_cap_neutralize_label_frame(
    labels: pd.DataFrame,
    daily_basic: pd.DataFrame,
    field: str,
    min_obs: int,
) -> pd.DataFrame:
    basics = daily_basic.copy()
    if isinstance(basics.index, pd.MultiIndex):
        names = list(basics.index.names)
        date_level = names.index("trade_date") if "trade_date" in names else 0
        symbol_level = names.index("ts_code") if "ts_code" in names else 1
        basics.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(basics.index.get_level_values(date_level)).normalize(),
                basics.index.get_level_values(symbol_level).astype(str).str.upper(),
            ],
            names=["datetime", "instrument"],
        )
    elif "trade_date" in basics.columns and "ts_code" in basics.columns:
        basics = basics.copy()
        basics["trade_date"] = pd.to_datetime(basics["trade_date"], errors="coerce").dt.normalize()
        basics["ts_code"] = basics["ts_code"].astype(str).str.upper()
        basics = basics.dropna(subset=["trade_date", "ts_code"]).set_index(["trade_date", "ts_code"])
        basics.index = basics.index.set_names(["datetime", "instrument"])
    else:
        return labels

    result = labels.copy()
    upper_columns = pd.Index(result.columns.astype(str).str.upper())
    for date, row in result.iterrows():
        date_key = pd.Timestamp(date).normalize()
        try:
            basics_daily = basics.xs(date_key, level=0)
        except KeyError:
            continue
        cap = pd.to_numeric(basics_daily[field].reindex(upper_columns), errors="coerce")
        residual = _residualize_row_by_market_cap(row, cap, min_obs)
        if residual is not None:
            result.loc[date] = residual.to_numpy(dtype=float)
    return result


def _residualize_row_by_market_cap(row: pd.Series, cap: pd.Series, min_obs: int) -> pd.Series | None:
    y = pd.to_numeric(row, errors="coerce").astype(float)
    x = np.log1p(pd.to_numeric(cap, errors="coerce").astype(float))
    x.index = y.index
    valid = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
    if int(valid.sum()) < min_obs:
        return None
    x_values = x.loc[valid].to_numpy(dtype=float)
    if float(np.nanstd(x_values)) <= 1e-12:
        return None
    design = np.column_stack([np.ones(len(x_values)), x_values])
    beta = np.linalg.lstsq(design, y.loc[valid].to_numpy(dtype=float), rcond=None)[0]
    residual = y.copy()
    residual.loc[valid] = y.loc[valid].to_numpy(dtype=float) - design @ beta
    return residual


def _prepare_prediction_matrix(
    predict_frame: pd.DataFrame,
    medians: pd.Series,
    stds: pd.Series,
    min_feature_count: int,
    feature_weights: pd.Series | None,
) -> dict[str, Any]:
    predict_counts = predict_frame.notna().sum(axis=1).to_numpy()
    predict_mask = predict_counts >= min_feature_count
    X_predict = _scale_features(predict_frame.loc[predict_mask], medians, stds, feature_weights)
    return {"X_predict": X_predict, "predict_mask": predict_mask}


def _scale_features(
    frame: pd.DataFrame,
    medians: pd.Series,
    stds: pd.Series,
    feature_weights: pd.Series | None = None,
) -> np.ndarray:
    scaled = frame.fillna(medians).sub(medians, axis=1).div(stds, axis=1)
    if feature_weights is not None and not feature_weights.empty:
        weights = feature_weights.reindex(scaled.columns).fillna(0.0)
        scaled = scaled.mul(weights, axis=1)
    return scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32, copy=False)


def _predict_ensemble(model_buffer: list[TrainedMLModel], predict_features: pd.DataFrame) -> pd.Series:
    if not model_buffer or predict_features.empty:
        return pd.Series(dtype=float, name="score")
    predictions: list[pd.Series] = []
    for trained in model_buffer:
        available_columns = [column for column in trained.feature_columns if column in predict_features.columns]
        if len(available_columns) != len(trained.feature_columns):
            continue
        predict_frame = predict_features[trained.feature_columns]
        prepared = _prepare_prediction_matrix(
            predict_frame,
            trained.medians,
            trained.stds,
            trained.min_feature_count,
            trained.feature_weights,
        )
        pred = pd.Series(np.nan, index=predict_frame.index, dtype=float)
        if len(prepared["X_predict"]):
            pred.loc[predict_frame.index[prepared["predict_mask"]]] = _predict_model(trained.model, prepared["X_predict"])
        predictions.append(pred)
    if not predictions:
        return pd.Series(np.nan, index=predict_features.index, name="score", dtype=float)
    return pd.concat(predictions, axis=1).mean(axis=1, skipna=True).rename("score")


def _evolve_features(
    base_feature_columns: list[str],
    train_end_date: pd.Timestamp,
    feature_ic_weights_by_date: dict[pd.Timestamp, pd.Series],
) -> tuple[list[str], pd.Series | None, bool]:
    if not feature_ic_weights_by_date:
        return list(base_feature_columns), None, False

    eligible_dates = [date for date in feature_ic_weights_by_date if date <= pd.Timestamp(train_end_date).normalize()]
    if not eligible_dates:
        return list(base_feature_columns), None, False
    weights = feature_ic_weights_by_date[max(eligible_dates)]
    weights = weights.reindex([column for column in weights.index if column in base_feature_columns]).dropna()
    if weights.empty:
        return list(base_feature_columns), None, False
    selected = [str(column) for column in weights.index]
    return selected, weights.astype(float), True


def _precompute_feature_ic_weights(
    numeric: pd.DataFrame,
    close: pd.DataFrame,
    base_feature_columns: list[str],
    horizon: int,
    cfg: dict[str, Any],
    enabled: bool,
) -> dict[pd.Timestamp, pd.Series]:
    if not enabled:
        return {}

    window = max(1, int(cfg.get("feature_ic_window", 252)))
    top_k = max(1, int(cfg.get("feature_ic_top_k", 30)))
    min_periods = max(1, int(cfg.get("feature_ic_min_periods", 60)))
    min_obs = max(1, int(cfg.get("feature_ic_min_obs", 20)))
    min_abs_ic = float(cfg.get("feature_ic_min_abs_ic", 0.02))
    method = str(cfg.get("feature_ic_method", "spearman"))

    try:
        daily_ic = calculate_factor_ic(numeric[base_feature_columns], close, horizon=horizon, method=method, min_obs=min_obs)
    except ValueError:
        return {}

    realized_lag = max(1, int(horizon))
    source = daily_ic.shift(realized_lag)
    rolling_mean = source.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = source.rolling(window=window, min_periods=min_periods).std(ddof=0)
    rolling_count = source.rolling(window=window, min_periods=min_periods).count()

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    for date in rolling_mean.index:
        mean_ic = rolling_mean.loc[date]
        count = rolling_count.loc[date]
        std_ic = rolling_std.loc[date].replace(0, np.nan)
        summary = pd.DataFrame(
            {
                "mean_ic": mean_ic,
                "std_ic": rolling_std.loc[date],
                "ic_ir": mean_ic / std_ic,
                "positive_ratio": (source.loc[source.index <= date].tail(window) > 0).mean(),
                "count": count,
            }
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        summary = summary.loc[summary["count"] >= min_periods]
        if summary.empty:
            continue
        weights = make_ic_weights(summary, top_k=top_k, min_abs_ic=min_abs_ic)
        weights = weights.reindex([column for column in weights.index if column in base_feature_columns]).dropna()
        if not weights.empty:
            weights_by_date[pd.Timestamp(date).normalize()] = weights.astype(float)
    return weights_by_date


def _fit_train_model(
    X_train: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    group: list[int] | None = None,
) -> tuple[Any, str]:
    requested = str(cfg.get("model_type", "auto")).strip().lower()
    candidates = [requested] if requested != "auto" else ["lightgbm", "xgboost", "sklearn_gbdt", "ridge_numpy"]
    for candidate in candidates:
        try:
            if candidate in {"lightgbm", "lgbm"}:
                return _fit_lightgbm_model(X_train, y, cfg, seed, group=group), "lightgbm"
            if candidate in {"xgboost", "xgb"}:
                return _fit_xgboost_model(X_train, y, cfg, seed), "xgboost"
            if candidate in {"sklearn_gbdt", "gbdt", "hist_gradient_boosting"}:
                return _fit_sklearn_gbdt_model(X_train, y, cfg, seed), "sklearn_gbdt"
            if candidate in {"ridge", "ridge_numpy", "linear"}:
                return _fit_ridge_numpy_model(X_train, y, cfg), "ridge_numpy"
        except (ImportError, ModuleNotFoundError):
            if requested != "auto" and not bool(cfg.get("fallback_on_missing_model", True)):
                raise
            continue
        except Exception as exc:
            if not _is_missing_model_dependency(exc) or (
                requested != "auto" and not bool(cfg.get("fallback_on_missing_model", True))
            ):
                raise
            continue
    return _fit_ridge_numpy_model(X_train, y, cfg), "ridge_numpy"


def _predict_model(model: Any, X_predict: np.ndarray) -> np.ndarray:
    if len(X_predict) == 0:
        return np.array([], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X_predict), dtype=float)
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                return probabilities[:, 1].astype(float)
        return np.asarray(model.predict(X_predict), dtype=float)


def _is_missing_model_dependency(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    patterns = [
        "no module named",
        "not installed",
        "is required",
        "cannot import",
        "missing optional dependency",
    ]
    return any(pattern in text for pattern in patterns)


def _fit_lightgbm_model(
    X_train: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    group: list[int] | None = None,
):
    objective = str(cfg.get("model_objective", "regression")).strip().lower()
    if objective in {"ranking", "rank", "lambdarank"}:
        if not group:
            raise ValueError("LightGBM ranking objective requires non-empty query groups.")
        from lightgbm import LGBMRanker

        model = LGBMRanker(
            n_estimators=int(cfg.get("n_estimators", 160)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            num_leaves=int(cfg.get("num_leaves", 31)),
            max_depth=int(cfg.get("max_depth", -1)),
            subsample=float(cfg.get("subsample", 0.85)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.85)),
            min_child_samples=int(cfg.get("min_child_samples", 50)),
            reg_alpha=float(cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            random_state=seed,
            n_jobs=int(cfg.get("n_jobs", -1)),
            verbosity=-1,
            objective="lambdarank",
        )
        model.fit(X_train, y.astype(int), group=group)
        return model
    if objective in {"classification", "binary", "binary_classification"}:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=int(cfg.get("n_estimators", 160)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            num_leaves=int(cfg.get("num_leaves", 31)),
            max_depth=int(cfg.get("max_depth", -1)),
            subsample=float(cfg.get("subsample", 0.85)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.85)),
            min_child_samples=int(cfg.get("min_child_samples", 50)),
            reg_alpha=float(cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            random_state=seed,
            n_jobs=int(cfg.get("n_jobs", -1)),
            verbosity=-1,
            class_weight=cfg.get("class_weight", "balanced"),
        )
        model.fit(X_train, y.astype(int))
        return model

    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=int(cfg.get("n_estimators", 160)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        num_leaves=int(cfg.get("num_leaves", 31)),
        max_depth=int(cfg.get("max_depth", -1)),
        subsample=float(cfg.get("subsample", 0.85)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.85)),
        min_child_samples=int(cfg.get("min_child_samples", 50)),
        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=int(cfg.get("n_jobs", -1)),
        verbosity=-1,
    )
    model.fit(X_train, y)
    return model


def _fit_xgboost_model(
    X_train: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
):
    objective = str(cfg.get("model_objective", "regression")).strip().lower()
    if objective in {"classification", "binary", "binary_classification"}:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=int(cfg.get("n_estimators", 160)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 4)),
            subsample=float(cfg.get("subsample", 0.85)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.85)),
            reg_alpha=float(cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            random_state=seed,
            n_jobs=int(cfg.get("n_jobs", -1)),
            eval_metric="logloss",
        )
        model.fit(X_train, y.astype(int))
        return model

    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=int(cfg.get("n_estimators", 160)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("max_depth", 4)),
        subsample=float(cfg.get("subsample", 0.85)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.85)),
        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=int(cfg.get("n_jobs", -1)),
    )
    model.fit(X_train, y)
    return model


def _fit_sklearn_gbdt_model(
    X_train: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
):
    objective = str(cfg.get("model_objective", "regression")).strip().lower()
    if objective in {"classification", "binary", "binary_classification"}:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=int(cfg.get("n_estimators", 160)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_leaf_nodes=int(cfg.get("num_leaves", 31)),
            l2_regularization=float(cfg.get("reg_lambda", 1.0)),
            random_state=seed,
        )
        model.fit(X_train, y.astype(int))
        return model

    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=int(cfg.get("n_estimators", 160)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_leaf_nodes=int(cfg.get("num_leaves", 31)),
        l2_regularization=float(cfg.get("reg_lambda", 1.0)),
        random_state=seed,
    )
    model.fit(X_train, y)
    return model


def _fit_ridge_numpy_model(
    X_train: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
) -> RidgeNumpyModel:
    alpha = float(cfg.get("ridge_alpha", 10.0))
    X_design = np.column_stack([np.ones(len(X_train), dtype=np.float32), X_train])
    reg = np.eye(X_design.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    xtx = X_design.T @ X_design + reg
    xty = X_design.T @ y
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return RidgeNumpyModel(coef=coef)


def _skipped_row(signal_date: pd.Timestamp, reason: str) -> dict[str, object]:
    return {
        "signal_date": pd.Timestamp(signal_date).date().isoformat(),
        "train_start": "",
        "train_end": "",
        "max_label_end": "",
        "label_horizon_sessions": 0,
        "train_rows_available": 0,
        "train_rows_used": 0,
        "predict_rows": 0,
        "model_used": "skipped",
        "no_lookahead": False,
        "skip_reason": reason,
        "ensemble_size": 0,
        "ensemble_models": "",
        "feature_count": 0,
        "feature_ic_evolved": False,
    }


def _diagnostic_row(
    signal_date: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    max_label_end: pd.Timestamp,
    horizon: int,
    available_rows: int,
    used_rows: int,
    predict_rows: int,
    model_used: str,
    lookahead_breached: bool,
    reason: str,
    ensemble_size: int = 0,
    ensemble_models: list[str] | None = None,
    feature_count: int = 0,
    feature_ic_evolved: bool = False,
) -> dict[str, object]:
    return {
        "signal_date": pd.Timestamp(signal_date).date().isoformat(),
        "train_start": pd.Timestamp(train_start).date().isoformat(),
        "train_end": pd.Timestamp(train_end).date().isoformat(),
        "max_label_end": pd.Timestamp(max_label_end).date().isoformat(),
        "label_horizon_sessions": int(horizon),
        "train_rows_available": int(available_rows),
        "train_rows_used": int(used_rows),
        "predict_rows": int(predict_rows),
        "model_used": model_used,
        "no_lookahead": not bool(lookahead_breached),
        "skip_reason": reason,
        "ensemble_size": int(ensemble_size),
        "ensemble_models": ",".join(ensemble_models or []),
        "feature_count": int(feature_count),
        "feature_ic_evolved": bool(feature_ic_evolved),
    }
