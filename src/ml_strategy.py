from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class MLStrategyResult:
    scores: pd.Series
    diagnostics: pd.DataFrame


def ml_strategy_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("ml_strategy", {}).get("enabled", False))


def build_ml_scores(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict[str, Any],
    signal_dates: Iterable[pd.Timestamp | str] | None = None,
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

    close = _close_frame(prices)
    if close.empty:
        raise ValueError("prices must contain close prices for ML labels.")

    horizon = max(1, int(cfg.get("label_horizon_sessions", cfg.get("label_horizon_days", 20))))
    forward_returns = close.shift(-horizon).divide(close).sub(1.0)
    label_series = forward_returns.stack(future_stack=True).rename("forward_return")
    label_series.index = label_series.index.set_names(["datetime", "instrument"])
    labels = label_series.reindex(factors.index).to_numpy(dtype=float)

    factor_dates = pd.DatetimeIndex(pd.to_datetime(factors.index.get_level_values(0)).normalize())
    factor_date_values = factor_dates.to_numpy(dtype="datetime64[ns]")
    price_dates = pd.DatetimeIndex(pd.to_datetime(close.index).normalize()).unique().sort_values()
    signals = _resolve_signal_dates(factor_dates, price_dates, cfg, signal_dates)

    train_years = float(cfg.get("train_years", 3.0))
    min_train_rows = int(cfg.get("min_train_rows", 20_000))
    max_train_rows = int(cfg.get("max_train_rows", 80_000))
    min_feature_fraction = float(cfg.get("min_feature_fraction", 0.5))
    min_feature_count = max(1, int(np.ceil(len(feature_columns) * min_feature_fraction)))
    seed = int(cfg.get("random_state", 42))

    score_parts: list[pd.Series] = []
    diagnostic_rows: list[dict[str, object]] = []
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
            train_positions = np.sort(rng.choice(candidate_positions, size=max_train_rows, replace=False))
        else:
            train_positions = candidate_positions

        predict_start = np.searchsorted(factor_date_values, signal_date.to_datetime64(), side="left")
        predict_end = np.searchsorted(factor_date_values, signal_date.to_datetime64(), side="right")
        if predict_start == predict_end:
            diagnostic_rows.append(_skipped_row(signal_date, "missing_signal_features"))
            continue

        train_frame = numeric.iloc[train_positions][feature_columns]
        predict_frame = numeric.iloc[predict_start:predict_end][feature_columns]
        y = labels[train_positions]
        prepared = _prepare_matrices(train_frame, predict_frame, y, min_feature_count, cfg)
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
                    len(predict_frame),
                    "skipped",
                    max_label_end >= signal_date,
                    "insufficient_clean_train_rows",
                )
            )
            continue
        if len(prepared["X_predict"]) == 0:
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
                )
            )
            continue

        predictions, model_used = _fit_predict_model(
            prepared["X_train"],
            prepared["y"],
            prepared["X_predict"],
            cfg,
            seed + signal_number,
        )
        full_predictions = np.full(len(predict_frame), np.nan, dtype=float)
        full_predictions[prepared["predict_mask"]] = predictions
        daily_scores = pd.Series(full_predictions, index=predict_frame.index, name="score")
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
                int(np.isfinite(full_predictions).sum()),
                model_used,
                max_label_end >= signal_date,
                "",
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
        columns = available
        feature_limit = cfg.get("feature_limit")
        if feature_limit is not None:
            columns = columns[: max(1, int(feature_limit))]
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


def _prepare_matrices(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    labels: np.ndarray,
    min_feature_count: int,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    train_counts = train_frame.notna().sum(axis=1).to_numpy()
    train_mask = (train_counts >= min_feature_count) & np.isfinite(labels)
    if not train_mask.any():
        return None

    y = labels[train_mask].astype(float)
    label_clip = cfg.get("label_clip")
    if label_clip is not None:
        clip = abs(float(label_clip))
        y = np.clip(y, -clip, clip)

    train_clean = train_frame.loc[train_mask]
    medians = train_clean.median(axis=0, skipna=True).fillna(0.0)
    stds = train_clean.std(axis=0, skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
    X_train = _scale_features(train_clean, medians, stds)

    predict_counts = predict_frame.notna().sum(axis=1).to_numpy()
    predict_mask = predict_counts >= min_feature_count
    X_predict = _scale_features(predict_frame.loc[predict_mask], medians, stds)
    return {"X_train": X_train, "y": y, "X_predict": X_predict, "predict_mask": predict_mask}


def _scale_features(frame: pd.DataFrame, medians: pd.Series, stds: pd.Series) -> np.ndarray:
    scaled = frame.fillna(medians).sub(medians, axis=1).div(stds, axis=1)
    return scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32, copy=False)


def _fit_predict_model(
    X_train: np.ndarray,
    y: np.ndarray,
    X_predict: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, str]:
    requested = str(cfg.get("model_type", "auto")).strip().lower()
    candidates = [requested] if requested != "auto" else ["lightgbm", "xgboost", "sklearn_gbdt", "ridge_numpy"]
    for candidate in candidates:
        try:
            if candidate in {"lightgbm", "lgbm"}:
                return _fit_predict_lightgbm(X_train, y, X_predict, cfg, seed), "lightgbm"
            if candidate in {"xgboost", "xgb"}:
                return _fit_predict_xgboost(X_train, y, X_predict, cfg, seed), "xgboost"
            if candidate in {"sklearn_gbdt", "gbdt", "hist_gradient_boosting"}:
                return _fit_predict_sklearn_gbdt(X_train, y, X_predict, cfg, seed), "sklearn_gbdt"
            if candidate in {"ridge", "ridge_numpy", "linear"}:
                return _fit_predict_ridge_numpy(X_train, y, X_predict, cfg), "ridge_numpy"
        except (ImportError, ModuleNotFoundError):
            if requested != "auto" and not bool(cfg.get("fallback_on_missing_model", True)):
                raise
            continue
    return _fit_predict_ridge_numpy(X_train, y, X_predict, cfg), "ridge_numpy"


def _fit_predict_lightgbm(
    X_train: np.ndarray,
    y: np.ndarray,
    X_predict: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
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
    return model.predict(X_predict).astype(float)


def _fit_predict_xgboost(
    X_train: np.ndarray,
    y: np.ndarray,
    X_predict: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
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
    return model.predict(X_predict).astype(float)


def _fit_predict_sklearn_gbdt(
    X_train: np.ndarray,
    y: np.ndarray,
    X_predict: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=int(cfg.get("n_estimators", 160)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_leaf_nodes=int(cfg.get("num_leaves", 31)),
        l2_regularization=float(cfg.get("reg_lambda", 1.0)),
        random_state=seed,
    )
    model.fit(X_train, y)
    return model.predict(X_predict).astype(float)


def _fit_predict_ridge_numpy(
    X_train: np.ndarray,
    y: np.ndarray,
    X_predict: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    alpha = float(cfg.get("ridge_alpha", 10.0))
    X_design = np.column_stack([np.ones(len(X_train), dtype=np.float32), X_train])
    X_pred_design = np.column_stack([np.ones(len(X_predict), dtype=np.float32), X_predict])
    reg = np.eye(X_design.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    xtx = X_design.T @ X_design + reg
    xty = X_design.T @ y
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return (X_pred_design @ coef).astype(float)


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
    }
