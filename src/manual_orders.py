from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from src.config_loader import load_config, resolve_path
from src.market_regime import defensive_exposure_for_date


@dataclass
class AccountState:
    total_asset: float
    cash: float
    max_position_pct: float | None
    lot_size: int
    star_market_lot_size: int
    source_file: str
    holdings_file: str
    holdings_loaded: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_account_state(config: dict | None = None) -> AccountState:
    cfg = config or load_config()
    account_cfg = dict(cfg.get("account", {}))
    account_path = resolve_path(account_cfg.get("file", "config/account.yaml"))
    if account_path.exists():
        with account_path.open("r", encoding="utf-8") as f:
            account_cfg.update(yaml.safe_load(f) or {})
    holdings_file = str(account_cfg.get("current_holdings_file", "config/current_holdings.csv"))
    holdings_path = resolve_path(holdings_file)
    return AccountState(
        total_asset=float(account_cfg.get("total_asset", cfg.get("backtest", {}).get("initial_capital", 1_000_000))),
        cash=float(account_cfg.get("cash", 0.0)),
        max_position_pct=_optional_float(account_cfg.get("max_position_pct")),
        lot_size=int(account_cfg.get("lot_size", 100)),
        star_market_lot_size=int(account_cfg.get("star_market_lot_size", 200)),
        source_file=str(account_path),
        holdings_file=str(holdings_path),
        holdings_loaded=holdings_path.exists(),
    )


def load_current_holdings(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    account_cfg = cfg.get("account", {})
    holdings_path = resolve_path(account_cfg.get("current_holdings_file", "config/current_holdings.csv"))
    if not holdings_path.exists():
        return pd.DataFrame(columns=["instrument", "shares"])
    frame = pd.read_csv(holdings_path)
    if "instrument" not in frame.columns and "ticker" in frame.columns:
        frame = frame.rename(columns={"ticker": "instrument"})
    if "instrument" not in frame.columns:
        return pd.DataFrame(columns=["instrument", "shares"])
    if "shares" not in frame.columns:
        frame["shares"] = pd.NA
    result = frame[["instrument", "shares"]].copy()
    result["instrument"] = result["instrument"].map(_normalize_instrument)
    result["shares"] = pd.to_numeric(result["shares"], errors="coerce")
    result = result[result["instrument"] != ""]
    return result.drop_duplicates("instrument", keep="last").reset_index(drop=True)


def generate_manual_orders(
    signal_df: pd.DataFrame,
    target_holdings: list[str],
    price_df: pd.DataFrame,
    signal_date: str,
    intended_trade_date: str | None,
    config: dict | None = None,
    account: AccountState | None = None,
    current_holdings: pd.DataFrame | None = None,
    is_executable: bool = True,
    block_reasons: list[str] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    account = account or load_account_state(cfg)
    current = current_holdings if current_holdings is not None else load_current_holdings(cfg)
    current_map = _current_share_map(current)
    action_map = _signal_action_map(signal_df)
    normalized_targets = _normalize_instruments(target_holdings)
    target_set = set(normalized_targets)
    all_symbols = sorted(set(action_map) | target_set | set(current_map), key=lambda code: (code not in target_set, code))

    rows: list[dict[str, Any]] = []
    target_exposure = defensive_exposure_for_date(price_df, cfg, signal_date)
    target_weight = _target_weight(normalized_targets, account, cfg, target_exposure)
    reference_date = _reference_price_date(price_df, signal_date, intended_trade_date)
    close = _price_row(price_df, "close", reference_date)
    signal_close = _price_row(price_df, "close", pd.Timestamp(signal_date))
    reference_from_signal_date = _reference_from_signal_date(signal_date, intended_trade_date, reference_date)
    reference_price_source = "signal_date_close" if reference_from_signal_date or not intended_trade_date else "intended_trade_date_close"
    final_targets = _target_share_plan(normalized_targets, close, account, target_weight, cfg)
    indicative_targets = _target_share_plan(normalized_targets, signal_close, account, target_weight, cfg)
    account_issues = validate_account_inputs(account, current, cfg)
    order_actionable = bool(is_executable) and not reference_from_signal_date and not account_issues
    stop_loss_pct = cfg.get("strategy", {}).get("stop_loss_pct")
    for instrument in all_symbols:
        action = action_map.get(instrument, "HOLD" if instrument in target_set else "SELL")
        reference_price = _reference_price(instrument, close)
        if instrument in current_map:
            current_shares = current_map.get(instrument)
        else:
            current_shares = 0.0 if account.holdings_loaded else None
        desired_weight = target_weight if instrument in target_set else 0.0
        target_value = account.total_asset * desired_weight
        final_target_shares = final_targets.get(instrument, 0 if instrument in target_set else 0)
        if instrument in target_set and reference_price is None:
            final_target_shares = None
        if reference_from_signal_date and instrument in target_set:
            final_target_shares = None
        indicative_target_shares = indicative_targets.get(instrument)
        target_shares = final_target_shares
        row_actionable = order_actionable and current_shares is not None and target_shares is not None and reference_price is not None
        order_shares = _order_shares(current_shares, target_shares, action) if row_actionable else None
        adv_10d = _adv_for_date(price_df, reference_date, instrument, cfg, window=10)
        order_notional = abs(float(order_shares or 0.0)) * reference_price if reference_price is not None else None
        capacity_ratio = float(order_notional / adv_10d) if order_notional is not None and adv_10d and adv_10d > 0 else None
        suggested_limit_price = _suggested_limit_price(action, reference_price, cfg)
        stop_loss_price = _stop_loss_price(reference_price, stop_loss_pct) if instrument in target_set else None
        sizing_warning = _sizing_warning(
            reference_from_signal_date=reference_from_signal_date,
            current_shares=current_shares,
            reference_price=reference_price,
            account_issues=account_issues,
        )
        rows.append(
            {
                "signal_date": signal_date,
                "intended_trade_date": intended_trade_date or "",
                "instrument": instrument,
                "action": action,
                "is_executable": bool(is_executable),
                "is_order_actionable": row_actionable,
                "target_weight": desired_weight,
                "target_value": target_value,
                "target_notional": target_value,
                "current_shares": current_shares,
                "indicative_target_shares": indicative_target_shares,
                "final_target_shares": final_target_shares,
                "target_shares": target_shares,
                "order_shares": order_shares,
                "reference_price_date": str(reference_date.date()),
                "reference_price_source": reference_price_source,
                "reference_price": reference_price,
                "suggested_limit_price": suggested_limit_price,
                "stop_loss_price": stop_loss_price,
                "is_limit_up": _is_limit_up(price_df, reference_date, instrument, cfg),
                "is_limit_down": _is_limit_down(price_df, reference_date, instrument, cfg),
                "is_st": _is_st(price_df, reference_date, instrument),
                "adv_10d": adv_10d,
                "capacity_ratio": capacity_ratio,
                "cash_after_orders_estimate": None,
                "sizing_warning": sizing_warning,
                "note": _order_note(
                    current_shares,
                    reference_price,
                    is_executable,
                    block_reasons,
                    reference_from_signal_date=reference_from_signal_date,
                    account_issues=account_issues,
                    row_actionable=row_actionable,
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["cash_after_orders_estimate"] = _cash_after_orders_estimate(result, account)
    return result


def save_manual_orders(orders: pd.DataFrame, signal_date: str, out_dir: str | Path, executable: bool = True) -> Path:
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "manual_orders" if executable else "manual_orders_candidate"
    path = output_dir / f"{prefix}_{signal_date}.csv"
    orders.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _current_share_map(current_holdings: pd.DataFrame) -> dict[str, float | None]:
    if current_holdings.empty:
        return {}
    result: dict[str, float | None] = {}
    for _, row in current_holdings.iterrows():
        instrument = _normalize_instrument(row["instrument"])
        if not instrument:
            continue
        shares = row.get("shares")
        result[instrument] = None if pd.isna(shares) else float(shares)
    return result


def _signal_action_map(signal_df: pd.DataFrame) -> dict[str, str]:
    if signal_df.empty or "instrument" not in signal_df.columns or "action" not in signal_df.columns:
        return {}
    result: dict[str, str] = {}
    for _, row in signal_df.iterrows():
        instrument = _normalize_instrument(row["instrument"])
        if instrument:
            result[instrument] = str(row["action"]).upper()
    return result


def _target_weight(target_holdings: list[str], account: AccountState, config: dict, exposure_scale: float = 1.0) -> float:
    if not target_holdings:
        return 0.0
    weight = 1.0 / len(target_holdings) * max(float(exposure_scale), 0.0)
    strategy_cap = config.get("strategy", {}).get("max_weight_per_stock")
    if strategy_cap is not None:
        weight = min(weight, float(strategy_cap))
    if account.max_position_pct is not None:
        weight = min(weight, account.max_position_pct)
    return float(weight)


def _price_row(price_df: pd.DataFrame, field: str, date: pd.Timestamp) -> pd.Series:
    if price_df.empty:
        return pd.Series(dtype=float)
    target = pd.Timestamp(date).normalize()
    normalized_index = pd.DatetimeIndex(pd.to_datetime(price_df.index).normalize())
    matches = normalized_index == target
    if not matches.any():
        return pd.Series(dtype=float)
    row_key = price_df.index[matches][0]
    if isinstance(price_df.columns, pd.MultiIndex):
        if field not in price_df.columns.get_level_values(0):
            return pd.Series(dtype=float)
        row = price_df.xs(field, level=0, axis=1).loc[row_key]
    else:
        row = price_df.loc[row_key]
    row.index = [_normalize_instrument(value) for value in row.index]
    row = row[row.index != ""]
    row = row[~row.index.duplicated(keep="last")]
    return row.astype(float)


def _reference_price(instrument: str, close: pd.Series) -> float | None:
    if instrument not in close.index or pd.isna(close.loc[instrument]):
        return None
    value = float(close.loc[instrument])
    return value if value > 0 else None


def _target_shares(instrument: str, target_value: float, reference_price: float | None, account: AccountState) -> int | None:
    if reference_price is None or reference_price <= 0:
        return None
    lot_size = account.star_market_lot_size if instrument.lower().startswith(("688", "689")) else account.lot_size
    return int(target_value / reference_price / lot_size) * lot_size


def _target_share_plan(target_holdings: list[str], close: pd.Series, account: AccountState, target_weight: float, config: dict) -> dict[str, int]:
    if not target_holdings or target_weight <= 0:
        return {instrument: 0 for instrument in target_holdings}
    plan: dict[str, int] = {}
    target_values: dict[str, float] = {}
    for instrument in target_holdings:
        reference_price = _reference_price(instrument, close)
        target_value = account.total_asset * target_weight
        target_values[instrument] = target_value
        plan[instrument] = _target_shares(instrument, target_value, reference_price, account) or 0

    remaining_cash = max(sum(target_values.values()) - _plan_value(plan, close), 0.0)
    overweight_tolerance = max(float(config.get("manual_orders", {}).get("cash_redistribution_overweight_tolerance", 0.10)), 0.0)
    while remaining_cash > 0:
        candidates: list[tuple[float, str, int, float]] = []
        for instrument in target_holdings:
            price = _reference_price(instrument, close)
            if price is None:
                continue
            lot_size = _lot_size(instrument, account)
            lot_cost = price * lot_size
            current_value = plan.get(instrument, 0) * price
            under_allocated = target_values.get(instrument, 0.0) - current_value
            max_value = target_values.get(instrument, 0.0) * (1 + overweight_tolerance)
            if lot_cost <= remaining_cash and current_value + lot_cost <= max_value:
                candidates.append((under_allocated, instrument, lot_size, lot_cost))
        if not candidates:
            break
        _under_allocated, instrument, lot_size, lot_cost = sorted(candidates, reverse=True)[0]
        plan[instrument] = plan.get(instrument, 0) + lot_size
        remaining_cash -= lot_cost
    return plan


def _plan_value(plan: dict[str, int], close: pd.Series) -> float:
    total = 0.0
    for instrument, shares in plan.items():
        price = _reference_price(instrument, close)
        if price is not None:
            total += shares * price
    return float(total)


def _lot_size(instrument: str, account: AccountState) -> int:
    return account.star_market_lot_size if instrument.lower().startswith(("688", "689")) else account.lot_size


def _order_shares(current_shares: float | None, target_shares: int | None, action: str) -> float | None:
    if target_shares is None:
        return None
    if current_shares is None:
        return float(target_shares) if action == "BUY" else None
    return float(target_shares - current_shares)


def _order_note(
    current_shares: float | None,
    reference_price: float | None,
    is_executable: bool,
    block_reasons: list[str] | None,
    reference_from_signal_date: bool = False,
    account_issues: list[str] | None = None,
    row_actionable: bool = True,
) -> str:
    notes: list[str] = []
    if not is_executable:
        notes.append("blocked:" + ",".join(block_reasons or ["quality_gate_failed"]))
    if reference_from_signal_date:
        notes.append("indicative_only_reference_price_from_signal_date")
    if account_issues:
        notes.append("account_issues:" + ",".join(account_issues))
    if current_shares is None:
        notes.append("current_shares_missing")
    if reference_price is None:
        notes.append("reference_price_missing")
    if not row_actionable:
        notes.append("order_not_actionable")
    return ";".join(notes)


def validate_account_inputs(account: AccountState, current_holdings: pd.DataFrame, config: dict | None = None) -> list[str]:
    cfg = config or {}
    issues: list[str] = []
    if account.total_asset <= 0:
        issues.append("account_total_asset_not_positive")
    if account.cash < 0:
        issues.append("account_cash_negative")
    if account.lot_size <= 0:
        issues.append("account_lot_size_not_positive")
    if account.star_market_lot_size <= 0:
        issues.append("account_star_market_lot_size_not_positive")
    if not account.holdings_loaded:
        issues.append("current_holdings_file_missing")
    issues.extend(validate_current_holdings(current_holdings, account, cfg))
    return issues


def validate_current_holdings(current_holdings: pd.DataFrame, account: AccountState, config: dict | None = None) -> list[str]:
    if current_holdings.empty:
        return []
    issues: list[str] = []
    seen: set[str] = set()
    for _, row in current_holdings.iterrows():
        instrument = _normalize_instrument(row.get("instrument", ""))
        if not _valid_instrument(instrument):
            issues.append(f"invalid_instrument:{instrument or '<blank>'}")
        if instrument in seen:
            issues.append(f"duplicate_instrument:{instrument}")
        seen.add(instrument)
        shares = pd.to_numeric(row.get("shares"), errors="coerce")
        if pd.isna(shares):
            issues.append(f"invalid_shares:{instrument}")
            continue
        if float(shares) < 0:
            issues.append(f"negative_shares:{instrument}")
        if float(shares) != int(float(shares)):
            issues.append(f"fractional_shares:{instrument}")
        lot_size = _lot_size(instrument, account)
        if lot_size > 0 and int(float(shares)) % lot_size != 0:
            issues.append(f"shares_not_lot_multiple:{instrument}")
    return issues


def _reference_price_date(price_df: pd.DataFrame, signal_date: str, intended_trade_date: str | None) -> pd.Timestamp:
    signal_ts = pd.Timestamp(signal_date).normalize()
    if intended_trade_date:
        intended_ts = pd.Timestamp(intended_trade_date).normalize()
        if _has_price_date(price_df, intended_ts):
            return intended_ts
    return signal_ts


def _reference_from_signal_date(
    signal_date: str,
    intended_trade_date: str | None,
    reference_date: pd.Timestamp,
) -> bool:
    if not intended_trade_date:
        return False
    intended_ts = pd.Timestamp(intended_trade_date).normalize()
    signal_ts = pd.Timestamp(signal_date).normalize()
    return intended_ts != signal_ts and pd.Timestamp(reference_date).normalize() == signal_ts


def _has_price_date(price_df: pd.DataFrame, date: pd.Timestamp) -> bool:
    if price_df.empty:
        return False
    normalized_index = pd.DatetimeIndex(pd.to_datetime(price_df.index).normalize())
    return bool((normalized_index == pd.Timestamp(date).normalize()).any())


def _suggested_limit_price(action: str, reference_price: float | None, config: dict) -> float | None:
    if reference_price is None:
        return None
    buffer = float(config.get("manual_orders", {}).get("limit_price_buffer", config.get("backtest", {}).get("slippage", 0.001)))
    if action == "BUY":
        return round(reference_price * (1 + max(buffer, 0.0)), 4)
    if action == "SELL":
        return round(reference_price * (1 - max(buffer, 0.0)), 4)
    return round(reference_price, 4)


def _stop_loss_price(reference_price: float | None, stop_loss_pct: object) -> float | None:
    if reference_price is None or stop_loss_pct is None:
        return None
    return round(reference_price * (1 - abs(float(stop_loss_pct))), 4)


def _adv_for_date(price_df: pd.DataFrame, date: pd.Timestamp, instrument: str, config: dict, window: int = 10) -> float | None:
    amount = _price_field(price_df, "amount")
    if amount.empty or instrument not in amount.columns:
        return None
    amount.index = pd.to_datetime(amount.index).normalize()
    target = pd.Timestamp(date).normalize()
    history = amount.loc[amount.index < target, instrument].dropna().tail(max(window, 1))
    if history.empty:
        return None
    amount_unit = float(config.get("backtest", {}).get("amount_unit", 1000.0))
    return float(history.mean() * amount_unit)


def _price_field(price_df: pd.DataFrame, field: str) -> pd.DataFrame:
    if price_df.empty or not isinstance(price_df.columns, pd.MultiIndex):
        return pd.DataFrame(index=price_df.index)
    fields = price_df.columns.get_level_values(0).astype(str).str.lower()
    if field.lower() not in set(fields):
        return pd.DataFrame(index=price_df.index)
    frame = price_df.loc[:, fields == field.lower()].copy()
    frame.columns = [_normalize_instrument(value) for value in frame.columns.get_level_values(1)]
    return frame.loc[:, frame.columns != ""]


def _is_limit_up(price_df: pd.DataFrame, date: pd.Timestamp, instrument: str, config: dict) -> bool:
    return _limit_state(price_df, date, instrument, config, side="up")


def _is_limit_down(price_df: pd.DataFrame, date: pd.Timestamp, instrument: str, config: dict) -> bool:
    return _limit_state(price_df, date, instrument, config, side="down")


def _limit_state(price_df: pd.DataFrame, date: pd.Timestamp, instrument: str, config: dict, side: str) -> bool:
    current = pd.Timestamp(date).normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(price_df.index).normalize()).unique().sort_values()
    prev_dates = dates[dates < current]
    if prev_dates.empty:
        return False
    previous = prev_dates[-1]
    close_prev = _reference_price(instrument, _price_row(price_df, "close", previous))
    if close_prev is None:
        return False
    threshold = _limit_threshold(instrument, config, side)
    if side == "up":
        probe = _reference_price(instrument, _price_row(price_df, "high", current))
        probe = probe if probe is not None else _reference_price(instrument, _price_row(price_df, "close", current))
        return bool(probe is not None and probe >= close_prev * (1 + threshold))
    probe = _reference_price(instrument, _price_row(price_df, "low", current))
    probe = probe if probe is not None else _reference_price(instrument, _price_row(price_df, "close", current))
    return bool(probe is not None and probe <= close_prev * (1 - threshold))


def _limit_threshold(instrument: str, config: dict, side: str) -> float:
    backtest_cfg = config.get("backtest", {})
    suffix = "up" if side == "up" else "down"
    lowered = instrument.lower()
    if lowered.startswith(("688", "689", "300", "301")):
        return float(backtest_cfg.get(f"growth_limit_{suffix}_threshold", backtest_cfg.get(f"star_limit_{suffix}_threshold", 0.199)))
    if lowered.startswith(("8", "4")):
        return float(backtest_cfg.get(f"bj_limit_{suffix}_threshold", 0.299))
    return float(backtest_cfg.get(f"limit_{suffix}_threshold", 0.099))


def _is_st(price_df: pd.DataFrame, date: pd.Timestamp, instrument: str) -> bool:
    st_field = _price_row(price_df, "is_st", date)
    if instrument in st_field.index and not pd.isna(st_field.loc[instrument]):
        return bool(st_field.loc[instrument])
    return False


def _cash_after_orders_estimate(orders: pd.DataFrame, account: AccountState) -> float | None:
    if orders.empty or "order_shares" not in orders.columns or "reference_price" not in orders.columns:
        return None
    actionable = orders[orders["is_order_actionable"].astype(bool)].copy()
    if actionable.empty:
        return None
    shares = pd.to_numeric(actionable["order_shares"], errors="coerce").fillna(0.0)
    prices = pd.to_numeric(actionable["reference_price"], errors="coerce").fillna(0.0)
    cash_delta = -(shares * prices).sum()
    return float(account.cash + cash_delta)


def _sizing_warning(
    reference_from_signal_date: bool,
    current_shares: float | None,
    reference_price: float | None,
    account_issues: list[str],
) -> str:
    warnings: list[str] = []
    if reference_from_signal_date:
        warnings.append("indicative_only")
    if current_shares is None:
        warnings.append("current_shares_missing")
    if reference_price is None:
        warnings.append("reference_price_missing")
    warnings.extend(account_issues)
    return ";".join(warnings)


def _valid_instrument(instrument: str) -> bool:
    return bool(re.match(r"^\d{6}\.(SH|SZ|BJ)$", instrument))


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalize_instruments(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        instrument = _normalize_instrument(value)
        if not instrument or instrument in seen:
            continue
        result.append(instrument)
        seen.add(instrument)
    return result
