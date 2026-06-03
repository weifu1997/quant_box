from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config_loader import load_config, resolve_path


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
    target_weight = _target_weight(normalized_targets, account, cfg)
    reference_date = _reference_price_date(price_df, signal_date, intended_trade_date)
    close = _price_row(price_df, "close", reference_date)
    reference_from_signal_date = _reference_from_signal_date(signal_date, intended_trade_date, reference_date)
    for instrument in all_symbols:
        action = action_map.get(instrument, "HOLD" if instrument in target_set else "SELL")
        reference_price = _reference_price(instrument, close)
        current_shares = current_map.get(instrument)
        desired_weight = target_weight if instrument in target_set else 0.0
        target_value = account.total_asset * desired_weight
        target_shares = _target_shares(instrument, target_value, reference_price, account)
        order_shares = _order_shares(current_shares, target_shares, action)
        rows.append(
            {
                "signal_date": signal_date,
                "intended_trade_date": intended_trade_date or "",
                "instrument": instrument,
                "action": action,
                "is_executable": bool(is_executable),
                "target_weight": desired_weight,
                "target_value": target_value,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "order_shares": order_shares,
                "reference_price_date": str(reference_date.date()),
                "reference_price": reference_price,
                "note": _order_note(
                    current_shares,
                    reference_price,
                    is_executable,
                    block_reasons,
                    reference_from_signal_date=reference_from_signal_date,
                ),
            }
        )
    return pd.DataFrame(rows)


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


def _target_weight(target_holdings: list[str], account: AccountState, config: dict) -> float:
    if not target_holdings:
        return 0.0
    weight = 1.0 / len(target_holdings)
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
) -> str:
    notes: list[str] = []
    if not is_executable:
        notes.append("blocked:" + ",".join(block_reasons or ["quality_gate_failed"]))
    if reference_from_signal_date:
        notes.append("reference_price_from_signal_date")
    if current_shares is None:
        notes.append("current_shares_missing")
    if reference_price is None:
        notes.append("reference_price_missing")
    return ";".join(notes)


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
