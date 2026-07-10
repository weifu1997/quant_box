"""Safe account and current-holdings management for the Web workspace."""

from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import yaml

from src.common import normalize_instrument
from src.config_loader import load_config, resolve_path
from src.manual_orders import AccountState, load_account_state, load_current_holdings, validate_account_inputs


def build_account_workspace(config: dict | None = None) -> dict[str, Any]:
    """Return account fields and holdings without exposing unrelated private config."""
    cfg = config or load_config()
    account = load_account_state(cfg)
    holdings = load_current_holdings(cfg)
    issues = validate_account_inputs(account, holdings, cfg)
    return {
        "version": 1,
        "status": "ready" if not issues else "needs_input",
        "message": "账户与持仓校验通过。" if not issues else "请完善账户或持仓信息。",
        "account": {
            "total_asset": account.total_asset,
            "cash": account.cash,
            "max_position_pct": account.max_position_pct,
            "lot_size": account.lot_size,
            "star_market_lot_size": account.star_market_lot_size,
        },
        "holdings": _records(holdings),
        "issues": issues,
        "account_file": account.source_file,
        "holdings_file": account.holdings_file,
        "account_file_exists": Path(account.source_file).exists(),
        "holdings_file_exists": Path(account.holdings_file).exists(),
    }


def preview_account_update(payload: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    """Validate submitted account and holdings without writing files."""
    cfg = config or load_config()
    account, holdings = _submitted_state(payload, cfg)
    issues = validate_account_inputs(account, holdings, cfg)
    return {
        "valid": not issues,
        "issues": issues,
        "account": account.to_dict(),
        "holdings": _records(holdings),
        "position_count": int(len(holdings)),
        "holding_shares": int(pd.to_numeric(holdings.get("shares", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
    }


def apply_account_update(payload: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    """Validate, back up, and atomically write account and holdings files."""
    if payload.get("confirm") is not True:
        raise ValueError("confirm must be true before account files can be updated")
    cfg = config or load_config()
    account, holdings = _submitted_state(payload, cfg)
    issues = validate_account_inputs(account, holdings, cfg)
    if issues:
        raise ValueError("Invalid account or holdings: " + ",".join(issues[:10]))

    account_path = resolve_path(cfg.get("account", {}).get("file", "config/account.yaml"))
    holdings_path = resolve_path(cfg.get("account", {}).get("current_holdings_file", "config/current_holdings.csv"))
    backup_dir = _backup_existing_files(account_path, holdings_path, cfg)
    account_payload = {
        "total_asset": account.total_asset,
        "cash": account.cash,
        "max_position_pct": account.max_position_pct,
        "lot_size": account.lot_size,
        "star_market_lot_size": account.star_market_lot_size,
    }
    _write_text_atomic(account_path, yaml.safe_dump(account_payload, allow_unicode=True, sort_keys=False))
    _write_csv_atomic(holdings_path, holdings)
    return {
        "status": "applied",
        "message": "账户与真实持仓已保存。",
        "account_file": str(account_path),
        "holdings_file": str(holdings_path),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "account": account_payload,
        "holdings": _records(holdings),
    }


def _submitted_state(payload: dict[str, Any], config: dict) -> tuple[AccountState, pd.DataFrame]:
    account_value = payload.get("account")
    holdings_value = payload.get("holdings")
    if not isinstance(account_value, dict):
        raise ValueError("account must be an object")
    if not isinstance(holdings_value, list):
        raise ValueError("holdings must be a list")

    account_path = resolve_path(config.get("account", {}).get("file", "config/account.yaml"))
    holdings_path = resolve_path(config.get("account", {}).get("current_holdings_file", "config/current_holdings.csv"))
    total_asset = _number(account_value.get("total_asset"), "total_asset")
    cash = _number(account_value.get("cash"), "cash")
    max_position_raw = account_value.get("max_position_pct")
    max_position_pct = None if max_position_raw is None or max_position_raw == "" else _number(max_position_raw, "max_position_pct")
    lot_size = _integer(account_value.get("lot_size"), "lot_size")
    star_market_lot_size = _integer(account_value.get("star_market_lot_size"), "star_market_lot_size")
    if max_position_pct is not None and not 0 <= max_position_pct <= 1:
        raise ValueError("max_position_pct must be between 0 and 1")
    account = AccountState(
        total_asset=total_asset,
        cash=cash,
        max_position_pct=max_position_pct,
        lot_size=lot_size,
        star_market_lot_size=star_market_lot_size,
        source_file=str(account_path),
        holdings_file=str(holdings_path),
        holdings_loaded=True,
    )

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(holdings_value):
        if not isinstance(row, dict):
            raise ValueError(f"holdings row {index} must be an object")
        instrument = normalize_instrument(row.get("instrument", ""))
        shares = _number(row.get("shares"), f"holdings[{index}].shares")
        rows.append({"instrument": instrument, "shares": shares})
    holdings = pd.DataFrame(rows, columns=["instrument", "shares"])
    if not holdings.empty:
        holdings["shares"] = holdings["shares"].map(lambda value: int(value) if float(value).is_integer() else value)
        holdings = holdings.sort_values("instrument").reset_index(drop=True)
    return account, holdings


def _backup_existing_files(account_path: Path, holdings_path: Path, config: dict) -> Path | None:
    existing = [path for path in (account_path, holdings_path) if path.exists()]
    if not existing:
        return None
    output_dir = resolve_path(config.get("outputs", {}).get("dir", "outputs"))
    backup_dir = output_dir / "account_backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _integer(value: Any, field: str) -> int:
    number = _number(value, field)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".csv", prefix=f".{path.stem}-", dir=path.parent, delete=False, encoding="utf-8-sig", newline="") as handle:
        temp_path = Path(handle.name)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=path.suffix, prefix=f".{path.stem}-", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temp_path = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
