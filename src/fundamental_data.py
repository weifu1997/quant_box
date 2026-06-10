"""Fundamental data cache, conservative screening, and report helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.common import close_price_frame, normalize_instrument
from src.config_loader import load_config, resolve_path
from src.data_fetcher import fetch_stock_universe
from src.data_fetcher_frames import normalize_daily_basic_frame
from src.trading_calendar import resolve_target_date_value
from src.tushare_client import TushareHttpClient, _format_tushare_date, _parse_tushare_dates, _retry_wait_seconds


logger = logging.getLogger(__name__)

FINA_INDICATOR_FIELDS = [
    "ts_code",
    "ann_date",
    "end_date",
    "roe",
    "roe_dt",
    "roe_waa",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "ocf_to_or",
    "ocf_to_opincome",
    "ocf_to_debt",
    "debt_to_assets",
    "assets_to_eqt",
    "current_ratio",
    "quick_ratio",
    "bps",
    "ocfps",
    "fcff",
    "fcfe",
    "fcff_ps",
    "fcfe_ps",
    "profit_dedt",
]
DIVIDEND_FIELDS = [
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
    "base_share",
]

# Tushare fina_indicator fields that are stored as percentages (e.g., 15.0 for 15%).
# Used by _ratio_series to decide whether to divide by 100 — replaces the fragile
# p75>1.5 statistical heuristic with an explicit field-level contract.
TUSHARE_PERCENT_FIELDS: frozenset[str] = frozenset(
    {
        "roe",
        "roe_dt",
        "roe_waa",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "ocf_to_or",
        "ocf_to_opincome",
        "ocf_to_debt",
        "debt_to_assets",
        "assets_to_eqt",
        "dv_ttm",
    }
)


@dataclass(frozen=True)
class FundamentalScreenResult:
    """Container for a fundamental screen table plus summary metadata."""

    frame: pd.DataFrame
    summary: dict[str, Any]


def fetch_fina_indicator(
    ts_code: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    client: TushareHttpClient | None = None,
    fields: Iterable[str] | str | None = None,
    retries: int = 5,
    retry_max_wait: float | None = None,
) -> pd.DataFrame:
    """Fetch Tushare fina_indicator rows for one instrument."""

    client = client or TushareHttpClient.from_config()
    params = {"ts_code": normalize_instrument(ts_code)}
    if start_date is not None:
        params["start_date"] = _format_tushare_date(start_date)
    if end_date is not None:
        params["end_date"] = _format_tushare_date(end_date)
    requested_fields = fields or FINA_INDICATOR_FIELDS
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = client.call("fina_indicator", params=params, fields=requested_fields)
            return normalize_fina_indicator_frame(frame, default_ts_code=params["ts_code"])
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(_retry_wait_seconds(attempt, retry_max_wait))
    raise ValueError(f"fina_indicator response is invalid for {params['ts_code']}: {last_error}") from last_error


def fetch_dividend(
    ts_code: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    client: TushareHttpClient | None = None,
    fields: Iterable[str] | str | None = None,
    retries: int = 5,
    retry_max_wait: float | None = None,
) -> pd.DataFrame:
    """Fetch Tushare dividend rows for one instrument."""

    client = client or TushareHttpClient.from_config()
    params = {"ts_code": normalize_instrument(ts_code)}
    if start_date is not None:
        params["start_date"] = _format_tushare_date(start_date)
    if end_date is not None:
        params["end_date"] = _format_tushare_date(end_date)
    requested_fields = fields or DIVIDEND_FIELDS
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = client.call("dividend", params=params, fields=requested_fields)
            return normalize_dividend_frame(frame, default_ts_code=params["ts_code"])
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(_retry_wait_seconds(attempt, retry_max_wait))
    raise ValueError(f"dividend response is invalid for {params['ts_code']}: {last_error}") from last_error


def update_fundamental_data(
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    symbols: Iterable[str] | None = None,
    client: TushareHttpClient | None = None,
    sleep_seconds: float = 0.0,
    retries: int | None = None,
    retry_max_wait: float | None = None,
    max_symbols: int | None = None,
    missing_only: bool = False,
    skip_failed: bool = True,
    update_fina_indicator: bool = True,
    update_dividend: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Update local fundamental parquet caches."""

    config = config or load_config()
    data_cfg = config.get("data", {})
    fundamental_cfg = config.get("fundamentals", {})
    start = pd.Timestamp(start_date or data_cfg.get("history_start_date") or data_cfg.get("start_date")).normalize()
    end = pd.Timestamp(resolve_target_date_value(end_date or data_cfg.get("end_date"), config=config)).normalize()
    retries = int(retries if retries is not None else data_cfg.get("retries", 5))
    retry_max_wait = retry_max_wait if retry_max_wait is not None else data_cfg.get("retry_max_wait", 30)
    retry_max_wait = float(retry_max_wait) if retry_max_wait is not None else None
    selected_symbols = _resolve_symbols(symbols, config, end)
    if max_symbols is not None and not missing_only:
        selected_symbols = selected_symbols[: max(0, int(max_symbols))]
    client = client or TushareHttpClient.from_config(config)

    outputs: dict[str, Path] = {}
    if update_fina_indicator:
        path = resolve_path(fundamental_cfg.get("fina_indicator_file", "data/fundamentals/fina_indicator.parquet"))
        fetch_symbols = _missing_symbols(path, selected_symbols, normalize_fina_indicator_frame) if missing_only else selected_symbols
        if max_symbols is not None and missing_only:
            fetch_symbols = fetch_symbols[: max(0, int(max_symbols))]
        outputs["fina_indicator"] = _update_symbol_cache(
            path,
            fetch_symbols,
            fetcher=fetch_fina_indicator,
            normalizer=normalize_fina_indicator_frame,
            start_date=start,
            end_date=end,
            client=client,
            sleep_seconds=sleep_seconds,
            retries=retries,
            retry_max_wait=retry_max_wait,
            skip_failed=skip_failed,
            dedupe_keys=["ts_code", "end_date", "ann_date"],
        )
    if update_dividend:
        path = resolve_path(fundamental_cfg.get("dividend_file", "data/fundamentals/dividend.parquet"))
        fetch_symbols = _missing_symbols(path, selected_symbols, normalize_dividend_frame) if missing_only else selected_symbols
        if max_symbols is not None and missing_only:
            fetch_symbols = fetch_symbols[: max(0, int(max_symbols))]
        outputs["dividend"] = _update_symbol_cache(
            path,
            fetch_symbols,
            fetcher=fetch_dividend,
            normalizer=normalize_dividend_frame,
            start_date=start,
            end_date=end,
            client=client,
            sleep_seconds=sleep_seconds,
            retries=retries,
            retry_max_wait=retry_max_wait,
            skip_failed=skip_failed,
            dedupe_keys=["ts_code", "end_date", "ann_date", "ex_date", "pay_date"],
        )
    return outputs


def normalize_fina_indicator_frame(df: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    """Normalize fina_indicator rows into stable columns and dtypes."""

    if df.empty:
        return pd.DataFrame(columns=FINA_INDICATOR_FIELDS)
    frame = df.rename(columns={"code": "ts_code", "date": "ann_date"}).copy()
    if "ts_code" not in frame.columns and default_ts_code:
        frame["ts_code"] = default_ts_code
    for column in FINA_INDICATOR_FIELDS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[FINA_INDICATOR_FIELDS]
    frame["ts_code"] = frame["ts_code"].map(normalize_instrument)
    for column in ["ann_date", "end_date"]:
        frame[column] = _parse_tushare_dates(frame[column])
    for column in FINA_INDICATOR_FIELDS:
        if column not in {"ts_code", "ann_date", "end_date"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "end_date"])
    frame = frame[frame["ts_code"] != ""]
    return frame.drop_duplicates(["ts_code", "end_date", "ann_date"], keep="last").sort_values(
        ["ts_code", "end_date", "ann_date"]
    ).reset_index(drop=True)


def normalize_dividend_frame(df: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    """Normalize dividend rows into stable columns and dtypes."""

    if df.empty:
        return pd.DataFrame(columns=DIVIDEND_FIELDS)
    frame = df.rename(columns={"code": "ts_code", "date": "ann_date"}).copy()
    if "ts_code" not in frame.columns and default_ts_code:
        frame["ts_code"] = default_ts_code
    for column in DIVIDEND_FIELDS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[DIVIDEND_FIELDS]
    frame["ts_code"] = frame["ts_code"].map(normalize_instrument)
    for column in ["end_date", "ann_date", "record_date", "ex_date", "pay_date", "div_listdate", "imp_ann_date", "base_date"]:
        frame[column] = _parse_tushare_dates(frame[column])
    for column in DIVIDEND_FIELDS:
        if column not in {
            "ts_code",
            "end_date",
            "ann_date",
            "div_proc",
            "record_date",
            "ex_date",
            "pay_date",
            "div_listdate",
            "imp_ann_date",
            "base_date",
        }:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ts_code"])
    frame = frame[frame["ts_code"] != ""]
    return frame.drop_duplicates(["ts_code", "end_date", "ann_date", "ex_date", "pay_date"], keep="last").sort_values(
        ["ts_code", "end_date", "ann_date", "ex_date"]
    ).reset_index(drop=True)


def build_fundamental_screen(
    config: dict[str, Any] | None = None,
    as_of: str | pd.Timestamp = "latest",
    daily_basic: pd.DataFrame | None = None,
    fina_indicator: pd.DataFrame | None = None,
    dividend: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    stock_basic: pd.DataFrame | None = None,
) -> FundamentalScreenResult:
    """Build a conservative quality/dividend/debt screen."""

    config = config or load_config()
    screen_cfg = config.get("fundamental_screen", {})
    data_cfg = config.get("data", {})
    fundamental_cfg = config.get("fundamentals", {})
    daily_basic = _load_table(data_cfg.get("daily_basic_file", "data/factors/daily_basic.parquet"), daily_basic)
    fina_indicator = _load_table(fundamental_cfg.get("fina_indicator_file", "data/fundamentals/fina_indicator.parquet"), fina_indicator)
    dividend = _load_table(fundamental_cfg.get("dividend_file", "data/fundamentals/dividend.parquet"), dividend)
    stock_basic = _load_table(data_cfg.get("constituents_file"), stock_basic)

    daily_basic = normalize_daily_basic_frame(daily_basic) if not daily_basic.empty else daily_basic
    fina_indicator = normalize_fina_indicator_frame(fina_indicator) if not fina_indicator.empty else fina_indicator
    dividend = normalize_dividend_frame(dividend) if not dividend.empty else dividend
    as_of_date = _resolve_screen_date(as_of, daily_basic, fina_indicator)
    daily_latest = _latest_by_symbol(daily_basic, "trade_date", as_of_date)
    fina_latest = _latest_fina_by_symbol(
        fina_indicator,
        as_of_date,
        int(fundamental_cfg.get("fallback_lag_days", 120)),
        prefer_annual=bool(screen_cfg.get("prefer_annual_fina", True)),
        max_annual_report_age_days=screen_cfg.get("max_annual_report_age_days", 550),
    )
    dividend_summary = _dividend_summary(dividend, as_of_date, int(screen_cfg.get("dividend_lookback_years", 5)))
    close_latest = _latest_close_by_symbol(config, prices, as_of_date)
    stock_info = _stock_info_by_symbol(stock_basic)

    frame = daily_latest.merge(stock_info, on="ts_code", how="left") if not stock_info.empty else daily_latest
    frame = frame.merge(fina_latest, on="ts_code", how="left", suffixes=("", "_fundamental"))
    frame = frame.merge(dividend_summary, on="ts_code", how="left")
    if not close_latest.empty:
        frame = frame.merge(close_latest, on="ts_code", how="left")
    frame = _add_screen_metrics(frame, screen_cfg)
    frame = _apply_screen_rules(frame, screen_cfg)
    frame = frame.sort_values(
        ["overall_pass", "total_score", "dividend_yield_ttm", "roe", "fcf_yield"],
        ascending=[False, False, False, False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    summary = _screen_summary(frame, as_of_date, daily_basic, fina_indicator, dividend, screen_cfg)
    return FundamentalScreenResult(frame=frame, summary=summary)


def write_fundamental_screen_outputs(
    result: FundamentalScreenResult,
    out_dir: str | Path,
    top_n: int | None = None,
    csv_file: str | Path | None = None,
    report_file: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write fundamental screen CSV and Markdown report."""

    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = str(result.summary.get("as_of_date", "latest"))
    csv_path = resolve_path(csv_file) if csv_file is not None else output_dir / f"fundamental_screen_{as_of}.csv"
    report_path = resolve_path(report_file) if report_file is not None else output_dir / "fundamental_screen_report.md"
    result.frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_fundamental_screen_report(result, top_n=top_n), encoding="utf-8")
    return csv_path, report_path


def summarize_fundamental_screen_result(
    result: FundamentalScreenResult,
    top_n: int = 10,
    csv_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact JSON-safe summary for daily project reports."""

    frame = result.frame
    summary = dict(result.summary)
    covered = frame[frame.get("review_status", "") != "INSUFFICIENT_DATA"] if not frame.empty else frame
    watch = frame[frame.get("review_status", "") == "WATCH"].head(top_n) if not frame.empty else frame
    passed = frame[frame.get("review_status", "") == "PASS"].head(top_n) if not frame.empty else frame
    payload: dict[str, Any] = {
        "enabled": True,
        "status": "ok",
        "as_of_date": summary.get("as_of_date", ""),
        "rows": int(summary.get("rows", 0) or 0),
        "covered_rows": int(len(covered)),
        "passed": int(summary.get("passed", 0) or 0),
        "watch": int((frame.get("review_status", pd.Series(dtype=object)) == "WATCH").sum()) if not frame.empty else 0,
        "fundamental_coverage": summary.get("fundamental_coverage", ""),
        "dividend_coverage": summary.get("dividend_coverage", ""),
        "review_status_counts": summary.get("review_status_counts", {}),
        "top_pass": _summary_records(passed),
        "top_watch": _summary_records(watch),
    }
    files = {}
    if csv_path is not None:
        files["csv"] = str(csv_path)
    if report_path is not None:
        files["report"] = str(report_path)
    if files:
        payload["files"] = files
    return payload


def render_fundamental_screen_report(result: FundamentalScreenResult, top_n: int | None = None) -> str:
    """Render a compact Markdown explanation report."""

    frame = result.frame
    summary = result.summary
    top_n = int(top_n or summary.get("top_n", 30))
    passed = frame[frame.get("overall_pass", False) == True].head(top_n) if not frame.empty else frame
    covered = frame[frame.get("review_status", "") != "INSUFFICIENT_DATA"] if not frame.empty else frame
    near_misses = covered[covered.get("overall_pass", False) != True].head(top_n) if not covered.empty else covered
    failed_reasons = _reason_counts(frame.get("failed_reasons", pd.Series(dtype=object)))
    covered_failed_reasons = _reason_counts(covered.get("failed_reasons", pd.Series(dtype=object))) if not covered.empty else []
    lines = [
        "# Fundamental Screen Report",
        "",
        f"- As of date: {summary.get('as_of_date', '')}",
        f"- Rows screened: {summary.get('rows', 0)}",
        f"- Passed: {summary.get('passed', 0)}",
        f"- Missing fundamental rows: {summary.get('missing_fundamental_rows', 0)}",
        f"- Missing dividend rows: {summary.get('missing_dividend_rows', 0)}",
        "",
        "## Data Coverage",
        "",
        f"- Fundamental coverage: {summary.get('fundamental_coverage', '')}",
        f"- Dividend coverage: {summary.get('dividend_coverage', '')}",
        f"- fina_indicator symbols: {summary.get('fina_indicator_symbols', 0)}",
        f"- dividend symbols: {summary.get('dividend_symbols', 0)}",
        "",
        "## Review Status Counts",
        "",
        *_mapping_lines(summary.get("review_status_counts", {})),
        "",
        "## Thresholds",
        "",
        *_mapping_lines(summary.get("thresholds", {})),
        "",
        "## Top Candidates",
        "",
        _candidate_table(passed, empty_message="No candidates passed the current thresholds."),
        "",
        "## Near Misses",
        "",
        _candidate_table(near_misses, empty_message="No covered near-miss rows are available."),
        "",
        "## Main Failure Reasons In Covered Rows",
        "",
        *([f"- {reason}: {count}" for reason, count in covered_failed_reasons[:12]] or ["- none"]),
        "",
        "## Main Failure Reasons",
        "",
        *[f"- {reason}: {count}" for reason, count in failed_reasons[:12]],
        "",
        "## Reason Guide",
        "",
        *_reason_guide_lines(),
        "",
    ]
    return "\n".join(lines)


def _update_symbol_cache(
    path: Path,
    symbols: list[str],
    *,
    fetcher,
    normalizer,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    client: TushareHttpClient,
    sleep_seconds: float,
    retries: int,
    retry_max_wait: float | None,
    skip_failed: bool,
    dedupe_keys: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    existing = normalizer(existing) if not existing.empty else existing
    frames: list[pd.DataFrame] = []
    for pos, symbol in enumerate(symbols, start=1):
        try:
            frame = fetcher(symbol, start_date=start_date, end_date=end_date, client=client, retries=retries, retry_max_wait=retry_max_wait)
        except (RuntimeError, ValueError) as exc:
            if not skip_failed:
                raise
            logger.error("Skipping %s fundamental fetch after failure: %s", symbol, exc)
            continue
        if not frame.empty:
            frames.append(frame)
        if sleep_seconds > 0 and pos < len(symbols):
            time.sleep(float(sleep_seconds))
    if frames:
        new_data = pd.concat(frames, ignore_index=True)
        combined = pd.concat([existing, new_data], ignore_index=True) if not existing.empty else new_data
    else:
        combined = existing
    combined = normalizer(combined)
    if not combined.empty:
        combined = combined.drop_duplicates(dedupe_keys, keep="last")
    combined.to_parquet(path, index=False)
    return path


def _missing_symbols(path: Path, symbols: list[str], normalizer) -> list[str]:
    if not path.exists():
        return symbols
    existing = pd.read_parquet(path)
    if existing.empty or "ts_code" not in existing.columns:
        return symbols
    existing = normalizer(existing)
    cached = set(existing["ts_code"].dropna().map(normalize_instrument))
    return [symbol for symbol in symbols if normalize_instrument(symbol) not in cached]


def _resolve_symbols(symbols: Iterable[str] | None, config: dict[str, Any], as_of_date: pd.Timestamp) -> list[str]:
    if symbols is not None:
        values = [normalize_instrument(symbol) for symbol in symbols]
        return [symbol for symbol in dict.fromkeys(values) if symbol]
    data_cfg = config.get("data", {})
    return fetch_stock_universe(
        universe=data_cfg.get("universe", "mainboard_a"),
        date=as_of_date,
        local_file=data_cfg.get("constituents_file"),
        save_metadata=False,
    )


def _load_table(path_value: str | Path | None, override: pd.DataFrame | None) -> pd.DataFrame:
    if override is not None:
        return override.copy()
    if not path_value:
        return pd.DataFrame()
    path = resolve_path(path_value)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _resolve_screen_date(as_of: str | pd.Timestamp, daily_basic: pd.DataFrame, fina_indicator: pd.DataFrame) -> pd.Timestamp:
    if str(as_of).lower() != "latest":
        return pd.Timestamp(as_of).normalize()
    dates: list[pd.Timestamp] = []
    if not daily_basic.empty and "trade_date" in daily_basic.columns:
        parsed = pd.to_datetime(daily_basic["trade_date"], errors="coerce").dropna()
        if not parsed.empty:
            dates.append(pd.Timestamp(parsed.max()).normalize())
    if not dates and not fina_indicator.empty and "ann_date" in fina_indicator.columns:
        parsed = pd.to_datetime(fina_indicator["ann_date"], errors="coerce").dropna()
        if not parsed.empty:
            dates.append(pd.Timestamp(parsed.max()).normalize())
    if not dates:
        raise ValueError("Cannot resolve latest fundamental screen date without daily_basic or fina_indicator rows.")
    return max(dates)


def _latest_by_symbol(frame: pd.DataFrame, date_column: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns or "ts_code" not in frame.columns:
        return pd.DataFrame(columns=["ts_code"])
    data = frame.copy()
    data["ts_code"] = data["ts_code"].map(normalize_instrument)
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce").dt.normalize()
    data = data[(data["ts_code"] != "") & data[date_column].notna() & (data[date_column] <= as_of_date)]
    if data.empty:
        return pd.DataFrame(columns=["ts_code"])
    data = data.sort_values(["ts_code", date_column])
    return data.drop_duplicates("ts_code", keep="last").reset_index(drop=True)


def _stock_info_by_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["ts_code", "name", "industry", "area", "market", "list_date"]
    if frame.empty or "ts_code" not in frame.columns:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    data["ts_code"] = data["ts_code"].map(normalize_instrument)
    for column in columns:
        if column not in data.columns:
            data[column] = pd.NA
    data = data[columns]
    data = data[(data["ts_code"] != "") & data["ts_code"].notna()]
    if data.empty:
        return pd.DataFrame(columns=columns)
    return data.drop_duplicates("ts_code", keep="last").reset_index(drop=True)


def _latest_fina_by_symbol(
    frame: pd.DataFrame,
    as_of_date: pd.Timestamp,
    fallback_lag_days: int,
    *,
    prefer_annual: bool = True,
    max_annual_report_age_days: int | None = 550,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts_code"])
    data = frame.copy()
    data["ts_code"] = data["ts_code"].map(normalize_instrument)
    data["ann_date"] = pd.to_datetime(data.get("ann_date"), errors="coerce").dt.normalize()
    data["end_date"] = pd.to_datetime(data.get("end_date"), errors="coerce").dt.normalize()
    announced = data["ann_date"].notna() & (data["ann_date"] <= as_of_date)
    fallback = data["ann_date"].isna() & data["end_date"].notna() & (data["end_date"] <= as_of_date - pd.Timedelta(days=fallback_lag_days))
    data = data[(data["ts_code"] != "") & (announced | fallback)]
    if data.empty:
        return pd.DataFrame(columns=["ts_code"])
    data = data.sort_values(["ts_code", "end_date", "ann_date"], na_position="first")
    latest = data.drop_duplicates("ts_code", keep="last")
    if not prefer_annual:
        return latest.reset_index(drop=True)

    annual = data[_is_annual_statement(data["end_date"])].copy()
    if max_annual_report_age_days is not None:
        min_end_date = as_of_date - pd.Timedelta(days=max(1, int(max_annual_report_age_days)))
        annual = annual[annual["end_date"] >= min_end_date]
    if annual.empty:
        return latest.reset_index(drop=True)
    annual_latest = annual.drop_duplicates("ts_code", keep="last")
    combined = pd.concat(
        [latest.loc[~latest["ts_code"].isin(set(annual_latest["ts_code"]))], annual_latest],
        ignore_index=True,
    )
    return combined.sort_values(["ts_code"]).reset_index(drop=True)


def _is_annual_statement(end_dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(end_dates, errors="coerce")
    return dates.dt.month.eq(12) & dates.dt.day.eq(31)


def _dividend_summary(frame: pd.DataFrame, as_of_date: pd.Timestamp, lookback_years: int) -> pd.DataFrame:
    columns = [
        "ts_code",
        "positive_dividend_years",
        "latest_cash_div_tax",
        "ttm_cash_div_tax",
        "latest_dividend_date",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    data["ts_code"] = data["ts_code"].map(normalize_instrument)
    effective = data.get("ex_date")
    if effective is None or pd.to_datetime(effective, errors="coerce").isna().all():
        effective = data.get("ann_date")
    data["effective_date"] = pd.to_datetime(effective, errors="coerce").dt.normalize()
    data["cash_div_tax"] = pd.to_numeric(data.get("cash_div_tax"), errors="coerce")
    data["cash_div"] = pd.to_numeric(data.get("cash_div"), errors="coerce")
    data["cash_value"] = data["cash_div_tax"].where(data["cash_div_tax"].notna(), data["cash_div"])
    data = data[(data["ts_code"] != "") & data["effective_date"].notna() & (data["effective_date"] <= as_of_date)]
    data = data[pd.to_numeric(data["cash_value"], errors="coerce").fillna(0.0) > 0.0]
    if data.empty:
        return pd.DataFrame(columns=columns)
    min_date = as_of_date - pd.DateOffset(years=max(1, int(lookback_years)))
    recent = data[data["effective_date"] >= min_date].copy()
    recent["dividend_year"] = pd.to_datetime(recent.get("end_date"), errors="coerce").dt.year
    recent["dividend_year"] = recent["dividend_year"].where(recent["dividend_year"].notna(), recent["effective_date"].dt.year)
    grouped = recent.groupby("ts_code", dropna=False)
    summary = grouped.agg(
        positive_dividend_years=("dividend_year", "nunique"),
        latest_cash_div_tax=("cash_value", "last"),
        latest_dividend_date=("effective_date", "max"),
    ).reset_index()
    ttm = data[data["effective_date"] >= as_of_date - pd.Timedelta(days=365)].groupby("ts_code")["cash_value"].sum()
    summary["ttm_cash_div_tax"] = summary["ts_code"].map(ttm)
    return summary[columns]


def _latest_close_by_symbol(config: dict[str, Any], prices: pd.DataFrame | None, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if prices is None:
        price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
        if not price_path.exists():
            return pd.DataFrame(columns=["ts_code", "close"])
        prices = pd.read_parquet(price_path)
    close = close_price_frame(prices)
    if close.empty:
        return pd.DataFrame(columns=["ts_code", "close"])
    close = close[close.index <= as_of_date]
    if close.empty:
        return pd.DataFrame(columns=["ts_code", "close"])
    latest = close.loc[close.index.max()].dropna()
    return pd.DataFrame({"ts_code": latest.index.map(normalize_instrument), "close": latest.to_numpy(dtype=float)})


def _add_screen_metrics(frame: pd.DataFrame, screen_cfg: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["roe"] = _first_present_ratio(result, ["roe_dt", "roe_waa", "roe"])
    result["debt_to_assets"] = _ratio_series(_numeric_column(result, "debt_to_assets"), field_name="debt_to_assets")
    result["ocf_to_opincome"] = _ratio_series(_numeric_column(result, "ocf_to_opincome"), field_name="ocf_to_opincome")
    result["ocf_to_or"] = _ratio_series(_numeric_column(result, "ocf_to_or"), field_name="ocf_to_or")
    # Primary: dv_ttm from daily_basic (Tushare percent field).
    # Fallback: ttm_cash_div_tax / close when dv_ttm is NaN.
    dv_yield = _ratio_series(_numeric_column(result, "dv_ttm"), field_name="dv_ttm")
    if "ttm_cash_div_tax" in result.columns and "close" in result.columns:
        ttm_div = _numeric_column(result, "ttm_cash_div_tax")
        close_price = _numeric_column(result, "close").where(lambda s: s > 0)
        computed_yield = ttm_div.divide(close_price)
        result["dividend_yield_ttm"] = dv_yield.where(dv_yield.notna(), computed_yield)
    else:
        result["dividend_yield_ttm"] = dv_yield
    result["dividend_payback_years"] = np.where(result["dividend_yield_ttm"] > 0, 1.0 / result["dividend_yield_ttm"], np.nan)
    result["positive_dividend_years"] = _numeric_column(result, "positive_dividend_years").fillna(0).astype(int)
    market_cap = _numeric_column(result, "total_mv") * float(screen_cfg.get("market_cap_unit", 10000.0))
    fcff = _numeric_column(result, "fcff") * float(screen_cfg.get("statement_amount_unit", 1.0))
    result["fcf_yield"] = fcff.divide(market_cap.where(market_cap > 0))
    if "fcff_ps" in result.columns and "close" in result.columns:
        ps_yield = _numeric_column(result, "fcff_ps").divide(_numeric_column(result, "close").where(lambda s: s > 0))
        result["fcf_yield"] = result["fcf_yield"].where(result["fcf_yield"].notna(), ps_yield)
    return result


def _apply_screen_rules(frame: pd.DataFrame, screen_cfg: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    min_roe = float(screen_cfg.get("min_roe", 0.08))
    max_debt_to_assets = float(screen_cfg.get("max_debt_to_assets", 0.60))
    min_dividend_yield = float(screen_cfg.get("min_dividend_yield", 0.015))
    min_positive_dividend_years = int(screen_cfg.get("min_positive_dividend_years", 2))
    min_ocf_to_opincome = float(screen_cfg.get("min_ocf_to_opincome", 0.80))
    min_fcf_yield = float(screen_cfg.get("min_fcf_yield", 0.0))
    watch_min_score = int(screen_cfg.get("watch_min_score", 4))
    max_pe_ttm = screen_cfg.get("max_pe_ttm", 30.0)
    max_pb = screen_cfg.get("max_pb", 5.0)

    roe = _numeric_column(result, "roe")
    ocf_to_opincome = _numeric_column(result, "ocf_to_opincome")
    fcf_yield = _numeric_column(result, "fcf_yield")
    fcff = _numeric_column(result, "fcff")
    debt_to_assets = _numeric_column(result, "debt_to_assets")
    dividend_yield = _numeric_column(result, "dividend_yield_ttm")
    dividend_years = _numeric_column(result, "positive_dividend_years")
    pe_ttm = _numeric_column(result, "pe_ttm")
    pb = _numeric_column(result, "pb")

    result["quality_pass"] = (
        (roe >= min_roe)
        & (
            (ocf_to_opincome >= min_ocf_to_opincome)
            | (fcf_yield >= min_fcf_yield)
            | (fcff > 0)
        )
    )
    result["debt_pass"] = debt_to_assets <= max_debt_to_assets
    result["dividend_yield_pass"] = dividend_yield >= min_dividend_yield
    result["dividend_record_pass"] = dividend_years >= min_positive_dividend_years
    result["dividend_pass"] = result["dividend_yield_pass"] & result["dividend_record_pass"]
    # NaN PE/PB (e.g., loss-making stocks with undefined PE) should not cause
    # valuation failure — missing valuation data means we simply have no opinion,
    # not that the stock is overvalued.  In pandas, NaN <= threshold returns
    # False (not NaN), so fillna cannot rescue it.  We must explicitly OR with
    # the NaN mask so that missing values are treated as "no opinion" → pass.
    result["valuation_pass"] = True
    if max_pe_ttm is not None:
        pe_ok = (pe_ttm <= float(max_pe_ttm)) | pe_ttm.isna()
        result["valuation_pass"] &= pe_ok
    if max_pb is not None:
        pb_ok = (pb <= float(max_pb)) | pb.isna()
        result["valuation_pass"] &= pb_ok
    result["overall_pass"] = result["quality_pass"] & result["debt_pass"] & result["dividend_pass"] & result["valuation_pass"]
    result["total_score"] = (
        result["quality_pass"].astype(int) * 2
        + result["debt_pass"].astype(int)
        + result["dividend_yield_pass"].astype(int)
        + result["dividend_record_pass"].astype(int)
        + result["valuation_pass"].astype(int)
        + (roe >= min_roe * 1.5).fillna(False).astype(int)
        + (dividend_yield >= min_dividend_yield * 1.5).fillna(False).astype(int)
    )
    result["failed_reasons"] = result.apply(lambda row: ";".join(_failed_reasons(row)), axis=1)
    result["review_status"] = result.apply(lambda row: _review_status(row, watch_min_score), axis=1)
    result["explanation"] = result.apply(_row_explanation, axis=1)
    return result


def _failed_reasons(row: pd.Series) -> list[str]:
    reasons = []
    if not bool(row.get("quality_pass", False)):
        reasons.append("quality")
    if not bool(row.get("debt_pass", False)):
        reasons.append("debt")
    if not bool(row.get("dividend_yield_pass", False)):
        reasons.append("dividend_yield")
    if not bool(row.get("dividend_record_pass", False)):
        reasons.append("dividend_record")
    if not bool(row.get("valuation_pass", False)):
        reasons.append("valuation")
    if pd.isna(row.get("end_date")):
        reasons.append("missing_fundamental")
    return reasons


def _review_status(row: pd.Series, watch_min_score: int) -> str:
    if pd.isna(row.get("end_date")):
        return "INSUFFICIENT_DATA"
    if bool(row.get("overall_pass", False)):
        return "PASS"
    if int(row.get("total_score", 0) or 0) >= watch_min_score:
        return "WATCH"
    return "REJECT"


def _row_explanation(row: pd.Series) -> str:
    payback = _num(row.get("dividend_payback_years"))
    parts = [
        f"roe={_pct(row.get('roe'))}",
        f"debt={_pct(row.get('debt_to_assets'))}",
        f"div_yield={_pct(row.get('dividend_yield_ttm'))}",
        f"payback={payback + 'y' if payback else ''}",
        f"pe={_num(row.get('pe_ttm'))}",
        f"pb={_num(row.get('pb'))}",
        f"fcf_yield={_pct(row.get('fcf_yield'))}",
    ]
    failed = row.get("failed_reasons", "")
    if failed:
        parts.append(f"fail={failed}")
    return ", ".join(parts)


def _screen_summary(
    frame: pd.DataFrame,
    as_of_date: pd.Timestamp,
    daily_basic: pd.DataFrame,
    fina_indicator: pd.DataFrame,
    dividend: pd.DataFrame,
    screen_cfg: dict[str, Any],
) -> dict[str, Any]:
    rows = int(len(frame))
    missing_fundamental = int(frame["end_date"].isna().sum()) if "end_date" in frame.columns else rows
    missing_dividend = int(frame["positive_dividend_years"].eq(0).sum()) if "positive_dividend_years" in frame.columns else rows
    status_counts = (
        frame["review_status"].fillna("UNKNOWN").astype(str).value_counts().sort_index().to_dict()
        if "review_status" in frame.columns
        else {}
    )
    return {
        "as_of_date": str(as_of_date.date()),
        "rows": rows,
        "passed": int(frame["overall_pass"].sum()) if "overall_pass" in frame.columns else 0,
        "missing_fundamental_rows": missing_fundamental,
        "missing_dividend_rows": missing_dividend,
        "fundamental_coverage": _pct((rows - missing_fundamental) / rows) if rows else "",
        "dividend_coverage": _pct((rows - missing_dividend) / rows) if rows else "",
        "daily_basic_rows": int(len(daily_basic)),
        "fina_indicator_rows": int(len(fina_indicator)),
        "dividend_rows": int(len(dividend)),
        "fina_indicator_symbols": int(fina_indicator["ts_code"].nunique()) if "ts_code" in fina_indicator.columns else 0,
        "dividend_symbols": int(dividend["ts_code"].nunique()) if "ts_code" in dividend.columns else 0,
        "review_status_counts": status_counts,
        "top_n": int(screen_cfg.get("top_n", 30)),
        "thresholds": {
            "min_roe": _pct(screen_cfg.get("min_roe", 0.08)),
            "max_debt_to_assets": _pct(screen_cfg.get("max_debt_to_assets", 0.60)),
            "min_dividend_yield": _pct(screen_cfg.get("min_dividend_yield", 0.015)),
            "min_positive_dividend_years": screen_cfg.get("min_positive_dividend_years", 2),
            "min_ocf_to_opincome": _pct(screen_cfg.get("min_ocf_to_opincome", 0.80)),
            "max_pe_ttm": screen_cfg.get("max_pe_ttm", 30.0),
            "max_pb": screen_cfg.get("max_pb", 5.0),
            "watch_min_score": screen_cfg.get("watch_min_score", 4),
            "prefer_annual_fina": bool(screen_cfg.get("prefer_annual_fina", True)),
            "max_annual_report_age_days": screen_cfg.get("max_annual_report_age_days", 550),
        },
    }


def _first_present_ratio(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            ratio = _ratio_series(_numeric_column(frame, column), field_name=column)
            result = result.where(result.notna(), ratio)
    return result


def _ratio_series(values: Any, always_percent: bool = False, field_name: str | None = None) -> pd.Series:
    """Convert a Tushare numeric series to ratio form (0-1 range).

    Uses an explicit field-level contract (TUSHARE_PERCENT_FIELDS) instead of a
    statistical heuristic. If ``field_name`` is in TUSHARE_PERCENT_FIELDS, the
    values are divided by 100 (Tushare stores percentages like 15.0 for 15%).
    If ``always_percent`` is True, division by 100 is forced regardless.
    If neither applies, values are returned unchanged (assumed already in ratio form).
    """
    if values is None:
        return pd.Series(dtype="float64")
    series = pd.to_numeric(values, errors="coerce")
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="float64")
    if always_percent:
        return series / 100.0
    if field_name is not None and field_name in TUSHARE_PERCENT_FIELDS:
        return series / 100.0
    return series


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _reason_counts(values: pd.Series) -> list[tuple[str, int]]:
    counter: dict[str, int] = {}
    for text in values.dropna().astype(str):
        for item in [part for part in text.split(";") if part]:
            counter[item] = counter.get(item, 0) + 1
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _candidate_table(frame: pd.DataFrame, empty_message: str = "No rows are available.") -> str:
    if frame.empty:
        return empty_message
    columns = [
        "ts_code",
        "name",
        "industry",
        "review_status",
        "total_score",
        "roe",
        "debt_to_assets",
        "dividend_yield_ttm",
        "dividend_payback_years",
        "pe_ttm",
        "pb",
        "fcf_yield",
        "positive_dividend_years",
        "failed_reasons",
        "explanation",
    ]
    existing = [column for column in columns if column in frame.columns]
    view = frame[existing].copy()
    for column in ["roe", "debt_to_assets", "dividend_yield_ttm", "fcf_yield"]:
        if column in view.columns:
            view[column] = view[column].map(_pct)
    for column in ["dividend_payback_years", "pe_ttm", "pb"]:
        if column in view.columns:
            view[column] = view[column].map(_num)
    return _markdown_table(view)


def _summary_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "ts_code",
        "name",
        "industry",
        "review_status",
        "total_score",
        "roe",
        "debt_to_assets",
        "dividend_yield_ttm",
        "pe_ttm",
        "pb",
        "failed_reasons",
    ]
    if frame.empty:
        return []
    existing = [column for column in columns if column in frame.columns]
    records = []
    for record in frame[existing].replace({np.nan: None}).to_dict(orient="records"):
        records.append({key: _json_safe_value(value) for key, value in record.items()})
    return records


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rows.append([_escape_markdown_cell(row.get(column)) for column in frame.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _escape_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _reason_guide_lines() -> list[str]:
    return [
        "- quality: ROE or cash conversion/free cash flow is below the conservative quality bar.",
        "- debt: debt-to-assets is above the balance-sheet safety bar.",
        "- dividend_yield: current dividend yield is below the conservative cash-return bar.",
        "- dividend_record: multi-year cash dividend record is too short or unavailable.",
        "- valuation: PE/PB does not satisfy the current payback sanity check.",
        "- missing_fundamental: no usable fundamental row is available as of the report date.",
    ]


def _mapping_lines(values: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}" for key, value in values.items()] or ["- none"]


def _pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _num(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""
