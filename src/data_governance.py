from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_config, resolve_path


@dataclass
class DataGovernanceReport:
    generated_at: str
    universe_file: str
    universe_rows: int
    universe_has_list_date: bool
    universe_has_delist_date: bool
    universe_has_list_status: bool
    universe_has_industry: bool
    delisted_rows: int
    st_calendar_file: str
    st_calendar_available: bool
    st_calendar_rows: int
    st_calendar_has_ts_code: bool
    st_calendar_start_date: str
    st_calendar_end_date: str
    st_filter_mode: str
    index_constituents_file: str
    index_constituents_available: bool
    index_constituents_rows: int
    index_constituents_has_trade_date: bool
    index_constituents_has_weight: bool
    index_constituents_start_date: str
    index_constituents_end_date: str
    index_constituents_unique_dates: int
    index_constituents_expected_months: int
    index_constituents_observed_months: int
    index_constituents_missing_months: int
    index_constituents_month_coverage: float
    daily_basic_file: str
    daily_basic_available: bool
    daily_basic_rows: int
    daily_basic_has_trade_date: bool
    daily_basic_has_ts_code: bool
    daily_basic_market_cap_field: str
    daily_basic_has_market_cap: bool
    daily_basic_start_date: str
    daily_basic_end_date: str
    daily_basic_unique_dates: int
    daily_basic_expected_dates: int
    daily_basic_covered_dates: int
    daily_basic_missing_dates: int
    daily_basic_date_coverage: float
    raw_adj_factor_sampled_files: int
    raw_adj_factor_files_with_column: int
    price_panel_start_date: str
    point_in_time_start_date: str
    factor_cache_meta_file: str
    factor_cache_meta_available: bool
    factor_cache_meta_start_date: str
    factor_cache_meta_end_date: str
    factor_cache_meta_symbols: int
    adj_factor_meta_file: str
    adj_factor_meta_available: bool
    adj_factor_meta_source: str
    adj_factor_meta_raw_file_count: int
    adj_factor_meta_files_with_adj_factor: int
    adj_factor_meta_symbols: int
    adj_factor_meta_missing_symbols: list[str]
    adj_factor_meta_end_date: str
    adj_factor_meta_digest: str
    issues: list[str]
    warnings: list[str]
    repair_actions: list[dict[str, Any]]

    @property
    def is_point_in_time_ready(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["is_point_in_time_ready"] = self.is_point_in_time_ready
        return result


def build_data_governance_report(config: dict | None = None, sample_raw_files: int = 50) -> DataGovernanceReport:
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    gov_cfg = cfg.get("data_governance", {})
    issues: list[str] = []
    warnings: list[str] = []
    data_start = _config_start_date(data_cfg)
    factor_meta_path = resolve_path(str(data_cfg.get("factor_meta_file") or cfg.get("factors", {}).get("cache_file", "data/factors/alpha158.parquet")) + ".meta.json")
    factor_meta = _read_json_if_exists(factor_meta_path)
    factor_symbols = factor_meta.get("symbols", []) if isinstance(factor_meta, dict) else []
    factor_meta_start_date = str(factor_meta.get("start_date", "")) if isinstance(factor_meta, dict) else ""
    factor_meta_end_date = str(factor_meta.get("end_date", "")) if isinstance(factor_meta, dict) else ""
    price_panel_dates = _price_panel_trade_dates(cfg)
    price_panel_start_date = _date_index_start_date(price_panel_dates)
    point_in_time_start = _point_in_time_start_date(data_start, factor_meta_start_date, price_panel_start_date)
    expected_price_dates = _filter_date_texts(price_panel_dates, point_in_time_start, factor_meta_end_date)
    if not factor_meta:
        warnings.append("factor_cache_meta_missing")

    universe_path = resolve_path(data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    universe = _read_csv_if_exists(universe_path)
    universe_rows = int(len(universe))
    universe_has_list_date = "list_date" in universe.columns
    universe_has_delist_date = "delist_date" in universe.columns
    universe_has_list_status = "list_status" in universe.columns
    universe_has_industry = "industry" in universe.columns
    delisted_rows = _delisted_rows(universe)
    if universe.empty:
        issues.append("universe_metadata_missing")
    if not universe_has_list_date:
        issues.append("universe_list_date_missing")
    if not universe_has_delist_date:
        issues.append("universe_delist_date_missing")
    if not universe_has_list_status:
        warnings.append("universe_list_status_missing")
    if not universe_has_industry:
        warnings.append("universe_industry_missing")
    if universe_has_delist_date and delisted_rows == 0:
        warnings.append("universe_has_no_delisted_rows_observed")

    st_path_value = gov_cfg.get("st_calendar_file") or data_cfg.get("st_calendar_file")
    st_path = resolve_path(st_path_value) if st_path_value else None
    st_calendar = _read_csv_if_exists(st_path) if st_path is not None else pd.DataFrame()
    st_calendar_available = not st_calendar.empty
    st_calendar_has_ts_code = "ts_code" in st_calendar.columns
    st_calendar_start_date, st_calendar_end_date = _date_range_from_columns(
        st_calendar,
        ["st_start_date", "start_date", "begin_date", "date", "ann_date", "st_end_date", "end_date"],
    )
    st_filter_mode = "historical_calendar" if st_calendar_available else "current_name_fallback" if data_cfg.get("exclude_st", True) else "disabled"
    if bool(data_cfg.get("exclude_st", True)) and not st_calendar_available:
        issues.append("st_calendar_missing_current_name_filter_only")
    if st_calendar_available:
        if not st_calendar_has_ts_code:
            issues.append("st_calendar_ts_code_missing")
        if not _has_any_column(st_calendar, ["st_start_date", "start_date", "begin_date", "date"]):
            issues.append("st_calendar_start_date_missing")
        if st_calendar_start_date and data_start and _date_after(st_calendar_start_date, data_start):
            issues.append(f"st_calendar_start_after_data_start:{st_calendar_start_date}>{data_start}")

    index_path = resolve_path(gov_cfg.get("index_constituents_file") or data_cfg.get("hs300_constituents_file", "data/raw/hs300_constituents.csv"))
    index_frame = _read_csv_if_exists(index_path)
    index_available = not index_frame.empty
    index_has_trade_date = "trade_date" in index_frame.columns
    index_has_weight = "weight" in index_frame.columns
    index_start, index_end = _date_range_text(index_frame.get("trade_date"))
    index_date_texts = _date_texts_from_series(index_frame.get("trade_date"))
    index_months = _month_texts_from_dates(index_date_texts)
    expected_index_months = _month_range_texts(point_in_time_start, factor_meta_end_date)
    index_expected_months = len(expected_index_months)
    index_observed_months = len(expected_index_months & index_months) if expected_index_months else 0
    index_missing_months = max(index_expected_months - index_observed_months, 0)
    index_month_coverage = _coverage_ratio(index_observed_months, index_expected_months)
    required_index_columns = [str(column) for column in gov_cfg.get("required_index_columns", ["index_code", "con_code", "trade_date", "weight"])]
    missing_index_columns = [column for column in required_index_columns if column not in index_frame.columns]
    if not index_available:
        issues.append("index_constituents_file_missing")
    elif missing_index_columns:
        issues.append("index_constituents_missing_columns:" + ",".join(missing_index_columns))
    if index_available and index_start and point_in_time_start and _month_after(index_start, point_in_time_start):
        issues.append(f"index_constituents_start_after_point_in_time_start:{index_start}>{point_in_time_start}")
    min_index_month_coverage = _float_value(gov_cfg.get("min_index_constituents_month_coverage", 1.0), 1.0)
    if index_available and index_expected_months and index_month_coverage < min_index_month_coverage:
        issues.append(
            "index_constituents_month_coverage_below_required:"
            f"{index_observed_months}/{index_expected_months}<{min_index_month_coverage:.2f}"
        )

    research_exposure_cfg = cfg.get("research", {}).get("exposure", {})
    daily_basic_path = resolve_path(
        research_exposure_cfg.get("daily_basic_file")
        or data_cfg.get("daily_basic_file", "data/factors/daily_basic.parquet")
    )
    daily_basic = _read_table_if_exists(daily_basic_path)
    daily_basic_available = not daily_basic.empty
    daily_basic_has_trade_date = "trade_date" in daily_basic.columns
    daily_basic_has_ts_code = "ts_code" in daily_basic.columns
    daily_basic_market_cap_field = str(research_exposure_cfg.get("market_cap_field", "circ_mv"))
    daily_basic_has_market_cap = daily_basic_market_cap_field in daily_basic.columns
    daily_basic_start, daily_basic_end = _date_range_text(daily_basic.get("trade_date"))
    daily_basic_date_texts = _date_texts_from_series(daily_basic.get("trade_date"))
    daily_basic_expected_dates = len(expected_price_dates)
    daily_basic_covered_dates = len(expected_price_dates & daily_basic_date_texts) if expected_price_dates else 0
    daily_basic_missing_dates = max(daily_basic_expected_dates - daily_basic_covered_dates, 0)
    daily_basic_date_coverage = _coverage_ratio(daily_basic_covered_dates, daily_basic_expected_dates)
    if not daily_basic_available:
        issues.append("daily_basic_missing_market_cap_exposure_unavailable")
    else:
        missing_daily_basic_columns = [
            column
            for column, present in {
                "trade_date": daily_basic_has_trade_date,
                "ts_code": daily_basic_has_ts_code,
                daily_basic_market_cap_field: daily_basic_has_market_cap,
            }.items()
            if not present
        ]
        if missing_daily_basic_columns:
            issues.append("daily_basic_missing_columns:" + ",".join(missing_daily_basic_columns))
        if daily_basic_start and point_in_time_start and _date_after(daily_basic_start, point_in_time_start):
            issues.append(f"daily_basic_start_after_point_in_time_start:{daily_basic_start}>{point_in_time_start}")
        min_daily_basic_coverage = _float_value(gov_cfg.get("min_daily_basic_date_coverage", 1.0), 1.0)
        if daily_basic_expected_dates and daily_basic_date_coverage < min_daily_basic_coverage:
            issues.append(
                "daily_basic_date_coverage_below_required:"
                f"{daily_basic_covered_dates}/{daily_basic_expected_dates}<{min_daily_basic_coverage:.2f}"
            )

    raw_dir = resolve_path(data_cfg.get("raw_dir", "data/raw"))
    raw_file_count = _count_raw_stock_files(raw_dir)
    sampled, with_adj_factor = _sample_raw_adj_factor(raw_dir, sample_raw_files)
    if sampled == 0:
        issues.append("raw_price_files_missing")
    elif with_adj_factor < sampled:
        issues.append(f"raw_adj_factor_missing_in_sample:{with_adj_factor}/{sampled}")

    if st_calendar_available and st_calendar_end_date and factor_meta_end_date and _date_after(factor_meta_end_date, st_calendar_end_date):
        warnings.append(f"st_calendar_end_before_factor_end:{st_calendar_end_date}<{factor_meta_end_date}")
    if daily_basic_end and factor_meta_end_date and _date_after(factor_meta_end_date, daily_basic_end):
        warnings.append(f"daily_basic_end_before_factor_end:{daily_basic_end}<{factor_meta_end_date}")
    if index_available and index_end and factor_meta_end_date and _month_after(factor_meta_end_date, index_end):
        issues.append(f"index_constituents_end_before_factor_end:{index_end}<{factor_meta_end_date}")

    adj_meta_path = resolve_path(gov_cfg.get("adj_factor_meta_file", "data/factors/adj_factor_meta.json"))
    adj_factor_meta = _read_json_if_exists(adj_meta_path)
    adj_factor_meta_available = bool(adj_factor_meta)
    if not adj_meta_path.exists():
        issues.append("adj_factor_version_meta_missing")
    elif not adj_factor_meta:
        issues.append("adj_factor_version_meta_invalid_json")
    adj_factor_meta_source = str(adj_factor_meta.get("source", "")) if isinstance(adj_factor_meta, dict) else ""
    adj_factor_meta_raw_file_count = _int_value(adj_factor_meta.get("raw_file_count")) if isinstance(adj_factor_meta, dict) else 0
    adj_factor_meta_files_with_adj_factor = (
        _int_value(adj_factor_meta.get("files_with_adj_factor")) if isinstance(adj_factor_meta, dict) else 0
    )
    adj_factor_meta_symbols = _int_value(adj_factor_meta.get("symbol_count")) if isinstance(adj_factor_meta, dict) else 0
    adj_factor_meta_end_date = str(adj_factor_meta.get("end_date", "")) if isinstance(adj_factor_meta, dict) else ""
    adj_factor_meta_digest = str(adj_factor_meta.get("digest", "")) if isinstance(adj_factor_meta, dict) else ""
    adj_factor_meta_missing_symbols = _adj_factor_missing_symbols(adj_factor_meta.get("issues", []) if isinstance(adj_factor_meta, dict) else [])
    if adj_factor_meta_available:
        if adj_factor_meta_source != "raw_csv_adj_factor":
            issues.append("adj_factor_version_meta_source_unknown")
        if raw_file_count and adj_factor_meta_raw_file_count != raw_file_count:
            issues.append(f"adj_factor_version_meta_file_count_mismatch:{adj_factor_meta_raw_file_count}/{raw_file_count}")
        if adj_factor_meta_raw_file_count and adj_factor_meta_files_with_adj_factor < adj_factor_meta_raw_file_count:
            issues.append(
                "adj_factor_version_meta_missing_files:"
                f"{adj_factor_meta_files_with_adj_factor}/{adj_factor_meta_raw_file_count}"
            )
        if not adj_factor_meta_digest:
            issues.append("adj_factor_version_meta_digest_missing")
        meta_issues = adj_factor_meta.get("issues", []) if isinstance(adj_factor_meta, dict) else []
        if meta_issues:
            issues.append("adj_factor_version_meta_issues:" + ",".join(map(str, meta_issues[:5])))

    repair_actions = _build_repair_actions(
        issues=issues,
        warnings=warnings,
        data_start=point_in_time_start,
        factor_end=factor_meta_end_date,
        daily_basic_path=daily_basic_path,
        index_path=index_path,
        adj_meta_path=adj_meta_path,
        raw_dir=raw_dir,
        data_cfg=data_cfg,
        adj_factor_missing_symbols=adj_factor_meta_missing_symbols,
    )

    return DataGovernanceReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        universe_file=str(universe_path),
        universe_rows=universe_rows,
        universe_has_list_date=universe_has_list_date,
        universe_has_delist_date=universe_has_delist_date,
        universe_has_list_status=universe_has_list_status,
        universe_has_industry=universe_has_industry,
        delisted_rows=delisted_rows,
        st_calendar_file=str(st_path or ""),
        st_calendar_available=st_calendar_available,
        st_calendar_rows=int(len(st_calendar)),
        st_calendar_has_ts_code=st_calendar_has_ts_code,
        st_calendar_start_date=st_calendar_start_date,
        st_calendar_end_date=st_calendar_end_date,
        st_filter_mode=st_filter_mode,
        index_constituents_file=str(index_path),
        index_constituents_available=index_available,
        index_constituents_rows=int(len(index_frame)),
        index_constituents_has_trade_date=index_has_trade_date,
        index_constituents_has_weight=index_has_weight,
        index_constituents_start_date=index_start,
        index_constituents_end_date=index_end,
        index_constituents_unique_dates=len(index_date_texts),
        index_constituents_expected_months=index_expected_months,
        index_constituents_observed_months=index_observed_months,
        index_constituents_missing_months=index_missing_months,
        index_constituents_month_coverage=index_month_coverage,
        daily_basic_file=str(daily_basic_path),
        daily_basic_available=daily_basic_available,
        daily_basic_rows=int(len(daily_basic)),
        daily_basic_has_trade_date=daily_basic_has_trade_date,
        daily_basic_has_ts_code=daily_basic_has_ts_code,
        daily_basic_market_cap_field=daily_basic_market_cap_field,
        daily_basic_has_market_cap=daily_basic_has_market_cap,
        daily_basic_start_date=daily_basic_start,
        daily_basic_end_date=daily_basic_end,
        daily_basic_unique_dates=len(daily_basic_date_texts),
        daily_basic_expected_dates=daily_basic_expected_dates,
        daily_basic_covered_dates=daily_basic_covered_dates,
        daily_basic_missing_dates=daily_basic_missing_dates,
        daily_basic_date_coverage=daily_basic_date_coverage,
        raw_adj_factor_sampled_files=sampled,
        raw_adj_factor_files_with_column=with_adj_factor,
        price_panel_start_date=price_panel_start_date,
        point_in_time_start_date=point_in_time_start,
        factor_cache_meta_file=str(factor_meta_path),
        factor_cache_meta_available=bool(factor_meta),
        factor_cache_meta_start_date=factor_meta_start_date,
        factor_cache_meta_end_date=factor_meta_end_date,
        factor_cache_meta_symbols=len(factor_symbols) if isinstance(factor_symbols, list) else 0,
        adj_factor_meta_file=str(adj_meta_path),
        adj_factor_meta_available=adj_factor_meta_available,
        adj_factor_meta_source=adj_factor_meta_source,
        adj_factor_meta_raw_file_count=adj_factor_meta_raw_file_count,
        adj_factor_meta_files_with_adj_factor=adj_factor_meta_files_with_adj_factor,
        adj_factor_meta_symbols=adj_factor_meta_symbols,
        adj_factor_meta_missing_symbols=adj_factor_meta_missing_symbols,
        adj_factor_meta_end_date=adj_factor_meta_end_date,
        adj_factor_meta_digest=adj_factor_meta_digest,
        issues=issues,
        warnings=warnings,
        repair_actions=repair_actions,
    )


def write_data_governance_report(report: DataGovernanceReport, out_dir: str | Path) -> Path:
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "data_governance_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_csv_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_table_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _adj_factor_missing_symbols(issues: Any) -> list[str]:
    if not isinstance(issues, list):
        return []
    symbols: list[str] = []
    for issue in issues:
        text = str(issue)
        prefix = "adj_factor_missing:"
        if not text.startswith(prefix):
            continue
        symbol = text[len(prefix) :].strip().upper()
        if _valid_symbol(symbol) and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _delisted_rows(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    count = 0
    if "list_status" in frame.columns:
        count += int((frame["list_status"].fillna("").astype(str).str.upper() == "D").sum())
    if "delist_date" in frame.columns:
        delisted = _coerce_dates(frame["delist_date"])
        count = max(count, int(delisted.notna().sum()))
    return count


def _has_any_column(frame: pd.DataFrame, columns: list[str]) -> bool:
    return any(column in frame.columns for column in columns)


def _date_range_text(series: pd.Series | None) -> tuple[str, str]:
    if series is None:
        return "", ""
    dates = _coerce_dates(series).dropna()
    if dates.empty:
        return "", ""
    return str(dates.min().date()), str(dates.max().date())


def _date_range_from_columns(frame: pd.DataFrame, columns: list[str]) -> tuple[str, str]:
    date_parts = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = _coerce_dates(frame[column]).dropna()
        if not values.empty:
            date_parts.append(values)
    if not date_parts:
        return "", ""
    dates = pd.concat(date_parts)
    return str(dates.min().date()), str(dates.max().date())


def _config_start_date(data_cfg: dict[str, Any]) -> str:
    value = data_cfg.get("history_start_date") or data_cfg.get("start_date")
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return str(pd.Timestamp(parsed).date())


def _price_panel_start_date(config: dict[str, Any]) -> str:
    return _date_index_start_date(_price_panel_trade_dates(config))


def _price_panel_trade_dates(config: dict[str, Any]) -> pd.DatetimeIndex:
    ic_cfg = config.get("ic")
    if not isinstance(ic_cfg, dict) or "price_file" not in ic_cfg:
        return pd.DatetimeIndex([])
    price_value = ic_cfg.get("price_file")
    if not price_value:
        return pd.DatetimeIndex([])
    path = resolve_path(price_value)
    if not path.exists():
        return pd.DatetimeIndex([])
    try:
        frame = pd.read_parquet(path, columns=[])
    except (OSError, ValueError, TypeError):
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            return pd.DatetimeIndex([])
    return _date_index_from_values(frame.index)


def _date_index_from_values(values: Any) -> pd.DatetimeIndex:
    if isinstance(values, pd.MultiIndex):
        candidates = [_date_index_from_values(values.get_level_values(pos)) for pos in range(values.nlevels)]
        return max(candidates, key=len) if candidates else pd.DatetimeIndex([])
    dates = _coerce_dates(values)
    index = pd.DatetimeIndex(dates.dropna()).normalize().unique()
    return index.sort_values()


def _date_index_start_date(dates: pd.DatetimeIndex) -> str:
    if dates.empty:
        return ""
    return str(pd.Timestamp(dates.min()).date())


def _filter_date_texts(dates: pd.DatetimeIndex, start: str, end: str) -> set[str]:
    if dates.empty:
        return set()
    start_date = pd.to_datetime(start, errors="coerce")
    end_date = pd.to_datetime(end, errors="coerce")
    selected = dates
    if not pd.isna(start_date):
        selected = selected[selected >= pd.Timestamp(start_date).normalize()]
    if not pd.isna(end_date):
        selected = selected[selected <= pd.Timestamp(end_date).normalize()]
    return {str(pd.Timestamp(date).date()) for date in selected}


def _date_texts_from_series(series: pd.Series | None) -> set[str]:
    if series is None:
        return set()
    return {str(pd.Timestamp(date).date()) for date in _date_index_from_values(series)}


def _month_texts_from_dates(date_texts: set[str]) -> set[str]:
    months: set[str] = set()
    for value in date_texts:
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            months.add(pd.Timestamp(parsed).strftime("%Y-%m"))
    return months


def _month_range_texts(start: str, end: str) -> set[str]:
    start_date = pd.to_datetime(start, errors="coerce")
    end_date = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date):
        return set()
    periods = pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M")
    return {str(period) for period in periods}


def _coverage_ratio(observed: int, expected: int) -> float:
    return float(observed / expected) if expected else 0.0


def _coerce_dates(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        raw = values.dropna()
    else:
        try:
            raw = pd.Series(list(values)).dropna()
        except TypeError:
            raw = pd.Series([values]).dropna()
    if raw.empty:
        return pd.Series([], dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(raw):
        return pd.to_datetime(raw, errors="coerce")
    text = raw.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    compact_date = compact.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if compact_date.any():
        parsed.loc[compact_date] = pd.to_datetime(compact.loc[compact_date], format="%Y%m%d", errors="coerce")
    if (~compact_date).any():
        fallback = pd.to_datetime(text.loc[~compact_date], errors="coerce")
        fallback_series = pd.Series(fallback, index=text.loc[~compact_date].index)
        if fallback_series.isna().any():
            mixed = pd.to_datetime(text.loc[~compact_date], errors="coerce", format="mixed")
            fallback_series = fallback_series.where(fallback_series.notna(), pd.Series(mixed, index=fallback_series.index))
        parsed.loc[~compact_date] = fallback_series
    return parsed


def _point_in_time_start_date(*dates: str) -> str:
    parsed = [pd.Timestamp(value).normalize() for value in pd.to_datetime(list(dates), errors="coerce") if not pd.isna(value)]
    if not parsed:
        return ""
    return str(max(parsed).date())


def _date_after(left: str, right: str) -> bool:
    left_date = pd.to_datetime(left, errors="coerce")
    right_date = pd.to_datetime(right, errors="coerce")
    if pd.isna(left_date) or pd.isna(right_date):
        return False
    return pd.Timestamp(left_date).normalize() > pd.Timestamp(right_date).normalize()


def _month_after(left: str, right: str) -> bool:
    left_date = pd.to_datetime(left, errors="coerce")
    right_date = pd.to_datetime(right, errors="coerce")
    if pd.isna(left_date) or pd.isna(right_date):
        return False
    return pd.Timestamp(left_date).to_period("M") > pd.Timestamp(right_date).to_period("M")


def _sample_raw_adj_factor(raw_dir: Path, sample_raw_files: int) -> tuple[int, int]:
    if not raw_dir.exists():
        return 0, 0
    files = sorted(path for path in raw_dir.glob("*.csv") if _is_stock_csv(path))[: max(sample_raw_files, 1)]
    with_column = 0
    for path in files:
        try:
            header = pd.read_csv(path, nrows=0)
        except pd.errors.EmptyDataError:
            continue
        if "adj_factor" in header.columns:
            with_column += 1
    return len(files), with_column


def _count_raw_stock_files(raw_dir: Path) -> int:
    if not raw_dir.exists():
        return 0
    return sum(1 for path in raw_dir.glob("*.csv") if _is_stock_csv(path))


def _build_repair_actions(
    issues: list[str],
    warnings: list[str],
    data_start: str,
    factor_end: str,
    daily_basic_path: Path,
    index_path: Path,
    adj_meta_path: Path,
    raw_dir: Path,
    data_cfg: dict[str, Any],
    adj_factor_missing_symbols: list[str] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    end_date = factor_end or str(data_cfg.get("end_date", "auto"))
    update_script = r".\.venv\Scripts\python.exe scripts\run_update_point_in_time_data.py"
    if any(issue.startswith("daily_basic_") for issue in issues) or any(warning.startswith("daily_basic_") for warning in warnings):
        actions.append(
            {
                "component": "daily_basic",
                "reason": "daily_basic_history_or_freshness_incomplete",
                "start_date": data_start,
                "end_date": end_date,
                "output": str(daily_basic_path),
                "commands": [
                    f"{update_script} --start-date {data_start} --end-date {end_date} --skip-index-constituents --skip-st-calendar"
                ],
            }
        )
    if any(issue.startswith("index_constituents_") for issue in issues):
        actions.append(
            {
                "component": "index_constituents",
                "reason": "index_constituents_history_or_freshness_incomplete",
                "start_date": data_start,
                "end_date": end_date,
                "output": str(index_path),
                "commands": [
                    f"{update_script} --start-date {data_start} --end-date {end_date} --skip-daily-basic --skip-st-calendar"
                ],
            }
        )
    if any(issue.startswith("st_calendar_") for issue in issues):
        actions.append(
            {
                "component": "st_calendar",
                "reason": "st_calendar_history_incomplete",
                "output": str(data_cfg.get("st_calendar_file", "data/raw/st_calendar.csv")),
                "commands": [f"{update_script} --skip-daily-basic --skip-index-constituents"],
            }
        )
    adj_issue_prefixes = ("raw_adj_factor_", "adj_factor_version_meta_")
    if any(issue.startswith(adj_issue_prefixes) for issue in issues):
        commands = []
        missing_symbols = list(adj_factor_missing_symbols or [])
        if any(
            issue.startswith("raw_adj_factor_")
            or issue.startswith("adj_factor_version_meta_missing_files")
            or issue.startswith("adj_factor_version_meta_issues")
            for issue in issues
        ):
            if missing_symbols:
                codes = " ".join(missing_symbols)
                commands.append(
                    r".\.venv\Scripts\python.exe scripts\run_update_data.py "
                    f"--codes {codes} --start-date {data_start} --end-date {end_date} --force-full"
                )
            else:
                commands.append(
                    r".\.venv\Scripts\python.exe scripts\run_update_data.py --include-existing "
                    f"--chunk-size {int(data_cfg.get('update_chunk_size', 300))} "
                    f"--sleep-seconds {float(data_cfg.get('update_sleep_seconds', 0))}"
                )
        commands.append(r".\.venv\Scripts\python.exe scripts\run_build_adj_factor_meta.py")
        action = {
            "component": "adj_factor_version",
            "reason": "raw_adj_factor_or_metadata_incomplete",
            "raw_dir": str(raw_dir),
            "output": str(adj_meta_path),
            "commands": commands,
        }
        if missing_symbols:
            action["missing_symbols"] = missing_symbols
        actions.append(action)
    return actions


def _is_stock_csv(path: Path) -> bool:
    name = path.name.upper()
    return len(name) == len("000001.SZ.CSV") and name[:6].isdigit() and name[6:] in {".SZ.CSV", ".SH.CSV"}


def _valid_symbol(value: str) -> bool:
    return len(value) == len("000001.SZ") and value[:6].isdigit() and value[6:] in {".SZ", ".SH"}
