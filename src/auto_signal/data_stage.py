"""Data refresh, factor preparation, and data-quality stage."""

from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.adj_factor_metadata import build_adj_factor_metadata, write_adj_factor_metadata
from src.data_converter import convert_to_qlib_format
from src.data_fetcher import update_daily_data_resumable
from src.data_governance import build_data_governance_report, write_data_governance_report
from src.data_health import build_data_health_report, write_data_health_report
from src.factor_calculator import load_or_compute_factors
from src.auto_signal.models import DataPreparationStageResult
from src.auto_signal.status import (
    can_reuse_conversion_outputs,
    stage,
    update_result_status,
    update_status_message,
)
from src.config_loader import resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataStageServices:
    update_daily_data_resumable: Callable[..., Any] = update_daily_data_resumable
    convert_to_qlib_format: Callable[..., Any] = convert_to_qlib_format
    load_or_compute_factors: Callable[..., Any] = load_or_compute_factors
    build_data_health_report: Callable[..., Any] = build_data_health_report
    write_data_health_report: Callable[..., Any] = write_data_health_report
    build_adj_factor_metadata: Callable[..., Any] = build_adj_factor_metadata
    write_adj_factor_metadata: Callable[..., Any] = write_adj_factor_metadata
    build_data_governance_report: Callable[..., Any] = build_data_governance_report
    write_data_governance_report: Callable[..., Any] = write_data_governance_report
    resolve_path: Callable[..., Any] = resolve_path
    can_reuse_conversion_outputs: Callable[..., Any] = can_reuse_conversion_outputs


def run_data_preparation_stage(
    args: Namespace,
    config: dict[str, Any],
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
    *,
    services: DataStageServices | None = None,
) -> DataPreparationStageResult:
    """Run update, conversion, factor loading, and data quality stages."""
    services = services or DataStageServices()
    update_info: dict[str, Any] | None = None
    if not args.skip_update:
        stage(status, out_dir, "update_data", "running")
        logger.info("Updating raw stock data that is missing or stale.")
        update_result = services.update_daily_data_resumable(
            start_date=args.start_date,
            end_date=end_date,
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
            max_chunks=args.max_chunks,
            include_existing=args.include_existing,
        )
        update_info = update_result_status(update_result)
        update_state = str(update_info.get("status") or "complete")
        if update_state not in {"complete", "partial", "error"}:
            update_state = "complete"
        stage(status, out_dir, "update_data", update_state, update_status_message(update_info))
        if update_state == "error" and not args.allow_unhealthy:
            raise RuntimeError(f"Data update failed: {update_info.get('last_error') or update_info}")
    else:
        stage(status, out_dir, "update_data", "skipped")

    if not args.skip_convert:
        if services.can_reuse_conversion_outputs(update_info, config, end_date):
            logger.info("Skipping conversion because no raw files changed and conversion outputs cover %s.", end_date)
            stage(status, out_dir, "convert_data", "skipped", "cache_current_no_raw_changes")
        else:
            stage(status, out_dir, "convert_data", "running")
            logger.info("Converting raw data to Qlib provider and price panels.")
            services.convert_to_qlib_format()
            stage(status, out_dir, "convert_data", "complete")
    else:
        stage(status, out_dir, "convert_data", "skipped")

    factor_file = config["factors"]["cache_file"]
    stage(status, out_dir, "compute_factors", "running")
    logger.info("Loading or computing factors.")
    factor_path = services.resolve_path(factor_file)
    if args.skip_factor:
        if not factor_path.exists():
            raise FileNotFoundError(f"Factor cache not found: {factor_path}")
        factors = pd.read_parquet(factor_path)
    else:
        factors = services.load_or_compute_factors(args.start_date, end_date, cache_file=factor_file, force=args.force_factor)
    stage(status, out_dir, "compute_factors", "complete")

    price_path = services.resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}. Run conversion first.")
    prices = pd.read_parquet(price_path)

    stage(status, out_dir, "data_health", "running")
    data_health = services.build_data_health_report(config, price_df=prices, factor_df=factors)
    health_json, health_csv = services.write_data_health_report(data_health, out_dir)
    artifacts.extend([health_json, health_csv])
    stage(status, out_dir, "data_health", "complete", "healthy" if data_health.is_healthy else ",".join(data_health.issues))
    data_gate = data_health.is_healthy or args.allow_unhealthy

    if not args.skip_adj_factor_meta:
        stage(status, out_dir, "adj_factor_meta", "running")
        adj_factor_meta = services.build_adj_factor_metadata(config)
        adj_factor_meta_path = services.write_adj_factor_metadata(adj_factor_meta, config)
        artifacts.append(adj_factor_meta_path)
        stage(
            status,
            out_dir,
            "adj_factor_meta",
            "complete",
            f"{adj_factor_meta.files_with_adj_factor}/{adj_factor_meta.raw_file_count}",
        )
    else:
        stage(status, out_dir, "adj_factor_meta", "skipped")

    stage(status, out_dir, "data_governance", "running")
    data_governance = services.build_data_governance_report(config)
    governance_path = services.write_data_governance_report(data_governance, out_dir)
    artifacts.append(governance_path)
    stage(
        status,
        out_dir,
        "data_governance",
        "complete",
        "point_in_time_ready" if data_governance.is_point_in_time_ready else ",".join(data_governance.issues),
    )
    governance_gate = data_governance.is_point_in_time_ready

    return DataPreparationStageResult(
        factor_file=factor_file,
        factors=factors,
        prices=prices,
        data_health=data_health,
        data_governance=data_governance,
        data_gate=data_gate,
        governance_gate=governance_gate,
        health_json=health_json,
        governance_path=governance_path,
    )
