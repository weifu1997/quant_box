"""模块说明：构建和保存复权因子元数据。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common import is_adj_factor_stock_csv as _is_adj_factor_stock_csv
from src.config_loader import load_config, resolve_path


@dataclass
class AdjFactorMetadata:
    """类说明：封装 AdjFactorMetadata 相关数据和行为。"""
    schema_version: int
    generated_at: str
    source: str
    raw_dir: str
    raw_file_count: int
    files_with_adj_factor: int
    rows_with_adj_factor: int
    symbol_count: int
    start_date: str
    end_date: str
    digest: str
    symbols: list[dict[str, Any]]
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        """函数说明：处理 to_dict 主要逻辑。"""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "source": self.source,
            "raw_dir": self.raw_dir,
            "raw_file_count": self.raw_file_count,
            "files_with_adj_factor": self.files_with_adj_factor,
            "rows_with_adj_factor": self.rows_with_adj_factor,
            "symbol_count": self.symbol_count,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "digest": self.digest,
            "symbols": self.symbols,
            "issues": self.issues,
        }


def build_adj_factor_metadata(config: dict | None = None) -> AdjFactorMetadata:
    """函数说明：构建 build_adj_factor_metadata 主要逻辑。"""
    cfg = config or load_config()
    raw_dir = resolve_path(cfg.get("data", {}).get("raw_dir", "data/raw"))
    files = sorted(path for path in raw_dir.glob("*.csv") if _is_adj_factor_stock_csv(path)) if raw_dir.exists() else []
    digest = hashlib.sha256()
    symbols: list[dict[str, Any]] = []
    issues: list[str] = []
    start_dates: list[str] = []
    end_dates: list[str] = []
    files_with_adj_factor = 0
    rows_with_adj_factor = 0

    for path in files:
        symbol, summary = _summarize_raw_adj_factor(path, digest)
        if summary is None:
            issues.append(f"adj_factor_missing:{symbol}")
            continue
        files_with_adj_factor += 1
        rows_with_adj_factor += int(summary["rows"])
        symbols.append(summary)
        start_dates.append(str(summary["start_date"]))
        end_dates.append(str(summary["end_date"]))

    if not files:
        issues.append("raw_price_files_missing")

    return AdjFactorMetadata(
        schema_version=1,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        source="raw_csv_adj_factor",
        raw_dir=str(raw_dir),
        raw_file_count=len(files),
        files_with_adj_factor=files_with_adj_factor,
        rows_with_adj_factor=rows_with_adj_factor,
        symbol_count=len(symbols),
        start_date=min(start_dates) if start_dates else "",
        end_date=max(end_dates) if end_dates else "",
        digest=digest.hexdigest(),
        symbols=symbols,
        issues=issues,
    )


def write_adj_factor_metadata(metadata: AdjFactorMetadata, config: dict | None = None, path: str | Path | None = None) -> Path:
    """函数说明：写入 write_adj_factor_metadata 主要逻辑。"""
    cfg = config or load_config()
    output_path = resolve_path(
        path
        or cfg.get("data_governance", {}).get("adj_factor_meta_file", "data/factors/adj_factor_meta.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _summarize_raw_adj_factor(path: Path, digest: hashlib._Hash) -> tuple[str, dict[str, Any] | None]:
    """函数说明：汇总 summarize_raw_adj_factor 的内部辅助逻辑。"""
    symbol = path.name[:-4].upper()
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError:
        return symbol, None

    rows = 0
    start_compact = ""
    end_compact = ""
    start_date = ""
    end_date = ""
    first_adj = 0.0
    last_adj = 0.0
    digest.update(symbol.encode("utf-8"))
    with handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "trade_date" not in reader.fieldnames or "adj_factor" not in reader.fieldnames:
            return symbol, None
        for row in reader:
            compact, iso_date = _normalize_date(row.get("trade_date", ""))
            if not compact:
                continue
            try:
                adj_value = float(str(row.get("adj_factor", "")).strip())
            except ValueError:
                continue
            rows += 1
            digest.update(f"|{compact}:{adj_value:.10g}".encode("utf-8"))
            if not start_compact or compact < start_compact:
                start_compact = compact
                start_date = iso_date
                first_adj = adj_value
            if not end_compact or compact >= end_compact:
                end_compact = compact
                end_date = iso_date
                last_adj = adj_value
    if rows == 0:
        return symbol, None

    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return symbol, {
        "instrument": symbol,
        "rows": rows,
        "start_date": start_date,
        "end_date": end_date,
        "first_adj_factor": first_adj,
        "last_adj_factor": last_adj,
        "file_mtime": mtime,
    }


def _normalize_date(value: object) -> tuple[str, str]:
    """函数说明：规范化 normalize_date 的内部辅助逻辑。"""
    compact = str(value).strip().replace("-", "")
    if len(compact) != 8 or not compact.isdigit():
        return "", ""
    return compact, f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
