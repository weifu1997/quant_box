from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import load_config, resolve_path


def compute_alpha158_factors(
    start_date: str,
    end_date: str,
    instruments: str | list[str] | None = None,
    provider_uri: str | Path | None = None,
) -> pd.DataFrame:
    try:
        import qlib
        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH
    except ImportError as exc:
        raise RuntimeError("pyqlib is required to compute Alpha158 factors. Install requirements first.") from exc

    config = load_config()
    qlib_cfg = config.get("qlib", {})
    provider = resolve_path(provider_uri or qlib_cfg["provider_uri"])
    region = qlib_cfg.get("region", "cn")
    instruments = instruments or qlib_cfg.get("instruments", "csi300")

    qlib.init(provider_uri=str(provider), region=region)
    handler = Alpha158(
        instruments=instruments,
        start_time=start_date,
        end_time=end_date,
        fit_start_time=start_date,
        fit_end_time=end_date,
    )
    dataset = DatasetH(handler, segments={"full": (start_date, end_date)})
    factors = dataset.prepare("full", col_set="feature")
    if not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("Expected Alpha158 result to use a MultiIndex of datetime/instrument.")
    return factors.sort_index()


def load_or_compute_factors(
    start_date: str,
    end_date: str,
    cache_file: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    config = load_config()
    path = resolve_path(cache_file or config["factors"]["cache_file"])
    if path.exists() and not force:
        return pd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    factors = compute_alpha158_factors(start_date, end_date)
    factors.to_parquet(path)
    return factors
