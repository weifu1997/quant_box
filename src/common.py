from __future__ import annotations

from pathlib import Path

import pandas as pd


PRICE_FIELD_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "vol", "amount", "vwap", "adj_factor", "is_st"})


def normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def looks_like_field_table(columns: pd.Index, price_fields: set[str] | frozenset[str] = PRICE_FIELD_COLUMNS) -> bool:
    labels = {str(column).strip().lower() for column in columns}
    return len(labels) > 1 and bool(labels & price_fields)


def is_stock_csv(path: Path) -> bool:
    name = path.name.upper()
    return len(name) == len("000001.SZ.CSV") and name[:6].isdigit() and name[6:] in {".SZ.CSV", ".SH.CSV"}
