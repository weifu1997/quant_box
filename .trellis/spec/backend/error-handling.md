# Error Handling

> Error handling conventions for data, configuration, network, and CLI workflows.

---

## Overview

The project favors explicit, diagnosable failures over silent coercion. Validation errors name the broken contract, external-service errors explain the operational fix, and optional inputs degrade to empty data only when the caller can safely continue.

Use these exception types consistently:

| Situation | Exception | Example |
| --- | --- | --- |
| Invalid user/config/data contract | `ValueError` | malformed MultiIndex, invalid OHLCV, unsupported strategy option |
| Required local artifact is missing | `FileNotFoundError` | missing price file, factor cache, candidate signal |
| External dependency or service failure | `RuntimeError` | Tushare proxy not configured, network failure, empty remote universe |
| Optimization time budget exceeded | `OptimizationTimeoutError` | walk-forward validation timeout with partial results |

---

## Validation Pattern

Validate at the boundary where data enters a module. Normalize first, then reject impossible final state.

Example from `src/data_fetcher_frames.py`:

```python
missing = [col for col in DAILY_FIELDS if col not in renamed.columns]
if missing:
    if renamed.empty:
        return pd.DataFrame(columns=DAILY_FIELDS)
    raise ValueError(f"Daily data is missing columns: {missing}")
```

Example from `src/data_fetcher_frames.py`:

```python
invalid = missing_prices | missing_flows | non_positive_prices | negative_flows | high_below_range | low_above_range
if invalid.any():
    examples = ", ".join(_daily_row_labels(frame.loc[invalid].head(5)))
    raise ValueError(f"Daily data has invalid OHLCV values in {int(invalid.sum())} rows: {examples}")
```

Include a short preview of bad rows or missing columns. Do not dump whole DataFrames into exceptions.

---

## DataFrame Contract Errors

Use precise messages for pandas shape requirements. Tests often assert these messages, so keep them stable when behavior is unchanged.

Common project messages:

```python
raise ValueError("score_panel must use MultiIndex: date/instrument.")
raise ValueError("factors must use MultiIndex: datetime/instrument.")
raise ValueError("price_df MultiIndex columns must be field/instrument.")
raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
```

If adding a new DataFrame boundary, specify:

- required index levels,
- required columns or fields,
- date normalization expectations,
- whether empty input is allowed.

---

## Missing File Handling

Return empty data only for optional files. Raise `FileNotFoundError` for required workflow inputs.

Good optional pattern from holdings readers:

```python
if not holdings_path.exists():
    return []
```

Good required pattern from signal/backtest paths:

```python
if not price_path.exists():
    raise FileNotFoundError(f"Price file not found: {price_path}. Run conversion first.")
```

When the fix is known, include it in the message, for example "Run scripts/run_convert_data.py first."

---

## External Service Errors

Tushare HTTP proxy failures are operational issues. Wrap request and response errors as `RuntimeError` with an actionable diagnosis.

Example from `src/tushare_client.py`:

```python
raise RuntimeError(
    "Failed to connect to tushare HTTP proxy "
    f"({endpoint}). Check the full proxy URL/path, firewall/network permission, "
    "and whether the proxy service is running."
) from exc
```

Do not expose tokens in errors. Use `describe_endpoint()` or `redacted_request_preview()` for diagnostics.

Retry loops should catch only expected transient errors (`RuntimeError`, `ValueError`) and either:

- continue with a recorded failed symbol/date when the workflow is designed to be resumable, or
- re-raise when `skip_failed=False` or a required global artifact cannot be produced.

---

## CLI Error Handling

Scripts should log progress and allow unexpected exceptions to surface with tracebacks. `scripts/run_backtest.py` wraps the main body with:

```python
try:
    ...
except Exception:
    logger.exception("Backtest failed. See the traceback and prior input summaries in this log.")
    raise
```

Use `SystemExit(1)` only for small user-facing checks where a clean CLI exit code is the intended interface, as in `scripts/run_update_data.py`.

Long workflows should write status artifacts before raising when possible:

- `_stage(..., "error", message)` in automatic signal flow,
- `failed_fetches.csv` for skipped raw-data symbols,
- `auto_failure_analysis.json` and related CSVs for blocked auto-signal runs.

---

## Error Matrix

| Condition | Behavior |
| --- | --- |
| `config/settings.yaml` is missing and default path was requested | Use in-memory `DEFAULT_CONFIG` |
| explicit config path missing | `FileNotFoundError` |
| known config section has wrong type | `ValueError("Invalid config: ... must be a mapping")` |
| unknown config key | `logger.warning`, do not reject |
| missing env var in `${VAR}` expansion | `ValueError` naming the missing variable |
| empty optional holdings file | empty list or empty DataFrame |
| missing required factor/price artifact | `FileNotFoundError` |
| malformed Tushare response shape | `ValueError` before normalization |
| Tushare proxy request failure | `RuntimeError` with endpoint and operational hints |
| invalid market OHLCV values | `ValueError` with count and up to 5 row labels |
| auto optimization timeout | `OptimizationTimeoutError` with partial results |

---

## Common Mistakes

### Catching `Exception` without preserving diagnostics

Only use broad catches for defensive non-blocking reports or fixture corruption, and include context. Prefer `logger.exception(...)` or `pytest.fail(...)` over swallowing the error.

### Replacing contract errors with generic messages

Tests use `assertRaisesRegex` for important messages. Keep messages specific enough to find the broken setting, file, column, or index.

### Logging secrets while debugging config

Never print or log Tushare tokens, account files, or holdings details beyond what a report intentionally contains. Use redacted helpers.
