# Real Data Fixtures

Tests in this project should use real A-share data by default.
`real_data.py` loads deterministic slices from the committed snapshot in
`tests/fixtures/data_snapshot/`:

- `prices/ohlcv_adjusted.parquet`
- `prices/close_adjusted.parquet`
- `factors/alpha158.parquet`
- `factors/daily_basic.parquet`

The snapshot is small enough to live in the repository, so CI should run these
tests instead of skipping them when a local `data/` directory is unavailable. If a
test asks for a window outside the snapshot, the fixture may use the local full
cache under `data/`; when neither source can satisfy the request, the test must
fail rather than fabricate market data or silently skip.

The snapshot was cut from one batch of instruments and one time range. Factor,
price, and daily-basic slices must be filtered to the same requested instruments
and dates before being returned. Tests that genuinely depend on `daily_basic`
should call `require_real_market_data(..., require_daily_basic=True)` so an empty
slice is reported as a data-contract failure.

Deterministic hand-built data is still appropriate for narrow tests where exact
values matter more than market realism:

- mathematical fixtures for IC, ranking, or correlation formulas
- API failure, timeout, empty response, and malformed schema paths
- missing file and corrupt cache handling
- A-share business constraints such as limit-up/limit-down, halts, missing quotes,
  and capacity limits
- case normalization, duplicate instrument handling, and intraday ordering

Those tests should make the purpose clear in the test name or local helper name,
for example `math_fixture`, `api_failure`, `schema_error`, `limit_down`, `halt`,
or `missing_quote`. Normal pipeline tests should not mock core data-producing
functions such as factor loading, score building, signal resampling, or backtest
execution.

Do not add a broad audit that bans `pd.date_range` or hand-built `DataFrame`
objects everywhere. Those constructs are valid in formula and calendar tests.
Future audit checks should focus on suspicious fake market panels in ordinary
pipeline tests: very small stock/date grids, impossible OHLCV relationships, and
tests that build market-like data without a clear math or business exception in
their name.
