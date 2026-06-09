"""模块说明：覆盖 test_no_fake_market_data 相关行为的测试用例。"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "tests" / "fixtures" / "data_snapshot"


def test_real_data_fixture_uses_committed_snapshot_without_skip() -> None:
    """函数说明：验证 test_real_data_fixture_uses_committed_snapshot_without_skip 覆盖的行为场景。"""
    source = (ROOT / "tests" / "fixtures" / "real_data.py").read_text(encoding="utf-8")

    assert "SNAPSHOT_ROOT" in source
    assert "pytest.skip" not in source


def test_committed_real_data_snapshot_is_present() -> None:
    """函数说明：验证 test_committed_real_data_snapshot_is_present 覆盖的行为场景。"""
    expected = [
        SNAPSHOT_ROOT / "manifest.json",
        SNAPSHOT_ROOT / "prices" / "ohlcv_adjusted.parquet",
        SNAPSHOT_ROOT / "prices" / "close_adjusted.parquet",
        SNAPSHOT_ROOT / "factors" / "alpha158.parquet",
        SNAPSHOT_ROOT / "factors" / "daily_basic.parquet",
    ]

    missing = [path.relative_to(ROOT).as_posix() for path in expected if not path.exists()]
    assert missing == []

    manifest = json.loads((SNAPSHOT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["instruments"]) >= 5
    assert manifest["rows"]["factors"] > 0
    assert manifest["rows"]["prices"] > 0
    assert manifest["rows"]["daily_basic"] > 0


def test_fake_tushare_client_daily_response_is_snapshot_derived() -> None:
    """函数说明：验证 test_fake_tushare_client_daily_response_is_snapshot_derived 覆盖的行为场景。"""
    source = (ROOT / "tests" / "test_data_fetcher.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    fake_client = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "FakeTushareClient"
    )
    client_source = ast.get_source_segment(source, fake_client) or ""

    assert "_real_tushare_daily_rows" in client_source
    assert "_real_tushare_adj_factor_rows" in client_source
    for forbidden in ['"open": 10.0', '"high": 11.0', '"low": 9.0', '"close": 10.5']:
        assert forbidden not in client_source


def test_high_risk_pipeline_tests_use_real_market_data_fixture() -> None:
    """函数说明：验证 test_high_risk_pipeline_tests_use_real_market_data_fixture 覆盖的行为场景。"""
    files = [
        ROOT / "tests" / "test_pipeline_integration.py",
        ROOT / "tests" / "test_signal_generator.py",
        ROOT / "tests" / "test_optimizer.py",
        ROOT / "tests" / "test_scoring.py",
        ROOT / "tests" / "test_factor_ic.py",
        ROOT / "tests" / "test_backtest.py",
        ROOT / "tests" / "test_fast_monthly_backtest.py",
        ROOT / "tests" / "test_factor_calculator.py",
        ROOT / "tests" / "test_rolling_ic.py",
        ROOT / "tests" / "test_ml_strategy.py",
    ]

    missing = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if "require_real_market_data" not in path.read_text(encoding="utf-8")
    ]
    assert missing == []
