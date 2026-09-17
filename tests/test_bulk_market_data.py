from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wave_segments.bulk_market_data import (
    BulkMarketBuilder,
    adjust_prices,
    audit_bulk_market_manifest,
    load_bulk_market_dataset,
)


class FakePro:
    def __init__(self): self.calls = []; self.fail_daily_once = True
    def trade_cal(self, **kwargs):
        self.calls.append(("trade_cal", kwargs))
        return pd.DataFrame({"cal_date": ["20240101", "20240102", "20240103"], "is_open": ["0", "1", "1"]})
    def daily(self, trade_date):
        self.calls.append(("daily", trade_date))
        if trade_date == "20240102" and self.fail_daily_once:
            self.fail_daily_once = False; raise RuntimeError("transient")
        return pd.DataFrame({"ts_code": ["A.SZ", "B.SZ"], "trade_date": [trade_date, trade_date], "open": [10, 20], "high": [11, 21], "low": [9, 19], "close": [10, 20]})
    def adj_factor(self, trade_date):
        self.calls.append(("adj_factor", trade_date))
        return pd.DataFrame({"ts_code": ["A.SZ", "B.SZ"], "trade_date": [trade_date, trade_date], "adj_factor": [2., 3.]})
    def daily_basic(self, trade_date):
        self.calls.append(("daily_basic", trade_date)); return pd.DataFrame({"ts_code": ["A.SZ"], "trade_date": [trade_date], "total_mv": [1]})
    def stk_limit(self, trade_date):
        self.calls.append(("stk_limit", trade_date)); raise RuntimeError("unavailable")
    def suspend_d(self, **kwargs):
        self.calls.append(("suspend_d", kwargs)); return pd.DataFrame()


def test_bulk_builder_uses_trade_date_partitions_retries_filters_and_resumes(tmp_path: Path):
    pro = FakePro()
    universe = pd.DataFrame({"ts_code": ["A.SZ", "B.SZ"], "list_date": ["20200101", "20240103"], "delist_date": [None, None]})
    builder = BulkMarketBuilder(pro, tmp_path, "20240101", "20240103", universe=universe,
                                optional_endpoints=["daily_basic", "stk_limit"], retries=2, sleep_seconds=0.0)
    manifest = builder.build()
    assert set(manifest["dates"]) == {"20240102", "20240103"}
    assert manifest["dates"]["20240102"]["endpoints"]["daily"]["attempts"] == 2
    assert manifest["dates"]["20240102"]["endpoints"]["stk_limit"]["status"] == "failed"
    daily_basic_sidecar = tmp_path / manifest["dates"]["20240102"]["endpoints"]["daily_basic"]["raw_partition"]
    assert daily_basic_sidecar.exists()
    first = pd.read_parquet(tmp_path / "trade_date=20240102" / "data.parquet")
    assert first.ts_code.tolist() == ["A.SZ"]
    assert set(pd.read_parquet(tmp_path / "trade_date=20240103" / "data.parquet").ts_code) == {"A.SZ", "B.SZ"}
    old_daily_calls = len([x for x in pro.calls if x[0] == "daily"])
    old_limit_calls = len([x for x in pro.calls if x[0] == "stk_limit"])
    builder.build()
    assert len([x for x in pro.calls if x[0] == "daily"]) == old_daily_calls
    assert len([x for x in pro.calls if x[0] == "stk_limit"]) == old_limit_calls + 4
    optional_audit = audit_bulk_market_manifest(tmp_path)
    assert optional_audit["summary"]["endpoint_status_counts"]["stk_limit"]["failed"] == 2
    assert not audit_bulk_market_manifest(tmp_path, required_endpoints=["stk_limit"])["summary"]["passed"]
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["price_storage"] == "raw_plus_adj_factor"
    assert saved["expected_open_dates"] == ["20240102", "20240103"]
    qfq = load_bulk_market_dataset(tmp_path, adjustment="qfq", anchor_date="20240103")
    assert set(qfq.trade_date) == {"20240102", "20240103"}


def test_adjustment_requires_explicit_safe_qfq_anchor():
    bars = pd.DataFrame({"open": [10.], "close": [12.], "adj_factor": [2.]})
    with pytest.raises(ValueError, match="explicit point-in-time"):
        adjust_prices(bars, mode="qfq")
    assert adjust_prices(bars, mode="qfq", anchor_factor=4.).close.iloc[0] == 6.
    assert adjust_prices(bars, mode="hfq").close.iloc[0] == 24.
    multi = pd.DataFrame({"ts_code": ["A", "B"], "close": [10., 10.], "adj_factor": [2., 4.]})
    adjusted = adjust_prices(multi, mode="qfq", anchor_factor=pd.Series({"A": 4., "B": 8.}))
    assert adjusted.close.tolist() == [5., 5.]


def test_suspend_endpoint_is_restricted_to_suspension_records(tmp_path: Path):
    pro = FakePro()
    manifest = BulkMarketBuilder(pro, tmp_path, "20240102", "20240102", optional_endpoints=["suspend_d"],
                                 sleep_seconds=0).build()
    call = next(value for name, value in pro.calls if name == "suspend_d")
    assert call["suspend_type"] == "S"
    raw = tmp_path / manifest["dates"]["20240102"]["endpoints"]["suspend_d"]["raw_partition"]
    assert raw.exists()


def test_bulk_manifest_audit_detects_missing_date_and_bad_factor(tmp_path: Path):
    pro = FakePro()
    BulkMarketBuilder(pro, tmp_path, "20240102", "20240103", sleep_seconds=0).build()
    good = audit_bulk_market_manifest(tmp_path)
    assert good["summary"]["passed"]
    assert good["summary"]["completed_partitions"] == 2

    universe = pd.DataFrame({"ts_code": ["A.SZ", "B.SZ", "C.SZ"], "list_date": ["20200101"] * 3})
    covered = audit_bulk_market_manifest(tmp_path, universe=universe)
    assert covered["summary"]["universe_coverage"]["median"] == pytest.approx(2 / 3)

    second = tmp_path / "trade_date=20240103" / "data.parquet"
    frame = pd.read_parquet(second)
    frame.loc[0, "adj_factor"] = float("nan")
    frame.to_parquet(second, index=False)
    bad = audit_bulk_market_manifest(tmp_path, expected_trade_dates=["20240102", "20240103", "20240104"])
    codes = {(issue["trade_date"], issue["code"]) for issue in bad["issues"]}
    assert ("20240103", "low_adj_factor_match") in codes
    assert ("20240104", "missing_manifest_date") in codes
    assert not bad["summary"]["passed"]


def test_bulk_builder_can_exit_after_a_bounded_resumable_chunk(tmp_path: Path):
    pro = FakePro()
    first = BulkMarketBuilder(pro, tmp_path, "20240102", "20240103", sleep_seconds=0,
                              max_dates_per_run=1).build()
    assert first["last_run_processed_dates"] == 1
    assert not first["build_complete"]
    assert "completed_at" not in first
    second = BulkMarketBuilder(pro, tmp_path, "20240102", "20240103", sleep_seconds=0,
                               max_dates_per_run=1).build()
    assert second["last_run_processed_dates"] == 1
    assert second["build_complete"]
    assert "completed_at" in second


def test_resuming_optional_endpoint_does_not_duplicate_premerged_columns(tmp_path: Path):
    pro = FakePro()
    initial = BulkMarketBuilder(pro, tmp_path, "20240102", "20240102", sleep_seconds=0).build()
    partition = tmp_path / initial["dates"]["20240102"]["partition"]
    frame = pd.read_parquet(partition)
    # Simulate a legacy partition whose merged metadata survived, while the
    # manifest/sidecar evidence for that endpoint did not.
    frame["total_mv"] = [1, None]
    frame["total_mv_daily_basic"] = [1, None]
    frame["close_daily_basic"] = [10, None]
    frame.to_parquet(partition, index=False)

    resumed = BulkMarketBuilder(pro, tmp_path, "20240102", "20240102",
                                optional_endpoints=["daily_basic"], sleep_seconds=0).build()
    assert resumed["build_complete"]
    saved = pd.read_parquet(partition)
    assert saved.columns.is_unique
    assert saved["total_mv_daily_basic"].tolist()[0] == 1
    assert saved["close_daily_basic"].tolist()[0] == 10
