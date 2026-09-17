from __future__ import annotations

import pandas as pd

from wave_segments.market_data import (
    align_financial_announcements,
    enrich_and_audit_market_data,
    normalize_market_metadata,
)


def _bars() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ"],
        "trade_date": ["20240102", "20240103", "20240102", "20240103"],
        "open": [10, 11, 20, 20], "high": [11, 12, 21, 21],
        "low": [9, 10, 19, 19], "close": [10, 12, 20, 19], "vol": [100, 110, 200, 210],
    })


def test_metadata_normalization_and_exact_asof_enrichment():
    daily = pd.DataFrame({"ts_code": ["A.SZ", "A.SZ"], "trade_date": ["20240102", "20240103"], "total_mv": [100, 120]})
    limits = pd.DataFrame({"ts_code": ["A.SZ", "A.SZ", "B.SZ"], "trade_date": ["20240102", "20240103", "20240103"], "up_limit": [11, 12, 22], "down_limit": [9, 10, 18]})
    suspended = pd.DataFrame({"ts_code": ["B.SZ"], "trade_date": ["20240103"], "suspend_type": ["S"]})
    membership = pd.DataFrame({
        "ts_code": ["A.SZ", "A.SZ", "B.SZ"], "in_date": ["20230101", "20240103", "20230101"],
        "out_date": ["20240102", None, None], "l3_code": ["OLD", "NEW", "FIN"],
        "l3_name": ["old", "new", "fin"],
    })
    universe = pd.DataFrame({"ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ", "C.SZ"], "trade_date": ["20240102", "20240103", "20240102", "20240103", "20240103"]})
    result = enrich_and_audit_market_data(_bars(), daily_basic=daily, stk_limit=limits, suspend_d=suspended, suspend_query_complete=True, sw_membership=membership, expected_universe=universe)
    out, report = result.enriched.sort_values(["symbol", "timestamp"]).reset_index(drop=True), result.report
    assert out.daily_basic_available.tolist() == [True, True, False, False]
    assert out.stk_limit_available.tolist() == [True, True, False, True]
    assert out.at_limit_up.tolist() == [False, True, False, False]
    assert out.is_suspended.tolist() == [False, False, False, True]
    assert out.selection_eligible.tolist() == [True, True, False, False]
    assert out.sw_industry_code.tolist() == ["OLD", "NEW", "FIN", "FIN"]
    assert report["sources"]["daily_basic"]["coverage"] == 0.5
    assert report["universe_completeness"]["missing_rows"] == 1
    assert report["universe_completeness"]["missing_by_timestamp"] == {"2024-01-03": 1}


def test_absent_sources_remain_unknown_and_never_forward_fill():
    result = enrich_and_audit_market_data(_bars())
    out = result.enriched
    assert not out.daily_basic_available.any()
    assert not out.stk_limit_available.any()
    assert out.is_suspended.isna().all()
    assert not out.selection_eligible.any()
    assert not out.sw_membership_available.any()
    assert result.report["universe_completeness"]["available"] is False


def test_metadata_rejects_missing_required_dates():
    try:
        normalize_market_metadata(pd.DataFrame({"ts_code": ["A.SZ"]}), kind="daily_basic")
    except ValueError as exc:
        assert "timestamp" in str(exc)
    else:
        raise AssertionError("metadata without a date must fail")


def test_sparse_suspension_source_needs_completeness_assertion_for_negative_rows():
    sparse = pd.DataFrame({"ts_code": ["A.SZ"], "trade_date": ["20240102"], "suspend_type": ["S"]})
    result = enrich_and_audit_market_data(_bars(), suspend_d=sparse)
    matched = result.enriched.symbol.eq("A.SZ") & result.enriched.timestamp.eq(pd.Timestamp("2024-01-02"))
    assert result.enriched.loc[matched, "is_suspended"].eq(True).all()
    assert result.enriched.loc[~matched, "is_suspended"].isna().all()
    assert not result.report["sources"]["suspend_d"]["negative_inference_allowed"]


def test_complete_empty_suspension_query_means_no_suspensions():
    result = enrich_and_audit_market_data(_bars(), suspend_d=pd.DataFrame(), suspend_query_complete=True)
    assert result.enriched["is_suspended"].eq(False).all()
    assert result.enriched["suspension_data_available"].all()


def test_financial_facts_are_visible_only_after_actual_announcement_and_revision():
    bars = pd.DataFrame({
        "symbol": ["A.SZ"] * 5,
        "timestamp": pd.to_datetime(["2024-01-10", "2024-02-20", "2024-03-10", "2024-04-15", "2024-04-30"]),
    })
    financials = pd.DataFrame({
        "ts_code": ["A.SZ", "A.SZ", "A.SZ"],
        "end_date": ["20230930", "20230930", "20231231"],
        "ann_date": ["20240112", "20240225", "20240401"],
        "f_ann_date": ["20240115", "20240301", "20240420"],
        "roe": [10.0, 9.5, 12.0],
    })
    aligned = align_financial_announcements(bars, financials, value_columns=["roe"])
    assert aligned.financial_available.tolist() == [False, True, True, True, True]
    assert pd.isna(aligned.loc[0, "financial_roe"])
    assert aligned.loc[1, "financial_roe"] == 10.0
    assert aligned.loc[2, "financial_roe"] == 9.5
    # The newer period is not exposed on scheduled ann_date; f_ann_date governs.
    assert aligned.loc[3, "financial_roe"] == 9.5
    assert aligned.loc[4, "financial_roe"] == 12.0


def test_market_quality_reports_financial_announcement_coverage():
    financials = pd.DataFrame({"ts_code": ["A.SZ"], "end_date": ["20230930"],
                               "ann_date": ["20240102"], "f_ann_date": ["20240102"], "roe": [8.0]})
    result = enrich_and_audit_market_data(_bars(), financials=financials, financial_value_columns=["roe"])
    assert result.report["sources"]["financials"]["effective_date_policy"].startswith("max(")
    assert result.enriched.loc[result.enriched.symbol.eq("A.SZ"), "financial_available"].all()
    assert not result.enriched.loc[result.enriched.symbol.eq("B.SZ"), "financial_available"].any()
