"""Point-in-time market metadata enrichment and data-quality audits.

This module deliberately does not infer unavailable historical fields.  A missing
source is represented as unknown in the audit rather than as a tradable/security
state.  All dated sources use an exact session match; Shenwan membership uses its
published in/out interval only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .schema import normalize_ohlcv

_DATE_ALIASES = {"ts_code": "symbol", "trade_date": "timestamp", "ann_date": "timestamp"}


def normalize_market_metadata(frame: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """Normalize Tushare symbol/date aliases without forward filling metadata."""
    if frame is None:
        return pd.DataFrame()
    out = frame.rename(columns={k: v for k, v in _DATE_ALIASES.items() if k in frame.columns}).copy()
    if out.empty:
        return out
    if "symbol" not in out.columns:
        raise ValueError(f"{kind} requires symbol or ts_code")
    out["symbol"] = out["symbol"].astype(str)
    if kind != "sw_membership":
        if "timestamp" not in out.columns:
            raise ValueError(f"{kind} requires timestamp/trade_date")
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.normalize()
        return out.dropna(subset=["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    for col in ("in_date", "out_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize() if col in out else pd.NaT
    if "in_date" not in out or out["in_date"].isna().all():
        raise ValueError("sw_membership requires a non-empty in_date")
    return out.dropna(subset=["symbol", "in_date"])


def _interval_membership(bars: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Match a bar only to membership rows active that day, never later rows."""
    if members.empty:
        return pd.DataFrame(index=bars.index)
    joined = bars[["symbol", "timestamp"]].reset_index(names="_bar_index").merge(members, on="symbol", how="left")
    active = joined[joined.in_date.le(joined.timestamp) & (joined.out_date.isna() | joined.out_date.ge(joined.timestamp))].copy()
    if active.empty:
        return pd.DataFrame(index=bars.index)
    # Revised classifications can overlap; most recently effective record wins.
    active = active.sort_values(["_bar_index", "in_date"]).drop_duplicates("_bar_index", keep="last").set_index("_bar_index")
    code = next((c for c in ("l3_code", "l2_code", "l1_code", "index_code") if c in active), None)
    name = next((c for c in ("l3_name", "l2_name", "l1_name", "index_name") if c in active), None)
    out = pd.DataFrame(index=bars.index)
    out["sw_membership_available"] = False
    out.loc[active.index, "sw_membership_available"] = True
    out["sw_industry_code"] = active[code].reindex(out.index) if code else pd.NA
    out["sw_industry_name"] = active[name].reindex(out.index) if name else pd.NA
    return out


def _coverage(frame: pd.DataFrame, col: str) -> dict[str, float | int]:
    count = int(frame[col].fillna(False).sum())
    total = len(frame)
    return {"matched_rows": count, "total_rows": total, "coverage": float(count / total) if total else 0.0}


def align_financial_announcements(
    bars: pd.DataFrame,
    financials: pd.DataFrame,
    *,
    value_columns: list[str] | None = None,
    availability_lag_days: int = 0,
) -> pd.DataFrame:
    """Align statement facts by their conservative public-availability date.

    Tushare statement tables commonly expose ``end_date``, ``ann_date`` and
    ``f_ann_date``. When both announcement dates exist, the later date is used;
    this is conservative when vendor semantics differ and prevents a scheduled
    date from revealing a later actual filing. Revisions replace the same report
    period only after their own effective date. At each bar, the latest report
    period actually available then is selected.
    """
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days cannot be negative")
    base = bars.copy()
    if not {"symbol", "timestamp"}.issubset(base.columns):
        raise ValueError("bars require normalized symbol and timestamp")
    events = financials.rename(columns={"ts_code": "symbol"} if "ts_code" in financials else {}).copy()
    if "symbol" not in events or "end_date" not in events:
        raise ValueError("financials require symbol/ts_code and end_date")
    date_cols = [c for c in ("ann_date", "f_ann_date") if c in events]
    if not date_cols:
        raise ValueError("financials require ann_date and/or f_ann_date")
    events["symbol"] = events["symbol"].astype(str)
    events["financial_period_end"] = pd.to_datetime(events["end_date"], errors="coerce").dt.normalize()
    parsed_dates = pd.concat(
        [pd.to_datetime(events[c], errors="coerce").dt.normalize().rename(c) for c in date_cols], axis=1
    )
    # max(axis=1) intentionally waits for the later of scheduled/actual fields.
    events["financial_effective_at"] = parsed_dates.max(axis=1) + pd.to_timedelta(availability_lag_days, unit="D")
    events["_source_order"] = np.arange(len(events))
    events = (events.dropna(subset=["symbol", "financial_period_end", "financial_effective_at"])
              .sort_values(["symbol", "financial_effective_at", "_source_order"])
              .drop_duplicates(["symbol", "financial_period_end", "financial_effective_at"], keep="last"))
    reserved = {"symbol", "end_date", "ann_date", "f_ann_date", "financial_period_end",
                "financial_effective_at", "_source_order"}
    values = value_columns or [c for c in events.columns if c not in reserved]
    missing_values = set(values) - set(events.columns)
    if missing_values:
        raise ValueError(f"financial value columns missing: {sorted(missing_values)}")

    selected = pd.Series(pd.NA, index=base.index, dtype="object")
    timestamps = pd.to_datetime(base["timestamp"], errors="coerce").dt.normalize()
    for symbol, bar_index in base.groupby("symbol", sort=False).groups.items():
        ordered_bars = sorted(bar_index, key=lambda i: timestamps.loc[i])
        symbol_events = events.loc[events["symbol"].eq(str(symbol))].sort_values(
            ["financial_effective_at", "financial_period_end", "_source_order"]
        )
        event_rows = list(symbol_events.index)
        pointer = 0
        available_by_period: dict[pd.Timestamp, Any] = {}
        for bar_i in ordered_bars:
            now = timestamps.loc[bar_i]
            while pointer < len(event_rows) and events.loc[event_rows[pointer], "financial_effective_at"] <= now:
                event_i = event_rows[pointer]
                period = events.loc[event_i, "financial_period_end"]
                if period <= now:
                    available_by_period[period] = event_i
                pointer += 1
            if available_by_period:
                selected.loc[bar_i] = available_by_period[max(available_by_period)]

    aligned = pd.DataFrame(index=base.index)
    aligned["financial_available"] = selected.notna()
    aligned["financial_period_end"] = pd.NaT
    aligned["financial_effective_at"] = pd.NaT
    for value in values:
        aligned[f"financial_{value}"] = pd.NA
    matched = selected.dropna()
    if len(matched):
        source_index = matched.astype(events.index.dtype, copy=False).to_numpy()
        target_index = matched.index
        aligned.loc[target_index, "financial_period_end"] = events.loc[source_index, "financial_period_end"].to_numpy()
        aligned.loc[target_index, "financial_effective_at"] = events.loc[source_index, "financial_effective_at"].to_numpy()
        for value in values:
            aligned.loc[target_index, f"financial_{value}"] = events.loc[source_index, value].to_numpy()
    return aligned


@dataclass(frozen=True)
class MarketDataQualityResult:
    enriched: pd.DataFrame
    report: dict[str, Any]


def enrich_and_audit_market_data(
    bars: pd.DataFrame,
    *,
    daily_basic: pd.DataFrame | None = None,
    stk_limit: pd.DataFrame | None = None,
    suspend_d: pd.DataFrame | None = None,
    suspend_query_complete: bool = False,
    sw_membership: pd.DataFrame | None = None,
    financials: pd.DataFrame | None = None,
    financial_value_columns: list[str] | None = None,
    financial_availability_lag_days: int = 0,
    expected_universe: pd.DataFrame | None = None,
) -> MarketDataQualityResult:
    """Exact-date enrich OHLCV and return an explicit point-in-time quality report.

    ``expected_universe`` is a timestamp/symbol panel. It must represent the
    historical eligible universe, not today's constituents. Completeness is not
    claimed when it is omitted.
    """
    out = normalize_ohlcv(bars).copy()
    out["timestamp"] = out["timestamp"].dt.normalize()
    report: dict[str, Any] = {"rows": len(out), "sources": {}, "missingness": {}}

    for source, raw in (("daily_basic", daily_basic), ("stk_limit", stk_limit)):
        if raw is None:
            out[f"{source}_available"] = False
            report["sources"][source] = {"provided": False, "coverage": 0.0}
            continue
        meta = normalize_market_metadata(raw, kind=source)
        if meta.empty:
            out[f"{source}_available"] = False
            report["sources"][source] = {"provided": True, "returned_rows": 0, "matched_rows": 0,
                                         "total_rows": len(out), "coverage": 0.0}
            continue
        if source == "daily_basic":
            keep = [c for c in meta.columns if c not in {"symbol", "timestamp"}]
            right = meta[["symbol", "timestamp", *keep]].rename(columns={c: f"daily_basic_{c}" for c in keep})
        else:
            rename = {"up_limit": "limit_up", "down_limit": "limit_down"}
            right = meta.rename(columns=rename)
            keep = [c for c in ("symbol", "timestamp", "limit_up", "limit_down") if c in right]
            right = right[keep]
        out = out.merge(right, on=["symbol", "timestamp"], how="left", validate="one_to_one")
        available = "daily_basic_available" if source == "daily_basic" else "stk_limit_available"
        out[available] = out[[c for c in right if c not in {"symbol", "timestamp"}]].notna().any(axis=1)
        report["sources"][source] = {"provided": True, **_coverage(out, available)}

    if "limit_up" in out:
        out["at_limit_up"] = out.limit_up.notna() & np.isclose(out.close, out.limit_up, rtol=1e-5, atol=1e-5)
        out["at_limit_down"] = out.limit_down.notna() & np.isclose(out.close, out.limit_down, rtol=1e-5, atol=1e-5)
    else:
        out["at_limit_up"] = False
        out["at_limit_down"] = False

    if suspend_d is None:
        out["suspension_data_available"] = False
        out["is_suspended"] = pd.NA
        report["sources"]["suspend_d"] = {"provided": False, "coverage": 0.0}
    else:
        suspended = normalize_market_metadata(suspend_d, kind="suspend_d")
        if suspended.empty:
            out["is_suspended"] = pd.Series(False if suspend_query_complete else pd.NA,
                                              index=out.index, dtype="boolean")
        else:
            flags = suspended[["symbol", "timestamp"]].drop_duplicates().assign(is_suspended=True)
            out = out.merge(flags, on=["symbol", "timestamp"], how="left", validate="one_to_one")
        matched = out["is_suspended"].eq(True)
        out["suspension_data_available"] = matched | bool(suspend_query_complete)
        out["is_suspended"] = matched.astype("boolean")
        if not suspend_query_complete:
            out.loc[~matched, "is_suspended"] = pd.NA
        report["sources"]["suspend_d"] = {
            "provided": True, "suspension_rows": int(matched.sum()),
            "negative_inference_allowed": bool(suspend_query_complete),
            "coverage": 1.0 if suspend_query_complete else float(matched.mean()),
        }

    if sw_membership is None:
        out["sw_membership_available"] = False
        out["sw_industry_code"] = pd.NA
        out["sw_industry_name"] = pd.NA
        report["sources"]["sw_membership"] = {"provided": False, "coverage": 0.0}
    else:
        members = normalize_market_metadata(sw_membership, kind="sw_membership")
        if members.empty:
            out["sw_membership_available"] = False
            out["sw_industry_code"] = pd.NA
            out["sw_industry_name"] = pd.NA
        else:
            membership = _interval_membership(out, members)
            for col in membership:
                out[col] = membership[col].to_numpy()
        report["sources"]["sw_membership"] = {"provided": True, **_coverage(out, "sw_membership_available")}

    if financials is None:
        out["financial_available"] = False
        report["sources"]["financials"] = {"provided": False, "coverage": 0.0}
    else:
        aligned_financials = align_financial_announcements(
            out, financials, value_columns=financial_value_columns,
            availability_lag_days=financial_availability_lag_days,
        )
        for col in aligned_financials:
            out[col] = aligned_financials[col].to_numpy()
        report["sources"]["financials"] = {
            "provided": True,
            "effective_date_policy": "max(ann_date, f_ann_date) + configured lag",
            "availability_lag_days": financial_availability_lag_days,
            **_coverage(out, "financial_available"),
        }

    report["missingness"] = {col: float(out[col].isna().mean()) for col in out.columns if out[col].isna().any()}
    suspension_known = out["suspension_data_available"].fillna(False)
    limits_known = out["stk_limit_available"].fillna(False)
    one_price_locked = np.isclose(out["high"], out["low"], rtol=1e-8, atol=1e-8) & (
        out["at_limit_up"] | out["at_limit_down"]
    )
    out["tradability_known"] = suspension_known & limits_known
    suspended_or_unknown = out["is_suspended"].astype("boolean").fillna(True)
    out["selection_eligible"] = (
        out["tradability_known"] & ~suspended_or_unknown
        & pd.to_numeric(out["volume"], errors="coerce").gt(0) & ~one_price_locked
    )
    report["tradability"] = {
        "known_coverage": float(out["tradability_known"].mean()) if len(out) else 0.0,
        "eligible_rows": int(out["selection_eligible"].sum()),
        "one_price_locked_rows": int(one_price_locked.sum()),
        "warning": "Entry-day screen only; exit execution and market impact require separate validation.",
    }
    if expected_universe is None:
        report["universe_completeness"] = {"available": False, "reason": "expected_universe not supplied"}
    else:
        expected = normalize_market_metadata(expected_universe, kind="expected_universe")[["symbol", "timestamp"]].drop_duplicates()
        observed = out[["symbol", "timestamp"]].drop_duplicates()
        missing = expected.merge(observed, on=["symbol", "timestamp"], how="left", indicator=True)
        missing = missing.loc[missing._merge.eq("left_only"), ["symbol", "timestamp"]]
        report["universe_completeness"] = {
            "available": True, "expected_rows": len(expected), "observed_rows": len(observed),
            "missing_rows": len(missing), "coverage": float(1 - len(missing) / len(expected)) if len(expected) else 0.0,
            "missing_by_timestamp": {str(k.date()): int(v) for k, v in missing.groupby("timestamp").size().items()},
        }
    return MarketDataQualityResult(enriched=out, report=report)
