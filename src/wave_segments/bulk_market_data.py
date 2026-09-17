"""Resumable, point-in-time Tushare full-market daily data downloads.

The Tushare ``daily`` and ``adj_factor`` endpoints accept ``trade_date`` and
return all securities for that session.  This module intentionally uses those
bulk endpoints rather than one ``pro_bar`` request per security.

Partitions retain vendor raw prices and ``adj_factor``.  A qfq series normally
uses a *later* factor as its denominator, so materialising qfq with the final
date as anchor leaks the chosen end-of-sample into historical training data.
Use :func:`adjust_prices` with an explicitly available anchor, or preserve the
two raw columns and choose an anchor inside each walk-forward training window.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

REQUIRED_ENDPOINTS = ("daily", "adj_factor")
OPTIONAL_ENDPOINTS = ("daily_basic", "stk_limit", "suspend_d")


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.strftime("%Y%m%d")


def _frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError("Tushare endpoint must return a pandas DataFrame")
    return value.copy()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def audit_bulk_market_manifest(
    output_dir: str | Path,
    *,
    expected_trade_dates: Iterable[str] | None = None,
    universe: pd.DataFrame | None = None,
    required_endpoints: Iterable[str] | None = None,
    verify_partitions: bool = True,
    minimum_adj_factor_match: float = 0.999,
    minimum_universe_coverage: float | None = None,
) -> dict[str, Any]:
    """Audit a bulk build without mutating or resuming it.

    Completeness is judged against an explicit exchange calendar (or the
    calendar persisted by newer builders), never merely against the dates that
    happen to appear in the manifest. Universe coverage is reported separately
    because listed-but-suspended securities legitimately have no ``daily`` row.
    """
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted = manifest.get("expected_open_dates")
    calendar_source = "argument" if expected_trade_dates is not None else "manifest"
    expected_values = expected_trade_dates if expected_trade_dates is not None else persisted
    if expected_values is None:
        raise ValueError("expected_trade_dates is required for an old manifest without expected_open_dates")
    expected = sorted({_date(day) for day in expected_values})
    universe_norm = normalize_security_universe(universe) if universe is not None else None

    issues: list[dict[str, Any]] = []
    per_date: dict[str, dict[str, Any]] = {}
    total_rows = total_factor_rows = total_factor_matched = 0
    universe_coverages: list[float] = []
    endpoint_names = tuple(dict.fromkeys((*REQUIRED_ENDPOINTS, *tuple(manifest.get("optional_endpoints", ())),
                                           *tuple(required_endpoints or ()))))
    required_names = set(required_endpoints) if required_endpoints is not None else set(REQUIRED_ENDPOINTS)
    endpoint_status_counts: dict[str, dict[str, int]] = {name: {} for name in endpoint_names}
    for day in expected:
        entry = manifest.get("dates", {}).get(day)
        result: dict[str, Any] = {"status": "missing"}
        if entry is None:
            issues.append({"trade_date": day, "code": "missing_manifest_date"})
            per_date[day] = result
            continue
        result["status"] = entry.get("status", "unknown")
        if entry.get("status") != "complete":
            issues.append({"trade_date": day, "code": "partition_not_complete", "status": entry.get("status")})
        for endpoint in endpoint_names:
            endpoint_state = entry.get("endpoints", {}).get(endpoint, {})
            status = endpoint_state.get("status", "missing")
            endpoint_status_counts[endpoint][status] = endpoint_status_counts[endpoint].get(status, 0) + 1
            if endpoint in required_names and status != "ok":
                issues.append({"trade_date": day, "code": "required_endpoint_not_ok",
                               "endpoint": endpoint, "status": status})
            if endpoint_state.get("suspected_truncated") or status == "truncated":
                issues.append({"trade_date": day, "code": "suspected_truncation", "endpoint": endpoint})
            if endpoint in OPTIONAL_ENDPOINTS and status == "ok":
                raw_relative = endpoint_state.get("raw_partition")
                if not raw_relative or not (root / raw_relative).exists():
                    issues.append({"trade_date": day, "code": "optional_raw_partition_missing", "endpoint": endpoint})
        relative = entry.get("partition")
        partition = root / relative if relative else None
        result["partition_exists"] = bool(partition and partition.exists())
        if not result["partition_exists"]:
            issues.append({"trade_date": day, "code": "partition_file_missing"})
            per_date[day] = result
            continue
        if verify_partitions:
            try:
                columns = pd.read_parquet(partition, columns=["ts_code", "trade_date", "adj_factor"])
                rows = len(columns)
                factor_matched = int(pd.to_numeric(columns["adj_factor"], errors="coerce").notna().sum())
                duplicate_keys = int(columns.duplicated(["ts_code", "trade_date"]).sum())
                result.update({"rows": rows, "adj_factor_matched_rows": factor_matched,
                               "adj_factor_match_rate": factor_matched / rows if rows else 0.0,
                               "duplicate_keys": duplicate_keys})
                total_rows += rows
                total_factor_rows += rows
                total_factor_matched += factor_matched
                if rows != int(entry.get("rows", rows)):
                    issues.append({"trade_date": day, "code": "manifest_row_count_mismatch",
                                   "manifest_rows": entry.get("rows"), "actual_rows": rows})
                if duplicate_keys:
                    issues.append({"trade_date": day, "code": "duplicate_security_date_keys",
                                   "count": duplicate_keys})
                if result["adj_factor_match_rate"] < minimum_adj_factor_match:
                    issues.append({"trade_date": day, "code": "low_adj_factor_match",
                                   "rate": result["adj_factor_match_rate"]})
                if universe_norm is not None:
                    stamp = pd.Timestamp(day)
                    active = int((universe_norm.list_date.le(stamp) &
                                  (universe_norm.delist_date.isna() | universe_norm.delist_date.ge(stamp))).sum())
                    coverage = rows / active if active else None
                    result.update({"active_universe": active, "universe_coverage": coverage})
                    if coverage is not None:
                        universe_coverages.append(float(coverage))
                    if minimum_universe_coverage is not None and (coverage is None or coverage < minimum_universe_coverage):
                        issues.append({"trade_date": day, "code": "low_universe_coverage", "rate": coverage})
            except Exception as exc:  # noqa: BLE001 - record corrupt/unreadable partitions and continue the audit
                issues.append({"trade_date": day, "code": "partition_read_error",
                               "error": f"{type(exc).__name__}: {exc}"})
        per_date[day] = result

    manifest_extra_dates = sorted(set(manifest.get("dates", {})) - set(expected))
    completed = sum(per_date[day].get("status") == "complete" and per_date[day].get("partition_exists")
                    for day in expected)
    summary = {
        "passed": not issues,
        "calendar_source": calendar_source,
        "expected_dates": len(expected),
        "completed_partitions": int(completed),
        "missing_or_incomplete_partitions": len(expected) - int(completed),
        "issue_count": len(issues),
        "manifest_extra_dates": len(manifest_extra_dates),
        "rows": total_rows if verify_partitions else None,
        "adj_factor_match_rate": (total_factor_matched / total_factor_rows
                                  if verify_partitions and total_factor_rows else None),
        "endpoint_status_counts": endpoint_status_counts,
    }
    if universe_coverages:
        coverage_array = np.asarray(universe_coverages, dtype=float)
        summary["universe_coverage"] = {
            "min": float(np.min(coverage_array)),
            "p01": float(np.quantile(coverage_array, .01)),
            "p05": float(np.quantile(coverage_array, .05)),
            "median": float(np.median(coverage_array)),
            "p95": float(np.quantile(coverage_array, .95)),
            "max": float(np.max(coverage_array)),
            "dates_below_90pct": int(np.sum(coverage_array < .90)),
            "note": "Listed-universe denominator includes suspended securities; low daily coverage is not alone a failed download.",
        }
    return {"summary": summary, "issues": issues, "extra_dates": manifest_extra_dates, "dates": per_date}


def read_security_universe(path: str | Path) -> pd.DataFrame:
    """Read a point-in-time security master with ts_code/list_date/delist_date."""
    source = Path(path)
    raw = pd.read_parquet(source) if source.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(source)
    return normalize_security_universe(raw)


def normalize_security_universe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ts_code" not in out and "symbol" in out:
        out = out.rename(columns={"symbol": "ts_code"})
    if "ts_code" not in out or "list_date" not in out:
        raise ValueError("universe requires ts_code (or symbol) and list_date")
    out["ts_code"] = out["ts_code"].astype(str)
    out["list_date"] = pd.to_datetime(out["list_date"], errors="coerce").dt.normalize()
    if "delist_date" not in out:
        out["delist_date"] = pd.NaT
    else:
        out["delist_date"] = pd.to_datetime(out["delist_date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["ts_code", "list_date"]).drop_duplicates("ts_code", keep="last")


def filter_listed_on(frame: pd.DataFrame, trade_date: str, universe: pd.DataFrame | None) -> pd.DataFrame:
    """Keep securities listed on this exact session (delist date remains active)."""
    if universe is None:
        return frame
    if "ts_code" not in frame:
        raise ValueError("daily response lacks ts_code")
    day = pd.Timestamp(trade_date)
    active = universe.loc[universe.list_date.le(day) & (universe.delist_date.isna() | universe.delist_date.ge(day)), ["ts_code"]]
    return frame.merge(active, on="ts_code", how="inner", validate="many_to_one")


def adjust_prices(frame: pd.DataFrame, *, mode: str, anchor_factor: float | pd.Series | None = None) -> pd.DataFrame:
    """Return price columns adjusted from raw Tushare prices and adj_factor.

    ``hfq`` is raw * factor (up to a positive scale).  ``qfq`` is raw * factor /
    anchor_factor. For qfq callers must supply an anchor factor known at the
    decision time; omitting it is refused to prevent accidental final-sample
    anchoring leakage.
    """
    if mode not in {"qfq", "hfq"}:
        raise ValueError("mode must be qfq or hfq")
    if "adj_factor" not in frame:
        raise ValueError("adj_factor is required")
    prices = [c for c in ("open", "high", "low", "close", "pre_close") if c in frame]
    factor = pd.to_numeric(frame["adj_factor"], errors="coerce")
    if mode == "qfq":
        if anchor_factor is None:
            raise ValueError("qfq requires an explicit point-in-time anchor_factor; retain raw prices + adj_factor otherwise")
        anchor = pd.to_numeric(anchor_factor, errors="coerce")
        if isinstance(anchor, pd.Series) and "ts_code" in frame and not anchor.index.equals(frame.index):
            anchor = frame["ts_code"].map(anchor)
        factor = factor / anchor
    out = frame.copy()
    out[prices] = out[prices].apply(pd.to_numeric, errors="coerce").mul(factor, axis=0)
    return out


@dataclass
class BulkMarketBuilder:
    """Fetch full-market daily partitions with retryable endpoint-level coverage."""
    pro: Any
    output_dir: str | Path
    start_date: str
    end_date: str
    universe: pd.DataFrame | None = None
    optional_endpoints: Iterable[str] = field(default_factory=tuple)
    retries: int = 3
    timeout_seconds: float = 60.0
    sleep_seconds: float = 0.5
    endpoint_row_limit: int = 6000
    max_dates_per_run: int | None = None
    progress: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.start_date, self.end_date = _date(self.start_date), _date(self.end_date)
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        self.optional_endpoints = tuple(self.optional_endpoints)
        unknown = set(self.optional_endpoints) - set(OPTIONAL_ENDPOINTS)
        if unknown:
            raise ValueError(f"unsupported optional endpoints: {sorted(unknown)}")
        if self.retries < 1 or self.timeout_seconds <= 0:
            raise ValueError("retries must be >= 1 and timeout_seconds must be positive")
        if self.max_dates_per_run is not None and self.max_dates_per_run < 1:
            raise ValueError("max_dates_per_run must be positive when supplied")
        if self.universe is not None:
            self.universe = normalize_security_universe(self.universe)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"format_version": 1, "provider": "tushare", "price_storage": "raw_plus_adj_factor", "dates": {}}

    def _call(self, endpoint: str, trade_date: str) -> tuple[pd.DataFrame, int]:
        method: Callable[..., Any] = getattr(self.pro, endpoint)
        error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            kwargs = {"trade_date": trade_date}
            if endpoint == "suspend_d":
                kwargs["suspend_type"] = "S"
            try:
                # Configure timeout on the provider HTTP client. A Python worker
                # thread cannot kill an already-running request and leaks after
                # ``future.cancel()``.
                result = _frame(method(**kwargs))
                return result, attempt
            except Exception as exc:  # noqa: BLE001 - vendor client exceptions are retried and recorded
                error = exc
            if attempt < self.retries and self.sleep_seconds:
                time.sleep(self.sleep_seconds * attempt)
        assert error is not None
        raise error

    def trade_dates(self) -> list[str]:
        calendar, _ = self._call_calendar()
        if not {"cal_date", "is_open"}.issubset(calendar.columns):
            raise ValueError("trade_cal response requires cal_date and is_open")
        cal_date = pd.to_datetime(calendar["cal_date"], errors="coerce")
        is_open = calendar["is_open"].astype(str).eq("1")
        return sorted(cal_date[is_open & cal_date.notna()].dt.strftime("%Y%m%d").tolist())

    def _call_calendar(self) -> tuple[pd.DataFrame, int]:
        method = self.pro.trade_cal
        error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                result = _frame(method(start_date=self.start_date, end_date=self.end_date, is_open="1"))
                return result, attempt
            except Exception as exc:  # noqa: BLE001 - vendor calendar exceptions are retried and recorded
                error = exc
            if attempt < self.retries and self.sleep_seconds:
                time.sleep(self.sleep_seconds * attempt)
        assert error is not None
        raise error

    def build(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        trade_dates = self.trade_dates()
        manifest.update({"start_date": self.start_date, "end_date": self.end_date,
                         "optional_endpoints": list(self.optional_endpoints),
                         "expected_open_dates": trade_dates,
                         "expected_open_date_count": len(trade_dates),
                         "qfq_note": "Not materialized: qfq needs a later anchor; raw prices and adj_factor are retained."})
        _atomic_json(self.manifest_path, manifest)
        processed_dates = 0
        for day in trade_dates:
            partition = self.output_dir / f"trade_date={day}" / "data.parquet"
            prior = manifest["dates"].get(day, {})
            requested = (*REQUIRED_ENDPOINTS, *self.optional_endpoints)
            prior_endpoint_states = prior.get("endpoints", {})

            def endpoint_cached(name: str, endpoint_states=prior_endpoint_states) -> bool:
                state = endpoint_states.get(name, {})
                if state.get("status") != "ok":
                    return False
                if name in OPTIONAL_ENDPOINTS:
                    raw_relative = state.get("raw_partition")
                    return bool(raw_relative and (self.output_dir / raw_relative).exists())
                return True

            prior_has_all = all(endpoint_cached(name) for name in requested)
            if prior.get("status") == "complete" and prior_has_all and partition.exists():
                if self.progress: self.progress({"event": "date_cached", "trade_date": day})
                continue
            if self.max_dates_per_run is not None and processed_dates >= self.max_dates_per_run:
                break
            prior_endpoints = prior_endpoint_states
            core_cached = (prior.get("status") == "complete" and partition.exists() and
                           all(prior_endpoints.get(name, {}).get("status") == "ok" for name in REQUIRED_ENDPOINTS))
            pending_optional = [name for name in self.optional_endpoints if not endpoint_cached(name)]
            sidecar_only = {
                name for name in pending_optional
                if prior_endpoints.get(name, {}).get("status") == "ok"
                and not prior_endpoints.get(name, {}).get("raw_partition")
            }
            # A failed/missing optional endpoint has contributed no trusted data,
            # so it can be retried against the cached core partition. Truncated
            # responses may already have partial columns and are rebuilt fully.
            incremental_optional = (core_cached and pending_optional and
                                    all(prior_endpoints.get(name, {}).get("status", "missing") in {"ok", "failed", "missing"}
                                        for name in pending_optional))
            endpoints_to_fetch = tuple(pending_optional) if incremental_optional else requested
            if self.progress: self.progress({"event": "date_started", "trade_date": day})
            entry: dict[str, Any] = {"status": "failed", "partition": str(partition.relative_to(self.output_dir)),
                                     "endpoints": deepcopy(prior_endpoints) if incremental_optional else {},
                                     "updated_at": datetime.now(timezone.utc).isoformat()}
            frames: dict[str, pd.DataFrame] = {}
            failed = False
            for endpoint in endpoints_to_fetch:
                if self.progress: self.progress({"event": "endpoint_started", "trade_date": day, "endpoint": endpoint})
                try:
                    data, attempts = self._call(endpoint, day)
                    frames[endpoint] = data
                    truncated = len(data) >= self.endpoint_row_limit
                    entry["endpoints"][endpoint] = {"status": "ok", "rows": len(data), "attempts": attempts,
                                                       "suspected_truncated": truncated}
                    if truncated:
                        entry["endpoints"][endpoint]["status"] = "truncated"
                        entry["endpoints"][endpoint]["error"] = f"row limit {self.endpoint_row_limit} reached"
                    if endpoint in REQUIRED_ENDPOINTS and (truncated or data.empty):
                        if data.empty:
                            entry["endpoints"][endpoint]["status"] = "failed"
                            entry["endpoints"][endpoint]["error"] = "required endpoint returned no rows"
                        failed = True
                    if endpoint in OPTIONAL_ENDPOINTS and entry["endpoints"][endpoint]["status"] == "ok":
                        raw_partition = self.output_dir / f"trade_date={day}" / f"{endpoint}.parquet"
                        _atomic_parquet(raw_partition, data)
                        entry["endpoints"][endpoint]["raw_partition"] = str(raw_partition.relative_to(self.output_dir))
                    if self.progress: self.progress({"event": "endpoint_finished", "trade_date": day, "endpoint": endpoint,
                                                     "status": entry["endpoints"][endpoint]["status"], "rows": len(data)})
                except Exception as exc:  # noqa: BLE001 - endpoint failures are recorded individually for auditability
                    entry["endpoints"][endpoint] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "attempts": self.retries}
                    if endpoint in REQUIRED_ENDPOINTS:
                        failed = True
                    if self.progress: self.progress({"event": "endpoint_finished", "trade_date": day, "endpoint": endpoint,
                                                     "status": "failed", "error": str(exc)})
            if not failed:
                if incremental_optional:
                    result = pd.read_parquet(partition)
                else:
                    daily = filter_listed_on(frames["daily"], day, self.universe)
                    if "ts_code" not in frames["adj_factor"] or "trade_date" not in frames["adj_factor"]:
                        failed = True
                        entry["endpoints"]["adj_factor"] = {"status": "failed", "error": "adj_factor lacks ts_code/trade_date"}
                        manifest["dates"][day] = entry
                        _atomic_json(self.manifest_path, manifest)
                        continue
                    result = daily.merge(frames["adj_factor"], on=["ts_code", "trade_date"], how="left", validate="one_to_one")
                    entry["endpoints"]["adj_factor"]["matched_rows"] = int(result["adj_factor"].notna().sum()) if "adj_factor" in result else 0
                    # Optional endpoint failures do not invalidate core daily/factor data.
                for endpoint in self.optional_endpoints:
                    if incremental_optional and endpoint not in frames:
                        continue
                    if endpoint in sidecar_only:
                        # Trusted merged columns already exist; repair only the
                        # missing raw endpoint evidence.
                        continue
                    if endpoint not in frames and endpoint not in entry["endpoints"]:
                        continue
                    result = result.drop(columns=[f"{endpoint}_query_ok"], errors="ignore")
                    if endpoint == "suspend_d":
                        result = result.drop(columns=["is_suspended"], errors="ignore")
                    if endpoint in frames:
                        meta = frames.get(endpoint)
                        query_ok = entry["endpoints"].get(endpoint, {}).get("status") == "ok"
                        result[f"{endpoint}_query_ok"] = query_ok
                        if meta is not None and {"ts_code", "trade_date"}.issubset(meta.columns):
                            # A resumed metadata endpoint may be absent from the
                            # manifest while its values were already merged into
                            # an older core partition. Avoid recreating columns
                            # that are already present under pandas' suffix name
                            # (e.g. close_daily_basic), which would otherwise
                            # produce duplicate Parquet field names.
                            existing = set(result.columns)
                            keys = {"ts_code", "trade_date"}
                            duplicate_suffixes = {
                                column for column in meta.columns
                                if column not in keys and f"{column}_{endpoint}" in existing
                            }
                            meta = meta.drop(columns=duplicate_suffixes, errors="ignore")
                            # Preserve the established convention of retaining
                            # overlapping source fields as *_<endpoint>.
                            suffix = f"_{endpoint}"
                            result = result.merge(meta, on=["ts_code", "trade_date"], how="left", suffixes=("", suffix))
                        if endpoint == "suspend_d":
                            suspension_keys = (set(zip(meta.ts_code.astype(str), meta.trade_date.astype(str)))
                                               if meta is not None and {"ts_code", "trade_date"}.issubset(meta.columns) else set())
                            result["is_suspended"] = [
                                ((str(code), str(date)) in suspension_keys) if query_ok else pd.NA
                                for code, date in zip(result.ts_code, result.trade_date)
                            ]
                _atomic_parquet(partition, result)
                entry["status"] = "complete"; entry["rows"] = len(result)
            manifest["dates"][day] = entry
            _atomic_json(self.manifest_path, manifest)
            processed_dates += 1
            if self.progress: self.progress({"event": "date_finished", "trade_date": day,
                                             "status": entry["status"], "rows": entry.get("rows", 0)})
        requested = (*REQUIRED_ENDPOINTS, *self.optional_endpoints)
        build_complete = all(
            manifest.get("dates", {}).get(day, {}).get("status") == "complete"
            and all(manifest["dates"][day].get("endpoints", {}).get(name, {}).get("status") == "ok"
                    for name in requested)
            and (self.output_dir / manifest["dates"][day].get("partition", "__missing__")).exists()
            for day in trade_dates
        )
        manifest["build_complete"] = build_complete
        manifest["last_run_processed_dates"] = processed_dates
        manifest["last_run_finished_at"] = datetime.now(timezone.utc).isoformat()
        if build_complete:
            manifest["completed_at"] = manifest["last_run_finished_at"]
        else:
            manifest.pop("completed_at", None)
        _atomic_json(self.manifest_path, manifest)
        return manifest


def load_bulk_market_dataset(
    output_dir: str | Path, *, adjustment: str | None = None,
    anchor_date: str | None = None,
) -> pd.DataFrame:
    """Load completed partitions and optionally materialize qfq/hfq prices.

    qfq requires ``anchor_date`` and loads no partition after that date. The
    latest factor available for each symbol on/before the anchor is used, which
    makes the chosen point-in-time normalization explicit and auditable.
    """
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    anchor = _date(anchor_date) if anchor_date is not None else None
    paths = []
    for day, entry in manifest.get("dates", {}).items():
        if entry.get("status") == "complete" and (anchor is None or day <= anchor):
            path = root / entry["partition"]
            if path.exists():
                paths.append(path)
    if not paths:
        raise RuntimeError("No completed bulk market partitions found")
    data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if adjustment == "qfq":
        if anchor is None:
            raise ValueError("qfq loading requires anchor_date")
        anchors = (data.sort_values("trade_date").dropna(subset=["adj_factor"])
                   .drop_duplicates("ts_code", keep="last").set_index("ts_code")["adj_factor"])
        data = adjust_prices(data, mode="qfq", anchor_factor=anchors)
    elif adjustment == "hfq":
        data = adjust_prices(data, mode="hfq")
    elif adjustment is not None:
        raise ValueError("adjustment must be qfq, hfq or None")
    return data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
