"""Resumable, bounded-memory construction of full-market segment candidates.

This is deliberately a *first-layer data preparation* job.  It runs the
existing variable-length segmenter and feature extractor, but does not fit a
clusterer on a bucket or assign business semantics.  Every resulting candidate
is ``UNKNOWN``/``unreviewed_candidate`` until the independent discovery or
human-label workflow handles it.

The source dataset is partitioned by trading date, whereas segmentation needs
one complete history per security.  The builder therefore first transposes the
date partitions into a small, stable-hash set of symbol buckets.  It then reads
and processes one bucket at a time.  No all-market OHLCV table, feature table,
or fitted model is ever held in memory.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from hashlib import blake2b
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .bulk_market_data import adjust_prices
from .config import SegmentationConfig
from .features import extract_segment_features
from .review import create_review_queue
from .schema import normalize_ohlcv
from .segmentation import segment_bars
from .stability import bootstrap_boundary_stability


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _bucket(symbol: object, bucket_count: int) -> int:
    """Stable process-independent bucket, unlike Python's randomized hash."""
    digest = blake2b(str(symbol).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % bucket_count


def _read_source_manifest(source_dir: Path) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"source bulk manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[tuple[str, Path]] = []
    for day, entry in manifest.get("dates", {}).items():
        if not entry.get("partition"):
            continue
        path = source_dir / entry["partition"]
        endpoints = entry.get("endpoints", {})
        # Optional metadata backfills are allowed to change a bulk builder's
        # overall status.  Candidate construction only consumes the frozen core
        # daily+adj_factor parquet, so accept a physical partition whenever its
        # required core endpoint statuses are OK (or an old manifest's completed
        # partition has no endpoint detail).
        required_ok = (all(endpoints.get(name, {}).get("status") == "ok" for name in ("daily", "adj_factor"))
                       if endpoints else entry.get("status") == "complete")
        if path.exists() and required_ok:
            rows.append((str(day), path))
    return manifest, sorted(rows)


def _core_fingerprint(manifest: dict[str, Any], sources: list[tuple[str, Path]]) -> str:
    """Hash core manifest evidence while ignoring optional metadata backfills."""
    evidence = []
    for day, path in sources:
        entry = manifest["dates"][day]
        endpoints = entry.get("endpoints", {})
        evidence.append({
            "day": day,
            "partition": str(path.relative_to(Path(manifest.get("_source_root", path.parents[1]))))
            if manifest.get("_source_root") else entry.get("partition"),
            "rows": entry.get("rows"),
            "daily_rows": endpoints.get("daily", {}).get("rows"),
            "adj_factor_rows": endpoints.get("adj_factor", {}).get("rows"),
            "adj_factor_matched": endpoints.get("adj_factor", {}).get("matched_rows"),
        })
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake2b(payload, digest_size=16).hexdigest()


def _prices_for_segmentation(frame: pd.DataFrame, adjustment: str, qfq_anchor_date: str | None) -> pd.DataFrame:
    """Adjust raw vendor prices with a point-in-time-safe policy.

    HFQ multiplies each bar by its own as-of factor.  QFQ is allowed only from
    the explicitly supplied anchor onward, so the denominator was available at
    every emitted segment end; pre-anchor bars are excluded rather than silently
    leaking an end-of-sample factor backwards.
    """
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    if adjustment == "hfq":
        return adjust_prices(work, mode="hfq")
    if adjustment != "qfq" or not qfq_anchor_date:
        raise ValueError("adjustment must be 'hfq', or qfq with an explicit qfq_anchor_date")
    anchor = pd.Timestamp(qfq_anchor_date).normalize()
    work = work.loc[work["trade_date"] >= anchor].copy()
    # The factor at the named anchor is a per-security known constant from that
    # date. Missing anchors deliberately exclude a security instead of falling
    # back to a later factor.
    anchor_factor = (work.loc[work["trade_date"].eq(anchor), ["ts_code", "adj_factor"]]
                     .drop_duplicates("ts_code", keep="last").set_index("ts_code")["adj_factor"])
    work = work.loc[work["ts_code"].isin(anchor_factor.index)].copy()
    return adjust_prices(work, mode="qfq", anchor_factor=anchor_factor)


def _bucket_output_paths(output_dir: str | Path, bucket: int) -> list[Path]:
    root, key = Path(output_dir), f"{bucket:03d}"
    return [root / name / f"bucket={key}" / "data.parquet"
            for name in ("candidate_segments", "segment_features", "review_candidates")]


def audit_fullmarket_segments(output_dir: str | Path, require_complete: bool = False) -> dict[str, Any]:
    """Audit bucket outputs without loading the full feature matrix at once.

    The audit treats ``UNKNOWN`` as an intentional semantic invariant for this
    candidate-building layer.  It also verifies row/id parity across the three
    per-bucket artifacts and checks segment geometry and boundary probabilities.
    """
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"full-market manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket_count = int(manifest.get("bucket_count", 0))
    issues: list[dict[str, Any]] = []
    totals = {"buckets_complete": 0, "segments": 0, "features": 0, "review_candidates": 0}
    seen_ids: set[str] = set()

    if require_complete and not manifest.get("build_complete", False):
        issues.append({"scope": "manifest", "code": "build_incomplete"})
    for bucket in range(bucket_count):
        key = f"{bucket:03d}"
        state = manifest.get("buckets", {}).get(key)
        if not state or state.get("status") != "complete":
            if require_complete:
                issues.append({"scope": key, "code": "bucket_incomplete",
                               "status": None if state is None else state.get("status")})
            continue
        totals["buckets_complete"] += 1
        paths = _bucket_output_paths(root, bucket)
        missing = [str(path.relative_to(root)) for path in paths if not path.exists()]
        if missing:
            issues.append({"scope": key, "code": "missing_outputs", "paths": missing})
            continue
        segments = pd.read_parquet(paths[0])
        features = pd.read_parquet(paths[1])
        review = pd.read_parquet(paths[2])
        totals["segments"] += len(segments)
        totals["features"] += len(features)
        totals["review_candidates"] += len(review)
        lengths = [len(segments), len(features), len(review)]
        if len(set(lengths)) != 1 or len(segments) != int(state.get("segments", -1)):
            issues.append({"scope": key, "code": "row_count_mismatch", "counts": lengths,
                           "manifest_segments": state.get("segments")})
        id_sets = []
        for name, frame in zip(("segments", "features", "review"), (segments, features, review)):
            if "segment_id" not in frame:
                issues.append({"scope": key, "code": "missing_segment_id", "artifact": name})
                id_sets.append(set())
                continue
            ids = frame["segment_id"].astype(str)
            if ids.duplicated().any():
                issues.append({"scope": key, "code": "duplicate_segment_id", "artifact": name,
                               "count": int(ids.duplicated().sum())})
            id_sets.append(set(ids))
        if not (id_sets[0] == id_sets[1] == id_sets[2]):
            issues.append({"scope": key, "code": "segment_id_set_mismatch"})
        overlap = seen_ids.intersection(id_sets[0])
        if overlap:
            issues.append({"scope": key, "code": "cross_bucket_duplicate_id", "count": len(overlap)})
        seen_ids.update(id_sets[0])

        for name, frame in (("segments", segments), ("features", features)):
            if not frame.empty and (frame.get("label", pd.Series(dtype=str)).ne("UNKNOWN").any()
                                    or frame.get("candidate_label", pd.Series(dtype=str)).ne("UNKNOWN").any()):
                issues.append({"scope": key, "code": "premature_semantic_label", "artifact": name})
        if not review.empty and review.get("predicted_label", pd.Series(dtype=str)).ne("UNKNOWN").any():
            issues.append({"scope": key, "code": "premature_semantic_label", "artifact": "review"})
        if not segments.empty:
            invalid_geometry = (
                pd.to_datetime(segments["start"]) > pd.to_datetime(segments["end"])
            ) | (segments["end_idx"] < segments["start_idx"]) | (
                segments["n_bars"] != segments["end_idx"] - segments["start_idx"] + 1
            )
            if invalid_geometry.any():
                issues.append({"scope": key, "code": "invalid_segment_geometry",
                               "count": int(invalid_geometry.sum())})
            for column in ("start_boundary_probability", "end_boundary_probability", "boundary_probability"):
                invalid = ~segments[column].between(0.0, 1.0, inclusive="both")
                if invalid.any():
                    issues.append({"scope": key, "code": "invalid_probability", "column": column,
                                   "count": int(invalid.sum())})

    return {
        "passed": not issues,
        "require_complete": require_complete,
        "manifest_build_complete": bool(manifest.get("build_complete", False)),
        "bucket_count": bucket_count,
        "summary": totals,
        "issues": issues,
    }


def _process_bucket_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Build exactly one bucket without reading or writing the parent manifest.

    This function is module-level so it is importable by Windows ``spawn``.
    ``task`` contains only pickle-safe primitives/dataclass dictionaries; in
    particular it intentionally does not contain a progress callback.
    """
    output_dir = Path(task["output_dir"])
    bucket = int(task["bucket"])
    key = f"{bucket:03d}"
    targets = _bucket_output_paths(output_dir, bucket)
    inputs = sorted((output_dir / "_symbol_shards" / f"bucket={key}").glob("shard=*.parquet"))
    try:
        if not inputs:
            return {"bucket": bucket, "status": "complete", "symbols": 0, "bars": 0, "segments": 0,
                    "failures": []}
        raw = pd.concat([pd.read_parquet(path) for path in inputs], ignore_index=True)
        bars = _prices_for_segmentation(raw, task["adjustment"], task.get("qfq_anchor_date"))
        bars = normalize_ohlcv(bars)
        config = SegmentationConfig(**task["segmentation"])
        segment_frames: list[pd.DataFrame] = []
        feature_frames: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        symbols = bars["symbol"].drop_duplicates().tolist()

        def process_group(group: pd.DataFrame) -> None:
            segment = segment_bars(group, config, use_changepoints=True)
            if segment.empty:
                return
            if config.bootstrap_iterations > 0:
                _, stability = bootstrap_boundary_stability(
                    group, config, iterations=config.bootstrap_iterations,
                    tolerance=config.bootstrap_tolerance,
                    price_noise=config.bootstrap_price_noise,
                    volume_noise=config.bootstrap_volume_noise,
                    random_state=42, use_changepoints=True,
                )
                lookup = stability.set_index(["symbol", "boundary_idx"])["stability_frequency"].to_dict()
                start_stability = [lookup.get((row.symbol, int(row.start_idx)), 1.0)
                                   for row in segment.itertuples()]
                end_stability = [lookup.get((row.symbol, int(row.end_idx)), 1.0)
                                 for row in segment.itertuples()]
                segment["detector_start_boundary_probability"] = segment["start_boundary_probability"]
                segment["detector_end_boundary_probability"] = segment["end_boundary_probability"]
                segment["detector_boundary_probability"] = segment["boundary_probability"]
                segment["start_boundary_stability"] = start_stability
                segment["end_boundary_stability"] = end_stability
                segment["bootstrap_boundary_stability"] = np.minimum(start_stability, end_stability)
                segment["start_boundary_probability"] = np.minimum(
                    segment["detector_start_boundary_probability"], segment["start_boundary_stability"]
                )
                segment["end_boundary_probability"] = np.minimum(
                    segment["detector_end_boundary_probability"], segment["end_boundary_stability"]
                )
                segment["boundary_probability"] = np.minimum(
                    segment["start_boundary_probability"], segment["end_boundary_probability"]
                )
                segment["boundary_uncertainty"] = 1.0 - segment["boundary_probability"]
            feature = extract_segment_features(group, segment)
            segment_frames.append(segment)
            feature_frames.append(feature)

        for offset in range(0, len(symbols), int(task["symbols_per_batch"])):
            names = symbols[offset:offset + int(task["symbols_per_batch"])]
            group = bars.loc[bars["symbol"].isin(names)]
            try:
                process_group(group)
            except Exception:  # noqa: BLE001 - isolate a bad batch, then retry its symbols individually
                # A bad symbol must not turn a whole bucket into a false
                # success; recover the unaffected members individually.
                for symbol in names:
                    try:
                        process_group(group.loc[group["symbol"].eq(symbol)])
                    except Exception as exc:  # noqa: BLE001 - report failures without aborting unaffected symbols
                        failures.append({"symbol": str(symbol), "error": f"{type(exc).__name__}: {exc}"})
        segments = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
        features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
        if not segments.empty:
            for frame in (segments, features):
                frame["candidate_label"] = "UNKNOWN"
                frame["label"] = "UNKNOWN"
                frame["candidate_status"] = "unreviewed_candidate"
                frame["unknown_reason"] = "unreviewed_candidate"
                frame["adjustment_mode"] = task["adjustment"]
                frame["qfq_anchor_date"] = task.get("qfq_anchor_date")
        review = create_review_queue(features)
        # Files are bucket-unique. Atomic replacement makes a stopped/retried
        # worker unable to expose a half-written parquet file.
        for target, frame in zip(targets, (segments, features, review)):
            _atomic_parquet(target, frame)
        return {"bucket": bucket, "status": "complete" if not failures else "partial",
                "symbols": int(bars.symbol.nunique()), "bars": len(bars), "segments": len(segments),
                "failures": failures}
    except Exception as exc:  # noqa: BLE001 - convert worker failures into auditable partial results
        # Expected worker errors are reported as an auditable partial bucket;
        # the parent remains the sole manifest writer and can retry later.
        return {"bucket": bucket, "status": "partial", "symbols": None, "bars": None, "segments": None,
                "failures": [], "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class FullMarketSegmentBuilder:
    """Build raw candidate segments/features from a bulk daily source.

    Output is a parquet dataset with one atomic file per symbol hash bucket:
    ``candidate_segments/``, ``segment_features/`` and ``review_candidates/``.
    Its manifest records the exact source-date set, making an in-progress data
    download impossible to accidentally mix with a completed derived run.
    """

    source_dir: str | Path
    output_dir: str | Path
    bucket_count: int = 64
    source_files_per_shard: int = 20
    symbols_per_batch: int = 250
    bucket_workers: int = 1
    adjustment: str = "hfq"
    qfq_anchor_date: str | None = None
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    max_shards_per_run: int | None = None
    max_buckets_per_run: int | None = None
    progress: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        self.source_dir, self.output_dir = Path(self.source_dir), Path(self.output_dir)
        if (self.bucket_count < 1 or self.source_files_per_shard < 1 or self.symbols_per_batch < 1
                or self.bucket_workers < 1):
            raise ValueError("bucket_count, source_files_per_shard, symbols_per_batch and bucket_workers must be positive")
        if any(value is not None and value < 1 for value in (self.max_shards_per_run, self.max_buckets_per_run)):
            raise ValueError("max_shards_per_run/max_buckets_per_run must be positive when supplied")
        if self.adjustment not in {"hfq", "qfq"}:
            raise ValueError("adjustment must be hfq or qfq")
        if self.adjustment == "qfq" and not self.qfq_anchor_date:
            raise ValueError("qfq requires qfq_anchor_date to prevent final-anchor leakage")

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def _settings(self) -> dict[str, Any]:
        return {
            "bucket_count": self.bucket_count,
            "source_files_per_shard": self.source_files_per_shard,
            "symbols_per_batch": self.symbols_per_batch,
            "adjustment": self.adjustment,
            "qfq_anchor_date": self.qfq_anchor_date,
            "segmentation": asdict(self.segmentation),
        }

    def _manifest(self, source_dates: list[str], core_fingerprint: str) -> dict[str, Any]:
        if self.manifest_path.exists():
            result = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if result.get("source_dates") != source_dates:
                raise RuntimeError(
                    "source complete-date set changed; use a new output directory after the bulk download is complete "
                    "to avoid mixing histories"
                )
            if result.get("core_fingerprint") not in {None, core_fingerprint}:
                raise RuntimeError("source core fingerprint changed; use a new output directory")
            previous_settings = result.get("settings")
            if previous_settings is not None and previous_settings != self._settings():
                raise RuntimeError("full-market builder settings changed; use a new output directory")
            # Upgrade a pre-signature manifest only when its legacy top-level
            # settings agree with this invocation.
            for key in ("bucket_count", "source_files_per_shard", "symbols_per_batch", "adjustment", "qfq_anchor_date"):
                if key in result and result[key] != self._settings()[key]:
                    raise RuntimeError("full-market builder settings changed; use a new output directory")
            result["settings"] = self._settings()
            result["core_fingerprint"] = core_fingerprint
            return result
        return {
            "format_version": 1,
            "source_dir": str(self.source_dir),
            "source_dates": source_dates,
            "bucket_count": self.bucket_count,
            "source_files_per_shard": self.source_files_per_shard,
            "symbols_per_batch": self.symbols_per_batch,
            "adjustment": self.adjustment,
            "qfq_anchor_date": self.qfq_anchor_date,
            "qfq_policy": "only anchor-and-later bars emitted" if self.adjustment == "qfq" else None,
            "settings": self._settings(),
            "core_fingerprint": core_fingerprint,
            "semantic_policy": "no automatic semantic label; all candidates are UNKNOWN until reviewed",
            "shards": {}, "buckets": {},
        }

    def _build_shards(self, manifest: dict[str, Any], sources: list[tuple[str, Path]]) -> int:
        processed = 0
        for index in range(0, len(sources), self.source_files_per_shard):
            chunk = index // self.source_files_per_shard
            key = f"{chunk:05d}"
            prior = manifest["shards"].get(key, {})
            expected = [day for day, _ in sources[index:index + self.source_files_per_shard]]
            if prior.get("status") == "complete" and prior.get("source_dates") == expected:
                continue
            if self.max_shards_per_run is not None and processed >= self.max_shards_per_run:
                break
            if self.progress: self.progress({"event": "shard_started", "shard": chunk, "source_dates": len(expected)})
            frames = []
            for day, path in sources[index:index + self.source_files_per_shard]:
                required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "adj_factor"}
                try:
                    data = pd.read_parquet(path, columns=sorted(required))
                except Exception as exc:
                    raise ValueError(f"{path} cannot supply required raw columns: {sorted(required)}") from exc
                frames.append(data)
            combined = pd.concat(frames, ignore_index=True)
            combined["_bucket"] = combined["ts_code"].map(lambda x: _bucket(x, self.bucket_count))
            rows_by_bucket: dict[str, int] = {}
            for bucket, group in combined.groupby("_bucket", sort=True):
                target = self.output_dir / "_symbol_shards" / f"bucket={int(bucket):03d}" / f"shard={key}.parquet"
                _atomic_parquet(target, group.drop(columns="_bucket"))
                rows_by_bucket[str(int(bucket))] = len(group)
            manifest["shards"][key] = {"status": "complete", "source_dates": expected,
                                       "rows": len(combined), "rows_by_bucket": rows_by_bucket}
            _atomic_json(self.manifest_path, manifest)
            processed += 1
            if self.progress: self.progress({"event": "shard_finished", "shard": chunk, "rows": len(combined)})
        return processed

    def _bucket_task(self, bucket: int) -> dict[str, Any]:
        """Return a spawn-safe task; callbacks and the manifest stay parent-only."""
        return {
            "output_dir": str(self.output_dir), "bucket": bucket, "adjustment": self.adjustment,
            "qfq_anchor_date": self.qfq_anchor_date, "segmentation": asdict(self.segmentation),
            "symbols_per_batch": self.symbols_per_batch,
        }

    def _commit_bucket_result(self, manifest: dict[str, Any], result: dict[str, Any]) -> None:
        """The sole manifest mutation point for both serial and parallel work."""
        bucket = int(result["bucket"])
        key = f"{bucket:03d}"
        state = {name: value for name, value in result.items() if name != "bucket"}
        manifest["buckets"][key] = state
        _atomic_json(self.manifest_path, manifest)
        if self.progress:
            self.progress({"event": "bucket_finished", "bucket": bucket, "status": state["status"],
                           "bars": state.get("bars"), "segments": state.get("segments"),
                           "failures": len(state.get("failures", [])), "error": state.get("error")})

    def _process_bucket(self, manifest: dict[str, Any], bucket: int) -> None:
        """Serial wrapper retained for predictable one-worker execution."""
        key = f"{bucket:03d}"
        prior = manifest["buckets"].get(key, {})
        targets = _bucket_output_paths(self.output_dir, bucket)
        if prior.get("status") == "complete" and all(path.exists() for path in targets):
            return
        inputs = list((self.output_dir / "_symbol_shards" / f"bucket={key}").glob("shard=*.parquet"))
        if self.progress:
            self.progress({"event": "bucket_started", "bucket": bucket, "shards": len(inputs)})
        self._commit_bucket_result(manifest, _process_bucket_worker(self._bucket_task(bucket)))

    def build(self) -> dict[str, Any]:
        source_manifest, sources = _read_source_manifest(self.source_dir)
        if not sources:
            raise RuntimeError("no complete source partitions available")
        source_dates = [day for day, _ in sources]
        manifest = self._manifest(source_dates, _core_fingerprint(source_manifest, sources))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.manifest_path, manifest)
        processed_shards = self._build_shards(manifest, sources)
        expected_shards = int(np.ceil(len(sources) / self.source_files_per_shard))
        shards_complete = len(manifest["shards"]) == expected_shards and all(
            entry.get("status") == "complete" for entry in manifest["shards"].values()
        )
        manifest["shards_complete"] = shards_complete
        manifest["last_run_processed_shards"] = processed_shards
        if not shards_complete:
            manifest["build_complete"] = False
            _atomic_json(self.manifest_path, manifest)
            return manifest
        pending = [bucket for bucket in range(self.bucket_count)
                   if manifest["buckets"].get(f"{bucket:03d}", {}).get("status") != "complete"]
        if self.max_buckets_per_run is not None:
            pending = pending[:self.max_buckets_per_run]
        completed = len(pending)
        if self.bucket_workers == 1:
            for bucket in pending:
                self._process_bucket(manifest, bucket)
        elif pending:
            # Explicit spawn works on Windows and also prevents inheriting a
            # parent callback/file handle under POSIX. Workers write only their
            # bucket-unique parquet paths; the parent observes futures and is
            # the sole owner of manifest.json.
            context = get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(self.bucket_workers, len(pending)), mp_context=context) as pool:
                futures = {}
                for bucket in pending:
                    inputs = list((self.output_dir / "_symbol_shards" / f"bucket={bucket:03d}").glob("shard=*.parquet"))
                    if self.progress:
                        self.progress({"event": "bucket_started", "bucket": bucket, "shards": len(inputs)})
                    futures[pool.submit(_process_bucket_worker, self._bucket_task(bucket))] = bucket
                for future in as_completed(futures):
                    bucket = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - keep crashed buckets visible and retryable
                        # A crashed process must remain retryable and visible;
                        # never claim an unwritten bucket as complete.
                        result = {"bucket": bucket, "status": "partial", "symbols": None, "bars": None,
                                  "segments": None, "failures": [],
                                  "error": f"worker_crash {type(exc).__name__}: {exc}"}
                    self._commit_bucket_result(manifest, result)
        manifest["build_complete"] = len(manifest["buckets"]) == self.bucket_count and all(
            entry.get("status") == "complete" for entry in manifest["buckets"].values()
        )
        manifest["last_run_processed_buckets"] = completed
        _atomic_json(self.manifest_path, manifest)
        return manifest
