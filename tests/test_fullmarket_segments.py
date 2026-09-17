import json

import pandas as pd
import pytest

from wave_segments.config import SegmentationConfig
from wave_segments.fullmarket_segments import (
    FullMarketSegmentBuilder,
    _process_bucket_worker,
    audit_fullmarket_segments,
)


def _source(tmp_path):
    root = tmp_path / "bulk"
    dates = {}
    for offset, day in enumerate(["20240102", "20240103", "20240104", "20240105", "20240108", "20240109"]):
        part = root / f"trade_date={day}" / "data.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for symbol, base in [("000001.SZ", 10.0), ("600000.SH", 20.0)]:
            close = base + offset * (0.4 if symbol.startswith("000") else -0.25)
            records.append({"ts_code": symbol, "trade_date": day, "open": close - .1, "high": close + .2,
                            "low": close - .3, "close": close, "vol": 1000 + offset, "adj_factor": 1 + offset * .01})
        pd.DataFrame(records).to_parquet(part, index=False)
        dates[day] = {"status": "complete", "partition": str(part.relative_to(root))}
    (root / "manifest.json").write_text(json.dumps({"dates": dates}), encoding="utf-8")
    return root


def test_fullmarket_builder_transposes_and_resumes_small_dataset(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "derived"
    builder = FullMarketSegmentBuilder(source, output, bucket_count=2, source_files_per_shard=2)
    manifest = builder.build()
    assert manifest["build_complete"]
    assert len(manifest["shards"]) == 3
    assert all(row["status"] == "complete" for row in manifest["buckets"].values())
    features = pd.concat([pd.read_parquet(path) for path in (output / "segment_features").glob("bucket=*/data.parquet")])
    review = pd.concat([pd.read_parquet(path) for path in (output / "review_candidates").glob("bucket=*/data.parquet")])
    assert set(features["adjustment_mode"]) == {"hfq"}
    assert set(features["label"]) == {"UNKNOWN"}
    assert set(features["candidate_status"]) == {"unreviewed_candidate"}
    assert set(review["predicted_label"]) == {"UNKNOWN"}
    assert set(review["candidate_status"]) == {"unreviewed_candidate"}
    assert len(review) == len(features)
    assert FullMarketSegmentBuilder(source, output, bucket_count=2, source_files_per_shard=2).build()["build_complete"]
    with pytest.raises(RuntimeError, match="settings changed"):
        FullMarketSegmentBuilder(source, output, bucket_count=3, source_files_per_shard=2).build()


def test_qfq_requires_anchor_and_does_not_emit_pre_anchor_bars(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="qfq_anchor_date"):
        FullMarketSegmentBuilder(source, tmp_path / "bad", adjustment="qfq")
    output = tmp_path / "qfq"
    result = FullMarketSegmentBuilder(source, output, bucket_count=1, source_files_per_shard=6,
                                      adjustment="qfq", qfq_anchor_date="20240105").build()
    features = pd.read_parquet(output / "segment_features" / "bucket=000" / "data.parquet")
    assert result["qfq_policy"] == "only anchor-and-later bars emitted"
    assert pd.to_datetime(features["start"]).min() >= pd.Timestamp("2024-01-05")


def test_fullmarket_bootstrap_setting_is_applied_to_candidate_boundaries(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "bootstrap"
    config = SegmentationConfig(
        min_bars=2, bootstrap_iterations=2, bootstrap_tolerance=3,
        use_adaptive_structure=False, use_ruptures=False,
        use_fractal_pivots=False, use_bocpd=False,
    )
    result = FullMarketSegmentBuilder(
        source, output, bucket_count=1, source_files_per_shard=6,
        segmentation=config,
    ).build()
    assert result["build_complete"]
    candidates = pd.read_parquet(output / "candidate_segments" / "bucket=000" / "data.parquet")
    assert {"detector_boundary_probability", "detector_start_boundary_probability",
            "detector_end_boundary_probability", "start_boundary_stability",
            "end_boundary_stability", "bootstrap_boundary_stability",
            "boundary_uncertainty"} <= set(candidates)
    assert candidates["bootstrap_boundary_stability"].between(0, 1).all()
    assert (candidates["boundary_probability"] <= candidates["detector_boundary_probability"]).all()
    assert (candidates["start_boundary_probability"] <= candidates["detector_start_boundary_probability"]).all()
    assert (candidates["end_boundary_probability"] <= candidates["detector_end_boundary_probability"]).all()


def test_optional_metadata_backfill_does_not_hide_valid_frozen_core(tmp_path):
    source = _source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["dates"].values():
        entry["status"] = "failed"  # an optional sidecar is still retrying
        entry["endpoints"] = {"daily": {"status": "ok"}, "adj_factor": {"status": "ok"},
                              "daily_basic": {"status": "failed"}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = FullMarketSegmentBuilder(source, tmp_path / "derived", bucket_count=1,
                                      source_files_per_shard=6).build()
    assert result["build_complete"]


def test_bucket_processing_waits_until_all_bounded_shards_exist(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "bounded"
    first = FullMarketSegmentBuilder(source, output, bucket_count=2, source_files_per_shard=2,
                                     max_shards_per_run=1).build()
    assert not first["shards_complete"]
    assert first["buckets"] == {}
    second = FullMarketSegmentBuilder(source, output, bucket_count=2, source_files_per_shard=2,
                                      max_shards_per_run=2).build()
    assert second["shards_complete"]
    assert second["build_complete"]


def _all_features(path):
    data = pd.concat([pd.read_parquet(part) for part in sorted((path / "segment_features").glob("bucket=*/data.parquet"))])
    return data.sort_values(["segment_id"]).reset_index(drop=True)


def test_spawn_bucket_workers_match_serial_and_resume_without_settings_conflict(tmp_path):
    source = _source(tmp_path)
    serial_path, parallel_path = tmp_path / "serial", tmp_path / "parallel"
    serial = FullMarketSegmentBuilder(source, serial_path, bucket_count=2, source_files_per_shard=2,
                                      bucket_workers=1).build()
    assert serial["build_complete"]

    # Limit is a total number of scheduled buckets, even when two workers are
    # available. The next invocation changes workers and resumes safely because
    # parallelism is intentionally not part of reproducibility settings.
    partial = FullMarketSegmentBuilder(source, parallel_path, bucket_count=2, source_files_per_shard=2,
                                      bucket_workers=2, max_buckets_per_run=1).build()
    assert not partial["build_complete"]
    assert len(partial["buckets"]) == 1
    resumed = FullMarketSegmentBuilder(source, parallel_path, bucket_count=2, source_files_per_shard=2,
                                       bucket_workers=2).build()
    assert resumed["build_complete"]
    assert all(value["status"] == "complete" for value in resumed["buckets"].values())
    # A serial resume must not be rejected solely because --workers changed.
    assert FullMarketSegmentBuilder(source, parallel_path, bucket_count=2, source_files_per_shard=2,
                                    bucket_workers=1).build()["build_complete"]
    pd.testing.assert_frame_equal(_all_features(serial_path), _all_features(parallel_path), check_like=True)


def test_bucket_worker_reports_corrupt_input_as_partial(tmp_path):
    root = tmp_path / "out"
    shard = root / "_symbol_shards" / "bucket=000" / "shard=00000.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"not parquet")
    result = _process_bucket_worker({"output_dir": str(root), "bucket": 0, "adjustment": "hfq",
                                     "qfq_anchor_date": None, "segmentation": {}, "symbols_per_batch": 1})
    assert result["status"] == "partial"
    assert "error" in result


def test_fullmarket_audit_checks_artifact_parity_and_unknown_policy(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "audited"
    FullMarketSegmentBuilder(source, output, bucket_count=2, source_files_per_shard=2).build()
    clean = audit_fullmarket_segments(output, require_complete=True)
    assert clean["passed"]
    assert clean["summary"]["segments"] == clean["summary"]["features"]

    path = next((output / "segment_features").glob("bucket=*/data.parquet"))
    features = pd.read_parquet(path)
    features.loc[features.index[0], "label"] = "UP"
    features.to_parquet(path, index=False)
    broken = audit_fullmarket_segments(output, require_complete=True)
    assert not broken["passed"]
    assert "premature_semantic_label" in {issue["code"] for issue in broken["issues"]}
