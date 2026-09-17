import json
from hashlib import blake2b

import pandas as pd

from wave_segments.annotation_app import append_annotation, load_bars_for_symbol, resolve_candidate_status


def test_annotations_are_append_only_and_keep_reviewers(tmp_path):
    path = tmp_path / "annotations.csv"
    append_annotation(path, {"segment_id": "S1", "annotator": "alice", "label": "上涨推进",
                             "candidate_status": "unreviewed_candidate"})
    result = append_annotation(path, {"segment_id": "S1", "annotator": "bob", "label": "UNKNOWN"})
    assert len(result) == 2
    assert set(result.annotator) == {"alice", "bob"}
    assert result.annotation_id.nunique() == 2
    assert result.iloc[0].candidate_status == "unreviewed_candidate"


def test_legacy_queue_status_is_inferred_without_conflating_unknown_types():
    assert resolve_candidate_status(pd.Series({
        "predicted_label": "UNKNOWN", "unknown_reason": "unreviewed_candidate",
    })) == "unreviewed_candidate"
    assert resolve_candidate_status(pd.Series({
        "predicted_label": "UNKNOWN", "unknown_reason": "high_entropy;low_density",
    })) == "model_abstention"
    assert resolve_candidate_status(pd.Series({"predicted_label": "CLUSTER_A"})) == "classified"


def test_load_bars_for_symbol_from_fullmarket_bucket_applies_hfq(tmp_path):
    root = tmp_path / "fullmarket"
    symbol = "600000.SH"
    bucket_count = 4
    digest = blake2b(symbol.encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") % bucket_count
    bucket_dir = root / "_symbol_shards" / f"bucket={bucket:03d}"
    bucket_dir.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"bucket_count": bucket_count}), encoding="utf-8")
    pd.DataFrame({
        "ts_code": [symbol, symbol, "OTHER.SH"],
        "trade_date": ["20240102", "20240103", "20240102"],
        "open": [10.0, 11.0, 50.0], "high": [12.0, 13.0, 55.0],
        "low": [9.0, 10.0, 48.0], "close": [11.0, 12.0, 52.0],
        "vol": [100.0, 110.0, 500.0], "adj_factor": [2.0, 3.0, 1.0],
    }).to_parquet(bucket_dir / "shard=00000.parquet", index=False)

    bars = load_bars_for_symbol(root, symbol)
    assert bars.symbol.unique().tolist() == [symbol]
    assert bars.timestamp.dt.strftime("%Y%m%d").tolist() == ["20240102", "20240103"]
    assert bars.close.tolist() == [22.0, 36.0]
    assert bars.volume.tolist() == [100.0, 110.0]
