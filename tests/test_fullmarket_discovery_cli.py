import pandas as pd
import pytest

from wave_segments.fullmarket_discovery_cli import run_fullmarket_discovery


def _features(n=60):
    rows = []
    for i in range(n):
        regime = i % 3
        rows.append({
            "segment_id": f"S{i}", "symbol": f"T{i % 6}",
            "start": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 40),
            "end": pd.Timestamp("2020-01-05") + pd.Timedelta(days=i * 40),
            "n_bars": 5, "duration_bars": 5 + regime * 3,
            "cumulative_return": (regime - 1) * .04 + (i % 4) * .0001,
            "return_volatility": .01 + regime * .01,
            "amplitude": .02 + regime * .02,
            "future_return_20d": 1000.0, "label": "UNKNOWN",
        })
    return pd.DataFrame(rows)


def test_offline_discovery_cli_outputs_review_status_and_reproducibility_evidence(tmp_path):
    source = tmp_path / "features"
    (source / "bucket=000").mkdir(parents=True)
    (source / "bucket=001").mkdir(parents=True)
    data = _features()
    data.iloc[:30].to_parquet(source / "bucket=000" / "data.parquet", index=False)
    data.iloc[30:].to_parquet(source / "bucket=001" / "data.parquet", index=False)
    output = tmp_path / "discovery"
    summary = run_fullmarket_discovery(
        source, output, fit_sample_size=48, random_state=3, n_components=3,
        ensemble_size=2, min_cluster_samples=2, min_cluster_symbols=1,
        min_cluster_years=2, max_iter=100, include_hdbscan=False,
    )
    assert summary["mode"] == "offline_transductive_review_only"
    assert summary["training_rows"] > 0 and summary["predicted_rows"] == len(data)
    assert summary["model_metadata"]["hdbscan_available"] is False
    assert (output / "discoverer.joblib").is_file()
    for bucket in ("bucket=000", "bucket=001"):
        predicted = pd.read_parquet(output / "discovery_labels" / bucket / "data.parquet")
        review = pd.read_parquet(output / "review_candidates" / bucket / "data.parquet")
        assert len(predicted) == len(review) == 30
        assert set(predicted.candidate_status) <= {"classified", "model_abstention"}
        assert set(review.candidate_status) <= {"classified", "model_abstention"}
        assert "future_return_20d" not in summary["model_metadata"]["feature_columns"]

    with pytest.raises(RuntimeError, match="settings changed"):
        run_fullmarket_discovery(
            source, output, fit_sample_size=48, random_state=4, n_components=3,
            ensemble_size=2, min_cluster_samples=2, min_cluster_symbols=1,
            min_cluster_years=2, max_iter=100, include_hdbscan=False,
        )
