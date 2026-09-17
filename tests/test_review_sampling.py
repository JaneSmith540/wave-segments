import pandas as pd

from wave_segments.review import build_stratified_review_sample, create_review_queue


def test_stratified_review_sample_is_reproducible_and_unknown_enriched():
    rows = []
    for i in range(60):
        rows.append({
            "segment_id": f"S{i}", "symbol": f"T{i % 6}",
            "start": pd.Timestamp(2021 + i % 3, 1, 1) + pd.Timedelta(days=i),
            "end": pd.Timestamp(2021 + i % 3, 1, 5) + pd.Timedelta(days=i),
            "label": "UNKNOWN" if i % 4 == 0 else f"CLUSTER_{i % 3}",
            "industry": f"I{i % 4}", "recognizability": i / 60,
        })
    data = pd.DataFrame(rows)
    first = build_stratified_review_sample(data, 30, unknown_share=.4, random_state=7)
    second = build_stratified_review_sample(data, 30, unknown_share=.4, random_state=7)
    assert first.segment_id.tolist() == second.segment_id.tolist()
    assert len(first) == 30
    assert first.predicted_label.eq("UNKNOWN").sum() == 12
    assert first.symbol.nunique() >= 5
    assert first.sampling_probability.between(0, 1).all()


def test_stratified_sample_does_not_lexically_truncate_year_strata():
    rows = []
    for year in range(2015, 2025):
        for number in range(100):
            rows.append({"segment_id": f"{year}-{number}", "symbol": f"S{number:03d}",
                         "start": f"{year}-01-02", "end": f"{year}-01-03", "label": "UNKNOWN"})
    sample = build_stratified_review_sample(pd.DataFrame(rows), 100, unknown_share=1.0, random_state=9)
    years = pd.to_datetime(sample.start_timestamp).dt.year
    assert set(years) == set(range(2015, 2025))
    assert years.value_counts().eq(10).all()
    assert sample.sampling_probability.eq(.1).all()


def test_review_queue_distinguishes_unreviewed_candidates_from_model_abstentions():
    frame = pd.DataFrame([
        {"segment_id": "candidate", "symbol": "A", "label": "UNKNOWN",
         "unknown_reason": "unreviewed_candidate"},
        {"segment_id": "abstention", "symbol": "B", "label": "UNKNOWN",
         "unknown_reason": "high_entropy;low_consensus"},
        {"segment_id": "cluster", "symbol": "C", "label": "CLUSTER_A",
         "unknown_reason": ""},
    ])
    queue = create_review_queue(frame).set_index("segment_id")
    assert queue.loc["candidate", "candidate_status"] == "unreviewed_candidate"
    assert queue.loc["abstention", "candidate_status"] == "model_abstention"
    assert queue.loc["cluster", "candidate_status"] == "classified"
