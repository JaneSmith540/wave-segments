import json

import numpy as np
import pandas as pd

from wave_segments.analytics import human_annotation_agreement
from wave_segments.annotation_app import resolve_boundary_correction
from wave_segments.validation import (annotation_agreement_report,
                                      boundary_annotation_agreement,
                                      resolve_human_annotations)


def _annotations():
    # Alice's older S1 vote must not count after she changes it to UP.
    return pd.DataFrame([
        {"segment_id": "S1", "annotator": "alice", "label": "DOWN", "reviewed_at": "2024-01-01", "start_idx": 10, "end_idx": 20, "sampling_probability": .5},
        {"segment_id": "S1", "annotator": "alice", "label": "UP", "reviewed_at": "2024-01-02", "start_idx": 10, "end_idx": 20, "sampling_probability": .5},
        {"segment_id": "S1", "annotator": "bob", "label": "UP", "start_idx": 12, "end_idx": 19, "sampling_probability": .5},
        {"segment_id": "S1", "annotator": "carol", "label": "UP", "start_idx": 11, "end_idx": 21, "sampling_probability": .5},
        {"segment_id": "S2", "annotator": "alice", "label": "UP", "start_idx": 30, "end_idx": 40, "sampling_probability": 1.0},
        {"segment_id": "S2", "annotator": "bob", "label": "DOWN", "start_idx": 35, "end_idx": 45, "sampling_probability": 1.0},
        {"segment_id": "S2", "annotator": "carol", "label": "DOWN", "start_idx": 34, "end_idx": 44, "sampling_probability": 1.0},
        # One rater is an explicitly unresolved ground-truth candidate.
        {"segment_id": "S3", "annotator": "alice", "label": "UP", "start_idx": 50, "end_idx": 60, "sampling_probability": 1.0},
    ])


def test_multi_rater_consensus_is_unknown_for_disagreement_or_single_vote_and_keeps_votes():
    consensus, summary = resolve_human_annotations(_annotations())
    indexed = consensus.set_index("segment_id")
    assert indexed.loc["S1", "consensus_label"] == "UP"
    assert indexed.loc["S2", "consensus_label"] == "UNKNOWN"
    assert indexed.loc["S3", "consensus_label"] == "UNKNOWN"
    assert indexed.loc["S2", "consensus_reason"] == "label_disagreement"
    assert indexed.loc["S3", "consensus_reason"] == "insufficient_independent_raters"
    assert json.loads(indexed.loc["S2", "vote_counts"]) == {"DOWN": 2, "UP": 1}
    assert summary["disputed_segments"] == 2
    assert np.isfinite(summary["pairwise_agreement"])
    assert np.isfinite(summary["cohen_kappa"])
    assert np.isfinite(summary["fleiss_kappa"])
    # S1 represents two population units, S2 and S3 one each.
    assert summary["weighted_annotated_segments"] == 4.0


def test_boundary_agreement_uses_bar_tolerance_corrections_and_inverse_probability_weights():
    annotations = _annotations()
    # A numeric correction is a corrected bar index and overrides its base one.
    annotations.loc[(annotations.segment_id == "S1") & (annotations.annotator == "bob"), "boundary_start_correction"] = 13
    result = boundary_annotation_agreement(annotations, tolerance=3)
    pairwise = result["pairwise"]
    alice_bob_start = pairwise[(pairwise.annotator_a == "alice") & (pairwise.annotator_b == "bob") & (pairwise.boundary == "start")].iloc[0]
    # S1 is a match (10 vs 13); S2 is not (30 vs 35), weighted TP=2 of total 3.
    assert np.isclose(alice_bob_start.weighted_true_positive, 2.0)
    assert np.isclose(alice_bob_start.precision, 2 / 3)
    assert np.isclose(alice_bob_start.recall, 2 / 3)
    assert np.isclose(alice_bob_start.f1, 2 / 3)
    assert result["tolerance_bars"] == 3


def test_annotation_report_is_available_from_analytics():
    direct = annotation_agreement_report(_annotations(), tolerance=3)
    via_analytics = human_annotation_agreement(_annotations(), tolerance=3)
    assert {"consensus", "label_agreement", "boundary_agreement"} <= set(direct)
    assert len(via_analytics["consensus"]) == 3
    assert via_analytics["boundary_agreement"]["scored_pairs"] == 6


def test_undated_legacy_event_does_not_override_a_dated_vote():
    rows = pd.DataFrame([
        {"segment_id": "S", "annotator": "a", "label": "UP", "reviewed_at": "2024-02-01"},
        {"segment_id": "S", "annotator": "a", "label": "DOWN", "reviewed_at": None},
        {"segment_id": "S", "annotator": "b", "label": "UP", "reviewed_at": "2024-02-02"},
    ])
    consensus, _ = resolve_human_annotations(rows)
    assert consensus.loc[0, "consensus_label"] == "UP"


def test_missing_sampling_probability_does_not_erase_a_boundary_review():
    rows = pd.DataFrame([
        {"segment_id": "S", "annotator": "a", "start_idx": 10, "end_idx": 20, "sampling_probability": np.nan},
        {"segment_id": "S", "annotator": "b", "start_idx": 11, "end_idx": 22, "sampling_probability": np.nan},
    ])
    assert boundary_annotation_agreement(rows, tolerance=3)["scored_pairs"] == 2


def test_boundary_date_is_converted_to_trading_bar_index():
    bars = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])})
    assert resolve_boundary_correction("2024-01-08", bars) == (1, "2024-01-08T00:00:00")
    assert resolve_boundary_correction("2", bars) == (2, None)
