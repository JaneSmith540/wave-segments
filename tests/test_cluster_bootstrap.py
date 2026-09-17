from types import SimpleNamespace

import numpy as np
import pandas as pd

from wave_segments.cluster_bootstrap import (
    align_bootstrap_to_reference,
    align_by_probability_overlap,
    remap_bootstrap_predictions,
    resample_symbols_with_replacement,
)


class IdentityScaler:
    scale_ = np.array([1.0])

    def inverse_transform(self, values):
        return np.asarray(values)


def test_symbol_cluster_bootstrap_is_seeded_and_resamples_whole_symbols():
    train = pd.DataFrame({"symbol": ["A", "A", "B", "B", "B"], "x": range(5)})
    a, unique_a = resample_symbols_with_replacement(train, random_state=11)
    b, unique_b = resample_symbols_with_replacement(train, random_state=11)
    pd.testing.assert_frame_equal(a, b)
    assert unique_a == unique_b
    assert set(a.symbol).issubset({"A", "B"})
    assert len(a) in {4, 5, 6}
    for symbol, group in a.groupby("symbol"):
        # Every sampled copy of a symbol contributes its complete block.
        assert len(group) in ({2, 4} if symbol == "A" else {3, 6, 9})


def test_cluster_ids_align_by_prototype_and_probabilities_follow_mapping():
    ref = SimpleNamespace(
        reference_means_=np.array([[0.], [10.]]), scaler=IdentityScaler(),
        cluster_labels_=["CLUSTER_A", "CLUSTER_B"],
    )
    boot = SimpleNamespace(
        reference_means_=np.array([[10.1], [-.1]]), scaler=IdentityScaler(),
        cluster_labels_=["CLUSTER_A", "CLUSTER_B"],
    )
    mapping, rms = align_bootstrap_to_reference(ref, boot)
    assert mapping == {"CLUSTER_A": "CLUSTER_B", "CLUSTER_B": "CLUSTER_A"}
    assert rms <= .11
    predictions = pd.DataFrame({
        "label": ["CLUSTER_A", "UNKNOWN"],
        "candidate_label": ["CLUSTER_A", "CLUSTER_B"],
        "prob_CLUSTER_A": [.8, .2], "prob_CLUSTER_B": [.2, .8],
    })
    aligned = remap_bootstrap_predictions(predictions, mapping, ["CLUSTER_A", "CLUSTER_B"])
    assert aligned.label.tolist() == ["CLUSTER_B", "UNKNOWN"]
    assert aligned.candidate_label.tolist() == ["CLUSTER_B", "CLUSTER_A"]
    assert aligned.prob_CLUSTER_A.tolist() == [.2, .8]
    assert aligned.prob_CLUSTER_B.tolist() == [.8, .2]


def test_soft_membership_alignment_recovers_permutation_on_shared_training_rows():
    reference = np.array([[.9, .1, 0], [.8, .2, 0], [0, .2, .8], [0, .1, .9]])
    # Bootstrap order is C, A, B rather than A, B, C.
    bootstrap = reference[:, [2, 0, 1]]
    mapping, similarity = align_by_probability_overlap(
        reference, bootstrap,
        ["CLUSTER_A", "CLUSTER_B", "CLUSTER_C"],
        ["BOOT_A", "BOOT_B", "BOOT_C"],
    )
    assert mapping == {"BOOT_A": "CLUSTER_C", "BOOT_B": "CLUSTER_A", "BOOT_C": "CLUSTER_B"}
    assert similarity > .99
