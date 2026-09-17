import pandas as pd

from wave_segments.classifier_diagnostics import classification_structure_summary


def test_structure_diagnostics_keep_model_versions_separate_and_ignore_unknown_transitions():
    data = pd.DataFrame([
        {"symbol": "AAA", "start_idx": 0, "end_idx": 2, "n_bars": 3, "label": "A", "oos_model_trained_at": "2020-01-01", "prob_A": .9, "prob_B": .1},
        {"symbol": "AAA", "start_idx": 2, "end_idx": 4, "n_bars": 3, "label": "UNKNOWN", "oos_model_trained_at": "2020-01-01", "prob_A": float("nan"), "prob_B": float("nan")},
        {"symbol": "AAA", "start_idx": 4, "end_idx": 6, "n_bars": 3, "label": "B", "oos_model_trained_at": "2020-01-01", "prob_A": .1, "prob_B": .9},
        # The same ticker with a new model version must not be connected to the prior row.
        {"symbol": "AAA", "start_idx": 6, "end_idx": 8, "n_bars": 3, "label": "A", "oos_model_trained_at": "2021-01-01", "prob_A": .9, "prob_B": .1},
    ])
    summary, transitions, hsmm = classification_structure_summary(data, min_duration_segments=1, max_duration_segments=3)
    assert len(summary) == 2
    assert summary.adjacent_identified_transition_pairs.sum() == 0
    assert transitions.transition_count.sum() == 0
    assert set(hsmm.oos_model_trained_at) == {"2020-01-01", "2021-01-01"}
    assert "future_return" not in set(summary.columns)


def test_structure_diagnostics_count_adjacent_identified_self_transition():
    data = pd.DataFrame([
        {"symbol": "AAA", "start_idx": 0, "end_idx": 2, "n_bars": 3, "label": "A", "prob_A": .9, "prob_B": .1},
        {"symbol": "AAA", "start_idx": 2, "end_idx": 5, "n_bars": 4, "label": "A", "prob_A": .8, "prob_B": .2},
        {"symbol": "AAA", "start_idx": 5, "end_idx": 7, "n_bars": 3, "label": "B", "prob_A": .1, "prob_B": .9},
    ])
    summary, transitions, _ = classification_structure_summary(data, min_duration_segments=1, max_duration_segments=3)
    assert summary.loc[0, "adjacent_identified_transition_pairs"] == 2
    assert summary.loc[0, "adjacent_self_transition_rate"] == .5
    assert summary.loc[0, "same_label_runs"] == 2
    assert summary.loc[0, "median_same_label_run_bars"] == 4.5  # shared pivot bars count once
    assert transitions.loc[(transitions.from_label == "A") & (transitions.to_label == "A"), "transition_count"].iloc[0] == 1
