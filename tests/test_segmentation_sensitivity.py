import pandas as pd

from wave_segments.segmentation_sensitivity import (
    compare_causal_boundaries,
    segment_distribution,
)


def test_boundary_parameter_agreement_within_tolerance_and_year_breakdown():
    reference = pd.DataFrame([
        {"symbol": "A", "start_idx": 0, "end_idx": 10, "start": "2020-12-01", "end": "2020-12-28", "n_bars": 11},
        {"symbol": "A", "start_idx": 10, "end_idx": 20, "start": "2020-12-28", "end": "2021-01-11", "n_bars": 11},
        {"symbol": "A", "start_idx": 20, "end_idx": 30, "start": "2021-01-11", "end": "2021-02-01", "n_bars": 11},
        {"symbol": "A", "start_idx": 30, "end_idx": 40, "start": "2021-02-01", "end": "2021-02-15", "n_bars": 11},
    ])
    changed = pd.DataFrame([
        {"symbol": "A", "start_idx": 0, "end_idx": 11, "start": "2020-12-01", "end": "2020-12-29", "n_bars": 12},
        {"symbol": "A", "start_idx": 11, "end_idx": 19, "start": "2020-12-29", "end": "2021-01-08", "n_bars": 9},
        {"symbol": "A", "start_idx": 19, "end_idx": 40, "start": "2021-01-08", "end": "2021-02-15", "n_bars": 22},
    ])
    overall, by_symbol, by_year = compare_causal_boundaries(reference, changed, tolerance=1)
    assert overall["matches"] == 3
    assert overall["reference_boundaries"] == 4
    assert overall["observed_boundaries"] == 3
    assert overall["precision_vs_reference"] == 1
    assert overall["recall_vs_reference"] == 3 / 4
    assert by_symbol.iloc[0].f1_agreement == overall["f1_agreement"]
    assert set(by_year.year) == {2020, 2021}


def test_segment_distribution_reports_variable_length_and_short_share():
    summary = segment_distribution(pd.DataFrame({"n_bars": [3, 5, 10]}), short_threshold=5)
    assert summary["segments"] == 3
    assert summary["median_bars"] == 5
    assert summary["short_segment_fraction"] == 1 / 3
