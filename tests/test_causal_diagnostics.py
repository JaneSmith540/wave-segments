import pandas as pd

from wave_segments.causal_diagnostics import build_causal_state_diagnostics


def test_causal_diagnostics_split_by_model_version_and_expand_unknown_reasons():
    states = pd.DataFrame({
        "symbol": ["A", "A", "B", "B"],
        "start": pd.to_datetime(["2021-01-01", "2021-06-01", "2022-01-01", "2022-06-01"]),
        "label": ["CLUSTER_A", "UNKNOWN", "CLUSTER_A", "UNKNOWN"],
        "is_unknown": [False, True, False, True],
        "unknown_reason": ["", "low_density;high_entropy", "", "purged_overlap"],
        "oos_model_trained_at": pd.to_datetime(["2021-04-01", "2021-04-01", "2022-04-01", "2022-04-01"]),
        "mixture_convergence_valid": [True, True, False, False],
    })
    reports = build_causal_state_diagnostics(states)
    by_version = reports["coverage_by_model_version"].set_index("model_version")
    assert len(by_version) == 2
    assert (by_version.coverage == .5).all()
    assert (by_version.fold_local_labels == "CLUSTER_A").all()
    assert (by_version.count_CLUSTER_A == 1).all()
    assert (by_version.count_UNKNOWN == 1).all()
    assert by_version.loc["2021-04-01", "mixture_convergence_valid"]
    assert not by_version.loc["2022-04-01", "mixture_convergence_valid"]
    reasons = reports["unknown_reasons_by_model_version"]
    assert set(reasons.reason) == {"low_density", "high_entropy", "purged_overlap"}
    by_year = reports["coverage_by_start_year"].set_index("start_year")
    assert set(by_year.index) == {"2021", "2022"}
