import numpy as np
import pandas as pd

from wave_segments.validation import (CalibratedSegmentClassifier, PurgedWalkForwardSplit,
                                      resolve_human_annotations, coverage_risk_curve)


def _segments(n=45):
    rows = []
    for sym in ["A", "B"]:
        for i in range(n):
            start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 5)
            rows.append({"symbol": sym, "start": start, "end": start + pd.Timedelta(days=2),
                         "segment_id": f"{sym}-{i}", "f1": (-1) ** i + i / 100, "f2": i % 3,
                         "consensus_label": "UP" if i % 2 else "DOWN"})
    return pd.DataFrame(rows)


def test_purged_walk_forward_has_no_bar_overlap():
    df = _segments()
    folds = list(PurgedWalkForwardSplit(2, min_train_segments=8).split(df))
    assert folds
    for fold in folds:
        assert df.loc[fold.train, "end"].max() < df.loc[fold.calibration, "start"].min()
        assert df.loc[fold.calibration, "end"].max() < df.loc[fold.test, "start"].min()
        for early, later in [(fold.train, fold.calibration), (fold.train, fold.test), (fold.calibration, fold.test)]:
            for _, a in df.loc[early].iterrows():
                for _, b in df.loc[later][df.loc[later].symbol.eq(a.symbol)].iterrows():
                    assert a.end < b.start or a.start > b.end


def test_annotations_calibration_metrics_and_sets():
    df = _segments()
    annotations = pd.DataFrame({"segment_id": ["A-0", "A-0", "A-1", "A-1"],
                                "annotator": ["x", "y", "x", "y"], "label": ["UP", "UP", "UP", "DOWN"]})
    consensus, agreement = resolve_human_annotations(annotations)
    assert consensus.set_index("segment_id").loc["A-1", "consensus_label"] == "UNKNOWN"
    assert agreement["disputed_segments"] == 1
    train, cal, test = df.iloc[:50], df.iloc[50:70], df.iloc[70:]
    model = CalibratedSegmentClassifier(method="sigmoid").fit(train, cal)
    model.choose_unknown_threshold(cal, target_error=.5)
    report = model.evaluate(test)
    assert {"macro_f1", "brier", "ece", "reliability", "coverage_risk"} <= set(report)
    assert report["reliability"]["count"].sum() == len(test)
    assert not coverage_risk_curve(test.consensus_label, model.predict_proba(test), model.classes_).empty
    model.fit_aps(cal)
    assert all(s for s in model.predict_sets(test.iloc[:3]))


def test_inverse_probability_weighted_coverage_and_risk():
    y = np.array(["A", "A", "B"])
    p = np.array([[.9, .1], [.4, .6], [.4, .6]])
    plain = coverage_risk_curve(y, p, ["A", "B"], thresholds=[.5])
    weighted = coverage_risk_curve(y, p, ["A", "B"], thresholds=[.5], sample_weight=[1, 4, 1])
    assert weighted.loc[0, "risk"] > plain.loc[0, "risk"]


def test_selective_evaluation_counts_true_unknowns_in_risk():
    df = _segments()
    train, cal, test = df.iloc[:50], df.iloc[50:70], df.iloc[70:].copy()
    model = CalibratedSegmentClassifier(method="sigmoid").fit(train, cal)
    model.unknown_threshold_ = 0.0
    test.loc[test.index[0], "consensus_label"] = "UNKNOWN"
    report = model.evaluate(test)
    assert report["selective_risk"] >= 1 / len(test)
    assert report["unknown_rejection_rate"] == 0.0
