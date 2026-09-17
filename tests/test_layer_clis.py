import json
import sys

import numpy as np
import pandas as pd

from wave_segments.classification_cli import main as classification_main
from wave_segments.selection_cli import main as selection_main


def test_classification_cli_writes_only_fold_report(tmp_path, monkeypatch):
    rows, annotations = [], []
    for symbol in ("A", "B"):
        for i in range(30):
            start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 4)
            segment_id, label = f"{symbol}-{i}", "UP" if i % 2 else "DOWN"
            rows.append({"segment_id": segment_id, "symbol": symbol, "start": start,
                         "end": start + pd.Timedelta(days=2), "feature_a": (-1) ** i, "feature_b": i % 3})
            annotations += [{"segment_id": segment_id, "annotator": reviewer, "label": label} for reviewer in ("x", "y")]
    features, labels, output = tmp_path / "features.parquet", tmp_path / "labels.csv", tmp_path / "report.json"
    pd.DataFrame(rows).to_parquet(features, index=False)
    pd.DataFrame(annotations).to_csv(labels, index=False)
    monkeypatch.setattr(sys, "argv", ["wave-validate-classifier", "--features", str(features),
                        "--annotations", str(labels), "--output", str(output), "--splits", "1",
                        "--target-error", "0.6", "--min-coverage", "0"])
    classification_main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["folds"]) == 1
    assert "agreement" in report


def test_selection_cli_writes_research_warning(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    bars, states = [], []
    for rank, symbol in enumerate(["A", "B", "C", "D", "E"]):
        for i, date in enumerate(dates):
            close = 100 + rank + i * (1 + rank / 10)
            bars.append({"symbol": symbol, "timestamp": date, "close": close, "low": close * .99})
            states.append({"symbol": symbol, "timestamp": date, "state_score": float(rank)})
    bars_path, states_path, output = tmp_path / "bars.parquet", tmp_path / "states.parquet", tmp_path / "selection.json"
    pd.DataFrame(bars).to_parquet(bars_path, index=False)
    pd.DataFrame(states).to_parquet(states_path, index=False)
    monkeypatch.setattr(sys, "argv", ["wave-validate-selection", "--bars", str(bars_path),
                        "--states", str(states_path), "--scores", "state_score", "--horizons", "2",
                        "--groups", "3", "--output", str(output)])
    selection_main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "state_score@2d" in report["results"]
    assert "Research metrics only" in report["warning"]
