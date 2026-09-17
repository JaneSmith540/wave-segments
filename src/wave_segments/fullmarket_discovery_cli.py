"""Bounded-memory, offline discovery over full-market segment features."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .discovery import DiscoveryConfig, MultiModelDiscoverer
from .review import create_review_queue


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _feature_parts(root: Path) -> list[Path]:
    paths = sorted(root.glob("bucket=*/data.parquet")) if root.is_dir() else [root]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"No parquet feature parts found under {root}")
    return paths


def _largest_remainder(total: int, weights: dict[int, int]) -> dict[int, int]:
    denominator = sum(weights.values())
    if denominator <= 0:
        return {}
    exact = {key: total * value / denominator for key, value in weights.items()}
    result = {key: int(np.floor(value)) for key, value in exact.items()}
    left = min(total, sum(weights.values())) - sum(result.values())
    for key in sorted(exact, key=lambda x: (-(exact[x] - result[x]), x))[:left]:
        result[key] += 1
    return result


def _fit_sample(parts: list[Path], sample_size: int, seed: int) -> pd.DataFrame:
    """Sample deterministically by year without concatenating the full market."""
    per_part_year: dict[Path, dict[int, int]] = {}
    year_counts: Counter[int] = Counter()
    for path in parts:
        dates = pd.to_datetime(pd.read_parquet(path, columns=["start"])["start"], errors="coerce")
        counts = dates.dt.year.dropna().astype(int).value_counts().to_dict()
        per_part_year[path] = {int(year): int(count) for year, count in counts.items()}
        year_counts.update(per_part_year[path])
    n_rows = sum(year_counts.values())
    target = min(int(sample_size), n_rows)
    quotas = _largest_remainder(target, dict(year_counts))
    chunks = []
    for part_index, path in enumerate(parts):
        frame = pd.read_parquet(path)
        years = pd.to_datetime(frame["start"], errors="coerce").dt.year
        for year, local_count in per_part_year[path].items():
            group = frame.loc[years.eq(year)]
            take = int(np.floor(quotas.get(year, 0) * local_count / year_counts[year]))
            if take:
                chunks.append(group.sample(n=min(take, len(group)), random_state=seed + part_index * 1009 + year))
    if not chunks:
        raise ValueError("Could not select any dated fit rows; segment start timestamps are required")
    return pd.concat(chunks, ignore_index=True).sort_values(["start", "symbol", "segment_id"]).reset_index(drop=True)


def run_fullmarket_discovery(
    features_path: str | Path,
    output_path: str | Path,
    *,
    fit_sample_size: int = 50_000,
    random_state: int = 42,
    n_components: int = 5,
    ensemble_size: int = 5,
    min_cluster_samples: int = 200,
    min_cluster_symbols: int = 50,
    min_cluster_years: int = 3,
    max_iter: int = 300,
    include_hdbscan: bool = True,
) -> dict[str, object]:
    source, output = Path(features_path), Path(output_path)
    if fit_sample_size < 1:
        raise ValueError("fit_sample_size must be positive")
    source_resolved, output_resolved = source.resolve(), output.resolve()
    if source_resolved == output_resolved or (source.is_dir() and source_resolved in output_resolved.parents):
        raise ValueError("discovery output must be separate from, not inside, the feature source")
    parts = _feature_parts(source)
    config = DiscoveryConfig(
        n_components=n_components, random_state=random_state, ensemble_size=ensemble_size,
        min_cluster_samples=min_cluster_samples, min_cluster_symbols=min_cluster_symbols,
        min_cluster_years=min_cluster_years, max_iter=max_iter, include_hdbscan=include_hdbscan,
    )
    settings = {"features_path": str(source.resolve()), "fit_sample_size": fit_sample_size,
                "random_state": random_state, "discovery_config": config.__dict__,
                "mode": "offline_transductive_review_only"}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("settings") != settings:
            raise RuntimeError("discovery settings changed; use a new output directory")
    elif any(output.iterdir()):
        raise RuntimeError("output directory is non-empty and has no discovery manifest; use a new directory")
    manifest = {"format_version": 1, "settings": settings, "status": "running",
                "input_parts": [{"path": str(p.relative_to(source)) if source.is_dir() else p.name,
                                 "bytes": p.stat().st_size} for p in parts]}
    _atomic_json(manifest_path, manifest)

    training = _fit_sample(parts, fit_sample_size, random_state)
    model = MultiModelDiscoverer(config).fit(training)
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - required by project runtime
        raise RuntimeError("joblib is required to persist the fitted discoverer") from exc
    model_path = output / "discoverer.joblib"
    temporary_model = model_path.with_name(f".{model_path.name}.{uuid4().hex}.tmp")
    joblib.dump(model, temporary_model)
    temporary_model.replace(model_path)

    label_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    output_parts = []
    for index, path in enumerate(parts):
        frame = pd.read_parquet(path)
        predicted = model.predict(frame)
        predicted["candidate_status"] = np.where(predicted["is_unknown"], "model_abstention", "classified")
        label_counts.update(predicted["label"].astype(str))
        reason_counts.update(reason for cell in predicted["unknown_reason"].astype(str)
                             for reason in cell.split(";") if reason)
        bucket_match = path.parent.name if source.is_dir() else "part=000"
        target = output / "discovery_labels" / bucket_match / "data.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        predicted.to_parquet(temp, index=False)
        temp.replace(target)
        queue = create_review_queue(predicted)
        review_path = output / "review_candidates" / bucket_match / "data.parquet"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_temp = review_path.with_name(f".{review_path.name}.{uuid4().hex}.tmp")
        queue.to_parquet(review_temp, index=False)
        review_temp.replace(review_path)
        output_parts.append(str(target.relative_to(output)))
        if (index + 1) % 8 == 0 or index + 1 == len(parts):
            print(f"predicted_parts={index + 1}/{len(parts)}", flush=True)

    sample_ids = "\n".join(training["segment_id"].astype(str).tolist()).encode("utf-8")
    summary = {
        "status": "complete", "mode": "offline_transductive_review_only",
        "training_rows": int(len(training)), "training_years": sorted(
            pd.to_datetime(training["start"]).dt.year.dropna().astype(int).unique().tolist()),
        "training_ids_sha256": sha256(sample_ids).hexdigest(),
        "predicted_rows": int(sum(label_counts.values())), "label_counts": dict(label_counts),
        "unknown_reason_counts": dict(reason_counts), "model_metadata": model.metadata(),
        "model_file": model_path.name, "output_parts": output_parts,
        "warning": "Temporary unsupervised clusters from a sample spanning the full historical period; not causal states, human truth, calibrated probabilities, or stock-selection evidence.",
    }
    _atomic_json(output / "summary.json", summary)
    manifest.update({"status": "complete", "training_rows": len(training),
                     "predicted_rows": int(sum(label_counts.values())),
                     "training_ids_sha256": summary["training_ids_sha256"],
                     "model_metadata": model.metadata(), "summary": "summary.json"})
    _atomic_json(manifest_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline full-market wave cluster discovery for review only")
    parser.add_argument("--features", required=True, help="Full-market segment_features parquet dataset or one parquet file")
    parser.add_argument("--output", required=True, help="New output directory; never overwrite the feature source")
    parser.add_argument("--fit-sample-size", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--min-cluster-samples", type=int, default=200)
    parser.add_argument("--min-cluster-symbols", type=int, default=50)
    parser.add_argument("--min-cluster-years", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--disable-hdbscan", action="store_true")
    args = parser.parse_args()
    summary = run_fullmarket_discovery(
        args.features, args.output, fit_sample_size=args.fit_sample_size,
        random_state=args.random_state, n_components=args.components,
        ensemble_size=args.ensemble_size, min_cluster_samples=args.min_cluster_samples,
        min_cluster_symbols=args.min_cluster_symbols, min_cluster_years=args.min_cluster_years,
        max_iter=args.max_iter, include_hdbscan=not args.disable_hdbscan,
    )
    print(f"status=complete rows={summary['predicted_rows']} training_rows={summary['training_rows']} "
          f"labels={summary['label_counts']} output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
