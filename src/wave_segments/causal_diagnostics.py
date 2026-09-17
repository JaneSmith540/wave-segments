"""Fold-aware diagnostics for past-only, temporary discovery states."""
from __future__ import annotations

import pandas as pd


def build_causal_state_diagnostics(states: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize coverage without treating fold-local cluster IDs as global classes."""
    required = {"symbol", "start", "label", "is_unknown", "unknown_reason", "oos_model_trained_at"}
    if missing := required - set(states.columns):
        raise ValueError(f"causal states missing diagnostic columns: {sorted(missing)}")
    data = states.copy()
    data["start"] = pd.to_datetime(data["start"], errors="coerce")
    data["start_year"] = data["start"].dt.year.astype("Int64").astype(str)
    data["model_version"] = pd.to_datetime(data["oos_model_trained_at"], errors="coerce").astype(str)
    data.loc[data["oos_model_trained_at"].isna(), "model_version"] = "UNTRAINED"

    def coverage(groups: list[str], *, include_fold_labels: bool = False) -> pd.DataFrame:
        rows = []
        for key, part in data.groupby(groups, dropna=False, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            row = dict(zip(groups, key))
            row.update({
                "segments": len(part),
                "identified": int((~part["is_unknown"].astype(bool)).sum()),
                "unknown": int(part["is_unknown"].astype(bool).sum()),
                "coverage": float((~part["is_unknown"].astype(bool)).mean()),
            })
            if include_fold_labels:
                # Cluster names only have meaning inside this fitted model version.
                row["fold_local_labels"] = ",".join(sorted(set(
                    part.loc[~part["is_unknown"].astype(bool), "label"].astype(str)
                )))
            rows.append(row)
        return pd.DataFrame(rows)

    unknown_rows = data.loc[data["is_unknown"].astype(bool), ["symbol", "start_year", "model_version", "unknown_reason"]]
    reasons = unknown_rows.assign(reason=unknown_rows["unknown_reason"].fillna("").str.split(";")).explode("reason")
    reasons = reasons.loc[reasons["reason"].ne("")].groupby(
        ["model_version", "reason"], as_index=False, dropna=False,
    ).size().rename(columns={"size": "segments"}).sort_values(
        ["model_version", "segments", "reason"], ascending=[True, False, True],
    ).reset_index(drop=True)
    by_version = coverage(["model_version"], include_fold_labels=True)
    label_counts = pd.crosstab(data["model_version"], data["label"]).add_prefix("count_").reset_index()
    by_version = by_version.merge(label_counts, on="model_version", how="left")
    if "mixture_convergence_valid" in data:
        fit_status = data.groupby("model_version", as_index=False, dropna=False).agg(
            mixture_convergence_valid=("mixture_convergence_valid", "first"),
        )
        by_version = by_version.merge(fit_status, on="model_version", how="left")
    return {
        "coverage_by_symbol": coverage(["symbol"]),
        "coverage_by_start_year": coverage(["start_year"]),
        "coverage_by_model_version": by_version,
        "unknown_reasons_by_model_version": reasons,
    }
