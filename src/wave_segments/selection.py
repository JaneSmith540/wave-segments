"""Research-only, point-in-time validation for a downstream stock-selection layer.

This module intentionally does *not* create orders or recommendations.  It turns a
frozen, as-of segment-state table into research features and measures whether a
caller-supplied cross-sectional score contains out-of-sample information.  The
classifier remains independent: no future-return target is ever passed back to it.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

_KEYS = ("symbol", "timestamp")


def _stack_all(frame: pd.DataFrame) -> pd.Series:
    """Stack a dense panel including missing cells across pandas versions."""
    try:
        return frame.stack(future_stack=True)
    except TypeError:  # pragma: no cover - pandas < 2.1
        return frame.stack(dropna=False)


def _require(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _rank_ic(group: pd.DataFrame, score: str, target: str) -> float:
    pair = group[[score, target]].replace([np.inf, -np.inf], np.nan).dropna()
    return pair[score].corr(pair[target], method="spearman") if len(pair) >= 3 else np.nan


def _hac_mean_t(values: pd.Series, max_lag: int) -> float:
    """Newey-West t statistic for a serially correlated mean."""
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        return np.nan
    centered = x - x.mean()
    lag = min(max(0, int(max_lag)), n - 1)
    long_run = float(centered @ centered / n)
    for j in range(1, lag + 1):
        gamma = float(centered[j:] @ centered[:-j] / n)
        long_run += 2 * (1 - j / (lag + 1)) * gamma
    se = np.sqrt(max(long_run, 0) / n)
    return float(x.mean() / se) if se > 0 else np.nan


def _neutralize(frame: pd.DataFrame, value: str, industry_col: str | None, size_col: str | None) -> pd.Series:
    """Return same-date OLS residuals; unavailable controls leave the value intact."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, g in frame.groupby("timestamp", sort=False):
        y = pd.to_numeric(g[value], errors="coerce")
        parts = [np.ones(len(g))]
        if size_col and size_col in g:
            parts.append(np.log(pd.to_numeric(g[size_col], errors="coerce").clip(lower=1)).to_numpy())
        if industry_col and industry_col in g:
            dummies = pd.get_dummies(g[industry_col].astype("category"), dtype=float).iloc[:, 1:]
            if not dummies.empty:
                parts.append(dummies.to_numpy())
        X = np.column_stack(parts)
        valid = np.isfinite(y.to_numpy()) & np.isfinite(X).all(axis=1)
        if valid.sum() <= X.shape[1]:
            out.loc[g.index] = y
        else:
            beta = np.linalg.lstsq(X[valid], y.to_numpy()[valid], rcond=None)[0]
            residual = np.full(len(g), np.nan)
            residual[valid] = y.to_numpy()[valid] - X[valid] @ beta
            out.loc[g.index] = residual
    return out


def build_selection_dataset(
    bars: pd.DataFrame,
    states: pd.DataFrame,
    horizons: Sequence[int] = (5, 20, 60),
    benchmark_symbol: str | None = None,
    industry_col: str | None = "industry",
    size_col: str | None = "market_cap",
    allow_same_timestamp_state: bool = False,
) -> pd.DataFrame:
    """Build point-in-time state features and strictly-forward evaluation targets.

    ``states`` needs ``symbol`` and an effective ``timestamp``.  It is backward
    as-of joined, so a state dated *after* a bar can never become a feature of that
    bar.  Target columns are named ``future_excess_{h}d``, ``future_mae_{h}d``,
    ``future_reward_risk_{h}d`` and ``future_rank_{h}d``. Targets use the common
    market-session calendar and are unavailable at feature time.
    """
    _require(bars, [*_KEYS, "close"], "bars")
    _require(states, _KEYS, "states")
    forbidden = [c for c in states.columns if any(t in c.lower() for t in ("future_", "forward_", "fwd_", "next_return", "hsmm_"))]
    if forbidden:
        raise ValueError(f"State features contain future/outcome columns: {forbidden}")
    if not horizons or any(int(h) < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    raw = bars.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.sort_values(["symbol", "timestamp"]).drop_duplicates(_KEYS, keep="last")
    state = states.copy()
    state["timestamp"] = pd.to_datetime(state["timestamp"])
    state = state.sort_values(["symbol", "timestamp"]).drop_duplicates(_KEYS, keep="last")
    # merge_asof only exposes states whose effective timestamp is <= decision date.
    feature = pd.merge_asof(raw.sort_values(["timestamp", "symbol"]), state.sort_values(["timestamp", "symbol"]),
                            on="timestamp", by="symbol", direction="backward",
                            allow_exact_matches=allow_same_timestamp_state, suffixes=("", "_state"))
    feature = feature.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    calendar = pd.Index(raw["timestamp"].drop_duplicates().sort_values())
    close_panel = raw.pivot(index="timestamp", columns="symbol", values="close").reindex(calendar)
    close_lookup = _stack_all(close_panel)
    current_keys = pd.MultiIndex.from_arrays([feature["timestamp"], feature["symbol"]])
    tradable_lookup = None
    if "selection_eligible" in raw:
        tradable_panel = raw.pivot(index="timestamp", columns="symbol", values="selection_eligible").reindex(calendar)
        tradable_lookup = _stack_all(tradable_panel)
    benchmark_returns: dict[int, pd.Series] = {}
    if benchmark_symbol is not None:
        bench = raw.loc[raw["symbol"] == benchmark_symbol, ["timestamp", "close"]].drop_duplicates("timestamp").set_index("timestamp")["close"].reindex(calendar)
        for h in horizons:
            benchmark_returns[h] = bench.shift(-h).div(bench).sub(1)
    for h in horizons:
        h = int(h)
        # Use the common market-session calendar. A suspended/missing future bar
        # stays missing instead of silently extending an h-session horizon.
        future_dates = pd.Series(calendar, index=calendar).shift(-h)
        target_dates = feature["timestamp"].map(future_dates)
        future_keys = pd.MultiIndex.from_arrays([target_dates, feature["symbol"]])
        future_close = pd.Series(close_lookup.reindex(future_keys).to_numpy(), index=feature.index)
        if tradable_lookup is not None:
            feature[f"target_tradable_{h}d"] = pd.Series(
                tradable_lookup.reindex(future_keys).to_numpy(), index=feature.index
            ).astype("boolean")
        gross = future_close.div(feature["close"])
        if "low" in raw:
            low_panel = raw.pivot(index="timestamp", columns="symbol", values="low").reindex(calendar)
            forward_low_panel = low_panel.shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1))
            forward_low = pd.Series(
                _stack_all(forward_low_panel).reindex(current_keys).to_numpy(), index=feature.index
            )
        else:
            forward_low = pd.Series(np.nan, index=feature.index)
        future_return = gross.sub(1)
        if benchmark_symbol is None:
            excess = future_return
        else:
            excess = future_return.sub(feature["timestamp"].map(benchmark_returns[h]))
        feature[f"future_excess_{h}d"] = excess
        feature[f"future_mae_{h}d"] = pd.Series(forward_low, index=feature.index).div(feature["close"]).sub(1)
        denom = feature[f"future_mae_{h}d"].abs().replace(0, np.nan)
        feature[f"future_reward_risk_{h}d"] = excess.div(denom)
        feature[f"future_rank_{h}d"] = feature.groupby("timestamp")[f"future_excess_{h}d"].rank(pct=True)
        # A target for the decision at t is not known until market session t+h.
        feature[f"target_available_at_{h}d"] = target_dates
    if industry_col and industry_col in feature:
        feature[industry_col] = feature[industry_col].astype("category")
    return feature.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _target_available_column(target_col: str) -> str:
    """Resolve the availability column belonging to a forward target name."""
    import re

    match = re.search(r"_(\d+)d$", target_col)
    if not match:
        raise ValueError(
            "target_col must end in a horizon such as 'future_excess_20d'; "
            "pass target_available_col explicitly otherwise"
        )
    return f"target_available_at_{match.group(1)}d"


def _ridge_fit(X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a small dependency-free ridge with an unpenalised intercept."""
    center = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = (X_train - center) / scale
    y_mean = float(y_train.mean())
    gram = train.T @ train + float(alpha) * np.eye(train.shape[1])
    coef = np.linalg.pinv(gram) @ train.T @ (y_train - y_mean)
    return center, scale, coef, y_mean


def _ridge_predict(model: tuple[np.ndarray, np.ndarray, np.ndarray, float], X: np.ndarray) -> np.ndarray:
    center, scale, coef, y_mean = model
    return ((X - center) / scale) @ coef + y_mean


def _ridge_fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_predict: np.ndarray, alpha: float
) -> np.ndarray:
    return _ridge_predict(_ridge_fit(X_train, y_train, alpha), X_predict)


def walk_forward_cross_sectional_scores(
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "future_excess_20d",
    *,
    target_available_col: str | None = None,
    min_train_dates: int = 60,
    min_train_rows: int | None = None,
    retrain_every: int = 5,
    ridge_alpha: float = 1.0,
    eligibility_col: str | None = None,
) -> pd.DataFrame:
    """Produce causally-valid, cross-sectional out-of-sample ridge scores.

    For each decision date, a fitted model may use only rows whose target was
    available *strictly before* that date.  This guards against both ordinary
    look-ahead and the subtler overlap created by forward-return horizons.  The
    input is never modified; returned ``oos_score`` is a downstream research
    artifact and is deliberately not fed into the state classifier.

    ``min_train_dates`` is counted after applying the availability filter.  A
    model is refit every ``retrain_every`` decision dates; a cached model is only
    reused after it has been established using the same causal rule.
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    if min_train_dates < 1 or retrain_every < 1 or ridge_alpha < 0:
        raise ValueError("min_train_dates/retrain_every must be positive and ridge_alpha non-negative")
    available_col = target_available_col or _target_available_column(target_col)
    _require(dataset, [*_KEYS, target_col, available_col, *feature_cols], "dataset")
    minimum_rows = int(min_train_rows) if min_train_rows is not None else max(10, len(feature_cols) + 2)
    if minimum_rows < len(feature_cols) + 2:
        raise ValueError("min_train_rows must allow an intercept plus the requested features")

    data = dataset.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data[available_col] = pd.to_datetime(data[available_col])
    data = data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    values = data.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(data[target_col], errors="coerce")
    valid = values.notna().all(axis=1) & target.notna() & data[available_col].notna()
    active_eligibility = eligibility_col or ("selection_eligible" if "selection_eligible" in data else None)
    if active_eligibility:
        _require(data, [active_eligibility], "dataset")
        valid &= data[active_eligibility].fillna(False).astype(bool)
    import re
    horizon = re.search(r"_(\d+)d$", target_col)
    target_tradable_col = f"target_tradable_{horizon.group(1)}d" if horizon else None
    if target_tradable_col and target_tradable_col in data:
        valid &= data[target_tradable_col].fillna(False).astype(bool)
    decision_dates = pd.Index(data["timestamp"].drop_duplicates().sort_values())
    result = data.copy()
    result["oos_score"] = np.nan
    result["oos_model_trained_at"] = pd.NaT
    result["oos_train_rows"] = 0
    result["oos_train_max_target_available_at"] = pd.NaT

    fitted: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float], pd.Timestamp, int, pd.Timestamp] | None = None
    for date_index, decision_date in enumerate(decision_dates):
        # Re-evaluate availability at every decision date.  The fit itself is
        # only refreshed periodically, but no labels become visible early.
        eligible = valid & (data[available_col] < decision_date)
        eligible_dates = data.loc[eligible, "timestamp"].nunique()
        should_refit = fitted is None or date_index % retrain_every == 0
        if should_refit and eligible_dates >= min_train_dates and int(eligible.sum()) >= minimum_rows:
            X_train = values.loc[eligible].to_numpy(dtype=float)
            y_train = target.loc[eligible].to_numpy(dtype=float)
            # Cache the fitted coefficients until the next scheduled refit.
            fitted = (_ridge_fit(X_train, y_train, ridge_alpha), decision_date,
                      int(eligible.sum()), data.loc[eligible, available_col].max())
        if fitted is None:
            continue
        day_rows = data["timestamp"].eq(decision_date) & values.notna().all(axis=1)
        if active_eligibility:
            day_rows &= data[active_eligibility].fillna(False).astype(bool)
        if not day_rows.any():
            continue
        ridge_model, trained_at, train_rows, max_available = fitted
        result.loc[day_rows, "oos_score"] = _ridge_predict(
            ridge_model, values.loc[day_rows].to_numpy(dtype=float)
        )
        result.loc[day_rows, "oos_model_trained_at"] = trained_at
        result.loc[day_rows, "oos_train_rows"] = train_rows
        result.loc[day_rows, "oos_train_max_target_available_at"] = max_available
    return result


# Short alias for notebooks; the explicit name above is preferred in production.
walk_forward_scores = walk_forward_cross_sectional_scores


def evaluate_selection(
    dataset: pd.DataFrame,
    score_col: str,
    target_col: str = "future_excess_20d",
    groups: int = 5,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    industry_col: str | None = "industry",
    size_col: str | None = "market_cap",
    rebalance_every: int | None = None,
    eligibility_col: str | None = None,
) -> dict[str, object]:
    """Evaluate an externally supplied score; returns metrics only, never trades."""
    _require(dataset, [*_KEYS, score_col, target_col], "dataset")
    if groups < 2:
        raise ValueError("groups must be at least 2")
    data = dataset.copy().sort_values(["timestamp", "symbol"])
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    active_eligibility = eligibility_col or ("selection_eligible" if "selection_eligible" in data else None)
    if active_eligibility:
        _require(data, [active_eligibility], "dataset")
        data = data[data[active_eligibility].fillna(False).astype(bool)].copy()
    import re
    horizon_match = re.search(r"_(\d+)d$", target_col)
    target_tradable_col = f"target_tradable_{horizon_match.group(1)}d" if horizon_match else None
    if target_tradable_col and target_tradable_col in data:
        data = data[data[target_tradable_col].fillna(False).astype(bool)].copy()
    data["score_neutral"] = _neutralize(data, score_col, industry_col, size_col)
    data["target_neutral"] = _neutralize(data, target_col, industry_col, size_col)
    data["group"] = data.groupby("timestamp")["score_neutral"].transform(
        lambda x: pd.qcut(x.rank(method="first"), groups, labels=False, duplicates="drop") if x.notna().sum() >= groups else np.nan)
    # Do not rely on pandas 2.2's ``include_groups`` option: the project supports
    # pandas 2.0 too, and the helper selects only the two relevant columns.
    ic = data.groupby("timestamp", group_keys=False)[["score_neutral", "target_neutral"]].apply(
        lambda g: _rank_ic(g, "score_neutral", "target_neutral")
    ).rename("rank_ic")
    grouped = data.dropna(subset=["group", target_col]).groupby(["timestamp", "group"])[target_col].mean().unstack("group")
    group_returns = grouped.mean(axis=0).rename("mean_return")
    monotonicity = group_returns.corr(pd.Series(group_returns.index, index=group_returns.index), method="spearman") if len(group_returns) > 1 else np.nan
    if rebalance_every is None:
        import re
        match = re.search(r"_(\d+)d$", target_col)
        rebalance_every = int(match.group(1)) if match else 1
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be positive")
    all_dates = pd.Index(data["timestamp"].drop_duplicates().sort_values())
    rebalance_dates = set(all_dates[::rebalance_every])
    weights = data[data["timestamp"].isin(rebalance_dates)].dropna(subset=["group"]).copy()
    extrema = weights.groupby("timestamp")["group"].agg(["min", "max"])
    weights = weights.join(extrema, on="timestamp")
    weights = weights[weights["group"].eq(weights["min"]) | weights["group"].eq(weights["max"])].copy()
    counts = weights.groupby(["timestamp", "group"])["symbol"].transform("size")
    weights["weight"] = np.where(weights["group"].eq(weights["max"]), 1.0 / counts, -1.0 / counts)
    pivot = weights.pivot_table(index="timestamp", columns="symbol", values="weight", aggfunc="last").fillna(0)
    turnover = pivot.diff().abs().sum(axis=1).div(2).fillna(0.0)
    non_overlapping = grouped[grouped.index.isin(rebalance_dates)]
    gross_long_short = non_overlapping.iloc[:, -1].sub(non_overlapping.iloc[:, 0]) if non_overlapping.shape[1] >= 2 else pd.Series(dtype=float)
    costs = turnover.reindex(gross_long_short.index).fillna(0).mul((transaction_cost_bps + slippage_bps) / 1e4)
    net = gross_long_short.sub(costs)
    wealth = (1 + net.fillna(0)).cumprod()
    max_drawdown = float((wealth / wealth.cummax() - 1).min()) if len(wealth) else np.nan
    year = pd.to_datetime(ic.index).year
    yearly = pd.DataFrame({"rank_ic": ic, "year": year}).groupby("year")["rank_ic"].agg(["mean", "std", "count"])
    yearly["icir"] = yearly["mean"].div(yearly["std"].replace(0, np.nan))
    regime = None
    if "market_regime" in data:
        regime = pd.Series({
            str(name): _rank_ic(group, "score_neutral", "target_neutral")
            for name, group in data.groupby("market_regime", observed=True)
        }, name="rank_ic")
    return {
        "daily_rank_ic": ic, "rank_ic": float(ic.mean()), "icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) else np.nan,
        "rank_ic_hac_t": _hac_mean_t(ic, rebalance_every - 1),
        "rank_ic_positive_rate": float(ic.dropna().gt(0).mean()) if ic.notna().any() else np.nan,
        "group_returns": group_returns, "group_returns_by_date": grouped, "monotonicity": float(monotonicity),
        "turnover": turnover, "gross_long_short": gross_long_short, "net_long_short": net,
        "mean_net_long_short": float(net.mean()) if len(net) else np.nan,
        "max_drawdown": max_drawdown, "yearly_stability": yearly,
        "rebalance_every": rebalance_every,
        "regime_rank_ic": regime, "dataset": data,
    }


def compare_baselines(dataset: pd.DataFrame, score_cols: Sequence[str], **evaluation_kwargs: object) -> pd.DataFrame:
    """Compare state scores with momentum/volatility baselines using identical targets."""
    rows = []
    for score in score_cols:
        result = evaluate_selection(dataset, score, **evaluation_kwargs)
        rows.append({"score": score, "rank_ic": result["rank_ic"], "icir": result["icir"],
                     "monotonicity": result["monotonicity"], "mean_net_long_short": result["mean_net_long_short"]})
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def evaluate_incremental_features(
    dataset: pd.DataFrame,
    baseline_features: Sequence[str],
    wave_features: Sequence[str],
    target_col: str = "future_excess_20d",
    *,
    min_train_dates: int = 60,
    min_train_rows: int | None = None,
    retrain_every: int = 5,
    ridge_alpha: float = 1.0,
    groups: int = 5,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, object]:
    """Paired OOS ablation: classic baseline versus baseline plus wave features.

    Both models use identical causal walk-forward rules and are evaluated only
    on their common scored rows. The HAC t statistic of the *daily IC
    difference* tests incremental information rather than comparing two
    unrelated headline IC values.
    """
    if not baseline_features or not wave_features:
        raise ValueError("baseline_features and wave_features must both be non-empty")
    augmented_features = list(dict.fromkeys([*baseline_features, *wave_features]))
    common = {
        "min_train_dates": min_train_dates,
        "min_train_rows": min_train_rows,
        "retrain_every": retrain_every,
        "ridge_alpha": ridge_alpha,
    }
    baseline = walk_forward_cross_sectional_scores(dataset, baseline_features, target_col, **common)
    augmented = walk_forward_cross_sectional_scores(dataset, augmented_features, target_col, **common)
    if not baseline.loc[:, _KEYS].equals(augmented.loc[:, _KEYS]):
        raise RuntimeError("walk-forward ablation produced misaligned rows")
    paired = dataset.copy().reset_index(drop=True)
    paired["baseline_oos_score"] = baseline["oos_score"].to_numpy()
    paired["augmented_oos_score"] = augmented["oos_score"].to_numpy()
    common_rows = paired["baseline_oos_score"].notna() & paired["augmented_oos_score"].notna()
    paired.loc[~common_rows, ["baseline_oos_score", "augmented_oos_score"]] = np.nan
    eval_kwargs = {
        "target_col": target_col,
        "groups": groups,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
    }
    baseline_report = evaluate_selection(paired, "baseline_oos_score", **eval_kwargs)
    augmented_report = evaluate_selection(paired, "augmented_oos_score", **eval_kwargs)
    daily = pd.concat({"baseline": baseline_report["daily_rank_ic"],
                       "augmented": augmented_report["daily_rank_ic"]}, axis=1).dropna()
    daily["incremental"] = daily["augmented"] - daily["baseline"]
    horizon_lag = max(0, int(baseline_report["rebalance_every"]) - 1)
    target_available = _target_available_column(target_col)
    base_scored = baseline["oos_score"].notna()
    aug_scored = augmented["oos_score"].notna()
    base_causal = bool((baseline.loc[base_scored, "oos_train_max_target_available_at"]
                        < baseline.loc[base_scored, "timestamp"]).all())
    aug_causal = bool((augmented.loc[aug_scored, "oos_train_max_target_available_at"]
                       < augmented.loc[aug_scored, "timestamp"]).all())
    # Presence check documents which availability series governed both fits.
    _require(dataset, [target_available], "dataset")
    delta = float(daily["incremental"].mean()) if len(daily) else np.nan
    delta_t = _hac_mean_t(daily["incremental"], horizon_lag)
    return {
        "baseline": baseline_report,
        "augmented": augmented_report,
        "daily_rank_ic_comparison": daily,
        "incremental_rank_ic": delta,
        "incremental_rank_ic_hac_t": delta_t,
        "overlap_scored_rows": int(common_rows.sum()),
        "baseline_causal_audit_passed": base_causal,
        "augmented_causal_audit_passed": aug_causal,
        "incremental_evidence_passed": bool(np.isfinite(delta) and delta > 0 and np.isfinite(delta_t) and delta_t >= 2.0),
    }


def assess_selection_evidence(
    report: dict[str, object],
    *,
    classification_metrics: dict[str, object] | None = None,
    min_scored_rows: int = 1000,
    min_rank_ic_hac_t: float = 2.0,
    min_rank_ic_positive_rate: float = 0.55,
    min_monotonicity: float = 0.50,
    max_selective_risk: float = 0.10,
    max_ece: float = 0.10,
    min_classification_coverage: float = 0.50,
    min_unknown_rejection_rate: float = 0.50,
) -> dict[str, object]:
    """Apply conservative research admission gates; this never emits orders.

    The defaults are deliberately demanding and should be pre-registered before
    looking at a new holdout. Passing means "worth a larger independent test",
    not that a signal is deployable or economically causal.
    """
    checks: dict[str, bool] = {}

    def finite(name: str) -> float | None:
        value = report.get(name)
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    checks["positive_rank_ic"] = (finite("rank_ic") or 0.0) > 0
    checks["hac_significance"] = (finite("rank_ic_hac_t") or float("-inf")) >= min_rank_ic_hac_t
    checks["positive_rate"] = (finite("rank_ic_positive_rate") or 0.0) >= min_rank_ic_positive_rate
    checks["quantile_monotonicity"] = (finite("monotonicity") or float("-inf")) >= min_monotonicity
    checks["positive_net_long_short"] = (finite("mean_net_long_short") or 0.0) > 0
    if "scored_rows" in report:
        checks["sample_size"] = (finite("scored_rows") or 0.0) >= min_scored_rows
    if "causal_audit_passed" in report:
        checks["causal_audit"] = bool(report["causal_audit_passed"])

    if classification_metrics is not None:
        def class_value(name: str) -> float | None:
            value = classification_metrics.get(name)
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return number if np.isfinite(number) else None

        checks["classification_selective_risk"] = (
            class_value("selective_risk") is not None and
            class_value("selective_risk") <= max_selective_risk  # type: ignore[operator]
        )
        checks["classification_calibration"] = (
            class_value("ece") is not None and class_value("ece") <= max_ece  # type: ignore[operator]
        )
        checks["classification_coverage"] = (
            class_value("coverage") is not None and
            class_value("coverage") >= min_classification_coverage  # type: ignore[operator]
        )
        checks["unknown_rejection"] = (
            class_value("unknown_rejection_rate") is not None and
            class_value("unknown_rejection_rate") >= min_unknown_rejection_rate  # type: ignore[operator]
        )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "status": "research_candidate" if not failed else "insufficient_evidence",
        "failed_checks": failed,
        "checks": checks,
        "disclaimer": "Admission to a larger independent test only; not a prediction, order, or trading signal.",
    }
