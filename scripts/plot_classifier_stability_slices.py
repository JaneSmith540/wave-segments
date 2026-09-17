"""Plot year and causal market-proxy summaries from the symbol bootstrap report."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    root = Path(args.report_dir)
    year = pd.read_csv(root / "stability_by_year.csv")
    regime = pd.read_csv(root / "stability_by_market_regime.csv")
    year = year.sort_values("slice_value", key=lambda values: values.astype(int))
    regime_order = ["BULL_PROXY", "SIDEWAYS_PROXY", "BEAR_PROXY", "UNKNOWN"]
    regime["_order"] = regime.slice_value.map({name: i for i, name in enumerate(regime_order)}).fillna(99)
    regime = regime.sort_values("_order")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    axes[0].plot(year.slice_value, year.identified_label_agreement_row_weighted,
                 marker="o", label="identified-class agreement")
    axes[0].plot(year.slice_value, year.coverage_after_bootstrap_row_weighted,
                 marker="s", label="coverage after resampling")
    axes[0].set(title="By state-availability year", xlabel="Year", ylabel="Rate", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()

    positions = range(len(regime))
    axes[1].bar([x - .18 for x in positions], regime.identified_label_agreement_row_weighted,
                width=.36, label="identified-class agreement")
    axes[1].bar([x + .18 for x in positions], regime.coverage_after_bootstrap_row_weighted,
                width=.36, label="coverage after resampling")
    axes[1].set(title="By causal sample-market proxy", xlabel="Trailing 20-session regime",
                ylabel="Rate", ylim=(0, 1), xticks=list(positions),
                xticklabels=regime.slice_value)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Training-resample stability slices — not classification accuracy")
    destination = root / "stability_by_year_and_regime.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
