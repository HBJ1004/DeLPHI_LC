"""Draw the paired direct-search comparison from its scored evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parents[2]
DATA = FOLLOWUP / "data/direct-workflow-benchmark"
SCORES = (DATA / "score.json", DATA / "score-loaded.json")
BUDGETS = DATA / "budget-equivalence.json"
FIGURE = FOLLOWUP.parent / "paper" / "figures" / "inversion-benchmark.pdf"


def main() -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in SCORES]
    if [row["object_id"] for row in reports[0]["object_rows"]] != [
        row["object_id"] for row in reports[1]["object_rows"]
    ]:
        raise ValueError("cold and loaded runs must contain the same objects")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), constrained_layout=True)
    blue, grey = "#0072B2", "#666666"

    edges = np.geomspace(0.1, 10.0, 41)
    for ax, report, title in zip(
        axes[:2], reports, ("(a) Fresh process for each asteroid", "(b) Models kept loaded"), strict=True
    ):
        rows = report["object_rows"]
        point = report["point_estimates"]
        interval = report["bootstrap_95_intervals"]
        classical_time = np.array([row["arms"]["classical"]["mean_wall_seconds"] for row in rows])
        delphi_time = np.array([row["arms"]["delphi"]["mean_wall_seconds"] for row in rows])
        ratio = classical_time / delphi_time
        ax.axvspan(1.0, edges[-1], color=blue, alpha=0.07, lw=0)
        ax.hist(ratio, bins=edges, color=blue, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(1.0, color=grey, ls="--", lw=1)
        ax.set(xscale="log", xlim=(edges[0], edges[-1]), xlabel="Classical time / DeLPHI time",
               ylabel="Number of asteroids", title=title)
        ax.set_xticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        mixed = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0.9, 0.74, "Classical\nfaster", transform=mixed, ha="right", va="center",
                color=grey, fontsize=9)
        ax.text(1.12, 0.74, "DeLPHI\nfaster", transform=mixed, ha="left", va="center",
                color=blue, fontsize=9)
        speed = interval["speed_ratio_classical_over_delphi"]
        ax.text(0.03, 0.97,
                f"DeLPHI faster for {int(np.sum(ratio > 1))} of {len(ratio)}\n"
                f"Ratio of means {point['speed_ratio_classical_over_delphi']:.2f} "
                f"({speed[0]:.2f}--{speed[1]:.2f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})

    # (c) Agreement with the DAMIT poles against the number of starts (exploratory, saved fits).
    budgets = json.loads(BUDGETS.read_text(encoding="utf-8"))
    starts = np.array([row["starts"] for row in budgets["budgets"]])
    error = np.array([row["mean_error_deg"] for row in budgets["budgets"]])
    delphi = budgets["delphi"]
    ax = axes[2]
    ax.plot(starts, error, "-o", color=grey, ms=4, lw=1.2, label="Classical search")
    ax.axhline(delphi["mean_error_deg"], color=blue, ls=":", lw=1.0)
    ax.plot([delphi["starts"]], [delphi["mean_error_deg"]], "D", color=blue, ms=7,
            label="DeLPHI, six starts")
    ax.set(xscale="log", xlabel="Number of starting poles",
           ylabel="Mean disagreement with DAMIT (deg)", title="(c) Agreement and number of starts")
    ax.set_xticks([6, 12, 24, 48, 96], ["6", "12", "24", "48", "96"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(12, 25)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    for ax in axes:
        ax.tick_params(direction="out", length=3.5, width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
