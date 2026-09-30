"""Draw the paired direct-search comparison from its scored evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


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

    for ax, report, title in zip(
        axes[:2], reports, ("(a) Fresh process for each asteroid", "(b) Models kept loaded"), strict=True
    ):
        rows = report["object_rows"]
        point = report["point_estimates"]
        interval = report["bootstrap_95_intervals"]
        classical_time = np.array([row["arms"]["classical"]["mean_wall_seconds"] for row in rows])
        delphi_time = np.array([row["arms"]["delphi"]["mean_wall_seconds"] for row in rows])
        ax.scatter(classical_time, delphi_time, s=18, color=blue, alpha=0.7,
                   edgecolors="white", linewidths=0.3)
        lower = 0.8 * min(classical_time.min(), delphi_time.min())
        upper = 1.2 * max(classical_time.max(), delphi_time.max())
        ax.plot([lower, upper], [lower, upper], "--", color=grey, lw=1)
        ax.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper),
               xlabel="Classical search time (s)", ylabel="DeLPHI search time (s)", title=title)
        ticks = [tick for tick in (1, 3, 10, 30) if lower <= tick <= upper]
        ax.set_xticks(ticks, [str(tick) for tick in ticks])
        ax.set_yticks(ticks, [str(tick) for tick in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_aspect("equal", adjustable="box")
        speed = interval["speed_ratio_classical_over_delphi"]
        ax.text(0.03, 0.96,
                f"Ratio of means {point['speed_ratio_classical_over_delphi']:.2f}\n"
                f"95% interval {speed[0]:.2f}--{speed[1]:.2f}",
                transform=ax.transAxes, va="top", ha="left",
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
