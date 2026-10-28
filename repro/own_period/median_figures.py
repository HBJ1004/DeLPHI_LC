"""Paper I median figures, from frozen evidence only (no inference or fitting).

F1 and F2 each supply six independent figure functions. A partial invocation
preserves the other entries in values.json. Example::

    taskset -c 12-23 nice -n 19 python -B -m repro.own_period.median_figures \
        --figures generalization-checks sampling-caps data-degradation latitude \
        candidate-count inversion-benchmark --out-dir /path/to/scratch/median-figures

Renderer bodies are extracted with AST, so worktree renderers' module-level
mkdir, argument parsing, and output paths are never executed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
FOLLOWUP = REPO.parent
DATA = FOLLOWUP / "data/own-period-20261021"
PAPER = FOLLOWUP.parent / "paper-rev-20261021"
SCRATCH = Path(
    "/tmp/claude-1000/-mnt-d-Downloads-Colab-Notebooks/"
    "3fa212d5-e372-4122-b9a4-788df746770f/scratchpad/p1map"
)
DEFAULT_OUT = SCRATCH / "median-figures"
MAPS_RENDERER = FOLLOWUP / "source-op-maps/repro/own_period/maps/render_figures.py"
LOWQ_RENDERER = FOLLOWUP / "source-op-lowq/repro/own_period/lowq_figures/render_main_figures.py"
RELIABILITY_RENDERER = DATA / "reliability/figures/render_main_figures.py"
INVERSION_RENDERER = REPO / "repro/direct_workflow_benchmark/plot_direct_workflow.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_source(path, name):
    source = path.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Renderer changed: expected exactly one {old!r}")
    return source.replace(old, new)


class Context:
    def __init__(self, out):
        from repro.own_period.median_stats import Evidence, bootstrap_indices, compute_statistics

        self.out = out
        self.evidence = Evidence()
        self.indices = bootstrap_indices
        self.statistics = compute_statistics
        self.frozen = self.read(DATA / "median-stats.json")
        self.macros = {r["mean_macro"]: r for r in self.frozen["entries"]}
        self.resamples = self.frozen["bootstrap_replicates"]
        self.sources = {}
        self.results = {}
        self.source(REPO / "repro/own_period/median_stats.py")
        self.source(REPO / "repro/compute_reference_baselines.py")
        self.source(DATA / "median-stats.json")
        self.source(PAPER / "sample701.tex")

    def source(self, path):
        path = Path(path)
        if str(path) not in self.sources:
            self.sources[str(path)] = digest(path)

    def read(self, path):
        # self.sources is initialized after the initial median-stats read.
        if hasattr(self, "sources"):
            self.source(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def figure(self, name):
        self.source(PAPER / "figures" / f"{name}.pdf")
        self.results[name] = {"points": [], "checks": [], "source_files": [],
                              "means_retained": [], "visual_message": "Unchanged."}
        return self.results[name]

    def check(self, result, label, actual, expected):
        passed = bool(np.allclose(actual, expected, atol=1e-10, rtol=0))
        result["checks"].append(dict(label=label, actual=actual, expected=expected,
                                      status="PASS" if passed else "FAIL"))
        if not passed:
            raise ValueError(f"{label}: {actual!r} != {expected!r}")

    def match(self, result, point, macro, *, interval=True):
        frozen = self.macros[macro]
        self.check(result, macro + ": median", point["median"], frozen["median"])
        self.check(result, macro + ": n", point["n"], frozen["n"])
        if "object_ids" in point:
            if set(point["object_ids"]) != set(frozen["object_ids"]):
                raise ValueError(f"{macro}: asteroid identities changed")
        if interval and frozen.get("interval") is not None:
            self.check(result, macro + ": CI95 endpoints", point["interval"], frozen["interval"])
        point.setdefault("median_stats_macros", []).append(macro)

    def stat(self, first, *, second=None, seed=None, folds=None, order="iid", rng=None):
        first = np.asarray(first, float)
        indices = None
        if seed is not None:
            indices = self.indices(len(first), seed, self.resamples, folds=folds, order=order)
        elif rng is not None:
            indices = rng.integers(0, len(first), (self.resamples, len(first)))
        result = self.statistics(first, second, indices=indices)
        if indices is not None:
            result["bootstrap"] = dict(seed=seed if seed is not None else 20260929,
                                       resamples=self.resamples, rng_and_order=order,
                                       unit="asteroid_within_fold" if folds is not None else "asteroid")
        return result

    def sources_for(self, result, paths):
        for path in paths:
            self.source(path)
            if str(path) not in result["source_files"]:
                result["source_files"].append(str(path))

    def canvas(self, name, rects):
        """Preserve the paper PDF's media box and normalized axes positions."""
        import matplotlib.pyplot as plt

        info = subprocess.check_output(["pdfinfo", str(PAPER / "figures" / f"{name}.pdf")],
                                       text=True)
        line = next(s for s in info.splitlines() if s.startswith("Page size:"))
        parts = line.split()
        size = [float(parts[2]), float(parts[4])]
        fig = plt.figure(figsize=np.array(size) / 72)
        axes = [fig.add_axes(rect) for rect in rects]
        self.results[name]["page_size_pt"] = size
        return fig, axes

    def save(self, fig, name):
        import matplotlib.pyplot as plt

        # No automatic cropping: the first five PDFs retain the paper media box.
        fig.savefig(self.out / f"{name}.pdf", bbox_inches=None, facecolor="white")
        fig.savefig(self.out / f"{name}.png", bbox_inches=None, dpi=200, facecolor="white")
        self.results[name]["axes"] = [dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                                           xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel())
                                      for ax in fig.axes]
        plt.close(fig)


def generalization_checks(ctx):
    """Six paired differences of medians, keeping the current successful cohorts."""
    result = ctx.figure("generalization-checks")
    main = ctx.evidence.pair_from_score("KThreeStandardImprovement")
    lowq = [ctx.evidence.additional_statistic(name, ctx.macros[name]) for name in
            ("FollowLowqFoundDifference", "FollowLowqGaiaFoundDifference",
             "FollowLowqAsasFoundDifference")]
    external = [ctx.evidence.pair_from_score(name) for name in
                ("FollowZtfFoundDifference", "FollowAlcdefStandardDifference")]
    rows = [main, *lowq, *external]
    macros = ["KThreeStandardImprovement", "FollowLowqFoundDifference",
              "FollowLowqGaiaFoundDifference", "FollowLowqAsasFoundDifference",
              "FollowZtfFoundDifference", "FollowAlcdefStandardDifference"]
    labels = ["Main DAMIT test asteroids (170/170 with a period)",
              "All lower-quality DAMIT (4,077/9,783 with a period)",
              "Gaia DR3 lower-quality subset (1,647/7,273 with a period)",
              "ASAS-SN lower-quality subset (94/147 with a period)",
              "Mapped ZTF (108/161 with a period)", "ALCDEF–Gaia (30/30 with a period)"]
    for row, macro, label in zip(rows, macros, labels, strict=True):
        ctx.match(result, row, macro)
        row["label"] = label
        ctx.sources_for(result, row["source_files"])
    result["points"] = rows
    # Use the frozen renderer's artist statements, with prepared median vectors.
    source = function_source(LOWQ_RENDERER, "render_transfer")
    source = "def draw():\n" + source[source.index("    y = np.arange"):]
    source = replace_once(source, '    fig, ax = plt.subplots(figsize=(6.4, 3.5))',
                          '    fig, (ax,) = ctx.canvas("generalization-checks", '
                          '[[0.395, 0.143, 0.593, 0.827]])')
    source = replace_once(source, "Standard-pole error minus DeLPHI error (degrees)",
                          "Standard-pole median minus DeLPHI median (degrees)")
    source = source[:source.index('    ax.set_title')]
    source += '    ax.grid(axis="x", alpha=0.2)\n'
    # Preserve the old automatic limits; extend only to include new CI endpoints.
    old_low = min(row["mean_interval_recomputed"][0] for row in rows)
    old_high = max(row["mean_interval_recomputed"][1] for row in rows)
    pad = (old_high - old_low) * 0.05
    difference = np.array([r["median"] for r in rows])
    low, high = np.array([r["interval"] for r in rows]).T
    limits = [min(old_low - pad, low.min() - pad), max(old_high + pad, high.max() + pad)]
    source += '    ax.set_xlim(limits)\n    ax.set_ylim(-0.25, 5.25)\n'
    source += '    ctx.save(fig, "generalization-checks")\n'
    import matplotlib.pyplot as plt

    namespace = dict(ctx=ctx, np=np, plt=plt, labels=labels, difference=difference,
                     low=low, high=high, limits=limits)
    exec(compile(source, str(LOWQ_RENDERER), "exec"), namespace)
    namespace["draw"]()
    ctx.sources_for(result, [LOWQ_RENDERER])
    result["caption_change"] = ('Replace "Mean oracle error ... minus that of DeLPHI" with '
                                 '"Median oracle error ... minus that of DeLPHI"; '
                                 'specify paired 95% bootstrap intervals for the difference of medians.')
    result["visual_message"] = (
        "All lower-quality DAMIT changes from red (mean difference -2.05 deg) to blue "
        "(median difference +0.91 deg, CI +0.13 to +1.59). ASAS-SN and ALCDEF "
        "median intervals include zero; their mean intervals were entirely negative."
    )


def reliability_condition(ctx, condition, *, bootstrap=False):
    """Average successful repeats per asteroid before taking its cohort median."""
    paired_path = DATA / "failed-excluded-derived/reliability-paired.json"
    scored_path = DATA / "reliability/score/scored-rows-own.json"
    paired = ctx.read(paired_path)["conditions"][condition]
    rows = paired.get("objects", [])
    result = dict(condition=condition, n=paired["n_objects"], n_failed=paired["n_failed"],
                  n_selections=paired["n_selections"], object_ids=[r["object_id"] for r in rows],
                  median=None, interval=None, standard_median=None)
    if not rows:
        return result
    # Match the source ordering: caps use first-success order, two_d uses sorted IDs.
    if bootstrap:
        scored = ctx.read(scored_path)
        cells = [r for r in scored if r["condition"] == condition and r["status"] == "ok"]
        by_id = {r["object_id"]: r for r in rows}
        ids = list(dict.fromkeys(r["object_id"] for r in cells))
        rows = [by_id[i] for i in ids]
        folds = {r["object_id"]: r["fold"] for r in cells}
        stats = ctx.stat([r["delphi_mean_deg"] for r in rows], seed=20260915,
                         folds=[folds[i] for i in ids], order="numpy_replicate_major")
        result.update(stats, object_ids=ids)
    else:
        result.update(ctx.stat([r["delphi_mean_deg"] for r in rows]))
    result["standard_median"] = float(np.median([r["standard_error_deg"] for r in rows]))
    result["mean_successful_repeats_convention"] = (
        "Arithmetic mean within each asteroid's successful selections, then cohort median."
    )
    return result


def sampling_caps(ctx):
    """Observation-cap medians and source fold-stratified asteroid intervals."""
    result = ctx.figure("sampling-caps")
    caps = ("10", "20", "50", "100", "200", "500", "None")
    rows = [reliability_condition(ctx, f"observation_cap-cap-{cap}", bootstrap=True) for cap in caps]
    summary_path = DATA / "reliability/score/reliability-own-period.json"
    summary = ctx.read(summary_path)["failed_excluded"]["conditions"]
    for row in rows:
        expected = summary[row["condition"]]
        for key, source_key in [("n", "n_objects_with_success"), ("n_failed", "n_failed_cells"),
                                ("n_selections", "n_cells")]:
            ctx.check(result, row["condition"] + ": " + key, row[key], expected[source_key])
        if row["n"]:
            ctx.check(result, row["condition"] + ": source median", row["median"],
                      expected["median_error_deg"])
            ctx.check(result, row["condition"] + ": source mean CI control",
                      row["mean_interval_recomputed"], [expected["ci95_low_deg"], expected["ci95_high_deg"]])
            row["median_stats_check"] = "PASS: same estimator/RNG as median-stats; no macro for this cap."
    # The three-object cap's comparator is also a frozen median-stats quantity.
    tiny = rows[1]
    ctx.check(result, "SamplingTinyDifference: paired median", tiny["standard_median"] - tiny["median"],
              ctx.macros["SamplingTinyDifference"]["median"])
    ctx.check(result, "All: standard median", rows[-1]["standard_median"],
              ctx.macros["KThreeStandardMean"]["median"])
    result["points"] = rows
    fig, (ax,) = ctx.canvas("sampling-caps", [[0.085, 0.25, 0.899, 0.713]])
    x = np.arange(len(rows))
    med = np.array([np.nan if r["median"] is None else r["median"] for r in rows])
    standard = [np.nan if r["standard_median"] is None else r["standard_median"] for r in rows]
    ci = np.array([[np.nan, np.nan] if r["interval"] is None else r["interval"] for r in rows]).T
    ax.plot(x, standard, "s:", color="#E69F00", lw=1.8, ms=4,
            label="Standard poles, same asteroids (median)")
    ax.errorbar(x, med, yerr=[med - ci[0], ci[1] - med], fmt="o-", color="#0072B2",
                ms=5, lw=1.3, capsize=3, label="DeLPHI (median)")
    labels = [("All" if cap == "None" else cap) +
              f'\n{r["n_failed"]}/{r["n_selections"]}\nn={r["n"]}'
              for cap, r in zip(caps, rows, strict=True)]
    ax.set_xticks(x, labels)
    ax.set(xlim=(-0.45, 6.35), ylim=(0, 45), ylabel="Median oracle error (degrees)",
           xlabel="Maximum total observations per asteroid\n"
                  "Failed searches / selections and asteroids with a period (n)")
    ax.text(0, 5, "No period", ha="center", fontsize=7)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ctx.save(fig, "sampling-caps")
    ctx.sources_for(result, [DATA / "failed-excluded-derived/reliability-paired.json", summary_path,
                             DATA / "reliability/score/scored-rows-own.json", RELIABILITY_RENDERER,
                             FOLLOWUP / "source-op-reliability/repro/own_period/reliability_score.py",
                             REPO / "lc_pipeline/k3/reliability_scoring.py"])
    result["means_retained"] = ["Successful selections are averaged within each asteroid, as in the caption."]
    result["caption_change"] = ('Replace "Mean oracle error" with "Median oracle error"; retain '
                                 'the same-asteroid standard poles, failure counts and 95% bootstrap wording.')
    result["visual_message"] = (
        "The ordering remains: 50--200 observations have large errors; All is much better. "
        "At cap 500 the DeLPHI median (19.80 deg) is below the standard-pole median "
        "(24.13 deg), whereas its mean was above the standard-pole mean. "
        "At cap 20 only three asteroids succeed and their source within-fold bootstrap is degenerate."
    )


def data_degradation(ctx):
    """The 5 by 5 grid, including every all-failed cell and its denominator."""
    result = ctx.figure("data-degradation")
    groups, caps = ("1", "3", "5", "10", "None"), ("5", "10", "20", "50", "None")
    summary_path = DATA / "reliability/score/reliability-own-period.json"
    summary = ctx.read(summary_path)["failed_excluded"]
    source_rows = summary["data_degradation"]
    source_rows = {r["condition"]: r for r in source_rows}
    grid = np.full((5, 5), np.nan)
    rows = []
    for i, group in enumerate(groups):
        for j, cap in enumerate(caps):
            condition = f"two_d-block_count-{group}-per_block_cap-{cap}"
            row = reliability_condition(ctx, condition)
            row.update(group=group, cap=cap, row=i, column=j)
            counts = summary["conditions"][condition]
            for key, source_key in [("n", "n_objects_with_success"), ("n_failed", "n_failed_cells"),
                                    ("n_selections", "n_cells")]:
                ctx.check(result, condition + ": " + key, row[key], counts[source_key])
            if row["n"]:
                grid[i, j] = row["median"]
                ctx.check(result, condition + ": source median", row["median"],
                          source_rows[condition]["median_error_deg"])
                ctx.check(result, condition + ": failures", row["n_failed"],
                          source_rows[condition]["failed_predictions"])
            rows.append(row)
    for condition, macro in [("two_d-block_count-1-per_block_cap-None", "SamplingOneFoundMean"),
                              ("two_d-block_count-10-per_block_cap-50", "FollowBlockTenCapFifty"),
                              ("two_d-block_count-10-per_block_cap-20", "FollowBlockTenCapTwenty")]:
        ctx.match(result, next(r for r in rows if r["condition"] == condition), macro, interval=False)
    result["points"] = rows
    result["source_note"] = (
        "The all/all cell is the original two_d-block_count-None-per_block_cap-None "
        "condition (median 13.185615777015663 deg), not the separate grid-full run "
        "summarized by FollowGridBaseline (13.185290985888038 deg). Both retain "
        "80 asteroids; their saved predictions differ slightly. The figure keeps its own condition."
    )
    fig, (ax, cax) = ctx.canvas("data-degradation",
                                [[0.09, 0.092, 0.74, 0.886], [0.875, 0.092, 0.035, 0.886]])
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#F0F0F0")
    old = [r["mean_error_deg"] for r in source_rows.values()]
    vmin, vmax = min(min(old), np.nanmin(grid)), max(max(old), np.nanmax(grid))
    image = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for row in rows:
        value = row["median"]
        text = "No period" if value is None else f"{value:.1f}°"
        text += f'\n{row["n_failed"]}/{row["n_selections"]} failed\nn={row["n"]}'
        row["drawn_text"] = text
        ax.text(row["column"], row["row"], text, ha="center", va="center", fontsize=8,
                color="white" if value is not None and value > 28 else "black")
    ax.set_xticks(range(5), ["5", "10", "20", "50", "all"])
    ax.set_yticks(range(5), ["1", "3", "5", "10", "all"])
    ax.set(xlabel="Maximum observations per 30-day group (all = no limit)",
           ylabel="30-day groups retained (all = complete record)")
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Median oracle error (degrees)")
    result["color_limits_deg"] = [vmin, vmax]
    ctx.save(fig, "data-degradation")
    ctx.sources_for(result, [DATA / "failed-excluded-derived/reliability-paired.json", summary_path,
                             DATA / "reliability/score/scored-rows-own.json", RELIABILITY_RENDERER])
    result["means_retained"] = ["Successful selections are averaged within each asteroid, as in the caption."]
    result["caption_change"] = 'Replace "Mean oracle error" with "Median oracle error"; retain "averaging successful selections within each asteroid".'
    result["visual_message"] = "Unchanged: more groups and observations improve errors; ten cells still have no period."


def latitude(ctx):
    """Six latitude bins with median bootstrap intervals and matched standard poles."""
    result = ctx.figure("latitude")
    main_path = DATA / "maps/own-period/analyses/mirror-symmetry.json"
    lowq_path = DATA / "lowq/scores-own/direction-excluded.json"
    main, lowq = ctx.read(main_path)["objects"], ctx.read(lowq_path)["series"]
    series = [("main", np.abs([r["reference_latitude_deg"] for r in main]),
               np.array([r["oracle_error_deg"] for r in main]), [r["object_id"] for r in main]),
              ("lower_quality", np.array(lowq["abs_latitude_deg"]), np.array(lowq["delphi_error_deg"]),
               lowq["object_id"]),
              ("standard", np.array(lowq["abs_latitude_deg"]), np.array(lowq["standard_error_deg"]),
               lowq["object_id"])]
    edges = np.arange(0, 91, 15)
    rng = np.random.default_rng(20260929)
    rows = []
    for name, lat, error, ids in series:
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            mask = (lat >= low) & ((lat < high) if high < 90 else (lat <= high))
            point = ctx.stat(error[mask], rng=rng if name != "standard" else None)
            point.update(series=name, bin_low_deg=int(low), bin_high_deg=int(high),
                         center_deg=float(low + 7.5), x_deg=float(low + 7.5 +
                         {"main": -1.2, "lower_quality": 1.2, "standard": 0}[name]),
                         object_ids=np.asarray(ids)[mask].tolist())
            if name in ("lower_quality", "standard") and low in (0, 75):
                macro = "InsightLowqFound" + ("Standard" if name == "standard" else "") + "Lat"
                ctx.match(result, point, macro + ("Low" if low == 0 else "High"))
            rows.append(point)
        ctx.check(result, name + ": binned asteroid count", sum(r["n"] for r in rows if r["series"] == name), len(ids))
    result["points"] = rows
    result["bootstrap_note"] = (
        "Seed 20260929, one NumPy stream shared across main bins then lower-quality bins. "
        "10,000 replicates, as median-stats.json specifies globally; old renderer used 2,000. "
        "No latitude-bin CI endpoints exist in median-stats.json."
    )
    fig, (ax,) = ctx.canvas("latitude", [[0.10, 0.15, 0.87, 0.64]])
    for name, color, marker, style, label in [
        ("main", "#0072B2", "o", "-", "DeLPHI median, 170/170 main-sample periods found"),
        ("lower_quality", "#333333", "s", "--", "DeLPHI median, 4,077/9,783 lower-quality periods found")]:
        data = [r for r in rows if r["series"] == name]
        med = np.array([r["median"] for r in data])
        ci = np.array([r["interval"] for r in data]).T
        ax.errorbar([r["x_deg"] for r in data], med, yerr=[med - ci[0], ci[1] - med],
                    fmt=marker, ls=style, color=color, ms=5, lw=1.3, capsize=2, label=label)
    standard = [r for r in rows if r["series"] == "standard"]
    ax.plot([r["x_deg"] for r in standard], [r["median"] for r in standard], ":", color="#E69F00",
            lw=1.8, label="Standard-pole median, same 4,077 asteroids")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower left",
              bbox_to_anchor=(0, 1.04), borderaxespad=0)
    ax.set_xticks(edges)
    top = max(45, max(r["interval"][1] for r in rows if r["interval"] is not None))
    ax.set(xlim=(0, 90), ylim=(0, top), ylabel="Median oracle error (degrees)",
           xlabel="Absolute ecliptic latitude of the reference pole (degrees)")
    ax.grid(alpha=0.2)
    ctx.save(fig, "latitude")
    ctx.sources_for(result, [main_path, lowq_path, MAPS_RENDERER])
    result["caption_change"] = 'Replace "Mean oracle error" with "Median oracle error"; keep the 15-degree bins and 95% bootstrap intervals.'
    result["visual_message"] = (
        "The lower-quality error still falls strongly with latitude; the main sample remains nonmonotonic. "
        "The standard-pole medians cross the lower-quality DeLPHI medians in the middle bins."
    )


def candidate_count(ctx):
    """Replace pole-error means; keep the original fixed search-cone geometry."""
    result = ctx.figure("candidate-count")
    path = DATA / "maps/own-period/analyses/candidate-count.json"
    summary = ctx.read(path)
    rows = summary["k"]
    standard = ctx.macros["KThreeStandardMean"]["median"]
    # The existing dashed-square median curve stays, coincident with the updated
    # solid-circle curve. One legend entry avoids implying two estimators.
    for row in rows:
        point = dict(panel="a", k=row["k"], n=170, median=row["median_error_deg"],
                     coincident_series=["updated solid circles", "existing dashed squares"])
        if row["k"] <= 3:
            ctx.match(result, point, {1: "CountOneMean", 2: "CountTwoMean", 3: "CountThreeMean"}[row["k"]])
        else:
            point["median_stats_check"] = "N/A: no macro; PASS frozen candidate-count median."
        result["points"].append(point)
        result["points"].append(dict(panel="b", k=row["k"],
                                     mean_cone_radius_deg=row["mean_error_deg"],
                                     p90_cone_radius_deg=row["p90_error_deg"],
                                     area_mean_radius_percent=100 * row["search_area_mean_radius"],
                                     area_p90_radius_percent=100 * row["search_area_p90_radius"]))
    result["points"].append(dict(panel="a", series="standard", n=170, median=standard))
    result["points"].append(dict(panel="b", series="standard", **summary["standard_poles"],
                                 area_mean_radius_percent=100 * summary["standard_poles"]["search_area_mean_radius"],
                                 area_p90_radius_percent=100 * summary["standard_poles"]["search_area_p90_radius"]))
    source = function_source(MAPS_RENDERER, "render_candidate_count")
    # Preserve all artist styles and subplot spacing from the maps renderer.
    source = replace_once(source, 'mean = np.array([row["mean_error_deg"] for row in rows])',
                          'mean = np.array([row["median_error_deg"] for row in rows])')
    source = replace_once(source, 'label="Mean"', 'label="Median"')
    # Target the existing square-series label without changing the circle legend.
    source = replace_once(source, 'alpha=0.8, label="Median"', 'alpha=0.8, label="_existing_median"')
    source = replace_once(source, 'ax.axhline(standard["mean_error_deg"]', 'ax.axhline(standard_median')
    source = replace_once(source, 'label="Standard poles, mean")\n    ax.set_xlabel',
                          'label="Standard poles, median")\n    ax.set_xlabel')
    source = replace_once(source, 'ax.set_ylabel("Oracle error (degrees)")',
                          'ax.set_ylabel("Median oracle error (degrees)")')
    source = replace_once(source, '    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))',
                          '    fig, axes = ctx.canvas("candidate-count", '
                          '[[0.074, 0.176, 0.411, 0.726], [0.575, 0.176, 0.411, 0.726]])')
    source = replace_once(source, '    fig.tight_layout()\n', '')
    source = replace_once(source, '    fig.savefig(FIG_OUT / "candidate-count.pdf", dpi=300, facecolor="white")\n    plt.close(fig)',
                          '    ctx.save(fig, "candidate-count")')
    import matplotlib.pyplot as plt

    namespace = dict(np=np, plt=plt, json=json, ANALYSES=path.parent, standard_median=standard, ctx=ctx)
    exec(compile(source, str(MAPS_RENDERER), "exec"), namespace)
    namespace["render_candidate_count"]()
    ctx.sources_for(result, [path, MAPS_RENDERER, *ctx.macros["KThreeStandardMean"]["source_files"]])
    result["means_retained"] = [
        "Panel (b): six fixed cone radii equal the original K-specific mean errors; the standard cone uses its original mean error. All areas and p90 radii remain frozen.",
    ]
    result["caption_change"] = ('In (a), replace "Mean and median oracle error" with "Median oracle error"; '
                                 'clarify that coincident solid-circle and dashed-square curves show the same median. '
                                 'In the final sentence, standard poles use the median in (a) and the original fixed cone radii in (b). '
                                 'Keep "mean" in (b).')
    result["visual_message"] = "The second candidate still gives the largest gain. The two error curves in (a) now coincide; (b) is unchanged."


def inversion_benchmark(ctx):
    """The amendment's three panels: two timing histograms and median pole errors."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, NullFormatter
    from matplotlib.transforms import blended_transform_factory

    result = ctx.figure("inversion-benchmark")
    folder = DATA / "benchmark"
    scores = [ctx.read(folder / name) for name in ("score.json", "score-loaded.json")]
    budget = ctx.read(folder / "budget-equivalence.json")
    ladder = ctx.read(folder / "classical-ladder-exploratory.json")["object_rows"]
    ids = [r["object_id"] for r in ladder]
    classical = []
    for row in budget["budgets"]:
        starts = row["starts"]
        stat = ctx.stat([r["classical"][str(starts)]["reference_error_degrees"] for r in ladder])
        stat.update(panel="c", starts=starts, series="classical", object_ids=ids)
        matching = [r for r in ctx.frozen["entries"]
                    if str(folder / "budget-equivalence.json") in r.get("source_files", [])
                    and f"budgets/{budget['budgets'].index(row)}/mean_error_deg" in r.get("input_source_comment", "")]
        if matching:
            ctx.match(result, stat, matching[0]["mean_macro"])
        else:
            stat["median_stats_check"] = "PASS: recomputed from saved ladder; no median-stats macro for this budget."
        classical.append(stat)
    loaded = {r["object_id"]: r for r in scores[1]["object_rows"]}
    de = ctx.stat([loaded[i]["arms"]["delphi"]["reference_error_degrees"] for i in ids])
    de.update(panel="c", starts=6, series="delphi", object_ids=ids)
    ctx.match(result, de, "BudgetDelphiError")
    ratios = [np.array([r["arms"]["classical"]["mean_wall_seconds"] /
                       r["arms"]["delphi"]["mean_wall_seconds"] for r in s["object_rows"]]) for s in scores]
    # Keep the renderer's log-bin spacing and extend its lower edge to include
    # two cold-run asteroids below 0.1; no observation is silently dropped.
    lower_exponent = min(-1.0, 0.05 * np.floor(np.log10(min(r.min() for r in ratios)) / 0.05))
    histogram_edges = 10.0 ** np.arange(lower_exponent, 1.00001, 0.05)
    for panel, report in zip(("a", "b"), scores, strict=True):
        rows = report["object_rows"]
        ratio = np.array([r["arms"]["classical"]["mean_wall_seconds"] /
                          r["arms"]["delphi"]["mean_wall_seconds"] for r in rows])
        counts, edges = np.histogram(ratio, bins=histogram_edges)
        result["points"].append(dict(panel=panel, object_ids=[r["object_id"] for r in rows],
                                     time_ratios=ratio.tolist(), histogram_counts=counts.tolist(),
                                     histogram_edges=edges.tolist(), n_delphi_faster=int((ratio > 1).sum()),
                                     n=len(rows), ratio_of_means=report["point_estimates"]["speed_ratio_classical_over_delphi"],
                                     interval=report["bootstrap_95_intervals"]["speed_ratio_classical_over_delphi"]))
        ctx.check(result, panel + ": histogram includes all asteroids", int(counts.sum()), len(rows))
    result["points"].extend([*classical, de])
    source = function_source(INVERSION_RENDERER, "main")
    source = replace_once(source, 'edges = np.geomspace(0.1, 10.0, 41)',
                          'edges = histogram_edges')
    source = replace_once(source, 'figsize=(10.4, 3.3)', 'figsize=(505 / 72, 3.0)')
    source = replace_once(source, '"font.size": 10', '"font.size": 9')
    source = replace_once(source, '"axes.labelsize": 10', '"axes.labelsize": 8')
    source = replace_once(source, '"axes.titlesize": 11', '"axes.titlesize": 8')
    source = replace_once(source, '"xtick.labelsize": 9', '"xtick.labelsize": 7')
    source = replace_once(source, '"ytick.labelsize": 9', '"ytick.labelsize": 7')
    source = replace_once(source, '"(a) Fresh process for each asteroid", "(b) Models kept loaded"',
                          '"(a) Fresh process\\nfor each asteroid", "(b) Models kept loaded"')
    source = source.replace('fontsize=9', 'fontsize=7')
    start = source.index('        ax.text(0.03, 0.97,')
    end = source.index('\n\n    # (c)', start)
    source = source[:start] + '''        ax.text(0.03, 0.97,
                f"DeLPHI faster: {int(np.sum(ratio > 1))}/{len(ratio)}\\n"
                f"Ratio of means {point['speed_ratio_classical_over_delphi']:.2f}\\n"
                f"95% interval {speed[0]:.2f}--{speed[1]:.2f}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})''' + source[end:]
    source = replace_once(source, 'error = np.array([row["mean_error_deg"] for row in budgets["budgets"]])',
                          'error = np.array(median_errors)')
    source = source.replace('delphi["mean_error_deg"]', 'delphi_median')
    source = replace_once(source, 'Mean disagreement with DAMIT (deg)', 'Median disagreement\\nwith DAMIT (deg)')
    source = replace_once(source, 'title="(c) Agreement and number of starts"', 'title="(c) Agreement and\\nnumber of starts"')
    source = replace_once(source, 'ax.set_ylim(12, 25)', 'ax.set_ylim(12, 55)')
    source = replace_once(source, 'loc="upper right", fontsize=8.5', 'loc="lower right", fontsize=7')
    source = replace_once(source, 'fig.savefig(FIGURE, bbox_inches="tight")\n    plt.close(fig)\n    print(FIGURE)',
                          'ctx.save(fig, "inversion-benchmark")')
    namespace = dict(np=np, plt=plt, json=json, SCORES=[folder / "score.json", folder / "score-loaded.json"],
                     BUDGETS=folder / "budget-equivalence.json", MaxNLocator=MaxNLocator,
                     NullFormatter=NullFormatter, blended_transform_factory=blended_transform_factory,
                     median_errors=[r["median"] for r in classical], delphi_median=de["median"],
                     histogram_edges=histogram_edges, ctx=ctx)
    with plt.rc_context():
        exec(compile(source, str(INVERSION_RENDERER), "exec"), namespace)
        namespace["main"]()
    result["page_size_pt"] = [505, 216]
    ctx.sources_for(result, [INVERSION_RENDERER, folder / "score.json", folder / "score-loaded.json",
                             folder / "budget-equivalence.json", folder / "classical-ladder-exploratory.json"])
    result["means_retained"] = [
        "Panels (a,b): per-asteroid time is the mean of three runs; annotations retain the ratio of mean times and its source bootstrap interval. Timing statistics are outside the pole-error conversion.",
    ]
    result["caption_change"] = 'In (c), replace "Mean disagreement" with "Median disagreement". Keep all timing wording, including "averaged over three runs".'
    result["visual_message"] = (
        "The amended PDF now matches the caption's three panels. Classical median disagreement is "
        "41.76--52.09 deg across saved budgets, versus 27.44 deg for six DeLPHI starts; "
        "more classical starts do not improve pole agreement at the fixed own period. "
        "The pole-error axis extends from the renderer's 12--25 to 12--55 deg. "
        "Timing histograms extend down to 0.0891 to include the two cold-run ratios below 0.1."
    )


NETTESTS_RENDERER = DATA / "nettests/figures/render_main_figures.py"
F2_NAMES = ("interpretation", "shapley", "training-test", "reference-reassignment",
            "input-removal", "period-sensitivity")


def f2_canvas(ctx, name, boxes, *, projections=None):
    """Axes boxes in PDF points, measured from the current paper's vector paths."""
    fig, _ = ctx.canvas(name, [])
    width, height = ctx.results[name]["page_size_pt"]
    projections = projections or [None] * len(boxes)
    axes = [fig.add_axes([x / width, y / height, w / width, h / height], projection=p)
            for (x, y, w, h), p in zip(boxes, projections, strict=True)]
    ctx.results[name]["original_axes_boxes_pt"] = boxes
    return fig, axes


def f2_definitions(path, names, namespace):
    """Load only named definitions; never execute another renderer's module."""
    source = path.read_text(encoding="utf-8")
    nodes = {n.name: n for n in ast.parse(source).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in names:
        exec(compile(ast.get_source_segment(source, nodes[name]), str(path), "exec"), namespace)


def interpretation(ctx):
    """Keep maps/mirror scatter; change the eight grouped errors in panel (d)."""
    import matplotlib.pyplot as plt

    result = ctx.figure("interpretation")
    folder = DATA / "maps/own-period/analyses"
    comparison = ctx.read(folder / "amplitude-aspect.json")
    symmetry = ctx.read(folder / "mirror-symmetry.json")["objects"]
    edges = (-1.0, 0.5, 0.7, 0.8, 1.0)
    labels = ("< 0.5", "0.5-0.7", "0.7-0.8", "> 0.8")
    statistics = {}
    paired_draws = {}
    for series, key, suffix in [("DeLPHI", "delphi_oracle_deg", "Delphi"),
                                ("Amplitude-aspect", "amplitude_aspect_oracle_deg", "AA")]:
        rng = np.random.default_rng(20260923)
        rows = []
        for i, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            chosen = [r for r in comparison if low <= r["map_correlation"] < high]
            values = np.array([r[key] for r in chosen])
            draws = rng.integers(0, len(chosen), size=(ctx.resamples, len(chosen)))
            if series == "DeLPHI":
                paired_draws[i] = draws
            else:
                ctx.check(result, labels[i] + ": paired bootstrap identities",
                          bool(np.array_equal(draws, paired_draws[i])), True)
            row = ctx.statistics(values, indices=draws)
            row.update(panel="d", series=series, label=f"{series}, {labels[i]}",
                       bin_low=low, bin_high=high, object_ids=[r["object_id"] for r in chosen],
                       errors_deg=values.tolist(), bootstrap=dict(seed=20260923, resamples=ctx.resamples,
                       rng_and_order="iid, bins in drawing order; reset per series", unit="asteroid"))
            if i in (0, 3):
                ctx.match(result, row, f'InterpAA{"Low" if i == 0 else "High"}{suffix}')
            else:
                row["median_stats_check"] = "PASS: same estimator/cohort; no macro for this intermediate bin."
            rows.append(row)
        statistics[series] = rows
        result["points"].extend(rows)
    result["bootstrap_note"] = (
        "Panel (d): paired identity resamples for the two maps within each original correlation bin, "
        "seed 20260923 (the renderer's seed), 10,000 replicates (median-stats.json). "
        "Frozen macros provide four endpoint-bin medians, but no individual-bin intervals."
    )
    fig, axes = f2_canvas(ctx, "interpretation", [
        [39.363125, 201.181812, 165.943816, 82.971908],
        [251.771209, 201.181812, 165.943816, 82.971908],
        [425.638625, 205.419579, 3.724819, 74.496374],
        [39.363125, 28.66, 173.763158, 148.146199],
        [261.779967, 28.66, 173.763158, 148.146199]],
        projections=["mollweide", "mollweide", None, None, None])
    source = function_source(MAPS_RENDERER, "render_interpretation")
    start = source.index('    fig = plt.figure')
    end = source.index('    both = np.r_', start)
    source = source[:start] + source[end:]
    source = replace_once(source, 'fig.add_subplot(outer[0, column], projection="mollweide")',
                          'axes[column]')
    source = replace_once(source, 'fig.add_subplot(outer[1, 0])', 'axes[3]')
    source = replace_once(source, 'fig.add_subplot(outer[1, 1])', 'axes[4]')
    source = replace_once(source, 'ax=fig.axes[:2], orientation="vertical", fraction=0.025,\n'
                          '                       pad=0.02, shrink=0.8', 'cax=axes[2], orientation="vertical"')
    start = source.index('        means, lows, highs = [], [], []')
    end = source.index('        ax.errorbar', start)
    source = source[:start] + '''        means = np.array([r["median"] for r in statistics[name]])
        limits = np.array([r["interval"] for r in statistics[name]])
        lows, highs = means - limits[:, 0], limits[:, 1] - means
''' + source[end:]
    source = replace_once(source, 'Mean oracle error (degrees)', 'Median oracle error (degrees)')
    source = source[:source.index('    fig.savefig')] + '''    ctx.save(fig, "interpretation")
'''
    namespace = dict(np=np, plt=plt, ctx=ctx, fig=fig, axes=axes, statistics=statistics,
                     ANALYSES=folder, ENSEMBLE=DATA / "maps/own-period/evaluations/real-oof-ensemble.npz",
                     INTERPRETATION_EXAMPLE_OBJECT="asteroid_121")
    f2_definitions(MAPS_RENDERER, ["axial_grid_vectors", "ecliptic_angles"], namespace)
    # Record the unchanged map and scatter values as well as the eight summaries.
    with np.load(namespace["ENSEMBLE"], allow_pickle=False) as ensemble, \
            np.load(folder / "amplitude-aspect.npz", allow_pickle=False) as physical:
        maps = []
        for panel, data, field in [("a", ensemble, "score_grids"),
                                    ("b", physical, "amplitude_aspect_maps")]:
            index = data["object_ids"].astype(str).tolist().index("asteroid_121")
            values = data[field][index].astype(float)
            maps.append(dict(panel=panel, object_id="asteroid_121", raw_scores=values.tolist(),
                             standardized_scores=((values - values.mean()) / values.std()).tolist()))
        result["unchanged_panels"] = dict(maps=maps, axial_grid_vectors=namespace["axial_grid_vectors"]().tolist(),
                                          mirror_scatter=symmetry, standardized_colour_limits=[-2.5, 2.5])
    exec(compile(source, str(MAPS_RENDERER), "exec"), namespace)
    namespace["render_interpretation"]()
    ctx.sources_for(result, [MAPS_RENDERER, folder / "amplitude-aspect.json", folder / "amplitude-aspect.npz",
                             folder / "mirror-symmetry.json", namespace["ENSEMBLE"]])
    result["means_retained"] = [
        "Panel (c)'s mean observing latitude and panels (a,b)'s score standardization are physical/map definitions, not cohort pole-error means."
    ]
    result["caption_change"] = 'In (d), replace "Mean oracle error" with "Median oracle error"; retain 95% bootstrap intervals.'
    result["visual_message"] = (
        "Panel (d) retains the advantage of DeLPHI at low map agreement and near agreement at high correlation. "
        "Its endpoint-bin medians are DeLPHI 15.76/13.70 deg and amplitude-aspect 33.78/14.55 deg. "
        "The original non-monotonic DeLPHI ordering across correlation bins remains; all intervals stay within 0--45 deg."
    )


def shapley(ctx):
    """Keep exact Shapley effects and shares; mark cohort medians in panel (b)."""
    import math
    from collections import defaultdict

    import matplotlib.pyplot as plt

    from lc_pipeline.k3.grid import axial_angular_error_deg, axial_healpix_grid

    result = ctx.figure("shapley")
    source_path = REPO / "repro/group_shapley.py"
    summary_path = REPO / "repro/group_shapley_summary.py"
    namespace = dict(np=np, math=math, axial_angular_error_deg=axial_angular_error_deg,
                     AXIS_GROUPS=(4, 5, 6))
    f2_definitions(source_path, ["FastModes", "shapley"], namespace)
    f2_definitions(summary_path, ["flat_map_masks"], namespace)
    masks = namespace["flat_map_masks"]()
    summary_file = DATA / "nettests/shapley-own-summary.json"
    summary = ctx.read(summary_file)
    paths = sorted((DATA / "nettests/shapley-own").glob("fold-*-seed-*.npz"))
    if len(paths) != 25:
        raise ValueError(f"Expected 25 frozen Shapley networks, found {len(paths)}")
    per_seed, per_object, per_object_p = defaultdict(list), defaultdict(list), defaultdict(list)
    grids = {}
    groups = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            seed = int(path.stem.split("seed-")[1])
            current = data["groups"].astype(str).tolist()
            if groups is not None and current != groups:
                raise ValueError("Shapley group order changed")
            groups = current
            percentile, error = data["percentile"].copy(), data["grid_error"].copy()
            percentile[:, masks] = 0.5
            nside = int(data["nside"])
            if nside not in grids:
                modes = namespace["FastModes"](axial_healpix_grid(nside))
                grids[nside] = modes.vectors[modes(np.zeros(len(modes.vectors)))]
            for index, name in enumerate(data["object_ids"].astype(str)):
                flat_axes, poles = grids[nside], ctx.evidence.poles[name]
                flat_error = float(np.degrees(np.arccos(np.clip(np.abs(flat_axes @ poles.T).max(), -1, 1))))
                error[index, masks] = flat_error
            values_p = namespace["shapley"](percentile)
            values_e = namespace["shapley"](error)
            per_seed[seed].append(values_p)
            for index, name in enumerate(data["object_ids"].astype(str)):
                per_object[name].append(values_e[index])
                per_object_p[name].append(values_p[index])
    ids = list(per_object)
    if set(ids) != set(ctx.evidence.ensemble["object_ids"].astype(str)) or any(len(v) != 5 for v in per_object.values()):
        raise ValueError("Shapley requires the original 170 asteroids, five networks each")
    errors = np.array([np.mean(v, axis=0) for v in per_object.values()])
    pooled = np.array([np.mean(v, axis=0) for v in per_object_p.values()])
    shares = []
    for _, values in sorted(per_seed.items()):
        values = np.concatenate(values)
        shares.append(100 * values.mean(axis=0) / values.sum(axis=1).mean())
    shares = np.array(shares)
    mean_share = 100 * pooled.mean(axis=0) / pooled.sum(axis=1).mean()
    medians = np.median(errors, axis=0)
    order = np.argsort(mean_share)
    controls = {r["group"]: r for r in summary["groups"]}
    jitter = np.random.default_rng(20260928)
    for y, index in enumerate(order):
        group = groups[index]
        control = controls[group]
        ctx.check(result, group + ": original panel (a) share", mean_share[index], 100 * control["percentile_share"])
        ctx.check(result, group + ": original panel (a) seed range", [shares[:, index].min(), shares[:, index].max()],
                  control["share_seed_range_percent"])
        ctx.check(result, group + ": recovered original mean effect", errors[:, index].mean(), control["grid_error_mean_deg"])
        result["points"].append(dict(label=group, panel="b", median=float(medians[index]), n=len(ids),
            object_ids=ids, effects_deg=errors[:, index].tolist(), scatter_y=(y + jitter.uniform(-0.22, 0.22, len(ids))).tolist(),
            original_mean=float(errors[:, index].mean()), share_percent=float(mean_share[index]),
            share_seed_values_percent=shares[:, index].tolist(), share_seed_range_percent=[shares[:, index].min(), shares[:, index].max()],
            median_stats_check="PASS: exact frozen Shapley effects recovered; no group-effect median macro exists."))
    fig, axes = f2_canvas(ctx, "shapley", [[141.0125, 33.7, 156.453333, 177.338125],
                                          [310.685833, 33.7, 195.566667, 177.338125]])
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    source = function_source(NETTESTS_RENDERER, "render_shapley")
    source = 'def draw():\n' + source[source.index('    labels = {'):]
    start = source.index('    groups = list(gs.GROUPS)')
    end = source.index('    y = np.arange', start)
    source = source[:start] + source[end:]
    source = replace_once(source, '[values.mean()] * 2', '[np.median(values)] * 2')
    source = replace_once(source, 'Change of oracle error (degrees)', 'Change of oracle error (degrees); black = median')
    source = source[:source.index('    fig.tight_layout()')] + '''    axes[0].set_xlim(original_limits[0])
    axes[1].set_xlim(original_limits[1])
    ctx.save(fig, "shapley")
'''
    # Recover the original automatic limits from the unchanged data artists.
    share_extrema = np.r_[0, mean_share, shares.ravel()]
    error_extrema = np.r_[0, errors.ravel()]
    original_limits = [(float(v.min() - .05 * np.ptp(v)), float(v.max() + .05 * np.ptp(v)))
                       for v in (share_extrema, error_extrema)]
    namespace.update(ctx=ctx, fig=fig, axes=axes, plt=plt, groups=groups, shares=shares,
                     mean_share=mean_share, order=order, errors=errors, original_limits=original_limits)
    exec(compile(source, str(NETTESTS_RENDERER), "exec"), namespace)
    namespace["draw"]()
    ctx.sources_for(result, [NETTESTS_RENDERER, source_path, summary_path, summary_file,
                             REPO / "lc_pipeline/k3/grid.py", *paths,
                             FOLLOWUP / "source/repro/data/damit-20250610T000301Z/catalog.jsonl"])
    result["means_retained"] = [
        "Panel (a) retains its signed percentile shares and five-seed ranges: these are additive attribution shares, not pole errors.",
        "Per-asteroid Shapley effects retain their exact coalition-weighted averages and five-network averages; only the across-asteroid black marker changes."
    ]
    result["source_note"] = "The 176-entry median-stats.json has no Shapley group-effect macro. All eight recovered original means and shares/ranges are checked against shapley-own-summary.json before computing the medians."
    result["caption_change"] = 'In (b), replace "with the mean marked in black" with "with the median marked in black"; panel (a) stays unchanged.'
    result["visual_message"] = (
        "Panel (a) and every per-asteroid point in (b) remain unchanged. The black markers now show typical "
        "asteroid effects. The strongest median reduction is now the Sun term (-10.93 deg), followed by "
        "brightness (-8.63 deg); their original mean reductions were -8.65 and -9.30 deg, respectively. "
        "No group's black marker changes sign."
    )


def training_test(ctx):
    """Per-asteroid scatter: retain the within-asteroid mean used by median-stats."""
    from collections import defaultdict

    import matplotlib.pyplot as plt

    result = ctx.figure("training-test")
    report_path = DATA / "nettests/train-vs-test-own-report.json"
    report = ctx.read(report_path)
    paths = sorted((DATA / "nettests/train-vs-test-own").glob("cells-fold-*.jsonl"))
    train, test = defaultdict(list), {}
    for path in paths:
        for row in map(json.loads, path.read_text().splitlines()):
            if row["role"] == "train":
                train[row["object_id"]].append(row["error_deg"])
            elif row["role"] == "test":
                test[row["object_id"]] = row["error_deg"]
    for row in report["pairs"]:
        oid = row["object_id"]
        ctx.check(result, oid + ": training mean", float(np.mean(train[oid])), row["mean_train_error_deg"])
        ctx.check(result, oid + ": training repeats", len(train[oid]), row["train_repetitions"])
        ctx.check(result, oid + ": single OOF error", test[oid], row["oof_error_deg"])
        result["points"].append(dict(label=oid, object_id=oid, x_test_error_deg=test[oid],
            y_training_mean_error_deg=float(np.mean(train[oid])), training_errors_deg=train[oid],
            median_stats_check="PASS: unchanged per-asteroid coordinates; the frozen population estimator preserves training means."))
    ctx.check(result, "Paired asteroid count", len(report["pairs"]), 169)
    result["validation_only_not_drawn"] = []
    for macro in ("CheckTrainMean", "CheckTestMean"):
        stats = ctx.evidence.resolve_statistic(ctx.macros[macro])
        ctx.match(result, stats, macro)
        result["validation_only_not_drawn"].append(stats)
        ctx.sources_for(result, stats["source_files"])
    pairs = np.array([(r["oof_error_deg"], r["mean_train_error_deg"]) for r in report["pairs"]])
    fig, (ax,) = f2_canvas(ctx, "training-test", [[35.73125, 33.7, 190.118125, 181.209004]])
    source = function_source(NETTESTS_RENDERER, "render_appendix_checks")
    source = 'def draw():\n' + source[source.index('    ax.scatter(pairs[:, 0]'):]
    source = source[:source.index('    fig.tight_layout()')] + '    ctx.save(fig, "training-test")\n'
    namespace = dict(np=np, plt=plt, ctx=ctx, fig=fig, ax=ax, pairs=pairs)
    exec(compile(source, str(NETTESTS_RENDERER), "exec"), namespace)
    namespace["draw"]()
    ctx.sources_for(result, [NETTESTS_RENDERER, report_path, *paths])
    result["means_retained"] = [
        "The y coordinates average training runs within each asteroid, as explicitly required by CheckTrainMean's frozen method. There is no across-asteroid mean or interval drawn to replace."
    ]
    result["caption_change"] = 'No change to the left-panel caption: retain "mean error as a training asteroid of the other runs". Only the companion right-panel caption changes to median.'
    result["visual_message"] = "Unchanged scatter, identity line and 0--90 deg axes. Population medians (8.8232 deg training, 13.5912 deg test) are checked, not added to the plot."
    result["statistic_label_exception"] = "No population statistic is drawn; individual coordinates retain the frozen within-asteroid training mean."


def reference_reassignment(ctx):
    """Reproduce all original stratified permutations, then histogram their medians."""
    from collections import defaultdict

    import matplotlib.pyplot as plt

    result = ctx.figure("reference-reassignment")
    folder = DATA / "maps/own-period"
    summary_path = folder / "analyses/reference-reassignment.json"
    source = ctx.read(summary_path)
    frozen_path = FOLLOWUP / "data/reliability-20260915/run-1/scored-rows.json"
    rows = [r for r in ctx.read(frozen_path)["payload"]
            if r["condition"] == "full" and r["repeat"] == 0 and r["status"] == "ok"]
    ensemble, poles = ctx.evidence.ensemble, ctx.evidence.poles
    index = {str(oid): i for i, oid in enumerate(ensemble["object_ids"])}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["fold"], len(poles[row["object_id"]]))].append(row["object_id"])
    values = np.empty((source["draws"], len(rows)))
    rng = np.random.default_rng(source["seed"])
    for k in range(source["draws"]):
        offset = 0
        for ids in groups.values():
            assigned = rng.permutation([poles[oid] for oid in ids])
            axes = np.asarray([ensemble["refined_axes"][index[oid]] for oid in ids], dtype=float)
            axes /= np.linalg.norm(axes, axis=2, keepdims=True)
            assigned /= np.linalg.norm(assigned, axis=2, keepdims=True)
            cosine = np.abs(np.einsum("iaj,irj->iar", axes, assigned)).max(axis=(1, 2))
            values[k, offset:offset + len(ids)] = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
            offset += len(ids)
    ctx.check(result, "All original mean-error permutation draws: maximum absolute discrepancy",
              float(np.max(np.abs(values.mean(axis=1) - source["null_draws_mean_error_deg"]))), 0)
    ids = [oid for group in groups.values() for oid in group]
    stats = ctx.evidence.randomized(values, ids, [], "Original paired reassignment stream.", source["seed"], source["draws"])
    ctx.match(result, stats, "CheckReassignMean")
    ctx.check(result, "CheckReassignLow: draw median quantile", stats["randomization_interval95"][0], ctx.macros["CheckReassignLow"]["median"])
    ctx.check(result, "CheckReassignHigh: draw median quantile", stats["randomization_interval95"][1], ctx.macros["CheckReassignHigh"]["median"])
    draw_medians = np.median(values, axis=1)
    counts, bins = np.histogram(draw_medians, bins=30)
    stats.update(label="Reference-permutation draw medians", draw_medians_deg=draw_medians.tolist(),
                 histogram_counts=counts.tolist(), histogram_edges=bins.tolist(), strata=[dict(fold=k[0], reference_count=k[1], object_ids=v) for k, v in groups.items()])
    result["points"].append(stats)
    observed = ctx.stat(ensemble["oracle_errors_deg"])
    observed.update(label="Actual reference poles", object_ids=ensemble["object_ids"].astype(str).tolist())
    ctx.match(result, observed, "KThreeOofMean", interval=False)
    result["points"].append(observed)
    ctx.check(result, "Histogram includes every original draw", int(counts.sum()), source["draws"])
    fig, (ax,) = f2_canvas(ctx, "reference-reassignment", [[45.2625, 33.7, 181.478125, 184.058125]])
    draw_source = function_source(MAPS_RENDERER, "render_reference_reassignment")
    draw_source = 'def draw():\n' + draw_source[draw_source.index('    ax.hist'):]
    draw_source = replace_once(draw_source, 'ax.hist(summary["null_draws_mean_error_deg"],', 'ax.hist(draw_medians,')
    draw_source = replace_once(draw_source, 'observed = summary["observed_mean_error_deg"]', 'observed = actual_median')
    draw_source = replace_once(draw_source, 'ax.set_xlim(10, max(summary["null_draws_mean_error_deg"]) + 2)',
                               'ax.set_xlim(10, max(max(summary["null_draws_mean_error_deg"]) + 2, draw_medians.max() + 2))\n'
                               '    ax.set_ylim(0, max(original_ymax, counts.max() * 1.05))')
    draw_source = replace_once(draw_source, 'Mean oracle error (degrees)', 'Median oracle error (degrees)')
    draw_source = draw_source[:draw_source.index('    fig.tight_layout()')] + '    ctx.save(fig, "reference-reassignment")\n'
    original_ymax = float(np.histogram(source["null_draws_mean_error_deg"], bins=30)[0].max() * 1.05)
    result["axis_changes"] = dict(original_ylim=[0, original_ymax],
                                  median_ylim=[0, max(original_ymax, float(counts.max() * 1.05))])
    namespace = dict(plt=plt, ctx=ctx, fig=fig, ax=ax, summary=source, draw_medians=draw_medians,
                     actual_median=observed["median"], counts=counts, original_ymax=original_ymax)
    exec(compile(draw_source, str(MAPS_RENDERER), "exec"), namespace)
    namespace["draw"]()
    ctx.sources_for(result, [MAPS_RENDERER, summary_path, frozen_path,
                             folder / "evaluations/real-oof-ensemble.npz",
                             FOLLOWUP / "source/repro/data/damit-20250610T000301Z/catalog.jsonl",
                             FOLLOWUP / "source-op-maps/repro/own_period/maps/analyses.py"])
    result["means_retained"] = ["The validation summary averages per-draw medians as median-stats.json requires; every histogram entry is one draw's asteroid median."]
    result["caption_change"] = 'Right panel: replace "mean oracle error over 10,000 reassignments" with "distribution of median oracle errors from 10,000 reassignments"; retain fixed candidates and permutation strata.'
    result["visual_message"] = (
        f'The actual median ({observed["median"]:.2f} deg) remains far below the reassignment draws '
        f'(central 95% range {stats["randomization_interval95"][0]:.2f}--{stats["randomization_interval95"][1]:.2f} deg). '
        'The null distribution is broader; these are randomization draws, not bootstrap intervals. '
        f'The histogram count axis extends from {original_ymax:.2f} to {max(original_ymax, counts.max() * 1.05):.2f} to include its tallest bin; the x limits stay unchanged.'
    )


def input_removal(ctx):
    """Difference of matched asteroid medians per seed; keep the seed-average bar."""
    import matplotlib.pyplot as plt

    result = ctx.figure("input-removal")
    folder = DATA / "nettests/controls-own"
    seeds = (17, 42, 137, 777, 2027)
    controls = [("All numerical inputs", "zero-numeric"), ("Sun and observer directions", "geometry"),
                ("All brightness information", "brightness"), ("Brightness in the 64 phase bins", "binned-brightness"),
                ("Fourier coefficients", "fourier-all"), ("Peak-to-peak amplitude", "amplitude"),
                ("Observation counts and coverage", "sampling-summary"), ("Period as a number", "period-scalar"),
                ("Sun term and product term", "observer-aspect-only"),
                ("Observer term and product term", "solar-aspect-only"), ("Product term only", "no-cross-term")]
    summary_path = DATA / "nettests/controls-own-summary.json"
    summary = ctx.read(summary_path)["summary"]["controls"]
    all_paths, gap_by_control, old_gaps = [], {}, []
    for label, control in controls:
        rows = []
        for seed in seeds:
            ablated, base, ids = [], [], []
            for fold in range(5):
                p, q = folder / f"real-fold-{fold}-seed-{seed}-{control}.npz", folder / f"real-fold-{fold}-seed-{seed}.npz"
                with np.load(p, allow_pickle=False) as a, np.load(q, allow_pickle=False) as b:
                    if not np.array_equal(a["object_ids"], b["object_ids"]):
                        raise ValueError(f"{control}: changed control pairing")
                    ids.extend(a["object_ids"].astype(str).tolist())
                    ablated.extend(a["oracle_errors_deg"])
                    base.extend(b["oracle_errors_deg"])
                all_paths.extend([p, q])
            if len(ids) != 170 or len(set(ids)) != 170:
                raise ValueError("Input removals require 170 unique paired asteroids per seed")
            stats = ctx.stat(ablated, second=base)
            stats.update(seed=seed, object_ids=ids, label=f"{label}, seed {seed}",
                         control=control, ablated_median=float(np.median(ablated)), baseline_median=float(np.median(base)))
            ctx.check(result, f"{control} seed {seed}: original mean control", stats["mean_recomputed"], summary[control]["gap_by_seed_deg"][str(seed)])
            stats["median_stats_check"] = "PASS: same paired per-seed estimator as median-stats Removal resolver; no macro for this plotted control."
            rows.append(stats)
            old_gaps.append(stats["mean_recomputed"])
        gaps = [r["median"] for r in rows]
        gap_by_control[control] = gaps
        rows.append(dict(label=label + ", black bar", control=control, median=float(np.mean(gaps)), n=170,
                         original_mean=float(np.mean([r["mean_recomputed"] for r in rows])),
                         seed_range=[min(gaps), max(gaps)], statistic="mean_of_seed_differences_of_medians",
                         median_stats_check="PASS: original five-seed mean/range convention; no plotted-control macro in median-stats.json."))
        result["points"].extend(rows)
    fig, (ax,) = f2_canvas(ctx, "input-removal", [[149.559375, 35.205859, 310.52, 253.877891]])
    source = function_source(NETTESTS_RENDERER, "render_appendix_checks")
    start = source.index('    # Okabe-Ito')
    source = 'def draw():\n' + source[start:source.index('    if not TRAIN_TEST_REPORT.exists():')]
    source = replace_once(source, '    fig, ax = plt.subplots(figsize=(6.6, 4.2))\n', '')
    start = source.index('        gaps = []')
    end = source.index('        for colour, marker, gap', start)
    source = source[:start] + '        gaps = gap_by_control[control]\n' + source[end:]
    source = replace_once(source, 'Change in mean oracle error after removal (degrees)',
                          'Change in median oracle error after removal (degrees)')
    source = replace_once(source, 'label="Mean"', 'label="Mean over seeds"')
    source = source[:source.index('    fig.tight_layout()')] + '''    ax.set_xlim(limits)
    ctx.save(fig, "input-removal")
'''
    # Preserve old auto-limits, extending only when a new median marker falls outside.
    old = np.r_[old_gaps, 0]
    limits = [float(old.min() - .05 * np.ptp(old)), float(old.max() + .05 * np.ptp(old))]
    original_limits = limits.copy()
    new = np.r_[np.array(list(gap_by_control.values())).ravel(), 0]
    if new.min() < limits[0]:
        limits[0] = float(new.min() - .05 * np.ptp(new))
    if new.max() > limits[1]:
        limits[1] = float(new.max() + .05 * np.ptp(new))
    result["axis_changes"] = dict(original_xlim=original_limits, median_xlim=limits)
    namespace = dict(np=np, plt=plt, ctx=ctx, fig=fig, ax=ax, seeds=seeds,
                     rows=[(label, folder, control) for label, control in controls],
                     gap_by_control=gap_by_control, limits=limits)
    exec(compile(source, str(NETTESTS_RENDERER), "exec"), namespace)
    namespace["draw"]()
    ctx.sources_for(result, [NETTESTS_RENDERER, summary_path,
                             FOLLOWUP / "source-op-nettests/repro/own_period/summarize_controls.py", *dict.fromkeys(all_paths)])
    result["means_retained"] = ["The black bar averages five seed-specific differences of medians, matching median-stats.json; seed ranges stay seed ranges."]
    result["source_note"] = "No Removal macro for these 11 plotted controls occurs in the 176-entry median-stats.json. All 55 original seed mean differences are recovered against controls-own-summary.json, with exact paired asteroid identities, before conversion."
    result["caption_change"] = 'Replace "Change in the mean oracle error" with "Change in the median oracle error"; clarify that each marker is median(removal) minus median(original) for one seed and the black bar is the mean over five seeds.'
    result["visual_message"] = (
        "Geometry and brightness remain the largest degradations; Fourier/amplitude changes remain small. "
        "Every control's seed-average bar stays positive. Removing the period scalar now improves seeds 777 "
        "and 42 slightly (-0.2511 and -0.0154 deg), whereas all five original mean differences were positive. "
        f'The x limits extend from {original_limits[0]:.3f}--{original_limits[1]:.3f} to '
        f'{limits[0]:.3f}--{limits[1]:.3f} deg to include the geometry/numerical median differences.'
    )


def period_sensitivity(ctx):
    """Published-period perturbations, preserving their original 170 asteroids."""
    from collections import defaultdict

    import matplotlib.pyplot as plt

    result = ctx.figure("period-sensitivity")
    path = FOLLOWUP / "data/reliability-20260915/run-1/scored-rows.json"
    payload = ctx.read(path)["payload"]
    table_path = path.with_name("reliability-report.csv")
    import csv

    with table_path.open(newline="", encoding="utf-8") as stream:
        table = {r["condition"]: r for r in csv.DictReader(stream) if r["model_id"] == "k3"}
    factors = ("0.5", "0.99", "0.999", "0.9999", None, "1.0001", "1.001", "1.01", "2.0")
    labels = ["0.5", "0.99", "0.999", "0.9999", "1", "1.0001", "1.001", "1.01", "2"]
    keys = ["full" if f is None else f"period-{f}" for f in factors]
    rows = []
    vectors = {}
    for condition, label in zip(keys, labels, strict=True):
        grouped = defaultdict(list)
        for r in payload:
            if r["condition"] == condition and r["status"] != "ineligible":
                grouped[r["object_id"]].append(r)
        ids = sorted(grouped)
        values = np.array([np.mean([90.0 if r["status"] == "failed" else r["error_deg"] for r in grouped[i]]) for i in ids])
        folds = [grouped[i][0]["fold"] for i in ids]
        stats = ctx.stat(values, seed=20260915, folds=folds, order="numpy_replicate_major")
        stats.update(label=label, condition=condition, object_ids=ids, errors_deg=values.tolist())
        expected = table[condition]
        ctx.check(result, condition + ": recovered original mean", stats["mean_recomputed"], float(expected["mean_error_deg"]))
        ctx.check(result, condition + ": original mean interval control", stats["mean_interval_recomputed"],
                  [float(expected["mean_error_bootstrap_lower_95"]), float(expected["mean_error_bootstrap_upper_95"])])
        ctx.check(result, condition + ": original asteroid count", len(ids), 170)
        if condition in ("period-0.5", "period-2.0"):
            ctx.match(result, stats, "PeriodHalf" if condition == "period-0.5" else "PeriodDouble")
        else:
            stats["median_stats_check"] = "PASS: original cohort/RNG; no absolute-error median macro for this factor."
        vectors[condition] = (values, folds, ids)
        rows.append(stats)
    paired = []
    for condition in keys:
        if condition in ("full", "period-0.5", "period-2.0"):
            continue
        values, folds, ids = vectors[condition]
        if ids != vectors["full"][2]:
            raise ValueError("Period factors changed the matched asteroid set")
        pair = ctx.stat(values, second=vectors["full"][0], seed=20260915, folds=folds, order="numpy_replicate_major")
        pair["condition"] = condition
        expected = ctx.macros["PeriodSensitivityMaxChange"]["condition_comparisons"][condition]
        ctx.check(result, condition + ": median-stats paired difference", pair["median"], expected["median"])
        ctx.check(result, condition + ": median-stats paired interval", pair["interval"], expected["interval"])
        paired.append(pair)
    ctx.check(result, "PeriodSensitivityMaxChange", max(abs(r["median"]) for r in paired), ctx.macros["PeriodSensitivityMaxChange"]["median"])
    result["validation_only_not_drawn"] = paired
    standard = ctx.stat(ctx.evidence.baseline["standard_errors_deg"])
    standard.update(label="Six standard starting poles", object_ids=ctx.evidence.baseline["object_ids"].astype(str).tolist())
    ctx.match(result, standard, "KThreeStandardMean")
    result["points"] = [*rows, standard]
    fig, (ax,) = f2_canvas(ctx, "period-sensitivity", [[37.10625, 34.66875, 353.23, 187.72]])
    source = function_source(MAPS_RENDERER, "render_period_sensitivity")
    source = 'def draw():\n' + source[source.index('    x = np.arange'):]
    source = replace_once(source, '    fig, ax = plt.subplots(figsize=(5.6, 3.3))\n', '')
    source = replace_once(source, '_standard_mean()', 'standard_median')
    source = replace_once(source, 'Six standard starting poles', 'Six standard starting poles (median)')
    source = replace_once(source, 'Mean oracle error (degrees)', 'Median oracle error (degrees)')
    source = source[:source.index('    fig.tight_layout()')] + '    ctx.save(fig, "period-sensitivity")\n'
    namespace = dict(np=np, plt=plt, ctx=ctx, fig=fig, ax=ax, keys=keys, labels=labels,
                     means=np.array([r["median"] for r in rows]),
                     low=np.array([r["interval"][0] for r in rows]), high=np.array([r["interval"][1] for r in rows]),
                     standard_median=standard["median"])
    exec(compile(source, str(MAPS_RENDERER), "exec"), namespace)
    namespace["draw"]()
    ctx.sources_for(result, [MAPS_RENDERER, path, table_path, REPO / "lc_pipeline/k3/reliability_scoring.py",
                             FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz"])
    result["means_retained"] = ["Repeated selections are averaged within asteroid before taking its cohort median, matching median-stats.json."]
    result["caption_change"] = 'Replace "Mean oracle error" with "Median oracle error"; retain the DAMIT-period sensitivity experiment, square marker and 95% bootstrap intervals.'
    result["visual_message"] = (
        "Half and double periods remain worse than the DAMIT period but better than standard poles. "
        "Medians are 17.04 and 14.86 deg; the largest change within 1% is 1.0161 deg. "
        "The original 0--30 deg axis includes all median intervals."
    )


FIGURES = {
    "generalization-checks": generalization_checks,
    "sampling-caps": sampling_caps,
    "data-degradation": data_degradation,
    "latitude": latitude,
    "candidate-count": candidate_count,
    "inversion-benchmark": inversion_benchmark,
    "interpretation": interpretation,
    "shapley": shapley,
    "training-test": training_test,
    "reference-reassignment": reference_reassignment,
    "input-removal": input_removal,
    "period-sensitivity": period_sensitivity,
}


def validate_artifacts(ctx, names):
    """Check the saved files, rather than relying on the in-memory figures."""
    from PIL import Image

    for name in names:
        path = ctx.out / f"{name}.pdf"
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        parts = next(s for s in info.splitlines() if s.startswith("Page size:")).split()
        size = [float(parts[2]), float(parts[4])]
        ctx.check(ctx.results[name], "Saved PDF media box (pt)", size,
                  ctx.results[name]["page_size_pt"])
        text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
        if "median" not in text.lower() or "Mean oracle error" in text:
            raise ValueError(f"{path}: incorrect statistic label")
        with Image.open(ctx.out / f"{name}.png") as preview:
            if preview.width < 500 or preview.height < 300:
                raise ValueError(f"{name}: undersized PNG preview {preview.size}")
        ctx.results[name]["artifact_validation"] = "PASS: readable PDF, expected media box, median labels, PNG preview."
    if any(name.endswith((".refine_chain", ".period_cost")) for name in sys.modules):
        raise ValueError("A forbidden live-run module was imported")


def write_report(ctx, names):
    lines = ["# F1 median figures", "", "PASS: six F1 figures generated from scratch." if len(names) == 6
             else "PASS: selected F1 figures regenerated.", "",
             "PDFs, PNG previews and every plotted value: `median-figures/`.",
             "No F2 outputs are required. No manuscript edits, solver runs, GPU use or commits.",
             "", "## Verification", "",
             "Each matching median-stats macro is checked at full precision (absolute tolerance 1e-10), "
             "including n and all available drawn CI endpoints. Quantities without a macro are "
             "recomputed from the same frozen identities and estimator; their absence is explicit below.",
             "Within-asteroid successful selections remain averaged before the across-asteroid median.", ""]
    for name in names:
        result = ctx.results[name]
        lines.extend([f"## {name}", "", result["visual_message"], "",
                      "Caption: " + result["caption_change"], "",
                      "Means retained: " + (" ".join(result["means_retained"]) or "None."), ""])
        if "bootstrap_note" in result:
            lines.extend([result["bootstrap_note"], ""])
        if "source_note" in result:
            lines.extend([result["source_note"], ""])
        lines.extend(["### Drawn values", "", "| Point | Median / plotted quantity | n | CI95 | Check |",
                      "|---|---:|---:|---|---|"])
        for point in result["points"]:
            label = point.get("label", point.get("condition", point.get("series", point.get("panel", "point"))))
            if "bin_low_deg" in point:
                label += f' [{point["bin_low_deg"]},{point["bin_high_deg"]}]'
            if "starts" in point:
                label += f' {point["starts"]} starts'
            if "k" in point:
                label += f' K={point["k"]}'
            value = point.get("median", point.get("ratio_of_means"))
            if value is None:
                value = "No period" if "condition" in point else json.dumps(point, ensure_ascii=False)
            elif isinstance(value, (int, float)):
                value = f"{value:.12g}"
            ci = point.get("interval")
            ci_text = "—" if ci is None else f"{ci[0]:.12g}, {ci[1]:.12g}"
            check = ", ".join(point.get("median_stats_macros", [])) or point.get("median_stats_check", "Frozen source / same estimator")
            lines.append(f'| {label} | {value} | {point.get("n", "—")} | {ci_text} | {check} |')
        lines.extend(["", "### Checks", ""])
        for check in result["checks"]:
            lines.append(f'- {check["status"]}: {check["label"]}; drawn/recomputed '
                         f'`{check["actual"]}`; expected `{check["expected"]}`.')
        lines.extend(["", "### Source files", ""])
        lines.extend(f"- `{path}`" for path in result["source_files"])
        lines.append("")
    lines.extend(["## Scope and artifacts", "", "Only `repro/own_period/median_figures.py` was created in the repository. "
                  "All input and manuscript hashes were unchanged during generation. "
                  "The six PDFs and six PNGs were checked for valid dimensions and median labels. "
                  "`values.json` records every point, interval, count, geometric radius, histogram bin and ratio.", ""])
    (ctx.out.parent / "f1-report.md").write_text("\n".join(lines), encoding="utf-8")


def validate_f2_training(ctx):
    """The training scatter has no across-asteroid statistic to label median."""
    from PIL import Image

    result = ctx.results["training-test"]
    path = ctx.out / "training-test.pdf"
    info = subprocess.check_output(["pdfinfo", str(path)], text=True)
    parts = next(s for s in info.splitlines() if s.startswith("Page size:")).split()
    ctx.check(result, "Saved PDF media box (pt)", [float(parts[2]), float(parts[4])], result["page_size_pt"])
    text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
    if "Oracle error as a training asteroid" not in text or "Oracle error as a test asteroid" not in text:
        raise ValueError("Training scatter's original coordinate labels changed")
    with Image.open(ctx.out / "training-test.png") as preview:
        if preview.width < 500 or preview.height < 300:
            raise ValueError("Training scatter preview is undersized")
    result["artifact_validation"] = "PASS: readable PDF, original media box/coordinate labels, PNG preview; no population statistic drawn."


def write_f2_report(ctx, names):
    """F2-only report; never overwrite the completed F1 report."""
    lines = ["# F2 median figures", "", f"PASS: {len(names)} F2 figures generated from frozen evidence.", "",
             "Only `repro/own_period/median_figures.py` is edited. PDFs, PNG previews and `values.json` "
             "are in `median-figures/`. No manuscript edits, solver runs, GPU use or commits.", "",
             "## Verification", "",
             "Matching median-stats.json medians, asteroid identities/counts and available intervals are "
             "checked at absolute tolerance 1e-10. The 176-entry file has no plotted Shapley-effect or "
             "removal-control macros, nor intermediate interpretation-bin medians or all period-factor "
             "absolute medians. Such points are explicitly source-checked, not claimed to match absent macros. "
             "Every drawn number (including unchanged scatter/map values, seed points, histogram bins and "
             "draw medians) is in values.json. Validation-only statistics are separately marked and not plotted.", "",
             "Axes boxes and media boxes follow the current paper PDFs. Fonts, colours, markers, panels and "
             "fixed data definitions are retained. Within-asteroid/network averages, additive Shapley "
             "shares and averages over training seeds/randomization draws follow the frozen estimator.", ""]
    for name in names:
        result = ctx.results[name]
        lines.extend([f"## {name}", "", result["visual_message"], "",
                      "Caption: " + result["caption_change"], "",
                      "Means retained: " + (" ".join(result["means_retained"]) or "None."), ""])
        for key in ("source_note", "bootstrap_note", "statistic_label_exception"):
            if key in result:
                lines.extend([result[key], ""])
        lines.extend(["### Drawn values", "", "| Point | Drawn value (degrees unless noted) | n | CI95 / seed range | Check |",
                      "|---|---:|---:|---|---|"])
        for point in result["points"]:
            label = point["label"]
            if "x_test_error_deg" in point:
                value = f'test={point["x_test_error_deg"]:.12g}; training mean={point["y_training_mean_error_deg"]:.12g}'
            else:
                value = f'{point["median"]:.12g}'
            ci = point.get("interval") or point.get("seed_range") or point.get("randomization_interval95")
            interval = "—" if ci is None else f"{ci[0]:.12g}, {ci[1]:.12g}"
            check = ", ".join(point.get("median_stats_macros", [])) or point.get("median_stats_check", "PASS: frozen source")
            lines.append(f'| {label} | {value} | {point.get("n", "—")} | {interval} | {check} |')
        if name == "shapley":
            lines.extend(["", "| Group | Original mean effect | Median effect | Panel (a) share (%) | Five-seed share range (%) |",
                          "|---|---:|---:|---:|---|"])
            for p in result["points"]:
                lo, hi = p["share_seed_range_percent"]
                lines.append(f'| {p["label"]} | {p["original_mean"]:.12g} | {p["median"]:.12g} | {p["share_percent"]:.12g} | {lo:.12g}, {hi:.12g} |')
        lines.extend(["", "### Checks", ""])
        for check in result["checks"]:
            lines.append(f'- {check["status"]}: {check["label"]}; actual `{check["actual"]}`; expected `{check["expected"]}`.')
        lines.extend(["", result["artifact_validation"], "", "### Source files", ""])
        lines.extend(f"- `{path}`" for path in result["source_files"])
        lines.append("")
    lines.extend(["## Execution", "", "Used `taskset -c 12-23 nice -n 19`, one Python process, CPU only. "
                  "All recorded read-only source, renderer, original-PDF and manuscript hashes were unchanged "
                  "during generation. Live refine-chain/period-cost modules were neither imported nor edited.", ""])
    (ctx.out.parent / "f2-report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", required=True, help="Figure names, separated by spaces or commas")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    names = list(dict.fromkeys(name for group in args.figures for name in group.split(",")))
    unknown = set(names) - FIGURES.keys()
    if unknown:
        parser.error(f"Unknown figures: {sorted(unknown)}; available: {list(FIGURES)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # All plotting caches are temporary and live inside the requested scratch output.
    with tempfile.TemporaryDirectory(prefix=".matplotlib-", dir=args.out_dir) as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                             "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                             "ytick.labelsize": 8, "pdf.fonttype": 42, "savefig.bbox": None})
        ctx = Context(args.out_dir)
        for name in names:
            print(f"Rendering {name}", flush=True)
            with plt.rc_context():
                if name in F2_NAMES:
                    plt.rcParams["savefig.dpi"] = 300
                FIGURES[name](ctx)
        validate_artifacts(ctx, [name for name in names if name != "training-test"])
        if "training-test" in names:
            validate_f2_training(ctx)
        for path, original_hash in ctx.sources.items():
            if digest(Path(path)) != original_hash:
                raise ValueError(f"Read-only input changed: {path}")
        values_path = args.out_dir / "values.json"
        values = json.loads(values_path.read_text()) if values_path.exists() else {"figures": {}}
        values.update(schema="delphi.paper1-median-figures.v1", median_stats_sha256=digest(DATA / "median-stats.json"),
                      paired_convention=ctx.frozen["paired_convention"], bootstrap_replicates=ctx.resamples)
        values["figures"].update(ctx.results)
        values["source_sha256"] = dict(values.get("source_sha256", {}), **ctx.sources)
        values_path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
        f1_names = [name for name in names if name not in F2_NAMES]
        f2_names = [name for name in names if name in F2_NAMES]
        if f1_names:
            write_report(ctx, f1_names)
        if f2_names:
            write_f2_report(ctx, f2_names)
        print(f"PASS: {len(names)} figures; {sum(len(r['checks']) for r in ctx.results.values())} numeric checks", flush=True)


if __name__ == "__main__":
    main()
