"""Summarize the end-to-end period pilot and draw its accuracy/cost figure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from physical import CATALOG


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(period, reference, tolerance=.01) -> bool:
    return (period is not None and math.isfinite(float(period))
            and abs(float(period) / reference - 1) <= tolerance + 1e-12)


def raw_method_rows(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for row in map(json.loads, path.read_text().splitlines()):
        if row.get("method") not in ("fourier_gpu", "pdm") or row.get("repeat") != 0 or row.get("mode") != "warm":
            continue
        key = row["object_id"], row["method"]
        if key in result:
            raise ValueError(f"duplicate raw method row: {key}")
        result[key] = row
    return result


def summarize(search_path: Path, candidate_path: Path, screen_path: Path, score_path: Path) -> dict:
    catalog = {row["object_id"]: row for row in map(json.loads, (CATALOG / "catalog.jsonl").read_text().splitlines())}
    searches = raw_method_rows(search_path)
    candidates = [json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()]
    screens = [json.loads(line) for line in screen_path.read_text().splitlines() if line.strip()]
    score_rows = {row["object_id"]: row for row in map(json.loads, score_path.read_text().splitlines())}
    if len(candidates) != 170 or len(screens) != 20 or len(score_rows) != 20:
        raise ValueError("expected 170 candidate rows and matching 20-object pilot rows")

    def reference(oid):
        return float(catalog[oid]["solutions"][0]["period_hours"])

    full = {}
    for method in ("fourier_gpu", "pdm"):
        rows = [searches[(row["object_id"], method)] for row in candidates]
        correct = sum(bool(row.get("ranked_periods")) and within(row["ranked_periods"][0], reference(row["object_id"])) for row in rows)
        full[method] = {
            "n": 170, "within_1pct_count": correct, "within_1pct_fraction": correct / 170,
            "mean_search_seconds": float(np.mean([row["seconds"] for row in rows])),
        }
    ceiling = sum(any(within(candidate["period_hours"], reference(row["object_id"])) for candidate in row["candidates"])
                  for row in candidates)
    full["alias_candidate_ceiling"] = {
        "n": 170, "within_1pct_count": ceiling, "within_1pct_fraction": ceiling / 170,
        "mean_search_seconds": float(np.mean([row["search_seconds"] for row in candidates])),
        "selected_accuracy": False,
        "mean_candidate_count": float(np.mean([row["candidate_count"] for row in candidates])),
        "maximum_candidate_count": max(row["candidate_count"] for row in candidates),
    }

    pilot = {}
    pilot_ids = [row["object_id"] for row in screens]
    for method in ("fourier_gpu", "pdm"):
        rows = [searches[(oid, method)] for oid in pilot_ids]
        correct = sum(bool(row.get("ranked_periods")) and within(row["ranked_periods"][0], reference(row["object_id"])) for row in rows)
        pilot[method] = {"n": 20, "within_1pct_count": correct, "within_1pct_fraction": correct / 20,
                         "mean_search_seconds": float(np.mean([row["seconds"] for row in rows]))}
    physical_correct = sum(within(row.get("selected_period_hours"), reference(row["object_id"])) for row in screens)
    physical_times = [row["screen_plus_search_seconds"] for row in screens]
    pilot["quick_physical"] = {
        "n": 20, "within_1pct_count": physical_correct, "within_1pct_fraction": physical_correct / 20,
        "ambiguous_count": sum(row["ambiguous"] for row in screens),
        "mean_pipeline_seconds": float(np.mean(physical_times)),
        "median_pipeline_seconds": float(np.median(physical_times)),
        "maximum_pipeline_seconds": float(np.max(physical_times)),
    }
    scored_correct, score_times = 0, []
    for oid in pilot_ids:
        row = score_rows[oid]
        chosen = max(row["scores"], key=lambda item: (item["peak_logit"], -item["candidate_index"]))
        scored_correct += within(chosen["period_hours"], reference(oid))
        score_times.append(sum(item["seconds"] for item in row["scores"]))
    pilot["delphi_peak_score"] = {
        "n": 20, "within_1pct_count": scored_correct, "within_1pct_fraction": scored_correct / 20,
        "mean_score_seconds": float(np.mean(score_times)),
        "mean_pipeline_seconds": float(np.mean([
            next(item["search_seconds"] for item in candidates if item["object_id"] == oid) + score_time
            for oid, score_time in zip(pilot_ids, score_times)
        ])),
        "calibrated_period_probability": False,
    }
    pilot_ceiling = sum(any(within(candidate["period_hours"], reference(row["object_id"])) for candidate in row["candidates"])
                        for row in candidates if row["object_id"] in set(pilot_ids))
    pilot["alias_candidate_ceiling"] = {"n": 20, "within_1pct_count": pilot_ceiling,
                                          "within_1pct_fraction": pilot_ceiling / 20,
                                          "selected_accuracy": False,
                                          "mean_search_seconds": float(np.mean([
                                              row["search_seconds"] for row in candidates
                                              if row["object_id"] in set(pilot_ids)
                                          ]))}
    passed = physical_correct >= pilot["fourier_gpu"]["within_1pct_count"] and pilot["quick_physical"]["ambiguous_count"] <= 2
    failures = []
    if physical_correct < pilot["fourier_gpu"]["within_1pct_count"]:
        failures.append("quick physical selection reduced period accuracy relative to GPU Fourier")
    if pilot["quick_physical"]["ambiguous_count"] > 2:
        failures.append("more than 10% of pilot objects remained ambiguous")
    if pilot["quick_physical"]["mean_pipeline_seconds"] >= 30:
        failures.append("mean pilot time exceeded 30 seconds per object")
    return {
        "schema": "delphi.end-to-end-period-benchmark.v1",
        "scope": "development experiment on previously inspected DAMIT objects",
        "full_170": full,
        "difficulty_enriched_pilot_20": pilot,
        "pilot_gate": {"passed": passed, "failures": failures, "full_physical_screen_started": False},
        "interpretation": (
            "The candidate union has high oracle coverage, but neither the quick physical fit nor the frozen DeLPHI "
            "peak score reliably selects the rotation period. The prespecified pilot gate therefore stops the full run."
        ),
        "inputs": {str(path): sha256(path) for path in (search_path, candidate_path, screen_path, score_path)},
        "script_sha256": sha256(Path(__file__)),
    }


def plot(summary: dict, pdf: Path, png: Path) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 15,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    panels = [
        (axes[0], summary["full_170"],
         [("pdm", "PDM"), ("fourier_gpu", "GPU Fourier"),
          ("alias_candidate_ceiling", "Alias candidates")],
         "All 170 objects", "Alias candidates show oracle coverage"),
        (axes[1], summary["difficulty_enriched_pilot_20"],
         [("fourier_gpu", "GPU Fourier"), ("delphi_peak_score", "DeLPHI score choice"),
          ("quick_physical", "Quick physical choice"),
          ("alias_candidate_ceiling", "Alias candidates")],
         "20-object difficulty-enriched pilot", "The pilot is not a representative subsample"),
    ]
    colors = {"pdm": "#7a7a7a", "fourier_gpu": "#2878b5", "quick_physical": "#c44e52",
              "delphi_peak_score": "#8172b2", "alias_candidate_ceiling": "#55a868"}
    for ax, data, entries, title, note in panels:
        xmax = max((data[key].get("mean_pipeline_seconds", data[key].get("mean_search_seconds", 0))
                    for key, _ in entries), default=1)
        for key, label in entries:
            seconds_key = "mean_pipeline_seconds" if "mean_pipeline_seconds" in data[key] else "mean_search_seconds"
            x = data[key][seconds_key]
            y = 100 * data[key]["within_1pct_fraction"]
            oracle = key == "alias_candidate_ceiling"
            ax.scatter(x, y, s=100, color="white" if oracle else colors[key], edgecolor=colors[key],
                       linewidth=2 if oracle else 1, zorder=3)
            if key == "alias_candidate_ceiling":
                x_offset, y_offset, horizontal, vertical = 8, -8, "left", "top"
            elif key == "fourier_gpu":
                x_offset, y_offset, horizontal, vertical = -8, -4, "right", "top"
            elif key == "quick_physical":
                x_offset, y_offset, horizontal, vertical = 8, 15, "left", "baseline"
            elif x > 0.85 * xmax:
                x_offset, y_offset, horizontal, vertical = -8, 5, "right", "baseline"
            else:
                x_offset, y_offset, horizontal, vertical = 8, 5, "left", "baseline"
            ax.annotate(
                f"{label}\n{data[key]['within_1pct_count']}/{data[key]['n']}",
                (x, y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal,
                va=vertical,
                fontsize=10,
            )
        ax.set_xlim(0, xmax * 1.75 + .5)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Mean measured time (s/object)")
        ax.set_ylabel("Period within 1% of DAMIT value (%)")
        ax.set_title(title)
        ax.grid(alpha=.25, linewidth=.7)
        ax.text(0, -.24, note, transform=ax.transAxes, fontsize=10, color="#444444")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def report(summary: dict) -> str:
    full, pilot = summary["full_170"], summary["difficulty_enriched_pilot_20"]
    failures = "\n".join(f"- {item}" for item in summary["pilot_gate"]["failures"])
    return f"""# End-to-end period pilot\n\n## Material Passport\n\n- Mode: code experiment execution and validation\n- Status: pilot gate failed; full physical screen not started\n- Scope: previously inspected 170-object DAMIT development sample\n- Reference use: scoring only, after candidate generation and selection\n\n## Full-sample fast search\n\n- GPU Fourier top-period accuracy: {full['fourier_gpu']['within_1pct_count']}/170 ({100*full['fourier_gpu']['within_1pct_fraction']:.1f}%), mean {full['fourier_gpu']['mean_search_seconds']:.2f} s/object.\n- PDM top-period accuracy: {full['pdm']['within_1pct_count']}/170 ({100*full['pdm']['within_1pct_fraction']:.1f}%), mean {full['pdm']['mean_search_seconds']:.2f} s/object.\n- Explicit P/2, P, and 2P candidate coverage: {full['alias_candidate_ceiling']['within_1pct_count']}/170 ({100*full['alias_candidate_ceiling']['within_1pct_fraction']:.1f}%). This is an oracle candidate-set ceiling, not automatic accuracy.\n- Candidate count: mean {full['alias_candidate_ceiling']['mean_candidate_count']:.2f}, maximum {full['alias_candidate_ceiling']['maximum_candidate_count']}.\n\n## Physical pilot\n\nThe pilot deliberately includes four difficult objects and is not representative of the 170-object sample.\n\n- GPU Fourier: {pilot['fourier_gpu']['within_1pct_count']}/20 correct.\n- Lowest-chi-squared quick physical fit: {pilot['quick_physical']['within_1pct_count']}/20 correct.\n- Frozen DeLPHI peak-score choice: {pilot['delphi_peak_score']['within_1pct_count']}/20 correct.\n- Candidate-set ceiling: {pilot['alias_candidate_ceiling']['within_1pct_count']}/20.\n- Ambiguous physical results: {pilot['quick_physical']['ambiguous_count']}/20.\n- Mean search plus physical-screen time: {pilot['quick_physical']['mean_pipeline_seconds']:.2f} s/object.\n\n## Gate verdict\n\n{failures}\n\nThe full 170-object physical screen was not started. The quick fit is useful for exposing ambiguity, but it does not safely choose between half-period and full-period solutions. These results are development evidence and should not be promoted to the manuscript as a successful end-to-end DeLPHI result.\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_directory / name for name in
               ("benchmark-summary.json", "benchmark-report.md", "accuracy-cost.pdf", "accuracy-cost.png")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("benchmark output already exists")
    result = summarize(args.search, args.candidates, args.screen, args.scores)
    outputs[0].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    outputs[1].write_text(report(result))
    plot(result, outputs[2], outputs[3])


if __name__ == "__main__":
    main()
