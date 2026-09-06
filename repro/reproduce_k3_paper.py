"""Rebuild K3 manuscript results and descriptive plots from frozen predictions.

Run with ``python -m repro.reproduce_k3_paper --help``. No training or
checkpoint selection occurs; all output directories must be new.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

from lc_pipeline.k3.figures import build_publication_figures
from lc_pipeline.k3.release import build_publication_release

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def reproduce(root: Path, comparator: Path, catalog: Path, output: Path) -> dict:
    """Verify release bindings, regenerate results, and export OOF diagnostics."""
    if output.exists():
        raise ValueError("output must be a new directory")
    original = json.loads((root / "release/publication-summary.json").read_text())
    downstream = original["downstream"]
    if digest(catalog) != downstream["catalog_sha256"]:
        raise ValueError("catalog differs from the benchmark-bound catalog")
    output.mkdir(parents=True)
    build_publication_release(
        artifact_root=root, comparator_path=comparator,
        output_json=output / "publication-summary.json",
        output_tex=output / "k3-results.tex",
    )
    for name in ("publication-summary.json", "k3-results.tex"):
        if digest(output / name) != digest(root / "release" / name):
            raise ValueError(f"regenerated {name} differs from frozen release")
    build_publication_figures(
        artifact_root=root, comparator_path=comparator,
        release_summary_path=output / "publication-summary.json",
        output_directory=output / "figures",
    )
    records = {row["object_id"]: row for row in map(json.loads, catalog.read_text().splitlines())}
    ensemble = root / "evaluations/real-oof-ensemble.npz"
    with np.load(ensemble, allow_pickle=False) as data:
        ids = data["object_ids"].astype(str)
        errors = data["oracle_errors_deg"].astype(float)
        folds = data["folds"].astype(int)
        timing = data["inference_wall_seconds"].astype(float)
        axes = data["refined_axes"].copy()
    # Recompute the axial oracle independently of the archived error column.
    recomputed = []
    for oid, predictions in zip(ids, axes, strict=True):
        targets = np.asarray([solution["vector"] for solution in records[oid]["solutions"]],
                             dtype=np.float64)
        predictions = predictions / np.linalg.norm(predictions, axis=1, keepdims=True)
        targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
        recomputed.append(np.degrees(np.arccos(np.clip(np.abs(predictions @ targets.T), 0, 1))).min())
    max_error_difference = float(np.max(np.abs(np.asarray(recomputed) - errors)))
    if max_error_difference > 1e-3:
        raise ValueError("independent oracle recomputation differs from archived errors")
    rows = []
    for oid, error, fold, seconds in zip(ids, errors, folds, timing, strict=True):
        record = records[oid]
        solutions = record["solutions"]
        rows.append({
            "object_id": oid, "fold": int(fold), "oracle_error_deg": float(error),
            "primary_abs_beta_deg": abs(solutions[0]["beta_deg"]),
            "period_hours": solutions[0]["period_hours"],
            "n_observations": record["lightcurve"]["n_observations"],
            "n_solutions": len(solutions),
            "primary_quality_flag": solutions[0]["quality_flag"],
            "inference_wall_seconds": float(seconds),
        })
    with (output / "oof-object-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def save(fig, name):
        fig.savefig(output / "figures" / name, bbox_inches="tight",
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    fig, panel = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
    panel.set(xlim=(0, 1), ylim=(0, 1))
    panel.axis("off")
    stages = ["Photometry + geometry\nExternally supplied period",
              "64 phase bins / epoch\nCircular CNN + set Transformer",
              "6,144 trial axes\nAxis–geometry dot products",
              "Five-model mean scores\n3 peaks + bounded refinement"]
    for index, label in enumerate(stages):
        x = .125 + .25 * index
        panel.text(x, .6, label, ha="center", va="center", fontsize=8,
                   bbox={"boxstyle": "round,pad=.5", "facecolor": "#eaf0f5"})
        if index < 3:
            panel.annotate("", xy=(x + .15, .6), xytext=(x + .1, .6),
                           arrowprops={"arrowstyle": "->"})
    panel.text(.5, .16, "Training: synthetic pretraining → real-object fine-tuning; positive axes vs sampled negatives",
               ha="center", fontsize=9)
    save(fig, "k3_architecture.pdf")

    fig, panels = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)
    features = [("primary_abs_beta_deg", "Primary reference |latitude| (deg)"),
                ("period_hours", "Supplied period (h)"),
                ("n_observations", "Observations"),
                ("n_solutions", "Qualifying reference solutions")]
    for panel, (key, label) in zip(panels.flat, features, strict=True):
        panel.scatter([row[key] for row in rows], errors, s=12, alpha=.6)
        panel.set(xlabel=label, ylabel="Oracle@3 error (deg)", ylim=(0, 90))
        if key in ("period_hours", "n_observations"):
            panel.set_xscale("log")
    save(fig, "k3_error_properties.pdf")

    controls = original["metrics"]["diagnostic_gap_deg_by_seed"]
    names = ["zero-numeric", "brightness", "geometry", "period-scalar", "sampling-summary"]
    values = np.array([[row[name] for name in names] for row in controls.values()])
    fig, panel = plt.subplots(figsize=(7, 3), constrained_layout=True)
    for row in values:
        panel.scatter(np.arange(len(names)), row, alpha=.65, s=22)
    panel.plot(np.arange(len(names)), values.mean(axis=0), "k_", markersize=16)
    panel.axhline(0, color="gray", linewidth=.7)
    panel.set(xticks=np.arange(len(names)), xticklabels=names,
              ylabel="Removal − original mean error (deg)")
    save(fig, "k3_control_sensitivity.pdf")

    fig, panel = plt.subplots(figsize=(6, 3), constrained_layout=True)
    panel.scatter(np.degrees(np.arctan2(axes[..., 1], axes[..., 0])).ravel(),
                  np.degrees(np.arcsin(np.clip(axes[..., 2], -1, 1))).ravel(), s=7, alpha=.4)
    panel.set(xlabel="Representative axis longitude (deg)",
              ylabel="Representative axis latitude (deg)", xlim=(-180, 180), ylim=(-90, 90))
    save(fig, "k3_candidate_sky.pdf")

    calibration = original["calibration"]
    fig, panel = plt.subplots(figsize=(4, 3), constrained_layout=True)
    panel.plot([.9, .95], [calibration["empirical_coverage90"],
                         calibration["empirical_coverage95"]], "o", label="170 OOF objects")
    panel.plot([.85, 1], [.85, 1], "--", color="gray")
    panel.set(xlabel="Nominal containment", ylabel="Empirical containment",
              xlim=(.85, 1.01), ylim=(.85, 1.01))
    panel.legend()
    save(fig, "k3_calibration.pdf")

    diagnostics = {
        "scope": "descriptive frozen OOF analyses; no model selection or causal inference",
        "folds": [{"fold": int(fold), "n": int(sum(folds == fold)),
                   "mean_error_deg": float(errors[folds == fold].mean()),
                   "median_error_deg": float(np.median(errors[folds == fold]))}
                  for fold in sorted(set(folds))],
        "inference_median_seconds": float(np.median(timing)),
        "inference_total_seconds": float(timing.sum()),
        "independent_oracle_max_difference_deg": max_error_difference,
        "source_sha256": {"ensemble": digest(ensemble), "catalog": digest(catalog)},
    }
    (output / "descriptive-results.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    report = {
        "schema": "delphi.k3-paper-reproduction.v1",
        "frozen_summary_and_macros_match": True,
        "files": {str(path.relative_to(output)): digest(path)
                  for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "comparators", "catalog", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.artifact_root, args.comparators, args.catalog, args.output)))


if __name__ == "__main__":
    main()
