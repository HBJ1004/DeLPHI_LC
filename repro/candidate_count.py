"""Number of candidates and the sky area they leave to search (CPU only).

The locked DeLPHI procedure reads three peaks from the averaged score map.
Here the same peak rule (one-ring local maxima taken in order of score, at
least 15 degrees apart, filled by score if too few) is applied with K = 1..6
to the saved averaged maps of the 170 test asteroids.  The networks are not
retrained for each K, and no final adjustment is applied, so K = 3
reproduces the stored grid oracle errors (169 of 170 asteroids exactly, see
the note in main()).

The search area follows the thesis manuscript: the fraction of a hemisphere of
evenly spread directions that lies within a cone around at least one end of
at least one candidate axis.  The cone radius is either the mean or the 90th
percentile of the oracle error at that K.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import compute_reference_baselines as base  # noqa: E402

from lc_pipeline.k3.grid import axial_healpix_grid, local_maximum_indices  # noqa: E402

FOLLOWUP = HERE.parents[1]
ENSEMBLE = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
OUTPUT = FOLLOWUP / "data/interpretation-20260926/candidate-count.json"
SEPARATION_DEG = 15.0
KS = (1, 2, 3, 4, 5, 6)


def peaks(scores, grid, count):
    vectors = grid.vectors
    chosen = []

    def consider(index):
        if index in chosen:
            return
        if all(np.degrees(np.arccos(min(1.0, abs(float(vectors[index] @ vectors[o]))))) >= SEPARATION_DEG
               for o in chosen):
            chosen.append(index)

    for index in local_maximum_indices(scores, grid):
        consider(int(index))
        if len(chosen) == count:
            break
    if len(chosen) < count:
        order = np.lexsort((np.arange(scores.size), -scores))
        for index in order:
            consider(int(index))
            if len(chosen) == count:
                break
    return vectors[chosen]


def fibonacci_hemisphere(count=8000):
    index = np.arange(count) + 0.5
    z = 1.0 - index / count            # upper hemisphere, uniform in z
    phi = np.pi * (1.0 + 5 ** 0.5) * index
    r = np.sqrt(1.0 - z * z)
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def covered_fraction(axes, radius_deg, sky):
    cosine = np.abs(sky @ np.asarray(axes).T)      # axial: both ends of each axis
    return float(np.mean(cosine.max(axis=1) >= np.cos(np.radians(radius_deg))))


def main():
    ensemble = np.load(ENSEMBLE)
    names = [str(value) for value in ensemble["object_ids"]]
    grid = axial_healpix_grid(32)
    poles = base.reference_poles()
    sky = fibonacci_hemisphere()
    rows = {}
    per_k_axes = {}
    for k in KS:
        errors, axes_list = [], []
        for index, name in enumerate(names):
            axes = peaks(ensemble["score_grids"][index].astype(np.float64), grid, k)
            axes_list.append(axes)
            errors.append(base.axial_error_deg(axes, poles[name]))
        errors = np.array(errors)
        per_k_axes[k] = axes_list
        rows[k] = {"errors": errors}
    # The saved maps are float32, whereas the stored errors came from float64
    # maps.  Where two peaks are nearly tied, rounding can change the chosen
    # peak.  This happens for one asteroid (asteroid_1719), and we allow only that.
    difference = np.abs(rows[3]["errors"] - ensemble["grid_oracle_errors_deg"])
    mismatched = [names[i] for i in np.flatnonzero(difference > 1e-3)]
    if len(mismatched) > 1:
        raise SystemExit(f"K=3 does not reproduce the stored grid oracle errors: {mismatched}")
    check = float(np.mean(rows[3]["errors"] - ensemble["grid_oracle_errors_deg"]))

    standard = base.standard_axes()
    standard_errors = np.array([base.axial_error_deg(standard, poles[name]) for name in names])
    summary = {"check_k3_mean_difference_deg": check, "check_k3_mismatched": mismatched, "k": []}
    for k in KS:
        errors = rows[k]["errors"]
        mean_radius = float(errors.mean())
        p90_radius = float(np.percentile(errors, 90))
        summary["k"].append({
            "k": k,
            "mean_error_deg": mean_radius,
            "median_error_deg": float(np.median(errors)),
            "within_20": float(np.mean(errors < 20.0)),
            "within_30": float(np.mean(errors < 30.0)),
            "p90_error_deg": p90_radius,
            "search_area_mean_radius": float(np.mean([covered_fraction(a, mean_radius, sky) for a in per_k_axes[k]])),
            "search_area_p90_radius": float(np.mean([covered_fraction(a, p90_radius, sky) for a in per_k_axes[k]])),
        })
    summary["standard_poles"] = {
        "axes": int(len(standard)),
        "mean_error_deg": float(standard_errors.mean()),
        "within_20": float(np.mean(standard_errors < 20.0)),
        "p90_error_deg": float(np.percentile(standard_errors, 90)),
        "search_area_mean_radius": covered_fraction(standard, float(standard_errors.mean()), sky),
        "search_area_p90_radius": covered_fraction(standard, float(np.percentile(standard_errors, 90)), sky),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
